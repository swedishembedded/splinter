# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for testing agent execution
# backends against stub models and tiny tasks with no network, for its
# clients. If your team needs expertise in hermetic agent test harnesses then
# you can procure our services by sending an email to info@swedishembedded.com.
"""The sandbox providers' contracts that can be checked here. The proot provider runs for real in the other test files;
the user-namespace provider cannot start in this container (no user namespaces), so only what it decides before any
process starts is checked: the network policy it hands its launcher, and what it refuses."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from harbor.models.task.config import EnvironmentConfig, NetworkPolicy
from harbor.models.trial.paths import TrialPaths

from splinter_harbor_worker.sandbox.proot import ProotEnvironment
from splinter_harbor_worker.sandbox.userns import UserNamespaceEnvironment


def make_env(tmp_path: Path, cls, *, network: str = "public", allowed: list[str] | None = None, mounts=None,
             **kwargs):
    (tmp_path / "environment").mkdir(exist_ok=True)
    trial = TrialPaths(tmp_path / "trial")
    trial.mkdir()
    policy = NetworkPolicy(network_mode=network, allowed_hosts=allowed or [])
    return cls(environment_dir=tmp_path / "environment", environment_name="t", session_id="t__env", trial_paths=trial,
               task_env_config=EnvironmentConfig(docker_image="local/toy:1", network_mode=network),
               network_policy=policy, images=str(tmp_path / "cache"), mounts=mounts, **kwargs)


def test_proot_cannot_enforce_a_network_policy_and_harbor_refuses_it(tmp_path) -> None:
    with pytest.raises(ValueError, match="not supported by proot"):
        make_env(tmp_path, ProotEnvironment, network="no-network")


def test_proot_refuses_a_read_only_mount(tmp_path) -> None:
    (tmp_path / "data").mkdir()
    mount = {"type": "bind", "source": str(tmp_path / "data"), "target": "/data", "read_only": True}
    env = make_env(tmp_path, ProotEnvironment, mounts=[mount])
    with pytest.raises(ValueError, match="read-only mounts"):
        env._extra_mounts()


def test_proot_keeps_every_path_translated_by_ptrace(tmp_path) -> None:
    env = make_env(tmp_path, ProotEnvironment)
    _, environment = env._argv({"binds": [], "cwd": "/", "command": "true", "env": {"A": "1"}})
    assert environment["PROOT_NO_SECCOMP"] == "1" and environment["A"] == "1"


def test_proot_refuses_a_proot_that_would_run_commands_on_the_host(tmp_path, local_images, monkeypatch) -> None:
    monkeypatch.setattr(ProotEnvironment, "_argv",
                        lambda self, spec: (["/bin/sh", "-c", spec["command"]], {"PATH": "/usr/bin:/bin"}))
    env = make_env(tmp_path, ProotEnvironment)
    with pytest.raises(RuntimeError, match="does not run commands in the image"):
        asyncio.run(env.start(force_build=False))


def test_a_sandbox_without_its_own_image_is_refused(tmp_path) -> None:
    (tmp_path / "environment").mkdir()
    trial = TrialPaths(tmp_path / "trial")
    trial.mkdir()
    for cls in (ProotEnvironment, UserNamespaceEnvironment):
        with pytest.raises(ValueError, match="own image"):
            cls(environment_dir=tmp_path / "environment", environment_name="t", session_id="s", trial_paths=trial,
                task_env_config=EnvironmentConfig(network_mode="public"),
                network_policy=NetworkPolicy(network_mode="public"), images=str(tmp_path / "cache"))


def test_userns_admits_only_the_gateway_port_on_the_gateway_address(tmp_path) -> None:
    env = make_env(tmp_path, UserNamespaceEnvironment, network="allowlist", allowed=["127.0.0.1"], allow_ports=8123)
    assert env._network_spec() == {"allow": [["127.0.0.1", 8123]]}


def test_userns_allows_no_destination_without_network_and_loopback_ports_on_a_public_network(tmp_path) -> None:
    (tmp_path / "n").mkdir()
    none = make_env(tmp_path / "n", UserNamespaceEnvironment, network="no-network")
    assert none._network_spec() == {"allow": []}
    (tmp_path / "p").mkdir()
    public = make_env(tmp_path / "p", UserNamespaceEnvironment, network="public", allow_ports=8123)
    spec = public._network_spec()
    assert spec["public"] is True and spec["allow"] == [["127.0.0.1", 8123]]


def test_userns_refuses_an_allowlist_without_ports(tmp_path) -> None:
    env = make_env(tmp_path, UserNamespaceEnvironment, network="allowlist", allowed=["127.0.0.1"])
    with pytest.raises(ValueError, match="allow_ports"):
        env._network_spec()


def test_a_task_that_imports_a_host_variable_is_refused(tmp_path) -> None:
    (tmp_path / "environment").mkdir()
    (tmp_path / "task.toml").write_text('[verifier.env]\nSECRET = "${SOME_HOST_SECRET}"\n')
    trial = TrialPaths(tmp_path / "trial")
    trial.mkdir()
    with pytest.raises(ValueError, match="SOME_HOST_SECRET"):
        ProotEnvironment(environment_dir=tmp_path / "environment", environment_name="t", session_id="s",
                         trial_paths=trial, task_env_config=EnvironmentConfig(docker_image="x", network_mode="public"),
                         network_policy=NetworkPolicy(network_mode="public"), images=str(tmp_path / "cache"))


def test_a_confined_command_reads_only_the_system_its_tree_and_its_binds(tmp_path) -> None:
    from splinter_harbor_worker.sandbox.confined import SYSTEM_READ, confinement_args

    image, scratch = tmp_path / "image", tmp_path / "proot-tmp"
    binds = [("/srv/kb", "/kb", True), (str(tmp_path / "logs"), "/logs", False)]
    args = confinement_args(image, scratch, binds, "/usr/bin/proot")
    reads = [args[i + 1] for i, a in enumerate(args) if a == "--read-only"]
    writes = [args[i + 1] for i, a in enumerate(args) if a == "--read-write"]
    assert "/" not in reads and not any(p in reads for p in ("/home", "/root", "/tmp", "/data", "/var"))
    assert set(SYSTEM_READ) <= set(reads) and "/srv/kb" in reads
    assert {str(image), str(scratch), str(tmp_path / "logs")} <= set(writes) and "/srv/kb" not in writes
