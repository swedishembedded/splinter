# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for testing agent execution
# backends against stub models and tiny tasks with no network, for its
# clients. If your team needs expertise in hermetic agent test harnesses then
# you can procure our services by sending an email to info@swedishembedded.com.
"""Fixtures: tiny Harbor tasks, a root filesystem the proot sandbox can run them in, a stub model server, and the
contract's JSON Schema. Nothing here touches the network."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
from collections.abc import Iterator
from pathlib import Path

import jsonschema
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from stub_model import StubModel  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
SCHEMA_DIR = REPO / "schemas" / "factory"
# Tools the task scripts and the fake agents need inside the image.
TOOLS = ("bash", "sh", "cat", "id", "ls", "touch", "mkdir", "grep", "cp", "env", "printf", "sleep", "chmod", "tr",
         "dirname", "test", "head")


def attempt_validator() -> jsonschema.Draft202012Validator:
    path = SCHEMA_DIR / "splinter.attempt.v1.schema.json"
    assert path.is_file(), f"the contract's schema {path} is missing; the tests validate every report against it"
    schema = json.loads(path.read_text())
    return jsonschema.Draft202012Validator(schema)


@pytest.fixture(scope="session")
def validator() -> jsonschema.Draft202012Validator:
    return attempt_validator()


def _make_root(root: Path) -> None:
    for sub in ("usr/bin", "usr/lib", "app", "etc", "tmp", "root", "proc", "dev"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    for link, target in (("bin", "usr/bin"), ("lib", "usr/lib"), ("sbin", "usr/bin")):
        if not (root / link).exists():
            os.symlink(target, root / link)
    (root / "etc" / "passwd").write_text("root:x:0:0:root:/root:/bin/bash\n")
    (root / "etc" / "group").write_text("root:x:0:\n")
    for tool in TOOLS:
        found = shutil.which(tool)
        if found is None:
            continue
        host = Path(found).resolve()
        target = root / "usr/bin" / tool
        if not target.exists():
            shutil.copyfile(host, target)
            target.chmod(0o755)
        listing = subprocess.run(["ldd", str(host)], capture_output=True, text=True, check=False).stdout
        for line in listing.splitlines():
            logical = next((w for w in line.split() if w.startswith("/")), None)
            if logical is None:
                continue
            path = root / "usr" / logical.removeprefix("/usr/").removeprefix("/")
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                shutil.copyfile(Path(logical).resolve(), path)
                path.chmod(0o755)
    if (root / "usr/lib64").is_dir() and not (root / "lib64").exists():
        os.symlink("usr/lib64", root / "lib64")


@pytest.fixture(scope="session")
def image_archive(tmp_path_factory) -> Path:
    """A root filesystem of host tools as one image archive."""
    work = tmp_path_factory.mktemp("image")
    root = work / "root"
    _make_root(root)
    archive = work / "image.tar"
    with tarfile.open(archive, "w") as out:
        for entry in sorted(root.iterdir()):
            out.add(entry, arcname=entry.name)
    return archive


@pytest.fixture
def local_images(monkeypatch, image_archive: Path):
    """The task's image is the local archive, never a registry pull."""
    from splinter_harbor_worker.rootfs.images import TaskImage
    import splinter_harbor_worker.sandbox.proot as proot
    import splinter_harbor_worker.sandbox.userns as userns

    image = TaskImage((image_archive,), {"PATH": "/usr/bin"}, "/app",
                      {"source": "pull", "reference": "local/toy:1", "manifest_digest": "sha256:" + "d" * 64})

    def ensure(docker_image, environment_dir, cache, **kwargs):
        return image

    monkeypatch.setattr(proot, "ensure_task_image", ensure)
    monkeypatch.setattr(userns, "ensure_task_image", ensure)
    return image


TASK_TOML = """\
schema_version = "1.4"

[task]
name = "splinter/{name}"
version = "1"

[agent]
timeout_sec = {agent_timeout}

[verifier]
timeout_sec = 60

[environment]
docker_image = "local/toy:1"
"""

TEST_SH = """\
#!/bin/bash
mkdir -p /logs/verifier
if grep -q expected /app/answer.txt 2>/dev/null; then echo 1 > /logs/verifier/reward.txt
else echo 0 > /logs/verifier/reward.txt; fi
"""

SOLVE_SH = """\
#!/bin/bash
echo expected > /app/answer.txt
"""


def make_task(root: Path, task_id: str, *, agent_timeout: int = 60) -> Path:
    """A Harbor task below ``root/tasks/<task_id>``: the agent's job is to write ``expected`` to /app/answer.txt."""
    directory = root / "tasks" / task_id
    name = task_id.replace("/", "-")
    (directory / "environment").mkdir(parents=True)
    (directory / "tests").mkdir()
    (directory / "solution").mkdir()
    (directory / "task.toml").write_text(TASK_TOML.format(name=name, agent_timeout=agent_timeout))
    (directory / "instruction.md").write_text("Write the word expected into /app/answer.txt.\n")
    (directory / "environment" / "Dockerfile").write_text("FROM scratch\nWORKDIR /app\n")
    (directory / "tests" / "test.sh").write_text(TEST_SH)
    (directory / "solution" / "solve.sh").write_text(SOLVE_SH)
    for script in (directory / "tests" / "test.sh", directory / "solution" / "solve.sh"):
        script.chmod(0o755)
    return directory


def make_spec(task_id: str, *, agent: str = "oracle", slot: int = 0, attempt: int = 0, pass_: str = "local",
              provider: str = "proot", network: str = "open", args: list[str] | None = None, env: dict | None = None, files: list[dict] | None = None,
              system_prompt: dict | None = None, agent_harness: dict | None = None, **fields) -> dict:
    """An ExecutionSpec of the contract's shape (an optional ``model``, ``agent`` args and so on come as ``fields``)."""
    spec = {
        "schema": "splinter.execution/1",
        "slot": {"pass": pass_, "task": task_id, "slot": slot},
        "attempt": attempt,
        "task": {"id": task_id, "name": f"splinter/{task_id.replace('/', '-')}", "digest": "blake3:" + "a" * 64},
        "task_dir": f"tasks/{task_id}",
        "agent": {"name": agent, **({"args": args} if args else {}), **({"env": env} if env else {}),
                  **({"files": files} if files else {}),
                  **({"system_prompt": system_prompt} if system_prompt else {}),
                  **({"harness": agent_harness} if agent_harness else {})},
        "sandbox": {"provider": provider, "network": network},
        "seed": 7,  # seed: the specification carries one; the worker only forwards it
        "timeout_policy": "graded",
    }
    for key, value in fields.items():
        spec[key] = {**spec[key], **value} if isinstance(spec.get(key), dict) and isinstance(value, dict) else value
    return spec


def write_specs(path: Path, specs: list[dict]) -> Path:
    path.write_text("".join(json.dumps(spec) + "\n" for spec in specs))
    return path


@pytest.fixture
def stub_model() -> Iterator[StubModel]:
    with StubModel() as server:
        yield server


@pytest.fixture(autouse=True)
def _no_ambient_worker_environment(monkeypatch) -> None:
    for name in ("SPLINTER_SVEN_BIN", "SPLINTER_CLAUDE_BIN", "SPLINTER_CODEX_BIN", "SPLINTER_GATEWAY_HOST",
                 "SPLINTER_CONFINE_BIN"):
        monkeypatch.delenv(name, raising=False)


def confine_binary() -> Path:
    """This checkout's ``splinter-confine`` (``make build/crate/splinter-sandbox``); a test that needs it fails, never
    skips, when it is not built."""
    root = Path(__file__).resolve().parents[3]
    for profile in ("release", "debug"):
        if (found := root / "target" / profile / "splinter-confine").is_file():
            return found
    raise AssertionError("build splinter-confine first: make build/crate/splinter-sandbox")


@pytest.fixture
def confine(monkeypatch) -> Path:
    binary = confine_binary()
    monkeypatch.setenv("SPLINTER_CONFINE_BIN", str(binary))
    from splinter_harbor_worker.sandbox.confined import kernel_confines
    from splinter_harbor_worker.sandbox.choice import probe
    kernel_confines.cache_clear()
    probe.cache_clear()
    return binary
