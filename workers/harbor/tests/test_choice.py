# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for running untrusted agents
# confined on any Linux host, container or not, for its clients. If your team
# needs expertise in kernel-level sandboxing then you can procure our services
# by sending an email to info@swedishembedded.com.
"""Which sandbox a trial gets: the strongest the machine has that honours the specification, never a weaker one in
silence."""

from __future__ import annotations

import pytest

from splinter_harbor_worker.errors import UsageError
from splinter_harbor_worker.sandbox.choice import choose
from splinter_harbor_worker.spec import Mount, SandboxSpec

EVERYTHING = {"docker": None, "userns": None, "confined": None, "proot": None}
CONTAINER_HOST = {"docker": "no reachable Docker daemon", "userns": "this host does not allow user namespaces",
                  "confined": None, "proot": None}


def sandbox(**fields) -> SandboxSpec:
    return SandboxSpec(**{"provider": "auto", "network": "open", **fields})


def test_auto_takes_the_strongest_provider_the_machine_has(monkeypatch) -> None:
    monkeypatch.setenv("SPLINTER_GATEWAY_HOST", "10.0.0.1")
    assert choose(sandbox(network="model-gateway-only"), EVERYTHING) == "docker"
    assert choose(sandbox(network="model-gateway-only"), {**EVERYTHING, "docker": "none"}) == "userns"
    assert choose(sandbox(network="model-gateway-only"), CONTAINER_HOST) == "confined"


def test_a_requirement_skips_the_providers_that_cannot_honour_it(monkeypatch) -> None:
    only_proot = {**CONTAINER_HOST, "confined": "the kernel has no Landlock"}
    assert choose(sandbox(), only_proot) == "proot"
    with pytest.raises(UsageError, match="proot: it cannot enforce network 'none'"):
        choose(sandbox(network="none"), only_proot)
    mount = Mount(source="kb", digest="sha256:" + "a" * 64, target="/kb", read_only=True)
    with pytest.raises(UsageError, match="proot: it cannot make read-only mounts"):
        choose(sandbox(mounts=[mount]), only_proot)


def test_strong_isolation_is_never_downgraded_to_confinement() -> None:
    with pytest.raises(UsageError) as refused:
        choose(sandbox(isolation="strong"), CONTAINER_HOST)
    assert "docker: no reachable Docker daemon" in str(refused.value)
    assert "confined: isolation 'strong' needs docker or userns" in str(refused.value)
    assert choose(sandbox(isolation="confined"), CONTAINER_HOST) == "confined"


def test_a_named_provider_is_checked_not_replaced() -> None:
    with pytest.raises(UsageError, match="userns: this host does not allow user namespaces"):
        choose(sandbox(provider="userns"), CONTAINER_HOST)
    assert choose(sandbox(provider="proot"), CONTAINER_HOST) == "proot"
