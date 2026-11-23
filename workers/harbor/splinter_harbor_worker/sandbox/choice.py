# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for running untrusted agents
# confined on any Linux host, container or not, for its clients. If your team
# needs expertise in kernel-level sandboxing then you can procure our services
# by sending an email to info@swedishembedded.com.
"""Which sandbox a trial runs in: the strongest provider this machine has that enforces everything it asks.

A specification names a provider, or ``auto``. The worker probes once what this machine can run (``probe``): a
reachable Docker daemon, user namespaces, kernel confinement (``splinter-confine``) and ``proot``. ``choose`` takes
the providers in order of strength, ``docker`` > ``userns`` > ``confined`` > ``proot``, and returns the first that is
available and enforces the specification: its network (``proot`` enforces only ``open``), read-only mounts (not
``proot``), and its minimum isolation (``strong``: docker or userns; ``confined``: also confined). A named provider is
checked the same way. Nothing is weakened silently: a specification no available provider can honour is refused with
each provider's reason, and the provider chosen, with what it enforces (``GUARANTEES``), is recorded with the trial.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from functools import cache

from splinter_harbor_worker.errors import UsageError
from splinter_harbor_worker.sandbox.confined import confine_binary, kernel_confines
from splinter_harbor_worker.spec import SandboxSpec

STRENGTH = ("docker", "userns", "confined", "proot")
GUARANTEES = {
    "docker": ("process-isolation", "filesystem-isolation", "network-policy", "read-only-mounts", "resource-limits"),
    "userns": ("process-isolation", "filesystem-isolation", "network-policy", "read-only-mounts"),
    "confined": ("filesystem-writes", "network-policy", "read-only-mounts"),
    "proot": (),
}
ISOLATION = {"strong": ("docker", "userns"), "confined": ("docker", "userns", "confined"), "any": STRENGTH}
_PROBE_TIMEOUT = 15


def _succeeds(argv: list[str]) -> bool:
    try:
        return subprocess.run(argv, capture_output=True, timeout=_PROBE_TIMEOUT, check=False).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@cache
def probe() -> dict[str, str | None]:
    """Each provider, and why this machine cannot run it (``None``: it can)."""
    confines = kernel_confines()
    return {
        "docker": None if shutil.which("docker") and _succeeds(["docker", "info"])
        else "no reachable Docker daemon",
        "userns": None if shutil.which("unshare") and _succeeds(["unshare", "--user", "--map-root-user", "true"])
        else "this host does not allow user namespaces",
        "confined": None if confines.get("landlock_abi") and shutil.which("proot")
        else ("splinter-confine is not installed" if confine_binary() is None
              else "the kernel has no Landlock, or proot is missing"),
        "proot": None if shutil.which("proot") else "proot is not installed",
    }


def refusal(provider: str, sandbox: SandboxSpec, available: dict[str, str | None]) -> str | None:
    """Why ``provider`` cannot run ``sandbox`` here, or ``None``."""
    if (missing := available.get(provider, f"unknown provider {provider!r}")) is not None:
        return missing
    if provider not in ISOLATION[sandbox.isolation]:
        return f"isolation {sandbox.isolation!r} needs {' or '.join(ISOLATION[sandbox.isolation])}"
    if provider == "proot" and sandbox.network != "open":
        return f"it cannot enforce network {sandbox.network!r}"
    if provider == "proot" and any(m.read_only for m in sandbox.mounts):
        return "it cannot make read-only mounts"
    if provider == "confined" and sandbox.network != "open" and not kernel_confines().get("network"):
        return f"this kernel cannot confine the network (Landlock below ABI 4), as {sandbox.network!r} needs"
    if provider == "docker" and sandbox.network == "model-gateway-only" and "SPLINTER_GATEWAY_HOST" not in os.environ:
        return "a gateway-only network on docker needs SPLINTER_GATEWAY_HOST"
    return None


def choose(sandbox: SandboxSpec, available: dict[str, str | None] | None = None,
           allowed: tuple[str, ...] = STRENGTH) -> str:
    """The provider ``sandbox`` runs on: the one it names, or for ``auto`` the strongest of ``allowed`` that honours
    it (``play`` allows only the providers that give a terminal).

    Raises ``UsageError`` naming each candidate's reason when none can."""
    available = probe() if available is None else available
    candidates = tuple(p for p in STRENGTH if p in allowed) if sandbox.provider == "auto" else (sandbox.provider,)
    reasons = {}
    for provider in candidates:
        reason = refusal(provider, sandbox, available)
        if reason is None:
            return provider
        reasons[provider] = reason
    detail = "; ".join(f"{p}: {r}" for p, r in reasons.items())
    raise UsageError(f"no sandbox here can run this specification ({detail})")
