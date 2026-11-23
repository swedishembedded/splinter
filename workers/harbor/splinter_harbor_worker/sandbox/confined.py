# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for running untrusted agents
# confined on any Linux host, container or not, for its clients. If your team
# needs expertise in kernel-level sandboxing then you can procure our services
# by sending an email to info@swedishembedded.com.
"""Harbor environment that runs a task in its own image with ``proot``, confined by the kernel.

For hosts that give neither user namespaces nor a container daemon (a hardened container), where plain
``splinter_harbor_worker.sandbox.proot`` could enforce no network policy and no read-only mount. Every command runs as
``proot`` does there, wrapped in ``splinter-confine`` (Splinter's launcher; Landlock and a seccomp filter, no
privilege), which confines ``proot`` and everything it traces:

* Filesystem writes go only to the sandbox's own image tree, its writable binds (the workspace, ``/logs``), proot's
  own scratch directory and the terminal and null devices. Read-only binds are read-only. The host is readable.
* Network: ``no-network`` leaves no internet socket at all; an allowlist (the model gateway) becomes TCP ``connect``
  to the allowed ports only, on any address, and no UDP or IPv6; ``public`` is the host's network, its loopback
  included (the namespace provider keeps a public task off the host's loopback; this one cannot).

What it does not give: process isolation (the host's processes are visible, as under plain proot) and read
confinement. ``GUARANTEES`` names what it enforces, and the worker records it with every trial.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from functools import cache
from pathlib import Path, PurePosixPath

from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.task.config import NetworkMode

from splinter_harbor_worker.sandbox.proot import ProotEnvironment
from splinter_harbor_worker.sandbox.userns import UserNamespaceEnvironment

CONFINE_BIN_ENV = "SPLINTER_CONFINE_BIN"
# Device nodes a command writes to under any sandbox.
_WRITABLE_DEVICES = ("/dev/null", "/dev/zero", "/dev/full", "/dev/tty", "/dev/pts", "/dev/ptmx", "/dev/shm")
GUARANTEES = ("filesystem-writes", "read-only-mounts", "network-policy")


def confine_binary() -> str | None:
    """The ``splinter-confine`` the worker runs: ``SPLINTER_CONFINE_BIN``, else the one on PATH."""
    return os.environ.get(CONFINE_BIN_ENV) or shutil.which("splinter-confine")


@cache
def kernel_confines() -> dict:
    """What this kernel enforces, as ``splinter-confine --probe`` reports it (``landlock_abi``, ``network``); empty
    without the launcher."""
    binary = confine_binary()
    if binary is None:
        return {}
    result = subprocess.run([binary, "--probe"], capture_output=True, text=True, check=False, timeout=10)
    return json.loads(result.stdout) if result.returncode == 0 else {}


class ConfinedEnvironment(ProotEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        binary = confine_binary()
        if binary is None:
            raise FileNotFoundError("the confined provider needs splinter-confine (installed with splinter, or "
                                    f"{CONFINE_BIN_ENV})")
        self._confine = binary

    @staticmethod
    def type() -> str:
        return "confined"

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(mounted=True, disable_internet=True, network_allowlist=True,
                                       network_allowlist_ipv4_addresses=True, dynamic_network_policy=True)

    def _extra_mounts(self) -> list[tuple[Path, PurePosixPath, bool]]:
        # Read-only binds are enforced by the confinement, so none is refused here (proot's refusal is skipped).
        return UserNamespaceEnvironment._extra_mounts(self)

    def _network_flag(self) -> str:
        policy = self._network_policy
        if policy.network_mode == NetworkMode.PUBLIC:
            return "open"
        spec = self._network_spec()
        ports = sorted({port for _, port in spec["allow"]})
        return f"tcp:{','.join(str(p) for p in ports)}" if ports else "none"

    def _argv(self, spec: dict) -> tuple[list[str], dict[str, str]]:
        argv, env = super()._argv(spec)
        scratch = self._sandbox / "proot-tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        writable = [str(self._image_root), str(scratch), *_WRITABLE_DEVICES]
        writable += [host for host, _, read_only in spec["binds"] if not read_only]
        confine = [self._confine, "--read-only", "/"]
        for path in writable:
            confine += ["--read-write", path]
        confine += ["--network", self._network_flag(), "--"]
        return [*confine, *argv], {**env, "PROOT_TMP_DIR": str(scratch)}
