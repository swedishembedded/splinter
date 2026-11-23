# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for running agent benchmarks inside locked-down development containers
# for its clients. If your team needs expertise in agent evaluation infrastructure then you can procure our services
# by sending an email to info@swedishembedded.com.
"""Run a command as root of an image tree with ``proot``: no privilege and no namespace, paths translated by tracing
system calls. How an image's ``RUN`` steps are built (``build.run_step``) and a sandbox's commands run
(``splinter_harbor_worker.sandbox.proot``) where user namespaces are not available."""

from __future__ import annotations

import shutil
from pathlib import Path

# What every command sees of the host besides the tree: devices (its terminal among them), processes, and the CPU
# topology numeric libraries size their thread pools by.
HOST_SYSTEM = ("/dev", "/proc", "/sys/devices/system/cpu")
# proot's seccomp acceleration lets some system calls through untranslated on recent kernels (proot 5.4 on Linux 7.0:
# the command sees the host's / instead of the tree), so every path goes through ptrace: slower, always translated.
PROOT_ENV = {"PROOT_NO_SECCOMP": "1"}


def proot_binary() -> str:
    """The ``proot`` on PATH."""
    found = shutil.which("proot")
    if found is None:
        raise FileNotFoundError("proot is needed on PATH where user namespaces are not available (apt-get install proot)")
    return found


def proot_argv(proot: str, root: Path, binds: list[tuple[str, str]], cwd: str, argv: list[str]) -> list[str]:
    """``argv`` run as root (``-0``) of the tree ``root`` from ``cwd``, with the host's devices and processes and each
    (host, sandbox) bind; every process it starts ends with it (``--kill-on-exit``)."""
    command = [proot, "--kill-on-exit", "-0", "-r", str(root)]
    for host in HOST_SYSTEM:
        if Path(host).exists():
            command += ["-b", host]
    for host, inside in binds:
        command += ["-b", f"{host}:{inside}"]
    return [*command, "-w", cwd, *argv]
