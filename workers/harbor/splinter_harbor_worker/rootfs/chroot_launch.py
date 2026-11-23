# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for reproducible container-image based
# build and test environments on hosts without a container daemon for its clients.
# If your team needs expertise in hermetic benchmark environments then you can
# procure our services by sending an email to info@swedishembedded.com.
"""Run one command as root inside an image tree, in fresh user, mount, pid, ipc and uts
namespaces. This is what executes a Dockerfile's ``RUN`` steps.

    unshare --user --map-root-user --mount --pid --fork --ipc --uts -- \\
        python3 -I chroot_launch.py SPEC_JSON

The tree is pivoted into (its own mount, ``/proc`` and the host's ``/dev`` added, the
host's ``/etc/resolv.conf`` bound over the tree's so names resolve), so nothing of the
host is reachable but the network, which is shared: steps fetch packages. The command
is root of a namespace that maps one identity, which is the invoking user outside it.

``spec``: ``root`` (the tree), ``cwd``, ``env``, ``argv``. Standard library only; the
mount helpers are those of ``userns_launch.py``, loaded by path because this script runs
with ``-I``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import sys


def _helpers():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sandbox", "userns_launch.py")
    module_spec = importlib.util.spec_from_file_location("userns_launch", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def main() -> int:
    spec = json.loads(sys.argv[1])
    root = spec["root"]
    sandbox = _helpers()
    sandbox.mount(None, "/", None, sandbox.MS_REC | sandbox.MS_PRIVATE)
    sandbox.mount(root, root, None, sandbox.MS_BIND | sandbox.MS_REC)  # pivot_root needs a mount point
    os.makedirs(root + "/proc", exist_ok=True)
    sandbox.mount("proc", root + "/proc", "proc", sandbox.MS_NOSUID | sandbox.MS_NODEV)
    sandbox.bind("/dev", root + "/dev", read_only=False)
    sandbox.bind("/etc/resolv.conf", root + "/etc/resolv.conf", read_only=True)
    os.makedirs(root + "/.oldroot", exist_ok=True)
    pivot = sandbox.PIVOT_ROOT[platform.machine()]
    sandbox._check(sandbox.libc.syscall(pivot, root.encode(), (root + "/.oldroot").encode()), "pivot_root")
    os.chdir("/")
    sandbox._check(sandbox.libc.umount2(b"/.oldroot", sandbox.MNT_DETACH), "umount old root")
    os.rmdir("/.oldroot")
    os.chdir(spec["cwd"])
    os.execvpe(spec["argv"][0], spec["argv"], spec["env"])
    return 127


if __name__ == "__main__":
    sys.exit(main())
