# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for running published agent benchmarks
# on hosts without a container daemon for its clients. If your team needs expertise
# in hermetic benchmark environments then you can procure our services by sending
# an email to info@swedishembedded.com.
"""Hold a sandbox's private, writable copy of its image for as long as the sandbox lives.

Run as ``unshare --user --map-root-user --mount --pid --fork --kill-child -- python3 -I userns_keeper.py`` with one
JSON line on standard
input: ``{"mountpoint": DIR, "layers": [ARCHIVE, ...], "size_mb": N, "preload": SHIM, "in_memory": BOOL}``. It changes
into ``mountpoint`` (with ``in_memory``, first a tmpfs of at most ``size_mb`` mounted there in its own mount namespace,
the task's storage limit) and unpacks the image's archives there in order (OCI layer semantics, ``rootfs/layers.py``), makes the tree usable in
a namespace that maps one identity, prints ``ready`` and waits until its standard input closes. Every command of the
sandbox enters this user, mount and PID namespace (``nsenter``) before building its own mount namespace
(``userns_launch.py``), so the
copy is the sandbox's root filesystem: writable like a container's, persistent across commands, invisible to every
other process of the host's commands that do not enter it, and gone when the sandbox stops (a tmpfs with the
namespace when the keeper exits; a directory on disk removed by the environment, ``userns.remove_tree``). The keeper is the init of the sandbox's PID namespace, as a container's init is: a
process a command leaves running keeps running after that command (a server the agent starts and the tests query),
orphans are reaped here, and everything in the namespace ends with the keeper.

Why not overlayfs over a shared unpacked tree: a lower layer on a shared filesystem (Lustre) is refused, and that disk's
inode quota cannot hold unpacked images anyway; the cache keeps only archives (``rootfs/images.py``). So each sandbox
unpacks its own copy: on the local disk the trial directory is on, or in memory when that directory is on a network
filesystem. Memory is the last choice: hundreds of concurrent copies of a few hundred megabytes exhausted a node's
memory limit while its local disk had room for all of them. On disk ``size_mb`` is not enforced.

Standard library and ``splinter_harbor_worker.rootfs.layers`` only.
"""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import signal
import sys
from pathlib import Path

MS_NOSUID, MS_NODEV, MS_REC, MS_PRIVATE = 2, 4, 16384, 1 << 18
PR_SET_PDEATHSIG, SIGKILL = 1, 9
# apt drops privileges to its _apt user for downloads, which a namespace mapping one identity cannot express.
APT_SANDBOX = "etc/apt/apt.conf.d/00bench-sandbox"
# The chown shim (rootfs/nochown.c), preloaded into every dynamically linked program of the copy: root of a
# namespace mapping one identity cannot chown to anyone else (the kernel answers EINVAL), which tar, dpkg and pip do.
PRELOAD = "usr/lib/bench/chown-tolerant.so"
# Bound read-only from the host by every command (name resolution, as a container runtime provides them).
HOST_FILES = ("etc/resolv.conf", "etc/hosts")

libc = ctypes.CDLL(None, use_errno=True)


def _mount(src: str | None, dst: str, fstype: str | None, flags: int, data: str | None = None) -> None:
    if libc.mount(src and src.encode(), dst.encode(), fstype and fstype.encode(), flags, data and data.encode()) != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"mount {fstype or src} -> {dst}: {os.strerror(err)}")


def prepare(root: str, preload: str | None) -> None:
    """Adjust the private copy for a namespace with one identity and for host-provided files."""
    if preload:
        os.makedirs(os.path.join(root, os.path.dirname(PRELOAD)), exist_ok=True)
        shutil.copyfile(preload, os.path.join(root, PRELOAD))
        os.chmod(os.path.join(root, PRELOAD), 0o755)
        with open(os.path.join(root, "etc", "ld.so.preload"), "a") as handle:
            handle.write(f"/{PRELOAD}\n")
    if os.path.isdir(os.path.join(root, "etc", "apt", "apt.conf.d")):
        with open(os.path.join(root, APT_SANDBOX), "w") as handle:
            handle.write('APT::Sandbox::User "root";\n')
    for relative in HOST_FILES:
        path = os.path.join(root, relative)
        if os.path.islink(path) or not os.path.isfile(path):
            if os.path.lexists(path):
                os.unlink(path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "w").close()


def _reap(*_) -> None:
    """As init of the PID namespace, collect every orphan that exits."""
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


def mount_tmpfs(mountpoint: str, size_mb: int) -> None:
    """A private tmpfs of at most ``size_mb`` at ``mountpoint``, in this process's own mount namespace."""
    _mount(None, "/", None, MS_REC | MS_PRIVATE)
    _mount("tmpfs", mountpoint, "tmpfs", MS_NOSUID | MS_NODEV, f"mode=0755,size={int(size_mb)}m")


def populate(root: str, layers: list, preload: str | None) -> None:
    """Fill the tree at ``root`` from the image's archives and adjust it, from inside it: the keeper changes into
    ``root`` and every path after that is relative. Resolving ``root``'s absolute path again for each of an image's
    files costs a metadata round trip on a network filesystem the mountpoint sits on (on Lustre 76 s per sandbox with
    64 starting at once, against 2.4 s relative), and the tree is the keeper's to hold anyway."""
    from splinter_harbor_worker.rootfs import layers as layer_archives  # the bench checkout this script belongs to

    os.chdir(root)
    layer_archives.unpack_layers([Path(layer) for layer in layers], Path("."))
    prepare(".", preload)


def main() -> int:
    spec = json.loads(sys.stdin.readline())
    libc.prctl(PR_SET_PDEATHSIG, SIGKILL, 0, 0, 0)
    root = spec["mountpoint"]
    if spec.get("in_memory", True):
        mount_tmpfs(root, spec["size_mb"])
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    try:
        populate(root, spec["layers"], spec.get("preload"))
    except Exception as exc:
        print(f"unpacking the image: {exc}", file=sys.stderr, flush=True)
        return 1
    signal.signal(signal.SIGCHLD, _reap)
    print("ready", flush=True)
    sys.stdin.read()  # until the environment stops (or its process dies)
    return 0


if __name__ == "__main__":
    sys.exit(main())
