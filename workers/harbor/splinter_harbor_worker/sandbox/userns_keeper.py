# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for running published agent benchmarks
# on hosts without a container daemon for its clients. If your team needs expertise
# in hermetic benchmark environments then you can procure our services by sending
# an email to info@swedishembedded.com.
"""Hold a sandbox's private, writable copy of its image for as long as the sandbox lives.

Run as ``unshare --user --map-root-user --mount --pid --fork --kill-child -- python3 -I userns_keeper.py`` with one
JSON line on standard input: ``{"mountpoint": DIR, "layers": [ARCHIVE, ...], "size_mb": N, "preload": SHIM,
"in_memory": BOOL, "lower_cache": DIR or null, "overlay": DIR, "disk_reason": TEXT or null}``. It builds the copy at
``mountpoint`` in one of three ways, makes the tree usable in a namespace that maps one identity, prints ``ready``
followed by a JSON report of how the copy was made (``{"mode": ...}``) and waits until its standard input closes:

* ``overlay`` (``lower_cache`` given): the image's layers but the last are unpacked once per node into a read-only
  tree under ``lower_cache``, keyed by those layers' identity (``lower_key``), and every sandbox of the image mounts
  an overlay in its own mount namespace: that tree below, its own upper and work directories under ``overlay``
  above. The last layer (a Dockerfile task's own files) is unpacked into the overlay, so its deletions of lower files
  become whiteouts of this sandbox alone. Hundreds of sandboxes starting at once each unpacking the same few hundred
  megabytes of installation made the local disk the bottleneck of a run; with the overlay each writes only the task's
  layer. An image of one layer is shared whole (the attempts of one task).
* ``disk``: every layer unpacked into ``mountpoint``, a directory on the local disk the trial directory is on. Used
  when the cache is on a network filesystem (``disk_reason``) or the overlay cannot be mounted (another filesystem,
  a kernel that refuses it): the report says why.
* ``memory`` (``in_memory``): a tmpfs of at most ``size_mb`` mounted at ``mountpoint``, only when the trial directory
  is on a network filesystem: hundreds of copies in memory exhausted a node's memory limit while its local disk had
  room for all of them.

Only ``memory`` enforces ``size_mb``; on disk, overlay or not, the copy is not bounded.

Every command of the sandbox enters this user, mount and PID namespace (``nsenter``) before building its own mount
namespace (``userns_launch.py``), so the copy is the sandbox's root filesystem: writable like a container's,
persistent across commands, invisible to every other process of the host's commands that do not enter it, and gone
when the sandbox stops (the tmpfs and the overlay mount with the namespace when the keeper exits; directories on disk
removed by the environment, ``userns.remove_image_copy``). The shared tree stays for the next sandbox. The keeper is
the init of the sandbox's PID namespace, as a container's init is: a process a command leaves running keeps running
after that command (a server the agent starts and the tests query), orphans are reaped here, and everything in the
namespace ends with the keeper.

Standard library and ``splinter_harbor_worker.rootfs`` (``layers``, ``locks``) only.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import shutil
import signal
import sys
import tempfile
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
# How a shared tree is unpacked and laid out: changing either changes every key, so an old tree is never reused.
LOWER_FORMAT = 1
# Characters the overlay's mount options cannot carry in a directory name (the kernel reads them as separators).
OVERLAY_SEPARATORS = (",", ":", "\\")

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


def split_layers(layers: list) -> tuple[list, list]:
    """(the layers shared through the node's tree, the layers each sandbox unpacks over it): all but the last, which
    is a Dockerfile task's own files or a pulled image's top layer; an image of one layer is shared whole."""
    return (layers[:-1], layers[-1:]) if len(layers) > 1 else (list(layers), [])


def lower_key(layers: list) -> str:
    """The identity of a set of lower layers: each archive's name (a blob's digest, or a build's name, which carries
    the digest of its recipe and context), size and modification time, so an archive rebuilt under its old name makes
    a new tree instead of reusing one unpacked from the old archive."""
    entries = []
    for layer in layers:
        status = os.stat(layer)
        entries.append([os.path.basename(layer), status.st_size, status.st_mtime_ns])
    return hashlib.sha256(json.dumps([LOWER_FORMAT, entries]).encode()).hexdigest()[:32]


def ensure_lower(cache: Path, layers: list) -> tuple[Path, bool]:
    """(the unpacked tree of ``layers`` under ``cache``, whether this call unpacked it). A tree is unpacked at most
    once per cache, however many sandboxes start at once: under an exclusive lock, into a temporary directory renamed
    into place when complete, so a tree that exists is whole and the others wait on the lock and reuse it. It is
    never written afterwards (an overlay never modifies its lower layer) and stays for the next sandbox."""
    from splinter_harbor_worker.rootfs.layers import remove_tree, unpack_layers
    from splinter_harbor_worker.rootfs.locks import locked

    key = lower_key(layers)
    tree = cache / key
    if tree.is_dir():
        return tree / "root", False
    with locked(cache / f"{key}.lock"):
        if tree.is_dir():
            return tree / "root", False
        for stale in cache.glob(f"{key}.partial-*"):  # a keeper killed while unpacking; only the lock holder writes
            remove_tree(stale)
        partial = Path(tempfile.mkdtemp(prefix=f"{key}.partial-", dir=cache))
        unpack_layers([Path(layer) for layer in layers], partial / "root")
        (partial / "layers.json").write_text(json.dumps([str(layer) for layer in layers], indent=2) + "\n")
        os.chmod(partial, 0o755)
        os.rename(partial, tree)
    return tree / "root", True


def mount_overlay(mountpoint: str, lower: Path, overlay: str) -> None:
    """An overlay at ``mountpoint`` in this process's own mount namespace: ``lower`` read-only below, the sandbox's
    own upper and work directories under ``overlay`` above. ``userxattr``: an unprivileged mount keeps the marks of
    whiteouts and opaque directories in ``user.overlay.*`` (``trusted.*`` is the host root's); without it deleting a
    directory of the lower tree fails with EIO. ``volatile``: the upper directory is the sandbox's scratch, never
    mounted again; without it every unmount (the keeper's namespace ending) syncs the whole filesystem under it and
    waits for all of the node's dirty pages, which a busy run keeps at tens of gigabytes."""
    upper, work = os.path.join(overlay, "upper"), os.path.join(overlay, "work")
    os.makedirs(upper, exist_ok=True)
    os.makedirs(work, exist_ok=True)
    _mount(None, "/", None, MS_REC | MS_PRIVATE)
    _mount("overlay", mountpoint, "overlay", MS_NOSUID | MS_NODEV,
           f"lowerdir={lower},upperdir={upper},workdir={work},userxattr,volatile")


def _overlay_refusal(spec: dict) -> str | None:
    """Why an overlay cannot name the directories ``spec`` gives, or None."""
    for path in (spec["lower_cache"], spec["overlay"]):
        if any(separator in os.path.abspath(path) for separator in OVERLAY_SEPARATORS):
            return f"{path}: overlay options cannot name a directory holding any of {' '.join(OVERLAY_SEPARATORS)}"
    return None


def build_copy(spec: dict) -> dict:
    """Make the sandbox's copy at ``spec["mountpoint"]``; the report of how (the module's three modes)."""
    root, layers, preload = spec["mountpoint"], spec["layers"], spec.get("preload")
    if spec.get("in_memory", True):
        mount_tmpfs(root, spec["size_mb"])
        populate(root, layers, preload)
        return {"mode": "memory"}
    reason = spec.get("disk_reason") or "no shared tree was asked for"
    if spec.get("lower_cache"):
        shared, own = split_layers(layers)
        try:
            refusal = _overlay_refusal(spec)
            if refusal:
                raise OSError(errno.EINVAL, refusal)
            lower, created = ensure_lower(Path(spec["lower_cache"]), shared)
            mount_overlay(root, lower, spec["overlay"])
        except OSError as exc:
            reason = f"the overlay over the shared tree could not be mounted ({exc}); the whole image is unpacked"
        else:
            populate(root, own, preload)  # after the mount: the keeper changes into the overlay, not the directory
            return {"mode": "overlay", "lower": str(lower), "lower_created": created}
    populate(root, layers, preload)
    return {"mode": "disk", "reason": reason}


def main() -> int:
    spec = json.loads(sys.stdin.readline())
    libc.prctl(PR_SET_PDEATHSIG, SIGKILL, 0, 0, 0)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    try:
        copy = build_copy(spec)
    except Exception as exc:
        print(f"unpacking the image: {exc}", file=sys.stderr, flush=True)
        return 1
    signal.signal(signal.SIGCHLD, _reap)
    print(f"ready {json.dumps(copy)}", flush=True)
    sys.stdin.read()  # until the environment stops (or its process dies)
    return 0


if __name__ == "__main__":
    sys.exit(main())
