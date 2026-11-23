# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for reproducible container-image based
# build and test environments on hosts without a container daemon for its clients.
# If your team needs expertise in hermetic benchmark environments then you can
# procure our services by sending an email to info@swedishembedded.com.
"""A content store that lets many root filesystems share their identical files.

``link_tree(source, target, store)`` recreates ``source`` at ``target`` with every
regular file a hard link to one inode in the store, keyed by content, permission bits and
modification time. Two images that both hold the same ``libc`` (or the same Ubuntu base)
then cost one copy and one inode of it, which matters on a disk with an inode quota.

The shared inodes are never written afterwards: roots are bound read-only into sandboxes
and a build always starts from a *copy* of its base, never from linked files. Setuid and
setgid bits are dropped on the way in.
"""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path

_BLOCK = 1 << 20


@dataclass
class LinkStats:
    files: int = 0  # regular files placed in the target
    stored: int = 0  # of those, new to the store
    directories: int = 0
    symlinks: int = 0


def link_tree(source: Path, target: Path, store: Path) -> LinkStats:
    """Recreate ``source`` at ``target`` (which must not exist) from the store.

    Files of ``source`` that are new to the store are linked into it (they become the
    store's inodes), so ``source`` can be deleted afterwards without losing them."""
    stats = LinkStats()
    target.mkdir(parents=True)
    _link_directory(source, target, store, stats)
    return stats


def _link_directory(source: Path, target: Path, store: Path, stats: LinkStats) -> None:
    with os.scandir(source) as entries:
        children = sorted(entries, key=lambda e: e.name)
    for entry in children:
        here, there = Path(entry.path), target / entry.name
        info = entry.stat(follow_symlinks=False)
        if stat.S_ISLNK(info.st_mode):
            os.symlink(os.readlink(here), there)
            os.utime(there, ns=(info.st_atime_ns, info.st_mtime_ns), follow_symlinks=False)
            stats.symlinks += 1
        elif stat.S_ISDIR(info.st_mode):
            there.mkdir()
            stats.directories += 1
            _link_directory(here, there, store, stats)
            os.chmod(there, stat.S_IMODE(info.st_mode) & 0o1777)
            os.utime(there, ns=(info.st_atime_ns, info.st_mtime_ns))
        elif stat.S_ISREG(info.st_mode):
            _place(here, there, info, store, stats)
        elif stat.S_ISFIFO(info.st_mode):
            os.mkfifo(there, stat.S_IMODE(info.st_mode) & 0o777)
        # sockets and devices are not part of an image


def _place(source: Path, target: Path, info: os.stat_result, store: Path, stats: LinkStats) -> None:
    mode = stat.S_IMODE(info.st_mode) & 0o1777
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while block := handle.read(_BLOCK):
            digest.update(block)
    hex_digest = digest.hexdigest()
    entry = store / hex_digest[:2] / f"{hex_digest}.{mode:o}.{int(info.st_mtime)}"
    if not entry.exists():
        entry.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(source, mode)
        os.utime(source, (info.st_mtime, info.st_mtime))
        try:
            os.link(source, entry)
        except FileExistsError:
            pass
        else:
            stats.stored += 1
    os.link(entry, target)
    stats.files += 1
