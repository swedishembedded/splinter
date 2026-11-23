# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for reproducible container-image based
# build and test environments on hosts without a container daemon for its clients.
# If your team needs expertise in hermetic benchmark environments then you can
# procure our services by sending an email to info@swedishembedded.com.
"""Unpack image layers into a directory without privileges.

Layers are applied in order with the overlay semantics of the OCI image format: a
``.wh.<name>`` entry deletes ``<name>`` of the layers below, ``.wh..wh..opq`` empties the
directory it sits in before the layer's own entries land, and any other entry replaces
what it finds. Ownership is not restored (the files belong to whoever unpacks them; the
sandboxes present every file as owned by the invoking user anyway), devices are skipped
and counted, setuid and setgid bits are dropped. Owners always keep read and write
access (directories search too), so an unpacked tree can be hashed, copied and deleted
by the user who made it however restrictive the image's modes were.

Every path is resolved *inside* the destination, following symlinks the way a chroot
would, so no entry can write outside it.
"""

from __future__ import annotations

import os
import shutil
import stat
import tarfile
from pathlib import Path, PurePosixPath

from splinter_harbor_worker.errors import ExternalToolError

_WHITEOUT = ".wh."
_OPAQUE = ".wh..wh..opq"
_MAX_LINKS = 40


def resolve_in_root(root: Path, relative: PurePosixPath, *, follow_last: bool = False) -> Path:
    """``root/relative`` with every symlink on the way (the last component only when
    ``follow_last``) resolved as if ``root`` were ``/``, so the result is always inside ``root``."""
    parts = [p for p in relative.parts if p not in ("", ".", "/")]
    resolved: list[str] = []
    hops = 0
    while parts:
        part = parts.pop(0)
        if part == "..":
            if resolved:  # as in a chroot, ".." at the root stays at the root
                resolved.pop()
            continue
        candidate = root.joinpath(*resolved, part)
        if candidate.is_symlink() and (parts or follow_last):
            hops += 1
            if hops > _MAX_LINKS:
                raise ValueError(f"{relative}: too many symbolic links")
            target = PurePosixPath(os.readlink(candidate))
            if target.is_absolute():
                resolved = []
            parts = [p for p in target.parts if p != "/"] + parts
            continue
        resolved.append(part)
    return root.joinpath(*resolved)


def unpack_layers(layers: list[Path], destination: Path) -> dict[str, int]:
    """Apply ``layers`` (lowest first) to ``destination``; counts of what was skipped."""
    destination.mkdir(parents=True, exist_ok=True)
    skipped = {"devices": 0}
    for layer in layers:
        try:
            with tarfile.open(layer, "r:*") as archive:
                _apply(archive, destination, skipped)
        except (tarfile.TarError, ValueError, OSError) as exc:
            raise ExternalToolError(f"{layer}: cannot unpack layer: {exc}") from None
    return skipped


def _apply(archive: tarfile.TarFile, root: Path, skipped: dict[str, int]) -> None:
    for member in archive:
        relative = PurePosixPath(member.name)
        if relative.is_absolute():
            relative = relative.relative_to("/")
        if not relative.parts or relative == PurePosixPath("."):
            continue
        name = relative.name
        parent = resolve_in_root(root, relative.parent, follow_last=True)
        if name == _OPAQUE:
            if parent.is_dir():
                for child in parent.iterdir():
                    _remove(child)
            continue
        if name.startswith(_WHITEOUT):
            _remove(parent / name.removeprefix(_WHITEOUT))
            continue
        parent.mkdir(parents=True, exist_ok=True)
        target = parent / name
        if member.isdir():
            if target.is_symlink() or (target.exists() and not target.is_dir()):
                _remove(target)
            target.mkdir(exist_ok=True)
            os.chmod(target, _mode(member, directory=True))
            os.utime(target, (member.mtime, member.mtime), follow_symlinks=False)
            continue
        _remove(target)
        if member.isreg():
            with archive.extractfile(member) as source, target.open("wb") as out:
                shutil.copyfileobj(source, out, 1 << 20)
            os.chmod(target, _mode(member, directory=False))
        elif member.issym():
            os.symlink(member.linkname, target)
        elif member.islnk():
            source = resolve_in_root(root, PurePosixPath(member.linkname))
            os.link(source, target, follow_symlinks=False)
            continue  # shares the inode, mode and times of the file it links to
        elif member.isfifo():
            os.mkfifo(target, _mode(member, directory=False))
        else:
            skipped["devices"] += 1
            continue
        os.utime(target, (member.mtime, member.mtime), follow_symlinks=False)


def _mode(member: tarfile.TarInfo, *, directory: bool) -> int:
    mode = member.mode & (0o777 | stat.S_ISVTX)
    return mode | (0o700 if directory else 0o600)


def _remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        make_deletable(path)
        shutil.rmtree(path)


def make_deletable(path: Path) -> None:
    """Give the owner full access to a tree, so it can be listed, copied and removed (a
    tree built in a user namespace may hold directories and files without it)."""
    mode = os.lstat(path).st_mode
    if stat.S_ISLNK(mode):
        return
    if stat.S_ISDIR(mode):
        os.chmod(path, mode | 0o700)
        for entry in os.scandir(path):
            make_deletable(Path(entry.path))
    else:
        os.chmod(path, mode | 0o600)


def remove_tree(path: Path) -> None:
    """Delete a tree that may hold unwritable entries; a missing tree is fine."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        make_deletable(path)
        shutil.rmtree(path)
