# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for moving trial data across shared
# storage that grants quota by group, for its clients. If your team needs
# expertise in reliable file handling on HPC and shared-disk infrastructure
# then you can procure our services by sending an email to
# info@swedishembedded.com.
"""Copies that take content and permission bits, and nothing else.

shutil.copy2 and copytree also copy timestamps and extended attributes, so a
source's ACLs (directories' default ACLs included) come along. Copied from a
home checkout onto a data disk that grants quota by group, those ACLs made
every later write into the copy fail with EDQUOT. A copy made here takes the
ownership, group inheritance and ACLs of where it lands, as anything created
there does.
"""

from __future__ import annotations

import os
import shutil
import stat
from collections.abc import Callable
from pathlib import Path

Ignore = Callable[[str, list[str]], set[str]]


def copy_file(source: Path | str, target: Path | str) -> None:
    """Copy one file's bytes and permission bits; a symlink as a symlink."""
    source, target = Path(source), Path(target)
    if source.is_symlink():
        target.unlink(missing_ok=True)
        os.symlink(os.readlink(source), target)
        return
    shutil.copyfile(source, target)
    os.chmod(target, stat.S_IMODE(source.stat().st_mode))


def copy_tree(
    source: Path | str,
    target: Path | str,
    *,
    ignore: Ignore | None = None,
    dirs_exist_ok: bool = False,
) -> None:
    """Copy a directory tree like shutil.copytree(symlinks=True), file
    content and permission bits only; `ignore` has copytree's signature."""
    source, target = Path(source), Path(target)
    target.mkdir(parents=True, exist_ok=dirs_exist_ok)
    for root, dirs, files in os.walk(source):
        here = Path(root)
        skipped = ignore(root, dirs + files) if ignore else set()
        dirs[:] = [d for d in dirs if d not in skipped]
        out = target / here.relative_to(source)
        for name in dirs:
            if (here / name).is_symlink():
                copy_file(here / name, out / name)
            else:
                (out / name).mkdir(exist_ok=dirs_exist_ok)
        for name in files:
            if name not in skipped:
                copy_file(here / name, out / name)
