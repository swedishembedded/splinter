# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for running published agent benchmarks on hosts without a container
# daemon for its clients. If your team needs expertise in sandboxing many agent trials on one machine then you can
# procure our services by sending an email to info@swedishembedded.com.
"""Where a sandbox's private copy of its image lives: on the local disk the trial directory is on, as a directory of
the sandbox that goes with it, and in memory only when that directory is on a network filesystem. Hundreds of
sandboxes each holding a few hundred megabytes of image in memory exhausted a node's memory limit, while its local
disk had room for all of them."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from splinter_harbor_worker.sandbox import userns
from splinter_harbor_worker.sandbox.userns import copy_in_memory, remove_tree

MOUNTINFO = """\
22 1 253:0 / / rw,relatime shared:1 - xfs /dev/mapper/root rw
40 22 253:3 / /srv/local rw,relatime shared:20 - xfs /dev/mapper/vg_data-lv_data rw
41 22 0:51 / /net rw,relatime shared:21 - lustre 10.0.0.1@o2ib:/proj rw
42 41 0:52 / /net/proj/disk rw,relatime shared:22 - lustre 10.0.0.1@o2ib:/disk rw
43 22 0:53 / /users rw,relatime shared:23 - nfs4 server:/users rw
"""


def test_a_copy_is_held_in_memory_only_on_a_network_filesystem():
    assert not copy_in_memory(Path("/srv/local/runs/x/sandbox-1/image"), MOUNTINFO)
    assert not copy_in_memory(Path("/var/tmp/t/image"), MOUNTINFO), "the root filesystem is a local disk"
    assert copy_in_memory(Path("/net/proj/disk/runs/sandbox-1/image"), MOUNTINFO)
    assert copy_in_memory(Path("/users/me/runs/sandbox-1/image"), MOUNTINFO)
    # A mount point matches whole path components: /netx is on the root filesystem, not under /net.
    assert not copy_in_memory(Path("/netx/image"), MOUNTINFO)


def _layer(path: Path) -> Path:
    with tarfile.open(path, "w") as archive:
        for name, data in (("etc/os-release", b"ID=test\n"), ("workspace/repo/a.txt", b"one\n")):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        locked = tarfile.TarInfo("locked")
        locked.type, locked.mode = tarfile.DIRTYPE, 0o555
        archive.addfile(locked)
    return path


def _userns_available() -> bool:
    return shutil.which("unshare") is not None and subprocess.run(
        ["unshare", "--user", "--map-root-user", "--mount", "true"], capture_output=True).returncode == 0


def test_a_copy_on_disk_is_a_tree_of_the_sandbox_that_stop_removes(tmp_path):
    if not _userns_available():
        pytest.fail("this host cannot create a user namespace; the sandbox needs one")
    root = tmp_path / "sandbox-1" / "image"
    root.mkdir(parents=True)
    spec = {"mountpoint": str(root), "layers": [str(_layer(tmp_path / "layer.tar"))], "size_mb": 64,
            "in_memory": False}
    keeper = subprocess.Popen(
        ["unshare", "--user", "--map-root-user", "--mount", "--pid", "--fork", "--kill-child", "--",
         sys.executable, "-I", str(userns._KEEPER)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    keeper.stdin.write(json.dumps(spec).encode() + b"\n")
    keeper.stdin.flush()
    assert keeper.stdout.readline().strip() == b"ready", keeper.stderr.read().decode()
    # The tree is on the host's disk, not in a mount only the keeper's namespace sees.
    assert (root / "workspace" / "repo" / "a.txt").read_text() == "one\n"
    keeper.stdin.close()
    keeper.wait(timeout=30)
    remove_tree(root)
    assert not root.exists(), "a read-only directory of the image does not keep the copy on disk"


def test_remove_tree_of_nothing_is_nothing(tmp_path):
    remove_tree(tmp_path / "absent")
    assert os.listdir(tmp_path) == []
