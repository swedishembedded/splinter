# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for running published agent benchmarks on hosts without a container
# daemon for its clients. If your team needs expertise in sandboxing many agent trials on one machine then you can
# procure our services by sending an email to info@swedishembedded.com.
"""Where a sandbox's private copy of its image lives: on the local disk the trial directory is on, as a directory of
the sandbox that goes with it, and in memory only when that directory is on a network filesystem. Hundreds of
sandboxes each holding a few hundred megabytes of image in memory exhausted a node's memory limit, while its local
disk had room for all of them. Hundreds of them each unpacking the same installation to that disk made the disk the
bottleneck, so the image's shared layers are unpacked once per node and every sandbox writes to an overlay over
them."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
from pathlib import Path

import pytest

from splinter_harbor_worker.sandbox import userns
from splinter_harbor_worker.sandbox.userns import copy_placement, on_network_filesystem, remove_image_copy, remove_tree

MOUNTINFO = """\
22 1 253:0 / / rw,relatime shared:1 - xfs /dev/mapper/root rw
40 22 253:3 / /srv/local rw,relatime shared:20 - xfs /dev/mapper/vg_data-lv_data rw
41 22 0:51 / /net rw,relatime shared:21 - lustre 10.0.0.1@o2ib:/proj rw
42 41 0:52 / /net/proj/disk rw,relatime shared:22 - lustre 10.0.0.1@o2ib:/disk rw
43 22 0:53 / /users rw,relatime shared:23 - nfs4 server:/users rw
"""


def test_a_copy_is_held_in_memory_only_on_a_network_filesystem():
    assert not on_network_filesystem(Path("/srv/local/runs/x/sandbox-1/image"), MOUNTINFO)
    assert not on_network_filesystem(Path("/var/tmp/t/image"), MOUNTINFO), "the root filesystem is a local disk"
    assert on_network_filesystem(Path("/net/proj/disk/runs/sandbox-1/image"), MOUNTINFO)
    assert on_network_filesystem(Path("/users/me/runs/sandbox-1/image"), MOUNTINFO)
    # A mount point matches whole path components: /netx is on the root filesystem, not under /net.
    assert not on_network_filesystem(Path("/netx/image"), MOUNTINFO)


def test_the_copy_is_an_overlay_over_the_node_cache_only_when_both_are_on_a_local_disk():
    local = copy_placement(Path("/srv/local/runs/sandbox-1/image"), Path("/srv/local/cache/images"), MOUNTINFO)
    assert local == {"in_memory": False, "lower_cache": "/srv/local/cache/images/unpacked", "disk_reason": None}
    remote_cache = copy_placement(Path("/srv/local/runs/sandbox-1/image"), Path("/net/proj/images"), MOUNTINFO)
    assert remote_cache["lower_cache"] is None and not remote_cache["in_memory"]
    assert "network filesystem" in remote_cache["disk_reason"], "a full copy on disk says why"
    remote_trial = copy_placement(Path("/users/me/runs/sandbox-1/image"), Path("/srv/local/cache/images"), MOUNTINFO)
    assert remote_trial["in_memory"] and remote_trial["lower_cache"] is None


def _tar(path: Path, files: dict[str, bytes], dirs: dict[str, int] | None = None) -> Path:
    with tarfile.open(path, "w") as archive:
        for name, mode in (dirs or {}).items():
            info = tarfile.TarInfo(name)
            info.type, info.mode = tarfile.DIRTYPE, mode
            archive.addfile(info)
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path


def _layer(path: Path) -> Path:
    return _tar(path, {"etc/os-release": b"ID=test\n", "workspace/repo/a.txt": b"one\n",
                       "etc/remove-me": b"lower\n", "opt/old/x": b"lower\n"}, {"locked": 0o555})


def _task_layer(path: Path) -> Path:
    return _tar(path, {"workspace/repo/task.txt": b"task\n"})


def _deleting_layer(path: Path) -> Path:
    """A layer that deletes a file of the layer below and empties a directory of it."""
    return _tar(path, {"etc/.wh.remove-me": b"", "opt/old/.wh..wh..opq": b"", "opt/old/y": b"task\n"})


def _userns_available() -> bool:
    return shutil.which("unshare") is not None and subprocess.run(
        ["unshare", "--user", "--map-root-user", "--mount", "true"], capture_output=True).returncode == 0


@pytest.fixture
def userns_host():
    if not _userns_available():
        pytest.fail("this host cannot create a user namespace; the sandbox needs one")


class Keeper:
    """A sandbox's keeper, started as the environment starts it."""

    def __init__(self, sandbox: Path, layers: list[Path], *, lower_cache: Path | None = None,
                 disk_reason: str | None = None, in_memory: bool = False):
        self.sandbox = sandbox
        self.root = sandbox / "image"
        self.root.mkdir(parents=True)
        self.spec = {"mountpoint": str(self.root), "overlay": str(sandbox / "overlay"),
                     "layers": [str(layer) for layer in layers], "size_mb": 64, "in_memory": in_memory,
                     "lower_cache": str(lower_cache) if lower_cache else None, "disk_reason": disk_reason}
        self.process = subprocess.Popen(
            ["unshare", "--user", "--map-root-user", "--mount", "--pid", "--fork", "--kill-child", "--",
             sys.executable, "-I", str(userns._KEEPER)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.copy: dict = {}

    def ready(self) -> dict:
        self.process.stdin.write(json.dumps(self.spec).encode() + b"\n")
        self.process.stdin.flush()
        word, _, report = self.process.stdout.readline().decode().strip().partition(" ")
        assert word == "ready", self.process.stderr.read().decode()
        self.copy = json.loads(report)
        return self.copy

    @property
    def pid(self) -> int:
        """The keeper itself, the one child of ``unshare --fork``."""
        children = Path(f"/proc/{self.process.pid}/task/{self.process.pid}/children").read_text().split()
        return int(children[0])

    def run(self, command: str) -> str:
        """``command`` in the keeper's namespaces, where the image copy is mounted."""
        result = subprocess.run(["nsenter", f"--target={self.pid}", "--user", "--mount", "--preserve-credentials",
                                 "--", "sh", "-c", command], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        return result.stdout

    def read(self, relative: str) -> str:
        return self.run(f"cat {self.root}/{relative}")

    def exists(self, relative: str) -> bool:
        return self.run(f"test -e {self.root}/{relative} && echo yes || echo no").strip() == "yes"

    def download(self, inside: str, target: Path) -> None:
        """Copy ``inside`` out of the sandbox through the launcher, as the environment's downloads do: it binds the
        image root onto itself and pivots into it."""
        target.parent.mkdir(parents=True, exist_ok=True)
        spec = {"root": str(self.root), "image": True, "binds": [], "symlinks": [], "cwd": "/", "env": {},
                "command": "", "uid": 0, "gid": 0, "network": {"allow": []},
                "late_binds": [(str(target.parent), "/dev/.splinter-transfer/out", False)],
                "transfer": {"direction": "out", "source": inside, "directory": False,
                             "target": f"/dev/.splinter-transfer/out/{target.name}"}}
        result = subprocess.run(
            ["nsenter", f"--target={self.pid}", "--user", "--mount", "--pid", "--preserve-credentials", "--",
             "unshare", "--mount", "--ipc", "--uts", "--", sys.executable, "-I", str(userns._LAUNCHER)],
            input=json.dumps(spec) + "\n", capture_output=True, text=True)
        assert result.returncode == 0, result.stderr

    def stop(self) -> None:
        self.process.stdin.close()
        self.process.wait(timeout=30)
        remove_image_copy(self.sandbox)


def test_a_copy_on_disk_is_a_tree_of_the_sandbox_that_stop_removes(tmp_path, userns_host):
    keeper = Keeper(tmp_path / "sandbox-1", [_layer(tmp_path / "layer.tar")], disk_reason="asked for")
    assert keeper.ready() == {"mode": "disk", "reason": "asked for"}
    # The tree is on the host's disk, not in a mount only the keeper's namespace sees.
    assert (keeper.root / "workspace" / "repo" / "a.txt").read_text() == "one\n"
    keeper.stop()
    assert not keeper.root.exists(), "a read-only directory of the image does not keep the copy on disk"


def test_sandboxes_of_one_image_share_its_lower_layers_unpacked_once(tmp_path, userns_host):
    cache = tmp_path / "unpacked"
    layers = [_layer(tmp_path / "layer.tar"), _task_layer(tmp_path / "task.tar")]
    first = Keeper(tmp_path / "sandbox-1", layers, lower_cache=cache)
    second = Keeper(tmp_path / "sandbox-2", layers, lower_cache=cache)
    assert first.ready()["mode"] == "overlay" and first.copy["lower_created"]
    assert second.ready()["mode"] == "overlay" and not second.copy["lower_created"], "the second reuses the tree"
    assert first.copy["lower"] == second.copy["lower"]
    assert len([p for p in cache.iterdir() if p.is_dir()]) == 1, "one unpacked tree for one set of lower layers"
    assert first.read("workspace/repo/task.txt") == "task\n", "the task layer is unpacked over the shared tree"
    first.run(f"echo mine > {first.root}/workspace/repo/a.txt && echo new > {first.root}/workspace/repo/b.txt")
    assert first.read("workspace/repo/a.txt") == "mine\n"
    assert second.read("workspace/repo/a.txt") == "one\n", "a sandbox never sees another's writes"
    assert not second.exists("workspace/repo/b.txt")
    lower = Path(first.copy["lower"])
    assert (lower / "workspace" / "repo" / "a.txt").read_text() == "one\n", "the shared tree is never written"
    first.download("/workspace/repo/a.txt", tmp_path / "out" / "a.txt")
    assert (tmp_path / "out" / "a.txt").read_text() == "mine\n", "the launcher pivots into the overlay"
    first.stop()
    second.stop()


def test_a_layer_that_deletes_a_lower_file_hides_it_in_its_sandbox_only(tmp_path, userns_host):
    cache = tmp_path / "unpacked"
    lower = _layer(tmp_path / "layer.tar")
    deleting = Keeper(tmp_path / "sandbox-1", [lower, _deleting_layer(tmp_path / "delete.tar")], lower_cache=cache)
    keeping = Keeper(tmp_path / "sandbox-2", [lower, _task_layer(tmp_path / "task.tar")], lower_cache=cache)
    assert deleting.ready()["mode"] == "overlay" and keeping.ready()["mode"] == "overlay"
    assert deleting.copy["lower"] == keeping.copy["lower"], "the lower layers decide the shared tree"
    assert not deleting.exists("etc/remove-me")
    assert not deleting.exists("opt/old/x") and deleting.read("opt/old/y") == "task\n", "an opaque directory"
    assert keeping.read("etc/remove-me") == "lower\n" and keeping.read("opt/old/x") == "lower\n"
    deleting.stop()
    keeping.stop()


def test_concurrent_first_starts_unpack_the_lower_layers_once(tmp_path, userns_host):
    cache = tmp_path / "unpacked"
    layers = [_layer(tmp_path / "layer.tar"), _task_layer(tmp_path / "task.tar")]
    keepers = [Keeper(tmp_path / f"sandbox-{n}", layers, lower_cache=cache) for n in range(8)]
    threads = [threading.Thread(target=keeper.ready) for keeper in keepers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert [k.copy.get("mode") for k in keepers] == ["overlay"] * len(keepers)
    assert sum(k.copy["lower_created"] for k in keepers) == 1
    assert len([p for p in cache.iterdir() if p.is_dir()]) == 1, "no partial tree is left behind"
    for keeper in keepers:
        assert keeper.read("workspace/repo/task.txt") == "task\n"
        keeper.stop()


def test_stop_removes_the_sandbox_layer_and_keeps_the_shared_tree(tmp_path, userns_host):
    cache = tmp_path / "unpacked"
    keeper = Keeper(tmp_path / "sandbox-1", [_layer(tmp_path / "layer.tar"), _task_layer(tmp_path / "task.tar")],
                    lower_cache=cache)
    lower = Path(keeper.ready()["lower"])
    keeper.run(f"echo mine > {keeper.root}/workspace/repo/a.txt && chmod 0500 {keeper.root}/workspace/repo")
    keeper.stop()
    assert not (keeper.sandbox / "image").exists() and not (keeper.sandbox / "overlay").exists()
    assert (lower / "workspace" / "repo" / "a.txt").read_text() == "one\n", "the next sandbox reuses it"


def test_an_overlay_that_cannot_be_mounted_falls_back_to_a_full_copy_and_says_why(tmp_path, userns_host):
    # Overlay options cannot name a directory with a comma; the kernel would read it as another option.
    keeper = Keeper(tmp_path / "sandbox-1", [_layer(tmp_path / "layer.tar"), _task_layer(tmp_path / "task.tar")],
                    lower_cache=tmp_path / "un,packed")
    copy = keeper.ready()
    assert copy["mode"] == "disk" and "overlay" in copy["reason"]
    assert (keeper.root / "workspace" / "repo" / "task.txt").read_text() == "task\n"
    keeper.stop()


def test_a_copy_in_memory_is_reported(tmp_path, userns_host):
    keeper = Keeper(tmp_path / "sandbox-1", [_layer(tmp_path / "layer.tar")], in_memory=True)
    assert keeper.ready() == {"mode": "memory"}
    assert keeper.read("etc/os-release") == "ID=test\n"
    assert not (keeper.root / "etc").exists(), "the tmpfs exists in the keeper's namespace only"
    keeper.stop()


def test_remove_tree_of_nothing_is_nothing(tmp_path):
    remove_tree(tmp_path / "absent")
    assert os.listdir(tmp_path) == []
