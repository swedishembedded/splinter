# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for running published agent benchmarks on hosts without a container
# daemon for its clients. If your team needs expertise in hermetic benchmark environments then you can procure our
# services by sending an email to info@swedishembedded.com.
"""A task image is the shared installation plus a layer of only what the task's files add."""

from __future__ import annotations

import tarfile

from splinter_harbor_worker.rootfs import dockerfile
from splinter_harbor_worker.rootfs.archive import _layer, prefix_name, shared_prefix

TASK = """FROM debian:trixie-slim
RUN apt-get update && apt-get install -y vim
COPY tool.py /usr/local/bin/tool.py
COPY workspace/ /workspace/repo/
WORKDIR /workspace/repo
"""


def plan(text: str) -> dockerfile.Plan:
    return dockerfile.parse(text)


def test_the_steps_before_the_first_copy_are_shared_when_only_copies_follow():
    assert shared_prefix(plan(TASK)) == 1
    assert shared_prefix(plan(TASK + "RUN touch /x\n")) is None, "a command on the task's files needs the whole tree"
    assert shared_prefix(plan("FROM debian:trixie-slim\nCOPY a /a\n")) is None, "nothing before the first COPY to share"


def test_the_shared_archive_is_named_by_what_decides_its_content_not_by_the_task():
    other = TASK.replace("COPY workspace/ /workspace/repo/", "COPY files/ /workspace/repo/")
    assert prefix_name(plan(TASK), "sha256:a", 1) == prefix_name(plan(other), "sha256:a", 1)
    assert prefix_name(plan(TASK), "sha256:a", 1) != prefix_name(plan(TASK), "sha256:b", 1)
    changed = TASK.replace("install -y vim", "install -y vim-nox")
    assert prefix_name(plan(TASK), "sha256:a", 1) != prefix_name(plan(changed), "sha256:a", 1)


def test_a_layer_holds_only_the_files_the_copies_write(tmp_path):
    context = tmp_path / "context"
    (context / "workspace").mkdir(parents=True)
    (context / "workspace" / "a.txt").write_text("one\n")
    (context / "tool.py").write_text("print()\n")
    shared = {"dirs": ["/usr", "/usr/local", "/usr/local/bin"], "symlinks": ["/bin"]}
    layer = _layer(plan(TASK).steps[1:], context, shared, tmp_path / "cache", "task-1", lambda m: None)
    with tarfile.open(layer) as archive:
        names = {m.name: m for m in archive}
    assert set(names) == {"usr/local/bin/tool.py", "workspace", "workspace/repo", "workspace/repo/a.txt"}, "the shared /usr/local/bin is not repeated"
    assert all(m.uid == 0 and m.gid == 0 for m in names.values())
    assert names["workspace/repo/a.txt"].size == 4


def test_a_destination_through_a_link_of_the_shared_tree_is_not_layered(tmp_path):
    context = tmp_path / "context"
    context.mkdir()
    (context / "tool.py").write_text("print()\n")
    shared = {"dirs": [], "symlinks": ["/usr/local"]}
    steps = plan("FROM d\nRUN x\nCOPY tool.py /usr/local/bin/tool.py\n").steps[1:]
    assert _layer(steps, context, shared, tmp_path / "cache", "task-2", lambda m: None) is None
