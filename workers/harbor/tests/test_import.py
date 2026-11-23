# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for turning agent runs made
# elsewhere into verified training material for its clients. If your team
# needs expertise in agent trajectory pipelines then you can procure our
# services by sending an email to info@swedishembedded.com.
"""``import``: the trials of a Harbor job run elsewhere, as attempt reports of the tasks given.

The job is a real one: the worker runs an oracle and a nop attempt of one task, and the import reads the Harbor job
that run left. A trial is matched to its task by Harbor's own content digest of the task directory, so a trial of a
task that changed since, or of a task not given, is refused with its reason, never guessed."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from conftest import make_spec, make_task, write_specs

from splinter_harbor_worker.cli import main

TASK = "train/001-write-answer"


@pytest.fixture(scope="module")
def job(tmp_path_factory, image_archive) -> tuple[Path, Path]:
    """(the tasks root, the Harbor job directory) of an oracle and a nop attempt of ``TASK``."""
    import splinter_harbor_worker.sandbox.proot as proot
    from splinter_harbor_worker.rootfs.images import TaskImage

    work = tmp_path_factory.mktemp("elsewhere")
    make_task(work, TASK)
    specs = write_specs(work / "specs.jsonl", [make_spec(TASK, agent="oracle", slot=0),
                                               make_spec(TASK, agent="nop", slot=1)])
    image = TaskImage((image_archive,), {"PATH": "/usr/bin"}, "/app", {"source": "pull", "reference": "local/toy:1"})
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(proot, "ensure_task_image", lambda *a, **k: image)
        assert main(["run", "--specs", str(specs), "--tasks", str(work), "--out", str(work / "out"),
                     "--image-cache", str(work / "cache")]) == 0
    # Harbor ran the worker's staged copy of the task (its task.toml carries the network the spec asked for), so the
    # job's tasks, as they were run, are those copies.
    ran = work / "ran"
    shutil.copytree(work / "out" / "staged" / "tasks" / f"{TASK.replace("/", "__")}.public.task", ran / TASK)
    return ran, work / "out" / "harbor" / "job"


def run_import(tasks: Path, jobs: list[Path], out: Path, ids: list[str]) -> int:
    listing = out.parent / f"{out.name}-tasks.jsonl"
    listing.write_text("".join(json.dumps({"id": i, "name": f"splinter/{i}", "digest": "blake3:" + "a" * 64}) + "\n"
                               for i in ids))
    arguments = ["import", "--pass", "earlier", "--tasks", str(tasks), "--task-list", str(listing), "--out", str(out)]
    for job in jobs:
        arguments += ["--job", str(job)]
    return main(arguments)


def reports(out: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((out / "attempts").rglob("*.json"))]


def refusals(out: Path) -> list[dict]:
    path = out / "refused.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []


def test_every_finished_trial_of_a_given_task_becomes_a_report_with_its_own_grade_evidence(job, tmp_path,
                                                                                            validator) -> None:
    tasks, harbor_job = job
    out = tmp_path / "out"
    assert run_import(tasks, [harbor_job], out, [TASK]) == 0
    found = reports(out)
    assert [(r["slot"], r["attempt"]) for r in found] == [
        ({"pass": "earlier", "task": TASK, "slot": 0}, 0), ({"pass": "earlier", "task": TASK, "slot": 1}, 0)]
    for report in found:
        validator.validate(report)
        # The trial is copied into the output, and every path is relative to it.
        assert report["native_dir"].startswith("harbor/0/")
        assert (out / report["native_dir"] / "result.json").is_file()
        assert report["sandbox"] is None, "an import ran nothing, so it names no sandbox of its own"
    rewards = sorted(r["evidence"]["result"]["verifier_result"]["rewards"]["reward"] for r in found)
    assert rewards == [0.0, 1.0]
    assert refusals(out) == []
    # A report is a function of the job alone: importing again writes the same.
    again = tmp_path / "again"
    assert run_import(tasks, [harbor_job], again, [TASK]) == 0
    assert reports(again) == found


def test_a_trial_of_a_task_that_changed_or_was_not_given_is_refused_with_its_reason(job, tmp_path) -> None:
    tasks, harbor_job = job
    changed = tmp_path / "tasks"
    shutil.copytree(tasks, changed)
    (changed / TASK / "instruction.md").write_text("Write something else.\n")
    out = tmp_path / "changed"
    assert run_import(changed, [harbor_job], out, [TASK]) == 0
    assert reports(out) == []
    refused = refusals(out)
    assert len(refused) == 2
    assert all(r["job"] == 0 and "digest" in r["reason"] for r in refused), refused

    make_task(tmp_path, "train/002-other")
    other = tmp_path / "other"
    assert run_import(tmp_path / "tasks", [harbor_job], other, ["train/002-other"]) == 0
    assert reports(other) == [] and len(refusals(other)) == 2


def test_several_jobs_number_a_tasks_trials_on_in_the_order_given(job, tmp_path) -> None:
    tasks, harbor_job = job
    out = tmp_path / "out"
    assert run_import(tasks, [harbor_job, harbor_job], out, [TASK]) == 0
    found = reports(out)
    assert [r["slot"]["slot"] for r in found] == [0, 1, 2, 3]
    assert [r["native_dir"].split("/")[1] for r in found] == ["0", "0", "1", "1"]


def test_an_output_directory_that_holds_another_import_is_refused(job, tmp_path) -> None:
    tasks, harbor_job = job
    out = tmp_path / "out"
    out.mkdir()
    (out / "stale").write_text("")
    assert run_import(tasks, [harbor_job], out, [TASK]) == 2
