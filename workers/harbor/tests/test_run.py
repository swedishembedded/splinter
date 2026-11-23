# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for testing agent execution
# backends against stub models and tiny tasks with no network, for its
# clients. If your team needs expertise in hermetic agent test harnesses then
# you can procure our services by sending an email to info@swedishembedded.com.
"""One Harbor job, run for real under the proot sandbox with the oracle and nop agents: what a finished attempt's report
holds, what the progress log says in what order, and that a crash after the trials is repaired without rerunning any."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from conftest import make_spec, make_task, write_specs

from splinter_harbor_worker.cli import main

TASK = "train/001-write-answer"


@pytest.fixture(scope="module")
def finished(tmp_path_factory, image_archive):
    """The oracle attempt (slot 0) and the nop attempt (slot 1) of one task, run as one job."""
    import splinter_harbor_worker.sandbox.proot as proot
    from splinter_harbor_worker.rootfs.images import TaskImage

    work = tmp_path_factory.mktemp("job")
    make_task(work, TASK)
    specs = write_specs(work / "specs.jsonl", [make_spec(TASK, agent="oracle", slot=0),
                                               make_spec(TASK, agent="nop", slot=1)])
    image = TaskImage((image_archive,), {"PATH": "/usr/bin"}, "/app", {"source": "pull", "reference": "local/toy:1"})
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(proot, "ensure_task_image", lambda *a, **k: image)
        code = main(["run", "--specs", str(specs), "--tasks", str(work), "--out", str(work / "out"),
                     "--concurrency", "2", "--image-cache", str(work / "cache")])
    return work, code


def report(work: Path, slot: int) -> dict:
    path = work / "out" / "attempts" / "local" / "train__001-write-answer" / f"{slot}-0.json"
    return json.loads(path.read_text())


def test_every_attempt_has_a_report_and_the_worker_exits_zero(finished, validator) -> None:
    work, code = finished
    assert code == 0
    for slot in (0, 1):
        validator.validate(report(work, slot))


def test_a_passing_oracle_trial_reports_full_reward(finished) -> None:
    work, _ = finished
    evidence = report(work, 0)["evidence"]
    assert evidence["result"]["verifier_result"]["rewards"] == {"reward": 1.0}
    assert evidence["result"]["exception_info"] is None
    assert evidence["leaks"] == [] and evidence["requests"] == []
    assert evidence["agent_timed_out"] is False


def test_a_failing_nop_trial_still_gets_a_report_with_reward_zero(finished) -> None:
    work, _ = finished
    evidence = report(work, 1)["evidence"]
    assert evidence["result"]["verifier_result"]["rewards"] == {"reward": 0.0}
    assert report(work, 1)["slot"] == {"pass": "local", "task": TASK, "slot": 1}


def test_paths_are_relative_to_the_output_directory_and_the_job_is_viewable(finished) -> None:
    work, _ = finished
    out = work / "out"
    first = report(work, 0)
    assert first["native_dir"].startswith("harbor/job/") and not Path(first["native_dir"]).is_absolute()
    assert (out / first["native_dir"] / "result.json").is_file()
    assert "harbor/job/" + first["native_dir"].rsplit("/", 1)[1] + "/result.json" in first["native_files"]
    assert all(not name.startswith("/") for name in first["native_files"])
    assert (out / "harbor" / "job" / "result.json").is_file()  # what `harbor view OUT/harbor` opens
    assert first["usage"] == {"prompt_tokens": None, "completion_tokens": None, "cached_tokens": None, "cost_usd": None}


def test_the_progress_log_orders_started_before_finished_per_attempt(finished) -> None:
    work, _ = finished
    events = [json.loads(line) for line in (work / "out" / "events.jsonl").read_text().splitlines()]
    assert all(set(e) == {"at", "kind", "slot", "attempt", "detail"} for e in events)
    assert events[0]["kind"] == "started" and events[0]["slot"] is None
    assert events[-1]["kind"] == "finished" and events[-1]["detail"]["unreported"] == []
    for slot in (0, 1):
        kinds = [e["kind"] for e in events if e["slot"] and e["slot"]["slot"] == slot]
        assert kinds == ["started", "attempt_finished"], kinds
    assert [e["at"] for e in events] == sorted(e["at"] for e in events)


def test_reconcile_regenerates_identical_reports_and_runs_nothing(finished, capsys) -> None:
    work, _ = finished
    out = work / "out"
    before = {slot: (out / "attempts" / "local" / "train__001-write-answer" / f"{slot}-0.json").read_bytes()
              for slot in (0, 1)}
    trials = sorted(p.name for p in (out / "harbor" / "job").iterdir() if p.is_dir())
    results = {t: (out / "harbor" / "job" / t / "result.json").read_bytes() for t in trials}
    shutil.rmtree(out / "attempts")
    assert main(["reconcile", "--out", str(out)]) == 0
    after = {slot: (out / "attempts" / "local" / "train__001-write-answer" / f"{slot}-0.json").read_bytes()
             for slot in (0, 1)}
    assert after == before
    assert {t: (out / "harbor" / "job" / t / "result.json").read_bytes() for t in trials} == results
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary == {"missing": [], "reported": 2}
    # idempotent: a second reconcile writes nothing new
    log = (out / "events.jsonl").read_text()
    assert main(["reconcile", "--out", str(out)]) == 0
    assert (out / "events.jsonl").read_text() == log


def test_reconcile_names_the_slots_whose_trial_never_finished(finished, capsys) -> None:
    work, _ = finished
    out = work / "out"
    spec = json.loads((out / "specs.jsonl").read_text().splitlines()[1])
    trial = next(p for p in (out / "harbor" / "job").iterdir() if p.is_dir() and (p / "agent").is_dir()
                 and json.loads((p / "result.json").read_text())["agent_info"]["name"] == "nop")
    saved = (trial / "result.json").read_bytes()
    (trial / "result.json").unlink()
    (out / "attempts" / "local" / "train__001-write-answer" / "1-0.json").unlink()
    try:
        assert main(["reconcile", "--out", str(out)]) == 1
        summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert summary["missing"] == [{"slot": spec["slot"], "attempt": 0}]
    finally:
        (trial / "result.json").write_bytes(saved)
        assert main(["reconcile", "--out", str(out)]) == 0


def test_a_task_pinned_read_only_is_staged_with_its_network_written(tmp_path) -> None:
    from splinter_harbor_worker.plan import stage_task
    from splinter_harbor_worker.spec import ExecutionSpec

    make_task(tmp_path, TASK)
    for path in (tmp_path / "tasks" / TASK).rglob("*"):
        path.chmod(path.stat().st_mode & ~0o222)
    spec = ExecutionSpec.model_validate(make_spec(TASK, network="none"))
    staged = stage_task(tmp_path / "out", tmp_path, spec)
    assert 'network_mode = "no-network"' in (staged / "task.toml").read_text()
