# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for keeping the evidence of every agent run, however it ended, for its
# clients. If your team needs expertise in agent execution infrastructure then you can procure our services by sending
# an email to info@swedishembedded.com.
"""A trial whose agent crashed partway (plan 5A): its report keeps the evidence it left, nothing invented.

The trial directory is built by hand, as Harbor leaves one when the agent process dies after some of its work: a
``result.json`` naming the exception, the trajectory the agent wrote before it died (with a subagent's), its standard
error and exit code. The report must point at that trajectory, count the turns of the agent and its subagent, carry
Harbor's result as written for the classifier to read (it decides the status, not the worker), list the files, and
report nothing it could not read as if it were there."""

from __future__ import annotations

import json
from pathlib import Path

from splinter_harbor_worker.evidence import build_report

CRASH = {"exception_type": "RuntimeError", "exception_message": "agent process exited with code 139"}


def crashed_trial(out: Path, trajectory: dict | str | None) -> Path:
    trial = out / "harbor" / "job" / "t__0"
    (trial / "agent").mkdir(parents=True)
    (trial / "result.json").write_text(json.dumps({
        "trial_name": "t__0", "task_name": "train/001", "exception_info": CRASH, "verifier_result": None,
        "started_at": "2026-11-24T00:00:00Z", "finished_at": "2026-11-24T00:00:09Z"}))
    (trial / "agent" / "stderr.txt").write_text("Segmentation fault\n")
    (trial / "agent" / "exit-code.txt").write_text("139\n")
    if trajectory is not None:
        text = trajectory if isinstance(trajectory, str) else json.dumps(trajectory)
        (trial / "agent" / "trajectory.json").write_text(text)
    return trial


PARTIAL = {
    "schema_version": "ATIF-v1.7",
    "steps": [
        {"step_id": 1, "source": "user", "message": "Fix the parser."},
        {"step_id": 2, "source": "agent", "message": "", "tool_calls": [
            {"tool_call_id": "c1", "function_name": "shell", "arguments": {"cmd": "ls"}}]},
    ],
    "subagent_trajectories": [{"session_id": "child-1", "steps": [
        {"step_id": 1, "source": "user", "message": "List the tests."},
        {"step_id": 2, "source": "agent", "message": "tests/test_parse.py"}]}],
}


def test_a_crashed_agents_partial_trajectory_and_its_subagents_turns_are_kept(tmp_path) -> None:
    out = tmp_path / "out"
    trial = crashed_trial(out, PARTIAL)
    attempt = build_report(out, trial, {"task": "train/001", "slot": 0}, attempt=0)
    report = attempt.report
    assert attempt.warnings == []
    assert report["trace"] == "harbor/job/t__0/agent/trajectory.json"
    assert report["evidence"]["turns"] == 2, "the agent's turn and its subagent's"
    assert report["evidence"]["result"]["exception_info"] == CRASH, "Harbor's result as written"
    assert report["evidence"]["agent_timed_out"] is False
    assert "harbor/job/t__0/agent/trajectory.json" in report["native_files"]
    kept = json.loads((out / report["trace"]).read_text())
    assert kept["subagent_trajectories"][0]["session_id"] == "child-1", "the child stays inside its parent"


def test_a_crash_that_left_no_readable_trajectory_says_so_and_invents_none(tmp_path) -> None:
    out = tmp_path / "out"
    torn = crashed_trial(out, '{"steps": [{"step_id": 1, "sou')
    attempt = build_report(out, torn, {"task": "train/001", "slot": 0}, attempt=0)
    assert attempt.report["evidence"]["turns"] is None, "no turns are counted from what could not be read"
    assert any("trajectory" in w for w in attempt.warnings), attempt.warnings
    absent = crashed_trial(tmp_path / "other", None)
    report = build_report(tmp_path / "other", absent, {"task": "train/001", "slot": 0}, attempt=0).report
    assert report["trace"] is None and report["evidence"]["turns"] is None


def test_a_model_looping_in_its_reasoning_or_in_a_tool_calls_arguments_is_reported_as_a_loop(tmp_path) -> None:
    """sven ends a run whose model loops again after its one retry, in its reasoning or inside a tool call's
    arguments; either is the model's failure, which the classifier counts, never an endpoint's refusal."""
    for name, line in [
        ("reasoning", "[sven:error] reasoning loop: the model is repeating itself in its reasoning"),
        ("tool-call", "[sven:agent:error] tool-call loop: the model is repeating itself inside a tool call's "
                      "arguments (the same block occurs 6 times in a row; ~88 argument tokens so far)"),
    ]:
        out = tmp_path / name
        trial = crashed_trial(out, None)
        (trial / "agent" / "stderr.txt").write_text(f"[sven:warn] asking again once\n{line}\n[sven:agent:aborted]\n")
        report = build_report(out, trial, {"task": "train/001", "slot": 0}, attempt=0).report
        assert report["evidence"]["agent_reasoning_loop"] is True, name
    plain = build_report(tmp_path / "crash", crashed_trial(tmp_path / "crash", None),
                         {"task": "train/001", "slot": 0}, attempt=0).report
    assert plain["evidence"]["agent_reasoning_loop"] is False
