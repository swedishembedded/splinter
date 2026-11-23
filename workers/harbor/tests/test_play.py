# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for testing human-guided agent sessions without a human, offline, for its
# clients. If your team needs expertise in testing interactive agent harnesses then you can procure our services by
# sending an email to info@swedishembedded.com.
"""``play``: a person drives the agent on the caller's terminal, then the trial is graded and reported as for ``run``.

The "person" is the test: it runs ``play`` in a child process on a pseudo-terminal (``pty.fork``), waits for what the
scripted interactive agent (a bash stand-in for ``sven``, started inside the proot sandbox like the real one) shows,
types a line and lets the agent quit. Nothing needs a network or a model."""

from __future__ import annotations

import hashlib
import json
import os
import pty
import re
import select
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from conftest import make_spec, make_task, write_specs

from splinter_harbor_worker.agents.claude_code import ClaudeCodeAgent
from splinter_harbor_worker.cli import main

TASK = "train/001-play"
TIMEOUT_SEC = 90

# An interactive agent that is a script: it shows a prompt, takes one line from its terminal (the person's typing), does
# the task with it and writes the trajectory Sven writes: the line typed is a user turn marked origin "human", and what a
# loaded trajectory held is counted (loaded_steps) so a continuation is visible.
FAKE_SVEN = r"""#!/bin/bash
if [ "$1" = "--version" ]; then echo "sven 0.0.0-play-test"; exit 0; fi
printf '%s\n' "$@" > /logs/agent/argv.txt
if [ -t 0 ] && [ -t 1 ]; then echo tty > /logs/agent/tty.txt; fi
load=""; out=""
while [ $# -gt 0 ]; do
  case "$1" in --trace) load=$2; out=$2; shift;; --output-trace) out=$2; shift;; esac
  shift
done
prior=0
if [ -n "$load" ]; then prior=$(grep -o step_id "$load" | grep -c .); fi
echo "READY>"
read -r line
echo "$line" > /app/answer.txt
printf '{"session_id":"s%s","loaded_steps":%s,"steps":[{"step_id":%s,"source":"user","message":"%s","extra":{"sven":{"origin":"human"}}}]}\n' \
  "$prior" "$prior" "$((prior + 1))" "$line" > "$out"
echo "BYE"
"""

FAKE_CLAUDE = r"""#!/bin/bash
printf '%s\n' "$@" > /logs/agent/argv.txt
echo "READY>"
read -r line
echo "$line" > /app/answer.txt
echo "BYE"
"""


def install_agent(tmp_path: Path, name: str, script: str) -> Path:
    path = tmp_path / "bin" / name
    path.parent.mkdir(exist_ok=True)
    path.write_text(script)
    path.chmod(0o755)
    return path


# Escape sequences, left out when reading the screen's text.
ANSI = re.compile(rb"\x1b(\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(\x07|\x1b\\)|[@-Z\\-_])")
# What a terminal answers when a full-screen program asks it (cursor position, device attributes).
TERMINAL_QUERIES = ((b"\x1b[6n", b"\x1b[1;1R"), (b"\x1b[c", b"\x1b[?62c"))


def play_on_terminal(argv: list[str], typed: bytes | None = b"expected\r",
                     steps: list[tuple[bytes, bytes]] | None = None) -> tuple[int, str]:
    """``main(argv)`` in a child process whose standard streams are a pseudo-terminal. The person types ``typed`` once
    the agent shows ``READY>``, or the text of each of ``steps`` once its pattern has been shown since the step before;
    the terminal answers the queries of a full-screen agent. The exit code and everything the terminal showed."""
    pid, master = pty.fork()
    if pid == 0:  # the worker, as a person would run it
        code = 1
        sys.stdout, sys.stderr = open(1, "w", closefd=False), open(2, "w", closefd=False)  # not pytest's capture
        try:
            code = main(argv)
        except BaseException as exc:  # noqa: BLE001 - say why on the terminal, then end the child
            os.write(2, f"child failed: {type(exc).__name__}: {exc}\n".encode())
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(code)
    script = list(steps or ([(b"READY>", typed)] if typed is not None else []))
    shown, mark, deadline = b"", 0, time.monotonic() + TIMEOUT_SEC
    while time.monotonic() < deadline:
        if select.select([master], [], [], 0.2)[0]:
            try:
                chunk = os.read(master, 1 << 16)
            except OSError:  # EIO: the child has gone
                break
            if not chunk:
                break
            shown += chunk
            for query, answer in TERMINAL_QUERIES:
                if query in chunk:
                    os.write(master, answer)
        if script and script[0][0] in ANSI.sub(b"", shown[mark:]):
            os.write(master, script.pop(0)[1])
            mark = len(shown)
    else:
        os.kill(pid, 9)
        pytest.fail(f"play did not end in {TIMEOUT_SEC} s; the terminal showed: {shown.decode(errors='replace')[-500:]}")
    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status), shown.decode(errors="replace")


def play_argv(work: Path, spec: dict, out: Path, *extra: str) -> list[str]:
    write_specs(work / "spec.json", [spec])
    return ["play", "--spec", str(work / "spec.json"), "--tasks", str(work), "--out", str(out),
            "--image-cache", str(work / "cache"), *extra]


def single_report(out: Path) -> dict:
    (path,) = (out / "attempts").rglob("*.json")
    return json.loads(path.read_text())


def tree_digest(directory: Path) -> dict[str, str]:
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob("*")) if p.is_file() and not p.is_symlink()}


@pytest.fixture(scope="module")
def played(tmp_path_factory, image_archive):
    """One played trial: the scripted Sven, the person types ``expected``."""
    import splinter_harbor_worker.sandbox.proot as proot
    from splinter_harbor_worker.rootfs.images import TaskImage

    work = tmp_path_factory.mktemp("play")
    make_task(work, TASK)
    install_agent(work, "sven", FAKE_SVEN)
    image = TaskImage((image_archive,), {"PATH": "/usr/bin"}, "/app", {"source": "pull", "reference": "local/toy:1"})
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(proot, "ensure_task_image", lambda *a, **k: image)
        patch.setenv("SPLINTER_SVEN_BIN", str(work / "bin" / "sven"))
        spec = make_spec(TASK, agent="sven")
        code, shown = play_on_terminal(play_argv(work, spec, work / "out"))
        yield work, spec, code, shown


def test_a_played_trial_is_graded_and_its_report_is_valid_and_marked_human_guided(played, validator) -> None:
    work, _, code, shown = played
    assert code == 0, shown
    report = single_report(work / "out")
    validator.validate(report)
    evidence = report["evidence"]
    assert evidence["result"]["verifier_result"]["rewards"] == {"reward": 1.0}
    # The marker Splinter's Harbor ingestion reads (is_human_guided): the trial's agent configuration.
    assert evidence["result"]["config"]["agent"]["kwargs"]["human_guided"] is True
    trial = work / "out" / report["native_dir"]
    # What the person typed is in Sven's own trajectory, marked as theirs; guidance.json counts it.
    steps = json.loads((trial / "agent" / "trajectory.json").read_text())["steps"]
    assert [s["extra"]["sven"]["origin"] for s in steps] == ["human"]
    guidance = json.loads((trial / "agent" / "guidance.json").read_text())
    assert guidance["mode"] == "human_guided" and guidance["messages"] == 1 and guidance["trace"] is True
    # The agent ran interactively on a terminal and was given the task as its first message.
    assert (trial / "agent" / "tty.txt").read_text().strip() == "tty"
    argv = (trial / "agent" / "argv.txt").read_text().rstrip("\n").splitlines()
    assert "--interactive" in argv and "--headless" not in argv
    assert argv[-1] == "Write the word expected into /app/answer.txt."
    assert "READY>" in shown and (work / "out" / "events.jsonl").is_file()
    assert "Running trials" not in shown  # Harbor's progress display stays off the person's screen


def test_continue_makes_a_new_trial_whose_trajectory_continues_the_old_one_and_leaves_the_old_one_untouched(
        played, validator, monkeypatch) -> None:
    work, spec, _, _ = played
    monkeypatch.setenv("SPLINTER_SVEN_BIN", str(work / "bin" / "sven"))  # the autouse fixture cleared it
    old_out = work / "out"
    old_report = single_report(old_out)
    old_trial = old_out / old_report["native_dir"]
    before = tree_digest(old_out)
    new_out = work / "out-continued"
    code, shown = play_on_terminal(play_argv(work, spec, new_out, "--continue", str(old_trial)), typed=b"expected\r")
    assert code == 0, shown
    assert tree_digest(old_out) == before  # the old trial, its report and its logs are byte-identical
    report = single_report(new_out)
    validator.validate(report)
    assert report["attempt"] == old_report["attempt"] + 1 and report["slot"] == old_report["slot"]
    assert report["native_dir"] != old_report["native_dir"]
    assert report["evidence"]["result"]["config"]["agent"]["kwargs"]["human_guided"] is True
    trial = new_out / report["native_dir"]
    trajectory = json.loads((trial / "agent" / "trajectory.json").read_text())
    assert trajectory["loaded_steps"] == 1  # the old conversation was loaded ...
    assert [s["step_id"] for s in trajectory["steps"]] == [2]  # ... and this one went on from it
    argv = (trial / "agent" / "argv.txt").read_text().rstrip("\n").splitlines()
    assert "--trace" in argv and "--output-trace" not in argv and argv[-1] != "Write the word expected into /app/answer.txt."
    guidance = json.loads((trial / "agent" / "guidance.json").read_text())
    assert guidance["continued_from"]["trial"] == old_trial.name
    assert guidance["continued_from"]["trajectory_sha256"] == hashlib.sha256(
        (old_trial / "agent" / "trajectory.json").read_bytes()).hexdigest()


def test_a_session_that_is_not_on_a_terminal_is_refused_with_exit_code_2(tmp_path) -> None:
    make_task(tmp_path, TASK)
    write_specs(tmp_path / "spec.json", [make_spec(TASK, agent="sven")])
    done = subprocess.run(
        [sys.executable, "-m", "splinter_harbor_worker.cli", "play", "--spec", str(tmp_path / "spec.json"), "--tasks",
         str(tmp_path), "--out", str(tmp_path / "out")], stdin=subprocess.DEVNULL, capture_output=True, text=True,
        check=False)
    assert done.returncode == 2 and "needs a terminal" in done.stderr
    assert not (tmp_path / "out" / "harbor").exists()


@pytest.mark.parametrize("agent", ["oracle", "nop"])
def test_an_agent_that_cannot_be_played_is_refused_with_exit_code_2(tmp_path, agent) -> None:
    make_task(tmp_path, TASK)
    code, shown = play_on_terminal(play_argv(tmp_path, make_spec(TASK, agent=agent), tmp_path / "out"), typed=None)
    assert code == 2 and f"agent '{agent}' cannot be played interactively" in shown
    assert not (tmp_path / "out" / "harbor").exists()


def test_a_claude_code_session_is_played_but_cannot_be_continued(tmp_path, monkeypatch, local_images, validator) -> None:
    make_task(tmp_path, TASK)
    monkeypatch.setenv("SPLINTER_CLAUDE_BIN", str(install_agent(tmp_path, ClaudeCodeAgent.BINARY, FAKE_CLAUDE)))
    spec = make_spec(TASK, agent="claude-code")
    code, shown = play_on_terminal(play_argv(tmp_path, spec, tmp_path / "out"))
    assert code == 0, shown
    report = single_report(tmp_path / "out")
    validator.validate(report)
    assert report["evidence"]["result"]["config"]["agent"]["kwargs"]["human_guided"] is True
    assert report["trace"] is None  # no trajectory: the marker is the configuration's
    trial = tmp_path / "out" / report["native_dir"]
    assert json.loads((trial / "agent" / "guidance.json").read_text())["trace"] is False
    code, shown = play_on_terminal(play_argv(tmp_path, spec, tmp_path / "again", "--continue", str(trial)), typed=None)
    assert code == 2 and "--continue needs an agent that leaves a trajectory" in shown


def test_a_person_types_into_the_real_sven_tui_and_continues_the_conversation_in_a_second_trial(
        tmp_path, monkeypatch, local_images, stub_model, validator) -> None:
    assert shutil.which("sven"), "this test runs the real sven: put it on PATH (or set SPLINTER_SVEN_BIN)"
    make_task(tmp_path, TASK)
    monkeypatch.setenv("STUB_TOKEN", "sk-test-upstream-credential")
    model = {"connection": "stub", "model": "stub-model", "protocol": "openai", "url": stub_model.base_url,
             "token_env": "STUB_TOKEN"}
    argv = play_argv(tmp_path, make_spec(TASK, agent="sven", model=model), tmp_path / "out")
    # The first model reply answers the task's instruction; the person then types a message and leaves Sven.
    code, shown = play_on_terminal(argv, steps=[(b"Done.", b"hello from a person\r"), (b"Done.", b"/quit\r")])
    assert code == 0, shown
    report = single_report(tmp_path / "out")
    validator.validate(report)
    assert report["evidence"]["result"]["config"]["agent"]["kwargs"]["human_guided"] is True
    trial = tmp_path / "out" / report["native_dir"]
    steps = json.loads((trial / "agent" / "trajectory.json").read_text())["steps"]
    humans = [s for s in steps if s.get("source") == "user" and (s.get("extra") or {}).get("sven", {}).get("origin") == "human"]
    assert [h["message"] for h in humans] == ["hello from a person"]
    assert json.loads((trial / "agent" / "guidance.json").read_text())["messages"] == 1
    # The same conversation, continued by a person in a new trial: Sven loads the old trajectory as its history.
    again = tmp_path / "again"
    code, shown = play_on_terminal(play_argv(tmp_path, make_spec(TASK, agent="sven", model=model), again,
                                             "--continue", str(trial)),
                                   steps=[(b"hello from a person", b"and then this\r"), (b"Done.", b"/quit\r")])
    assert code == 0, shown
    second = single_report(again)
    validator.validate(second)
    assert second["attempt"] == 1
    resumed = json.loads((again / second["native_dir"] / "agent" / "trajectory.json").read_text())["steps"]
    typed = [s["message"] for s in resumed if (s.get("extra") or {}).get("sven", {}).get("origin") == "human"]
    assert typed == ["hello from a person", "and then this"]
    # ... and the model saw that history before the new message.
    seen = json.dumps(stub_model.posts()[-1].body["messages"])
    assert "hello from a person" in seen and "and then this" in seen
