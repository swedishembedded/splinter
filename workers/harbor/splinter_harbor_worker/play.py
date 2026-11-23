# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for putting a person in the loop of an agent's work, recorded as faithfully
# as an automatic run, for its clients. If your team needs expertise in human-guided agent evaluation and in turning a
# person's sessions into training data then you can procure our services by sending an email to
# info@swedishembedded.com.
"""``splinter-harbor-worker play``: one ExecutionSpec as a Harbor trial that a person drives.

The trial is staged, sandboxed, routed through the model gateway and graded exactly as ``run`` does it (``run_job``), with
one difference: the agent is started on a terminal attached to the caller's, so the person types into the agent's own
interface (Sven's TUI, Claude Code's) and ends the session by quitting the agent. The verifier then runs as usual and the
worker writes the same ``AttemptReport``. What marks the trial as guided is in ``guidance.py``.

``--continue TRIAL_DIR`` goes on with the conversation of an earlier played Sven trial: its trajectory is loaded as the
history of a NEW trial (the next attempt of the same slot, in a new output directory), whose trajectory is the whole
conversation. The earlier trial is only read. The new trial starts from the task's fresh workspace: the conversation
continues, the files the earlier session changed do not come with it.

Nothing is resumed: a person's session cannot be replayed, so an output directory that already holds results is refused.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from splinter_harbor_worker import runtime
from splinter_harbor_worker.errors import UsageError, WorkerError
from splinter_harbor_worker.events import EventLog
from splinter_harbor_worker.guidance import HUMAN_GUIDED_KWARG
from splinter_harbor_worker.plan import Guided, preflight, resolve_sandboxes, trial_name
from splinter_harbor_worker.runner import SPECS_COPY, default_image_cache, report_path_relative, run_job
from splinter_harbor_worker.seat import TerminalSeat
from splinter_harbor_worker.spec import ExecutionSpec, load_specs, parse_specs

# Agents with an interface a person can drive, and the sandboxes that give them a terminal.
INTERACTIVE_AGENTS = ("sven", "claude-code")
CONTINUABLE_AGENTS = ("sven",)  # the agents that leave a trajectory to load
TERMINAL_PROVIDERS = ("userns", "confined", "proot")
CONTINUED_DIR = "staged/continued"


def _read_spec(path: Path) -> tuple[dict, ExecutionSpec]:
    """The one specification of ``path`` (a single JSON object), validated as ``run`` validates its specifications."""
    try:
        raw = json.loads(path.read_text())
    except OSError as exc:
        raise UsageError(f"cannot read {path}: {exc}") from None
    except json.JSONDecodeError as exc:
        raise UsageError(f"{path}: play takes exactly one ExecutionSpec as one JSON object: {exc}") from None
    if not isinstance(raw, dict):
        raise UsageError(f"{path}: play takes exactly one ExecutionSpec as one JSON object")
    (spec,) = parse_specs(json.dumps(raw), source=str(path))
    return raw, spec


def _refuse_unplayable(spec: ExecutionSpec) -> None:
    if spec.agent.name not in INTERACTIVE_AGENTS:
        raise UsageError(f"agent {spec.agent.name!r} cannot be played interactively; play runs: "
                         f"{', '.join(INTERACTIVE_AGENTS)}")
    if spec.sandbox.provider not in TERMINAL_PROVIDERS:
        raise UsageError(f"sandbox provider {spec.sandbox.provider!r} gives the agent no terminal for a person; "
                         f"play runs: {', '.join(TERMINAL_PROVIDERS)}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _earlier_spec(trial: Path) -> ExecutionSpec:
    """The specification the earlier played trial ran, from the ``specs.jsonl`` of the output directory it is in."""
    layout = trial.parent.name == "job" and trial.parents[1].name == "harbor"
    copy = trial.parents[2] / SPECS_COPY if layout else None
    if copy is None or not copy.is_file():
        raise UsageError(f"--continue {trial}: not a trial directory of a worker's output "
                         "(OUT_DIR/harbor/job/<trial> beside OUT_DIR/specs.jsonl)")
    for earlier in load_specs(copy, environ={}, check_environment=False):
        if trial_name(earlier) == trial.name:
            return earlier
    raise UsageError(f"--continue {trial}: {copy} names no attempt that ran as this trial")


def _continuation(trial: Path, spec: ExecutionSpec) -> tuple[ExecutionSpec, Path, dict]:
    """The earlier trial's specification, its trajectory and the record of where it came from, for a spec that is to
    continue it; anything that cannot be continued is refused."""
    if spec.agent.name not in CONTINUABLE_AGENTS:
        raise UsageError(f"--continue needs an agent that leaves a trajectory to load ({', '.join(CONTINUABLE_AGENTS)}); "
                         f"the specification runs {spec.agent.name!r}")
    trial = trial.resolve()
    earlier = _earlier_spec(trial)
    try:
        result = json.loads((trial / "result.json").read_text())
    except (OSError, json.JSONDecodeError):
        raise UsageError(f"--continue {trial}: the trial has no readable result.json; it did not finish") from None
    kwargs = ((result.get("config") or {}).get("agent") or {}).get("kwargs") or {}
    if kwargs.get(HUMAN_GUIDED_KWARG) is not True:
        raise UsageError(f"--continue {trial}: not a played trial (its configuration is not human_guided)")
    trajectory = trial / "agent" / "trajectory.json"
    if not trajectory.is_file():
        raise UsageError(f"--continue {trial}: the trial left no agent/trajectory.json to continue from")
    if earlier.slot != spec.slot or earlier.agent.name != spec.agent.name:
        raise UsageError(f"--continue {trial}: it played {earlier.agent.name} on slot {earlier.slot.label()}; the "
                         f"specification is {spec.agent.name} on slot {spec.slot.label()}")
    source = {"trial": trial.name, "attempt": earlier.attempt, "trajectory_sha256": _sha256(trajectory)}
    return earlier, trajectory, source


def _fresh_output(out: Path) -> None:
    for name in (SPECS_COPY, "harbor", "attempts"):
        if (out / name).exists():
            raise UsageError(f"{out} already holds results ({name}); a played trial is never resumed or mixed with "
                             "another: use a new --out")


def _stage_continued(out: Path, trajectory: Path, source: dict) -> Path:
    """A copy of the earlier trajectory in the new output directory, so the new trial is complete without the old one."""
    target = out / CONTINUED_DIR / f"{source['trial']}.trajectory.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(trajectory, target)
    return target


def play(spec_path: Path, tasks_root: Path, out: Path, *, files_root: Path | None = None,
         image_cache: Path | None = None, continue_from: Path | None = None, stdin: int = 0, stdout: int = 1) -> int:
    """Play the specification; the exit code: 0 when the attempt has a report, 1 when the worker failed or it has
    none, 2 for a refused specification, argument or terminal."""
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    events = EventLog(out / "events.jsonl")
    try:
        if not (os.isatty(stdin) and os.isatty(stdout)):
            raise UsageError("play needs a terminal: standard input and output must be a TTY, because a person types "
                             "into the agent")
        raw, spec = _read_spec(spec_path)
        (spec,) = resolve_sandboxes([spec], allowed=TERMINAL_PROVIDERS)
        raw["sandbox"]["provider"] = spec.sandbox.provider
        _refuse_unplayable(spec)
        tasks_root = tasks_root.resolve()
        files_root = (files_root or tasks_root).resolve()
        image_cache = (image_cache or default_image_cache()).resolve()
        if not tasks_root.is_dir():
            raise UsageError(f"--tasks {tasks_root} is not a directory")
        trajectory = source = None
        if continue_from is not None:
            earlier, trajectory, source = _continuation(continue_from, spec)
            # A new attempt of the same slot: never the earlier trial's own identity.
            raw["attempt"] = max(spec.attempt, earlier.attempt + 1)
            spec = parse_specs(json.dumps(raw), source=str(spec_path))[0]
        checked = preflight([spec], tasks_root, files_root)
        _fresh_output(out)
        guided = Guided(_stage_continued(out, trajectory, source), source) if trajectory is not None else Guided()
        (out / SPECS_COPY).write_text(json.dumps(raw) + "\n")
        print(f"splinter-harbor-worker: playing {spec.slot.label()} attempt {spec.attempt} with {spec.agent.name}; "
              "quit the agent to end the session and have it graded", file=sys.stderr, flush=True)
        runtime.publish_seat(TerminalSeat(stdin, stdout))
        # The seat writes to the terminal by its descriptors; Harbor's progress display and log lines, which would
        # redraw over the agent's screen, go to the null device for the length of the session (they are in the trial's logs).
        try:
            with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                code = asyncio.run(run_job([spec], tasks_root, files_root, out, 1, image_cache, checked, events,
                                           guided=guided))
        finally:
            runtime.publish_seat(None)
        print(f"splinter-harbor-worker: report {out / report_path_relative(spec)}" if code == 0
              else "splinter-harbor-worker: the session produced no report", file=sys.stderr)
        return code
    except WorkerError as exc:
        events.emit("error", detail=str(exc))
        print(f"splinter-harbor-worker: {exc}", file=sys.stderr)
        return exc.exit_code
    except Exception as exc:  # the worker itself failed: say so in the event log too
        events.emit("error", detail=f"{type(exc).__name__}: {exc}")
        raise
