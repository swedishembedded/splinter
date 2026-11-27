# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements evaluations in which an infrastructure
# failure is never mistaken for the agent's failure, for its clients. If your
# team needs expertise in trustworthy benchmark scoring then you can procure
# our services by sending an email to info@swedishembedded.com.
"""What a finished Harbor trial directory says, as the ``splinter.attempt/1`` report.

The worker never grades: it reports the raw facts the controller's one classifier reads (``TrialEvidence``): Harbor's
``result.json``, the verifier's ``verification.json``, the model requests the gateway logged, what the agent reported
about itself on its standard error and exit code, the leak audit's findings, and the turns and model responses its own
logs record. Anything unreadable is reported as a warning next to the report and left absent in it, never invented.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from splinter_harbor_worker.audit import audit_trajectory, reference_packages
from splinter_harbor_worker.gateway import REQUEST_LOG
from splinter_harbor_worker.sandbox.choice import GUARANTEES

ATTEMPT_SCHEMA = "splinter.attempt/1"
GATEWAY_DIR = "gateway"
MAX_FINDINGS = 5  # leak findings kept per trial (what matters is whether there are any)
MAX_NATIVE_FILES = 5000
TIMEOUT_EXIT_CODE = 124  # `timeout(1)` and Sven both exit 124 when the time limit ended the run

# Sven's lines when a step or the whole run hits its time limit, e.g. `[sven:error] Step 1 timed out after 1680s`.
_AGENT_TIMEOUT = re.compile(r"^\[sven:(?:agent:)?error\].*\btimed out\b", re.M)
# ... when the run's completion-token or turn budget ended it, e.g. `[sven:error] run budget exhausted: ...`.
_AGENT_BUDGET = re.compile(r"^\[sven:(?:agent:)?error\].*\bbudget exhausted\b", re.M)
# ... when its reasoning, or the arguments of one of its tool calls, looped again after the one retry it gets, e.g.
# `[sven:error] reasoning loop: ...` or `[sven:agent:error] tool-call loop: ...`.
_AGENT_REASONING_LOOP = re.compile(r"^\[sven:(?:agent:)?error\] (?:reasoning|tool-call) loop:", re.M)


@dataclass
class Attempt:
    """A trial directory's report and the problems found reading it."""

    report: dict
    warnings: list[str] = field(default_factory=list)


def _read_json(path: Path, warnings: list[str], what: str):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(errors="replace"))
    except json.JSONDecodeError as exc:
        warnings.append(f"{what} {path.name} is not valid JSON: {exc}")
        return None


def _agent_stderr(trial: Path) -> str:
    stderr = trial / "agent" / "stderr.txt"
    return stderr.read_text(errors="replace") if stderr.is_file() else ""


def _exit_code(trial: Path) -> int | None:
    path = trial / "agent" / "exit-code.txt"
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def gateway_requests(trial: Path, warnings: list[str]) -> list[dict]:
    """The model requests the gateway logged for the trial, in order, as ``ModelRequest`` objects."""
    log = trial / GATEWAY_DIR / REQUEST_LOG
    if not log.is_file():
        return []
    requests = []
    for number, line in enumerate(log.read_text(errors="replace").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            warnings.append(f"{GATEWAY_DIR}/{REQUEST_LOG}:{number} is not valid JSON")
            continue
        requests.append({key: entry.get(key) for key in
                         ("status", "error", "prompt_tokens", "completion_tokens", "cached_tokens")})
    return requests


def _agent_responses(trial: Path, warnings: list[str]) -> tuple[int | None, int | None]:
    """The model responses the agent's own request log records and how many of them failed; ``(None, None)`` without
    a readable log."""
    log = trial / "agent" / "requests.jsonl"
    if not log.is_file():
        return None, None
    responses = errors = 0
    try:
        for line in log.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if isinstance(entry, dict) and entry.get("kind") == "response":
                responses += 1
                errors += bool((entry.get("response") or {}).get("error"))
    except (json.JSONDecodeError, AttributeError):
        warnings.append("agent/requests.jsonl is not readable")
        return None, None
    return responses, errors


def _trajectory(trial: Path, packages: tuple[str, ...], warnings: list[str]) -> tuple[int | None, list[str]]:
    """(the agent's turns, the leak audit's findings) from the ATIF trajectory, when the agent wrote one."""
    document = _read_json(trial / "agent" / "trajectory.json", warnings, "the trajectory")
    if not isinstance(document, dict):
        return None, []
    subagents = document.get("subagent_trajectories") or []
    steps = [*document.get("steps", []), *(s for child in subagents for s in child.get("steps", []))]
    turns = sum(1 for step in steps if step.get("source") == "agent")
    return turns, audit_trajectory(document, packages=packages)[:MAX_FINDINGS]


def _usage(requests: list[dict], result: dict) -> dict:
    """What the gateway's requests spent; a figure no request reported is absent (``None``), never 0."""
    def total(name: str):
        values = [r[name] for r in requests if r.get(name) is not None]
        return sum(values) if values else None

    cost = (result.get("agent_result") or {}).get("cost_usd")
    return {"prompt_tokens": total("prompt_tokens"), "completion_tokens": total("completion_tokens"),
            "cached_tokens": total("cached_tokens"), "cost_usd": cost}


def _native_files(out: Path, trial: Path) -> list[str]:
    """The backend's own files of the trial, relative to ``out``; a sandbox's private tree is not among them."""
    found = []
    for path in sorted(trial.rglob("*")):
        relative = path.relative_to(trial)
        if relative.parts[0].startswith("sandbox-") or not path.is_file():
            continue
        found.append(path.relative_to(out).as_posix())
        if len(found) >= MAX_NATIVE_FILES:
            break
    return found


def finished_trial(trial: Path) -> bool:
    """Whether Harbor finished the trial: it writes ``result.json`` last, and a trial it is still running has none."""
    result = trial / "result.json"
    if not result.is_file():
        return False
    try:
        return "trial_name" in json.loads(result.read_text())
    except (OSError, json.JSONDecodeError):
        return False


def build_report(out: Path, trial: Path, slot: dict, attempt: int, task_dir: Path | None = None,
                 sandbox: str | None = None) -> Attempt:
    """The ``splinter.attempt/1`` report of the finished trial directory ``trial`` (below ``out``)."""
    warnings: list[str] = []
    result = json.loads((trial / "result.json").read_text())
    verification = _read_json(trial / "verifier" / "verification.json", warnings, "the verification record")
    if verification is not None and not (isinstance(verification, dict) and "verdict" in verification):
        warnings.append("verifier/verification.json has no verdict")
        verification = None
    stderr = _agent_stderr(trial)
    timed_out_exit = _exit_code(trial) == TIMEOUT_EXIT_CODE
    packages = reference_packages(task_dir) if task_dir is not None else ()
    turns, leaks = _trajectory(trial, packages, warnings)
    responses, response_errors = _agent_responses(trial, warnings)
    requests = gateway_requests(trial, warnings)
    trace = trial / "agent" / "trajectory.json"
    report = {
        "schema": ATTEMPT_SCHEMA,
        "slot": slot,
        "attempt": attempt,
        "evidence": {
            "result": result,
            "verification": verification,
            "agent_timed_out": bool(_AGENT_TIMEOUT.search(stderr)) or timed_out_exit,
            "agent_budget_exhausted": bool(_AGENT_BUDGET.search(stderr)),
            "agent_reasoning_loop": bool(_AGENT_REASONING_LOOP.search(stderr)),
            "leaks": leaks,
            "turns": turns,
            "model_responses": responses,
            "model_errors": response_errors,
            "requests": requests,
        },
        "native_dir": trial.relative_to(out).as_posix(),
        "trace": trace.relative_to(out).as_posix() if trace.is_file() else None,
        "native_files": _native_files(out, trial),
        "agent_version": (result.get("agent_info") or {}).get("version"),
        "started_at": result.get("started_at"),
        "finished_at": result.get("finished_at"),
        "usage": _usage(requests, result),
        # The provider the trial ran on and what it enforced (sandbox/choice.py).
        "sandbox": None if sandbox is None else {"provider": sandbox, "guarantees": list(GUARANTEES[sandbox])},
    }
    return Attempt(report, warnings)
