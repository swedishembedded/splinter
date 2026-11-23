# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements execution backends that survive a crash
# without losing a finished attempt or repeating one, for its clients. If your
# team needs expertise in resumable, idempotent batch execution then you can
# procure our services by sending an email to info@swedishembedded.com.
"""The worker's two commands: ``run`` and ``reconcile``.

``run`` refuses what it cannot honour, stages the tasks, starts the model gateway and runs the specifications as ONE
native Harbor job under ``OUT/harbor/job``. Harbor's public job API takes the cross product of tasks, agents and
attempts; the worker's attempts are an arbitrary list of (slot, attempt) pairs, so the job is created with exactly the
trials the specifications name (``_init_trial_configs``, Harbor's one hook for it, pinned by the exact Harbor version).
Each finished attempt gets its report at once; a report is a pure function of the trial directory, so ``reconcile``
rebuilds the identical one after a crash without running anything.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from harbor.job import Job
from harbor.trial.hooks import TrialEvent, TrialHookEvent

from splinter_harbor_worker import runtime
from splinter_harbor_worker.errors import UsageError, WorkerError
from splinter_harbor_worker.events import EventLog
from splinter_harbor_worker.evidence import build_report, finished_trial
from splinter_harbor_worker.gateway import REQUEST_LOG, GatewayServer, ModelGateway, Upstream
from splinter_harbor_worker.plan import (
    JOB_NAME, GatewayAddress, Guided, job_config, preflight, report_path, stage_task, stage_tools,
    staged_task_dir, trial_config, trial_name, unenforced_limits,
)
from splinter_harbor_worker.spec import ExecutionSpec, load_specs

SPECS_COPY = "specs.jsonl"
DEFAULT_CONCURRENCY = 4


def default_image_cache() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "splinter-harbor-worker" / "images"


def _planned_job(trial_configs):
    class PlannedJob(Job):
        """A Harbor job whose trials are the given ones, not the cross product of its tasks and agents."""

        def _init_trial_configs(self) -> None:
            self._trial_configs = [c.model_copy(update={"job_id": self._id}, deep=True) for c in trial_configs]

    return PlannedJob


def trial_dir(out: Path, spec: ExecutionSpec) -> Path:
    return out / "harbor" / JOB_NAME / trial_name(spec)


def write_report(out: Path, spec: ExecutionSpec, events: EventLog | None = None) -> bool:
    """Write the report of ``spec``'s trial if Harbor finished it; whether one was written. The write is atomic and
    the content depends only on the trial directory, so writing it again changes nothing."""
    trial = trial_dir(out, spec)
    if not finished_trial(trial):
        return False
    attempt = build_report(out, trial, spec.slot.as_json(), spec.attempt, staged_task_dir(out, spec))
    path = report_path(out, spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(attempt.report, indent=1, sort_keys=True) + "\n")
    partial.replace(path)
    if events is not None:
        for warning in attempt.warnings:
            events.emit("error", slot=spec.slot.as_json(), attempt=spec.attempt, detail=warning)
    return True


def _announce_finished(events: EventLog, spec: ExecutionSpec, **detail) -> None:
    events.emit("attempt_finished", slot=spec.slot.as_json(), attempt=spec.attempt,
                detail={"trial": trial_name(spec), "report": report_path_relative(spec), **detail})


def report_path_relative(spec: ExecutionSpec) -> str:
    return report_path(Path("."), spec).as_posix().removeprefix("./")


@dataclass
class Outcome:
    reported: list[ExecutionSpec]
    unreported: list[ExecutionSpec]


def sweep(out: Path, specs: list[ExecutionSpec], events: EventLog, origin: str) -> Outcome:
    """A report for every spec whose trial Harbor finished and that has none yet; one already on disk is kept. Each
    report written here is announced (``origin`` says why it was not written when the trial ended)."""
    reported, unreported = [], []
    for spec in specs:
        if report_path(out, spec).is_file():
            reported.append(spec)
        elif write_report(out, spec, events):
            reported.append(spec)
            _announce_finished(events, spec, origin=origin)
        else:
            unreported.append(spec)
    return Outcome(reported, unreported)


def _register_models(gateway: ModelGateway, specs: list[ExecutionSpec], out: Path) -> None:
    for spec in specs:
        if spec.model is None:
            continue
        token = os.environ.get(spec.model.token_env) if spec.model.token_env else None
        upstream = Upstream(spec.model.url, spec.model.model, spec.model.protocol, token)
        gateway.register(trial_name(spec), upstream, trial_dir(out, spec) / "gateway" / REQUEST_LOG)


def _copy_specs(out: Path, specs_path: Path) -> None:
    """Keep the specifications beside the results, so ``reconcile`` needs nothing else; a directory that already holds
    another run's is refused rather than mixed."""
    target = out / SPECS_COPY
    text = specs_path.read_text()
    if target.is_file() and target.read_text() != text:
        raise UsageError(f"{out} already holds the results of different specifications; use a new --out")
    out.mkdir(parents=True, exist_ok=True)
    target.write_text(text)


def run(specs_path: Path, tasks_root: Path, out: Path, *, concurrency: int = DEFAULT_CONCURRENCY,
        files_root: Path | None = None, image_cache: Path | None = None) -> int:
    """Run the specifications; the exit code: 0 when every attempt has a report, 1 when the worker failed or an
    attempt has none, 2 for a refused specification or arguments."""
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    events = EventLog(out / "events.jsonl")
    try:
        specs = load_specs(specs_path)
        tasks_root = tasks_root.resolve()
        files_root = (files_root or tasks_root).resolve()
        image_cache = (image_cache or default_image_cache()).resolve()
        if not tasks_root.is_dir():
            raise UsageError(f"--tasks {tasks_root} is not a directory")
        if concurrency < 1:
            raise UsageError("--concurrency must be at least 1")
        checked = preflight(specs, tasks_root, files_root)
        _copy_specs(out, specs_path)
        return asyncio.run(run_job(specs, tasks_root, files_root, out, concurrency, image_cache, checked, events))
    except WorkerError as exc:
        events.emit("error", detail=str(exc))
        print(f"splinter-harbor-worker: {exc}", file=sys.stderr)
        return exc.exit_code
    except Exception as exc:  # the worker itself failed: say so in the event log too
        events.emit("error", detail=f"{type(exc).__name__}: {exc}")
        raise


async def run_job(specs, tasks_root, files_root, out, concurrency, image_cache, checked, events,
                  guided: Guided | None = None) -> int:
    """Stage and run ``specs`` as the one Harbor job; ``guided`` makes every trial one a person plays."""
    (out / "staged").mkdir(exist_ok=True)
    staged = {spec.key: stage_task(out, tasks_root, spec) for spec in specs}
    stage_tools(out, checked.binaries)
    by_trial = {trial_name(spec): spec for spec in specs}
    events.emit("started", detail={"attempts": len(specs), "concurrency": concurrency,
                                   "unenforced_limits": {s.slot.label(): lim for s in specs
                                                         if (lim := unenforced_limits(s))}})
    server = None
    gateway = None
    if any(spec.model is not None for spec in specs):
        gateway = ModelGateway()
        _register_models(gateway, specs, out)
        server = GatewayServer(gateway, host=checked.gateway_host)
        server.start()
    runtime.publish_gateway(gateway)
    address = GatewayAddress(checked.gateway_host, server.port) if server else None
    try:
        configs = [trial_config(spec, out=out, staged=staged[spec.key], files_root=files_root, image_cache=image_cache,
                                gateway=address, guided=guided) for spec in specs]
        job = await _planned_job(configs).create(job_config(out, list(staged.values()), concurrency))
        # A trial Harbor finished before a crash keeps its result and is not run again; its report may be missing.
        sweep(out, specs, events, "resumed")

        async def started(event: TrialHookEvent) -> None:
            spec = by_trial[event.trial_name]
            events.emit("started", slot=spec.slot.as_json(), attempt=spec.attempt, detail={"trial": event.trial_name})

        async def ended(event: TrialHookEvent) -> None:
            spec = by_trial[event.trial_name]
            if await asyncio.to_thread(write_report, out, spec, events):
                _announce_finished(events, spec)

        job.add_hook(TrialEvent.START, started)
        job.add_hook(TrialEvent.END, ended)
        await job.run()
    finally:
        runtime.publish_gateway(None)
        if server is not None:
            server.stop()
    outcome = sweep(out, specs, events, "swept")
    events.emit("finished", detail={"reported": len(outcome.reported),
                                    "unreported": [s.slot.label() + f"#{s.attempt}" for s in outcome.unreported]})
    return 0 if not outcome.unreported else 1


def reconcile(out: Path) -> int:
    """Write the missing reports for every trial Harbor already finished in ``out``, running nothing, and say which
    attempts have no finished trial. Idempotent. Exit 0 when every attempt has a report afterwards, else 1."""
    out = out.resolve()
    copy = out / SPECS_COPY
    if not copy.is_file():
        raise UsageError(f"{out} holds no {SPECS_COPY}: it is not the output of a run")
    specs = load_specs(copy, environ={}, check_environment=False)
    events = EventLog(out / "events.jsonl")
    outcome = sweep(out, specs, events, "reconciled")
    missing = [{"slot": s.slot.as_json(), "attempt": s.attempt} for s in outcome.unreported]
    print(json.dumps({"reported": len(outcome.reported), "missing": missing}, sort_keys=True))
    return 0 if not missing else 1
