# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for turning agent runs made
# elsewhere into verified training material for its clients. If your team
# needs expertise in agent trajectory pipelines then you can procure our
# services by sending an email to info@swedishembedded.com.
"""``import``: the trials of Harbor jobs run elsewhere, as ``splinter.attempt/1`` reports of the tasks given.

Nothing runs. Each trial is matched to a given task by Harbor's own content digest of the task directory: the
``task.digest`` its ``lock.json`` records, else (a job of a Harbor that wrote no lock) the ``task_checksum`` of its
``result.json``. A trial is refused, with its reason, when Harbor never finished it or when its digest is no given
task's: the task changed since the trial ran, or it is not one the caller selects. An accepted trial is copied to
``OUT/harbor/<job index>/<trial>`` and reported exactly as a trial the worker ran is (``evidence.build_report``),
the leak audit included. A task's trials fill its slots in the order of the jobs given and, within a job, of their
names, so the reports are a function of the jobs and the tasks alone. ``OUT/refused.jsonl`` lists the refusals.
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

from splinter_harbor_worker.errors import UsageError
from splinter_harbor_worker.events import EventLog
from splinter_harbor_worker.evidence import build_report, finished_trial

REFUSED = "refused.jsonl"
_DIGEST_PREFIX = "sha256:"


class _TaskDigests:
    """The given tasks by Harbor's content digests, each computed once and only when a trial asks for it."""

    def __init__(self, tasks_root: Path, ids: list[str]) -> None:
        self._dirs = {}
        for task_id in ids:
            directory = (tasks_root / task_id).resolve()
            if not directory.is_relative_to(tasks_root) or not (directory / "task.toml").is_file():
                raise UsageError(f"task {task_id} is not a Harbor task under {tasks_root}")
            self._dirs[task_id] = directory
        self._content: dict[str, str] | None = None
        self._checksum: dict[str, str] | None = None

    def directory(self, task_id: str) -> Path:
        return self._dirs[task_id]

    def by_content(self, digest: str) -> str | None:
        """The task whose content hash (a trial lock's ``task.digest``) is ``digest``."""
        if self._content is None:
            from harbor.publisher.packager import Packager

            self._content = {Packager.compute_content_hash(d)[0]: i for i, d in self._dirs.items()}
        return self._content.get(digest.removeprefix(_DIGEST_PREFIX))

    def by_checksum(self, checksum: str) -> str | None:
        """The task whose directory hash (an older trial's ``task_checksum``) is ``checksum``."""
        if self._checksum is None:
            from dirhash import dirhash

            self._checksum = {dirhash(d, "sha256"): i for i, d in self._dirs.items()}
        return self._checksum.get(checksum)


def _trials(job: Path) -> list[Path]:
    """The trial directories of a Harbor job: those Harbor wrote a trial configuration or a result into."""
    return sorted(p for p in job.iterdir()
                  if p.is_dir() and ((p / "config.json").is_file() or (p / "result.json").is_file()))


def _read(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _match(trial: Path, tasks: _TaskDigests) -> tuple[str | None, str]:
    """(the given task the trial ran, else None; the reason when none)."""
    lock = _read(trial / "lock.json")
    if lock is not None:
        task = lock.get("task") or {}
        digest = task.get("digest")
        if not isinstance(digest, str):
            return None, "its lock.json records no task digest"
        found = tasks.by_content(digest)
        return found, "" if found else (
            f"its task {task.get('name')!r} (digest {digest}) is none of the tasks given: "
            "the task changed since the trial ran, or it is not selected")
    result = _read(trial / "result.json") or {}
    checksum = result.get("task_checksum")
    if not isinstance(checksum, str):
        return None, "it records neither a lock.json task digest nor a task_checksum"
    found = tasks.by_checksum(checksum)
    return found, "" if found else (
        f"its task {result.get('task_name')!r} (checksum digest {checksum}) is none of the tasks given: "
        "the task changed since the trial ran, or it is not selected")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
    partial.replace(path)


def import_jobs(pass_: str, tasks_root: Path, task_list: Path, jobs: list[Path], out: Path) -> int:
    """Import ``jobs`` into ``out`` (new or empty) as reports of ``pass_`` over the tasks ``task_list`` names (one
    JSON object with an ``id`` per line, a directory under ``tasks_root``). Exit 0; a refused trial is a line of
    ``refused.jsonl``, not an error."""
    tasks_root = tasks_root.resolve()
    out = out.resolve()
    if out.exists() and any(out.iterdir()):
        raise UsageError(f"{out} is not empty; an import writes into a new directory")
    if not jobs:
        raise UsageError("no --job to import")
    for job in jobs:
        if not job.is_dir():
            raise UsageError(f"--job {job} is not a directory")
    try:
        ids = [json.loads(line)["id"] for line in task_list.read_text().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise UsageError(f"--task-list {task_list}: {exc}") from exc
    tasks = _TaskDigests(tasks_root, ids)
    out.mkdir(parents=True, exist_ok=True)
    events = EventLog(out / "events.jsonl")
    slots: dict[str, int] = defaultdict(int)
    refused = []
    imported = 0
    for index, job in enumerate(jobs):
        for trial in _trials(job):
            if not finished_trial(trial):
                refused.append({"job": index, "trial": trial.name, "reason": "Harbor never finished it"})
                continue
            task_id, why = _match(trial, tasks)
            if task_id is None:
                refused.append({"job": index, "trial": trial.name, "reason": why})
                continue
            copy = out / "harbor" / str(index) / trial.name
            shutil.copytree(trial, copy, symlinks=True)
            slot = {"pass": pass_, "task": task_id, "slot": slots[task_id]}
            slots[task_id] += 1
            attempt = build_report(out, copy, slot, 0, tasks.directory(task_id))
            for warning in attempt.warnings:
                events.emit("error", slot=slot, attempt=0, detail=warning)
            _write_json(out / "attempts" / pass_ / task_id.replace("/", "__") / f"{slot['slot']}-0.json",
                        attempt.report)
            imported += 1
    (out / REFUSED).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in refused))
    print(json.dumps({"imported": imported, "refused": len(refused)}, sort_keys=True))
    return 0
