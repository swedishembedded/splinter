# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for running many sandboxed agent
# trials on one machine without the bookkeeping becoming the bottleneck, for
# its clients. If your team needs expertise in high-throughput agent execution
# then you can procure our services by sending an email to
# info@swedishembedded.com.
"""The job's progress file: Harbor rewrites it on every trial's start and end, holding one lock across the write, so a
write that waits for anything makes every trial's start wait for every other trial's bookkeeping."""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime

from harbor.models.job.config import JobConfig
from harbor.models.job.result import JobResult, JobStats
from harbor.models.trial.config import TaskConfig

from splinter_harbor_worker.runner import NewestText, _planned_job


def test_a_progress_write_never_suspends_the_hook_that_asks_for_it(tmp_path) -> None:
    """Harbor awaits the write inside the lock that every trial's start and end takes; it returns at once (the
    coroutine finishes on its first step), and the file holds the newest progress once the job's final write is done."""
    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "instruction.md").write_text("nothing\n")
    (task / "task.toml").write_text('version = "1.0"\n')
    config = JobConfig(job_name="job", jobs_dir=tmp_path / "jobs", quiet=True, tasks=[TaskConfig(path=task)])
    job = asyncio.run(_planned_job([]).create(config))
    job._job_result = JobResult(id=job._id, started_at=datetime.now(), n_total_trials=0, stats=JobStats())

    asked = job._write_job_result_async(exclude_trial_results=True)
    try:
        asked.send(None)
    except StopIteration:
        pass
    else:
        asked.close()
        raise AssertionError("the progress write suspended the hook that holds Harbor's lock")
    job._write_job_result(exclude_trial_results=True)
    assert JobResult.model_validate_json(job._job_result_path.read_text()).id == job._id


def test_the_newest_text_wins_and_flush_waits_for_it(tmp_path) -> None:
    path = tmp_path / "result.json"
    gate = threading.Event()
    written: list[str] = []

    class Slow(NewestText):
        def _write(self, text: str) -> None:
            gate.wait(10)
            written.append(text)
            super()._write(text)

    file = Slow(path)
    file.put("first")
    file.put("second")
    file.put("third")
    gate.set()
    file.flush()
    assert path.read_text() == "third"
    assert written[-1] == "third" and "second" not in written
