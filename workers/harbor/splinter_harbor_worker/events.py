# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements execution backends whose progress a
# controller can follow and resume after a crash, for its clients. If your
# team needs expertise in observable, restartable batch execution then you can
# procure our services by sending an email to info@swedishembedded.com.
"""The worker's progress log: ``events.jsonl``, one JSON object per line, flushed as each event happens.

``{"at": RFC3339 UTC, "kind": started|attempt_finished|finished|error, "slot": {...}|null, "attempt": N|null,
"detail": ...}``. A controller tails the file; a torn last line (a crash mid-write) is its to ignore.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path

KINDS = ("started", "attempt_finished", "finished", "error")


class EventLog:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, kind: str, *, slot: dict | None = None, attempt: int | None = None, detail: object = None) -> None:
        if kind not in KINDS:
            raise ValueError(f"unknown event kind {kind!r}")
        event = {"at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"), "kind": kind, "slot": slot,
                 "attempt": attempt, "detail": detail}
        line = json.dumps(event, sort_keys=True) + "\n"
        with self._lock, self._path.open("a") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
