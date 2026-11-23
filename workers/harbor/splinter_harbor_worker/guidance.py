# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for recording a person's guidance of an agent so that it never passes for
# the agent's own work, for its clients. If your team needs expertise in the provenance of agent trajectories then you
# can procure our services by sending an email to info@swedishembedded.com.
"""How a played trial says that a person typed into it.

Three records, so that no reader has to read a message to know who wrote it:

* the trial's Harbor configuration: ``config.agent.kwargs.human_guided`` is true (``HUMAN_GUIDED_KWARG``). It is in
  ``result.json``, hence in the report's ``evidence.result``, and it is what Splinter's Harbor ingestion reads
  (``is_human_guided`` in ``crates/pipelines/src/arm_compare.rs``);
* Sven's own trajectory: each user turn the person typed carries ``extra.sven.origin: human``;
* ``agent/guidance.json`` (``splinter.guidance/1``), written when the session ends: the number of those turns and when in
  the agent's work each was sent. An agent without a trajectory (Claude Code) has ``trace: false`` and no count.

In a continued session Sven drops the mark of every user turn it reloads, so the worker restores it
(``restore_human_marks``) and says how many it restored.
"""

from __future__ import annotations

SCHEMA = "splinter.guidance/1"
GUIDANCE_FILE = "guidance.json"
HUMAN_GUIDED_KWARG = "human_guided"
HUMAN_ORIGIN = "human"  # Sven's extra.sven.origin of a turn a person typed


def is_human(step: dict) -> bool:
    """Whether an ATIF step is a turn a person typed."""
    return ((step.get("extra") or {}).get("sven") or {}).get("origin") == HUMAN_ORIGIN


def restore_human_marks(trajectory: dict, earlier: dict) -> int:
    """Mark, in the trajectory of a continued session, the user turns a person typed, and how many were missing the mark.

    Sven, given an earlier trajectory to continue, writes the whole conversation back without the ``origin`` of its user
    turns. A user turn is the person's if it was in ``earlier`` and marked there, or is newer than everything in
    ``earlier`` (a continued session sends no instruction of its own, so every new user turn was typed)."""
    marked = {step.get("step_id"): is_human(step) for step in earlier.get("steps") or []}
    last = max((i for i in marked if isinstance(i, int)), default=0)
    restored = 0
    for step in trajectory.get("steps") or []:
        step_id = step.get("step_id")
        if step.get("source") != "user" or is_human(step):
            continue
        if marked.get(step_id) or (isinstance(step_id, int) and step_id > last):
            extra = step.get("extra") or {}
            step["extra"] = {**extra, "sven": {**(extra.get("sven") or {}), "origin": HUMAN_ORIGIN}}
            restored += 1
    return restored


def guidance_record(trajectory: dict | None, seat: str, continued_from: dict | None = None,
                    marks_restored: int = 0) -> dict:
    """The guidance record of a session whose ATIF trajectory is ``trajectory`` (``None``: the agent leaves none).
    ``continued_from`` names the earlier trial this one continues; ``marks_restored`` how many turns got their
    ``origin`` from ``restore_human_marks``."""
    record = {"schema": SCHEMA, "mode": "human_guided", "seat": seat, "trace": trajectory is not None}
    if trajectory is not None:
        turns, agent_steps = [], 0
        for step in trajectory.get("steps") or []:
            if step.get("source") == "agent":
                agent_steps += 1
            elif step.get("source") == "user" and is_human(step):
                turns.append({"step_id": step.get("step_id"), "after_agent_steps": agent_steps})
        record |= {"messages": len(turns), "human_turns": turns}
    if continued_from is not None:
        record["continued_from"] = {**continued_from, "marks_restored": marks_restored}
    return record
