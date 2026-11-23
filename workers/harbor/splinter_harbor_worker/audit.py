# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements evaluations in which a trajectory that went
# looking for the answer is never mistaken for one that found it, for its
# clients. If your team needs expertise in leak-proof agent benchmarks then
# you can procure our services by sending an email to info@swedishembedded.com.
"""Leak audit of an agent trajectory.

The sandbox is built so that hidden tests and reference solutions are not reachable, but a trajectory that *tried* to
reach them is still not a trace to learn from, whatever its reward. The audit flags every tool call whose arguments go
looking for them: the verifier's inputs and outputs, and git commands that dig through history beyond the checked-out
commit. For a task whose reference solution patches a library, a tool call that reads an installed copy of that library
(``dist-packages`` or ``site-packages``) is flagged too: the answer is not the agent's work.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_PATTERNS = [
    (re.compile(r"(^|[\s'\"=:(])/(tests|solution)(/|\b)"), "hidden tests or reference solution"),
    (re.compile(r"/logs/verifier"), "verifier output"),
    (re.compile(r"\bgit\s+(log|rev-list|show-branch)\b[^\n]*--(all|reflog|branches|remotes|tags)"),
     "history beyond HEAD"),
    (re.compile(r"\bgit\s+(reflog|fsck|cat-file\s+--batch-all-objects|count-objects)\b"), "history beyond HEAD"),
    (re.compile(r"\.git/(objects|refs|logs|packed-refs)"), "raw git storage"),
]


def installed_copy_pattern(packages: tuple[str, ...]) -> re.Pattern[str]:
    """Paths into a package directory of an installation that name one of ``packages`` (an import name; its
    distribution metadata, ``Name-1.0.dist-info``, counts)."""
    names = "|".join(re.escape(name) for name in packages)
    return re.compile(rf"(dist|site)-packages/({names})([/.\-\s'\"]|$)", re.IGNORECASE)


def reference_packages(task_dir: Path) -> tuple[str, ...]:
    """The top-level Python packages a task's reference solution (or reverse patch) edits: the library under test.
    Empty for a task without such a patch."""
    names: dict[str, None] = {}
    for patch in sorted((task_dir / "solution").glob("*.patch")):
        for path in re.findall(r"^diff --git a/(\S+) b/", patch.read_text(errors="replace"), re.M):
            parts = Path(path).parts
            parts = parts[1:] if parts[0] == "src" else parts
            if len(parts) > 1 and parts[0].isidentifier():
                names[parts[0]] = None
    return tuple(names)


def audit_trajectory(trajectory: dict, *, packages: tuple[str, ...] = ()) -> list[str]:
    """One finding per offending tool call: ``"<reason>: <arguments>"``."""
    patterns = list(_PATTERNS)
    if packages:
        patterns.append((installed_copy_pattern(packages), "an installed copy of the library under test"))
    findings = []
    for call in _tool_calls(trajectory):
        arguments = call.get("arguments")
        text = arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
        flat = _argument_strings(arguments)
        for pattern, reason in patterns:
            if any(pattern.search(value) for value in flat):
                findings.append(f"{reason}: {text}")
                break
    return findings


def _tool_calls(trajectory: dict):
    for step in trajectory.get("steps", []):
        yield from step.get("tool_calls") or []
    for child in trajectory.get("subagent_trajectories") or []:
        yield from _tool_calls(child)


def _argument_strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _argument_strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _argument_strings(v)]
    return []
