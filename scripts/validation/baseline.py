#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for knowing exactly which checks of a learning system pass, fail or could
# not run on a given revision and machine, for its clients. If your team needs expertise in validating machine
# learning pipelines then you can procure our services by sending an email to info@swedishembedded.com.
"""Freeze a test baseline: every target run once under its own timeout, on this revision and machine.

    python3 -I scripts/validation/baseline.py --out target/validation/baseline-results.json
    python3 -I scripts/validation/baseline.py --only test/crate/splinter-data --only test/conformance

Each target is a command with a timeout. A cargo test target's counts are read from its `test result:` lines; a
failing test is run once more on its own, and a test that passes then is reported flaky, not failed. A target whose
prerequisite is missing is `missing_prerequisite` with the reason: neither a pass nor a fail. The report records the
revisions of Splinter and its sibling repositories, the toolchain and the host, so a result is never read against
another revision.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SIBLINGS = ("brain", "sven", "bench")
RESULT = re.compile(r"test result: \w+\. (\d+) passed; (\d+) failed; (\d+) ignored")
FAILED = re.compile(r"^test (\S+) \.\.\. FAILED", re.M)


def sh(*args: str, cwd: Path = ROOT) -> str:
    try:
        return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        return ""


def packages() -> list[str]:
    """The workspace's crates, as the Makefile lists them (samples excluded: they are not Splinter's contracts)."""
    names = []
    for manifest in sorted((ROOT / "crates").glob("*/Cargo.toml")):
        match = re.search(r'^\[package\][^\[]*?^name\s*=\s*"([^"]+)"', manifest.read_text(), re.M | re.S)
        if match:
            names.append(match.group(1))
    return names


def targets(brain: str | None) -> list[dict]:
    """Every target of the baseline: (name, command, timeout in seconds, prerequisite check)."""
    cargo = ["cargo", "test", "--release"]
    out = [{"name": "check", "cmd": ["make", "check"], "timeout": 1800}]
    for package in packages():
        out.append({
            "name": f"test/crate/{package}",
            "cmd": [*cargo, "-p", package, "--", "--test-threads=8"],
            "timeout": 1500,
            "cargo": package,
        })
    out += [
        {"name": "test/conformance", "cmd": ["make", "test/conformance"], "timeout": 1200},
        {"name": "test/conformance/integration", "cmd": ["make", "test/conformance/integration"], "timeout": 2400},
        {"name": "test/families", "cmd": ["make", "test/families"], "timeout": 1800},
        {"name": "test/package", "cmd": ["make", "test/package"], "timeout": 300, "needs": ["bats"]},
        {
            "name": "test/worker",
            "cmd": ["make", "test/worker"],
            "timeout": 1800,
            "needs": ["sven"],
            "needs_file": "workers/harbor/.venv/bin/python",
        },
        {
            "name": "test/methods",
            "cmd": ["make", "test/methods", f"BRAIN_BIN={brain or ''}"],
            "timeout": 3000,
            "needs_file": brain,
        },
    ]
    return out


def missing(target: dict) -> str | None:
    for tool in target.get("needs", []):
        if shutil.which(tool) is None:
            return f"{tool} is not on PATH"
    if "needs_file" in target:
        path = target["needs_file"]
        if not path or not (ROOT / path).exists():
            return f"no {path or 'path given'}"
    return None


def run(cmd: list[str], timeout: int, log: Path) -> tuple[int | None, float, str]:
    start = time.monotonic()
    env = {**os.environ, "CARGO_BUILD_JOBS": os.environ.get("CARGO_BUILD_JOBS", "8")}
    with log.open("w") as sink:
        try:
            proc = subprocess.run(cmd, cwd=ROOT, stdout=sink, stderr=subprocess.STDOUT, timeout=timeout, env=env,
                                  start_new_session=True, check=False)
            code: int | None = proc.returncode
        except subprocess.TimeoutExpired:
            code = None
    return code, round(time.monotonic() - start, 1), log.read_text(errors="replace")


def counts(text: str) -> dict:
    total = [0, 0, 0]
    for match in RESULT.finditer(text):
        for i in range(3):
            total[i] += int(match.group(i + 1))
    return {"passed": total[0], "failed": total[1], "ignored": total[2]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="target/validation/baseline-results.json")
    parser.add_argument("--only", action="append", default=[], help="run only these targets (repeatable)")
    parser.add_argument("--brain", default=os.environ.get("BRAIN_BIN") or "../brain/target/release/brain")
    args = parser.parse_args()

    out = ROOT / args.out
    logs = out.parent / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "splinter.validation-baseline/v1",
        "revisions": {"splinter": sh("git", "rev-parse", "HEAD")}
        | {name: sh("git", "rev-parse", "HEAD", cwd=ROOT.parent / name) or None for name in SIBLINGS},
        "dirty": bool(sh("git", "status", "--porcelain")),
        "environment": {
            "rustc": sh("rustc", "--version"),
            "cargo": sh("cargo", "--version"),
            "python": platform.python_version(),
            "os": " ".join(platform.uname()[i] for i in (0, 2, 4)),
            "cpus": os.cpu_count(),
            "gpu": shutil.which("nvidia-smi") is not None,
        },
        "targets": [],
    }
    for target in targets(args.brain):
        if args.only and target["name"] not in args.only:
            continue
        row: dict = {"name": target["name"], "command": " ".join(target["cmd"]), "timeout_s": target["timeout"]}
        why = missing(target)
        if why:
            row |= {"status": "missing_prerequisite", "reason": why}
        else:
            log = logs / (target["name"].replace("/", "_") + ".log")
            code, seconds, text = run(target["cmd"], target["timeout"], log)
            row |= {"seconds": seconds, "exit": code, "log": str(log.relative_to(ROOT))} | counts(text)
            failed = sorted(set(FAILED.findall(text)))
            flaky = []
            if failed and "cargo" in target:
                again = subprocess.run(
                    ["cargo", "test", "--release", "-p", target["cargo"], "--", "--exact", *failed],
                    cwd=ROOT, capture_output=True, text=True, timeout=target["timeout"], check=False)
                still = set(FAILED.findall(again.stdout + again.stderr))
                flaky = [name for name in failed if name not in still]
                failed = [name for name in failed if name in still]
            row |= {"failed_tests": failed, "flaky_tests": flaky}
            row["status"] = ("timeout" if code is None else "pass" if code == 0
                             else "flaky" if flaky and not failed else "fail")
        report["targets"].append(row)
        print(f"{row['name']}: {row['status']} {row.get('seconds', '')}s", file=sys.stderr, flush=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
