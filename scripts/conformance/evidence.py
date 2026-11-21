#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements evidence that a learning system's algorithms
# behave as specified, for its clients. If your team needs expertise in
# validating training and evaluation pipelines, you can procure our services
# by sending an email to info@swedishembedded.com.
"""Run the proving tests of the validation map and write the evidence file.

The map is the table in the contributing guide's validation page (one row per
algorithm: its kind, its spec, its proving tests, its tier). Each proving test
is written there as `package/target/filter` (target `lib`, `all`, or an
integration test binary). This runner runs every test of the rows of the tier
asked for and writes one JSON line per row to the evidence file: the
algorithm, its kind, the backend and precision the run proves (Splinter's
algorithms are host arithmetic: `cpu`, `f64`), the tests and how many ran and
passed, the commit and the time. A filter that matches no test fails its row:
a check that never ran certifies nothing.

Run: python3 -I scripts/conformance/evidence.py [--tier per-PR|integration]
[--cargo CARGO] [--out PATH] (make test/conformance, make
test/conformance/integration). Exit 0 when every row
of the tier passed, 1 otherwise.
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
# doc-reference: the map's one source is this page; a moved page fails the run.
MAP = os.path.join(ROOT, "docs", "contributing", "validation.md")
TEST = re.compile(r"`(splinter-[a-z-]+)/([a-z_]+)/([^`]*)`")
RAN = re.compile(r"test result: (ok|FAILED)\. (\d+) passed; (\d+) failed")


def rows(tier):
    """The map's rows of `tier`: (algorithm, kind, tests)."""
    found = []
    with open(MAP, encoding="utf-8") as f:
        for line in f:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) != 6 or not TEST.search(cells[3]):
                continue
            algorithm, kind, _spec, tests, row_tier, _mutants = cells
            if row_tier == tier:
                found.append((algorithm, kind, TEST.findall(tests)))
    return found


def build(cargo, packages):
    """Builds the tests of `packages` in one cargo invocation (one feature
    resolution, so no unit is compiled twice) and returns their executables
    by package, as (target kind, target name, path, package directory)."""
    command = [cargo, "test", "--release", "--offline", "--no-run", "--message-format=json"]
    for package in sorted(packages):
        command += ["-p", package]
    done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if done.returncode != 0:
        sys.stderr.write(done.stderr[-4000:])
        raise SystemExit("evidence: the tests do not build")
    executables = {}
    for line in done.stdout.splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if message.get("reason") != "compiler-artifact" or not message.get("executable"):
            continue
        package = message["package_id"]
        name = next((p for p in packages if f"/{p}" in package or f"#{p}@" in package), None)
        if name is None:
            continue
        target = message["target"]
        executables.setdefault(name, []).append(
            (
                target["kind"][0],
                target["name"],
                message["executable"],
                # cargo runs a test binary in its package's directory.
                os.path.dirname(message["manifest_path"]),
            )
        )
    return executables


def run_test(executables, package, target, name_filter):
    """Runs the tests of `package`'s `target` (`lib`, `all` or an integration
    test's name) whose names hold `name_filter`."""
    passed = failed = 0
    ok = True
    for kind, name, path, cwd in executables.get(package, []):
        if target == "lib" and kind != "lib":
            continue
        if target not in ("lib", "all") and name != target:
            continue
        command = [path] + ([name_filter] if name_filter else [])
        done = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
        ok &= done.returncode == 0
        for _, good, bad in RAN.findall(done.stdout):
            passed += int(good)
            failed += int(bad)
    return {
        "test": f"{package}/{target}/{name_filter}",
        "passed": passed,
        "failed": failed,
        "ok": ok and failed == 0 and passed > 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tier", default="per-PR")
    parser.add_argument("--cargo", default="cargo")
    parser.add_argument("--out", default=os.path.join(ROOT, "target", "conformance", "evidence.jsonl"))
    args = parser.parse_args()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT, capture_output=True, text=True,
        ).stdout.strip()
    )
    when = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    selected = rows(args.tier)
    if not selected:
        print(f"evidence: the map has no row of tier {args.tier}", file=sys.stderr)
        return 1
    executables = build(args.cargo, {t[0] for _, _, tests in selected for t in tests})
    cache = {}
    lines, all_ok = [], True
    for algorithm, kind, tests in selected:
        results = []
        for test in tests:
            if test not in cache:
                cache[test] = run_test(executables, *test)
            results.append(cache[test])
        ok = all(r["ok"] for r in results)
        all_ok &= ok
        lines.append({
            "algorithm": algorithm,
            "kind": kind,
            "backend": "cpu",
            "precision": "f64",
            "tier": args.tier,
            "ok": ok,
            "tests": results,
            "commit": commit,
            "dirty": dirty,
            "at": when,
        })
        print(f"{'ok  ' if ok else 'FAIL'} {kind:<12} {algorithm}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    print(f"evidence: {sum(l['ok'] for l in lines)}/{len(lines)} rows proven, written to {args.out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
