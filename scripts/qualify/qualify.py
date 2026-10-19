#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements qualification of learning agent systems
# against a matrix fixed before the first run, for its clients. If your team
# needs expertise in designing a qualification matrix and ablations for an
# agent that improves itself, you can procure our services by sending an
# email to info@swedishembedded.com.

"""One recorded command that runs the qualification matrix and the ablations.

    python3 scripts/qualify/qualify.py plan.toml --out RUN_DIR --var splinter=PATH ...

The plan (see plan.toml) declares every check: the command that produces the
evidence, how long it may take, which variables and programs it needs, and the
assertions the evidence must satisfy, each a path into the JSON the command
prints, an operator and a value. Nothing is decided after a run starts:

* the plan is read, validated and its digest written to `plan.lock.json`
  before the first command runs, so a threshold cannot be tuned afterwards
  (`--expect-plan SHA256` refuses a plan that is not the registered one);
* a check ends PASS, FAIL or BLOCKED, never anything softer. BLOCKED means
  something the check needs is absent (a variable not given, a program not
  found): it is reported with the reason and is not a pass. An assertion
  whose path is missing from the evidence FAILS: unmeasured is not passing;
* an ablation names the check it removes one thing from and the metric both
  report; the effect (full minus ablated) is recorded, whichever way it
  goes, and an ablation whose metric is unmeasured on either side FAILS.
  No feature is judged by preference: the measured effect is the finding;
* a check that needs a GPU runs under the reservation helper (`card`), which
  must be given, else the check is BLOCKED.

The report is `report.json` (every check, every assertion with its actual
value, the revisions and the host) and `report.md`, beside each check's own
stdout and stderr. Exit status: 0 when every required check passed, 1 when any
failed, 2 when none failed and some are blocked.
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

OPERATORS = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    "contains": lambda a, b: b in a,
}
PLACEHOLDER = re.compile(r"\{([a-z][a-z0-9_]*)\}")
ENV_ALLOWLIST = ("PATH", "HOME", "LANG", "TERM", "USER", "CARGO_HOME", "RUSTUP_HOME", "LD_LIBRARY_PATH")


class PlanError(Exception):
    """The plan is not valid."""


def lookup(document, path):
    """The value at `path` (`a.b[0].c`) in `document`; raises KeyError when absent."""
    current = document
    for part in re.findall(r"[^.\[\]]+|\[\d+\]", path):
        if part.startswith("["):
            current = current[int(part[1:-1])]
        else:
            current = current[part]
    return current


def variables_of(argv, *extra):
    names = set()
    for text in list(argv) + [e for e in extra if e]:
        names.update(PLACEHOLDER.findall(text))
    return names


def validate(plan):
    """Raises PlanError unless the plan is well formed."""
    seen = set()
    entries = [("check", c) for c in plan.get("check", [])] + [("ablation", a) for a in plan.get("ablation", [])]
    if not entries:
        raise PlanError("the plan declares no check")
    for kind, entry in entries:
        for field in ("id", "title"):
            if not entry.get(field):
                raise PlanError(f"a {kind} has no {field}")
        if entry["id"] in seen:
            raise PlanError(f"two entries are called {entry['id']!r}")
        seen.add(entry["id"])
        if bool(entry.get("command")) == bool(entry.get("result")):
            raise PlanError(f"{entry['id']}: give either a command or a result file, not both or neither")
        for assertion in entry.get("assert", []):
            if assertion.get("op") not in OPERATORS and assertion.get("op") != "exists":
                raise PlanError(f"{entry['id']}: unknown operator {assertion.get('op')!r}")
            if "path" not in assertion:
                raise PlanError(f"{entry['id']}: an assertion has no path")
        if kind == "check" and not entry.get("assert") and not entry.get("exit_is_evidence"):
            raise PlanError(f"{entry['id']}: a check asserts something, or says its exit status is the evidence")
        if entry.get("exit_is_evidence") and entry.get("result"):
            raise PlanError(f"{entry['id']}: a result file has no exit status")
    for ablation in plan.get("ablation", []):
        if ablation.get("of") not in {c["id"] for c in plan.get("check", [])}:
            raise PlanError(f"{ablation['id']}: it removes something from {ablation.get('of')!r}, which is no check")
        if not ablation.get("metric"):
            raise PlanError(f"{ablation['id']}: name the metric the effect is measured on")
        removed = next(c for c in plan["check"] if c["id"] == ablation["of"])
        if not removed.get("metric"):
            raise PlanError(f"{ablation['of']}: an ablation compares with it, so it names its metric")
    declared = set(plan.get("vars", {}))
    for _, entry in entries:
        missing = variables_of(entry.get("command", []), entry.get("result"), entry.get("cwd")) - declared - {"out", "check"}
        if missing:
            raise PlanError(f"{entry['id']}: undeclared variable(s) {sorted(missing)}")


def resolve(text, values):
    return PLACEHOLDER.sub(lambda m: values[m.group(1)], text)


def run_entry(entry, values, out, card, timeout_cap):
    """Runs one check or ablation and returns its record."""
    record = {"id": entry["id"], "title": entry["title"], "status": "FAIL", "reason": None, "assertions": []}
    directory = out / entry["id"]
    directory.mkdir(parents=True, exist_ok=True)
    values = {**values, "out": str(directory), "check": entry["id"]}
    needed = variables_of(entry.get("command", []), entry.get("result"), entry.get("cwd"))
    absent = sorted(n for n in needed if n not in ("out", "check") and not values.get(n))
    if absent:
        record.update(status="BLOCKED", reason=f"variable(s) not given: {', '.join(absent)}")
        return record
    document = None
    if entry.get("result"):
        path = Path(resolve(entry["result"], values))
        record["result"] = str(path)
        if not path.is_file():
            record.update(status="BLOCKED", reason=f"the registered result {path} does not exist yet")
            return record
        text = path.read_text()
    else:
        argv = [resolve(a, values) for a in entry["command"]]
        for program in entry.get("requires", []):
            if shutil.which(resolve(program, values)) is None and not Path(resolve(program, values)).exists():
                record.update(status="BLOCKED", reason=f"program not found: {resolve(program, values)}")
                return record
        if entry.get("card_gb"):
            if not card:
                record.update(status="BLOCKED", reason="needs a GPU reservation: --var card=PATH")
                return record
            argv = [card, "--gb", str(entry["card_gb"]), "--wait", "--"] + argv
        record["argv"] = argv
        env = {k: v for k, v in os.environ.items() if k in ENV_ALLOWLIST}
        started = time.monotonic()
        try:
            done = subprocess.run(
                argv, capture_output=True, text=True, env=env,
                cwd=resolve(entry.get("cwd", "."), values),
                timeout=min(entry.get("timeout_s", 3600), timeout_cap),
            )
        except subprocess.TimeoutExpired as expired:
            record.update(reason=f"timed out after {expired.timeout} s", seconds=round(time.monotonic() - started, 1))
            return record
        except FileNotFoundError as missing:
            record.update(status="BLOCKED", reason=f"cannot start: {missing}")
            return record
        record["seconds"] = round(time.monotonic() - started, 1)
        record["exit_code"] = done.returncode
        (directory / "stdout.txt").write_text(done.stdout)
        (directory / "stderr.txt").write_text(done.stderr)
        if done.returncode != 0 and not entry.get("allow_failure"):
            record["reason"] = f"exit code {done.returncode}"
            return record
        text = done.stdout
    if entry.get("json", True) and not entry.get("exit_is_evidence"):
        try:
            document = json.loads(text)
        except json.JSONDecodeError as error:
            record["reason"] = f"the evidence is not JSON: {error}"
            return record
    ok = True
    for assertion in entry.get("assert", []):
        result = {"path": assertion["path"], "op": assertion["op"], "expected": assertion.get("value")}
        try:
            actual = lookup(document, assertion["path"])
            result["actual"] = actual
            result["ok"] = True if assertion["op"] == "exists" else bool(OPERATORS[assertion["op"]](actual, assertion["value"]))
        except (KeyError, IndexError, TypeError):
            result.update(actual=None, ok=False, reason="not measured: the path is not in the evidence")
        ok = ok and result["ok"]
        record["assertions"].append(result)
    if entry.get("metric"):
        try:
            record["metric"] = lookup(document, entry["metric"])
        except (KeyError, IndexError, TypeError):
            record["metric"] = None
    record["status"] = "PASS" if ok else "FAIL"
    if not ok:
        record["reason"] = "an assertion did not hold"
    return record


def effect_of(ablation, records):
    """Fills in the effect an ablation had on the metric of the check it removes something from."""
    full = records.get(ablation["of"], {})
    ablated = ablation["_record"]
    full_metric = full.get("metric")
    ablated_metric = ablated.get("metric")
    measured = isinstance(full_metric, (int, float)) and isinstance(ablated_metric, (int, float))
    ablated["full"] = full_metric
    ablated["ablated"] = ablated_metric
    ablated["effect"] = (full_metric - ablated_metric) if measured else None
    if ablated["status"] == "PASS" and not measured:
        ablated["status"] = "BLOCKED" if full.get("status") == "BLOCKED" else "FAIL"
        ablated["reason"] = (
            f"the check {ablation['of']} it removes something from is blocked"
            if full.get("status") == "BLOCKED"
            else "the metric is unmeasured on one side: no effect can be stated"
        )


def revisions(values):
    found = {}
    for name in ("splinter_dir", "sven_dir", "brain_dir"):
        directory = values.get(name)
        if directory:
            done = subprocess.run(["git", "-C", directory, "rev-parse", "HEAD"], capture_output=True, text=True)
            found[name.removesuffix("_dir")] = done.stdout.strip() if done.returncode == 0 else None
    return found


def host():
    info = {"python": sys.version.split()[0]}
    if shutil.which("nvidia-smi"):
        done = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True
        )
        info["gpus"] = done.stdout.strip().splitlines() if done.returncode == 0 else None
    return info


def markdown(report):
    lines = [f"# Qualification {report['plan_sha256'][:12]}", "", f"Started {report['started']}", ""]
    lines += ["| id | status | reason | effect |", "|---|---|---|---|"]
    for r in report["results"]:
        effect = r.get("effect")
        lines.append(f"| {r['id']} | {r['status']} | {r.get('reason') or ''} | {'' if effect is None else round(effect, 4)} |")
    counts = report["counts"]
    lines += ["", f"PASS {counts['PASS']}, FAIL {counts['FAIL']}, BLOCKED {counts['BLOCKED']}", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="where the run's evidence is written (not in git)")
    parser.add_argument("--var", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--only", default="", help="comma-separated ids to run")
    parser.add_argument("--expect-plan", default="", metavar="SHA256", help="refuse any other plan")
    parser.add_argument("--timeout-cap", type=int, default=24 * 3600, help="the longest any one command may run, in seconds")
    parser.add_argument("--list", action="store_true", help="validate and list the entries; run nothing")
    args = parser.parse_args(argv)

    text = args.plan.read_bytes()
    digest = hashlib.sha256(text).hexdigest()
    try:
        plan = tomllib.loads(text.decode())
        validate(plan)
    except (PlanError, tomllib.TOMLDecodeError) as error:
        print(f"qualify: the plan is not valid: {error}", file=sys.stderr)
        return 3
    if args.expect_plan and args.expect_plan != digest:
        print(f"qualify: the plan is {digest}, not the registered {args.expect_plan}", file=sys.stderr)
        return 3
    if args.list:
        for entry in plan.get("check", []) + plan.get("ablation", []):
            print(f"{entry['id']}\t{entry['title']}")
        return 0
    values = dict(plan.get("vars", {}))
    for item in args.var:
        name, _, value = item.partition("=")
        values[name] = value
    values = {k: str(v) for k, v in values.items()}
    only = {i for i in args.only.split(",") if i}
    args.out.mkdir(parents=True, exist_ok=True)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    # Registered before anything runs.
    (args.out / "plan.lock.json").write_text(
        json.dumps({"plan_sha256": digest, "started": started, "vars": sorted(values)}, indent=2) + "\n"
    )
    records = {}
    results = []
    for entry in plan.get("check", []):
        if only and entry["id"] not in only:
            continue
        records[entry["id"]] = run_entry(entry, values, args.out, values.get("card"), args.timeout_cap)
        results.append(records[entry["id"]])
    for ablation in plan.get("ablation", []):
        if only and ablation["id"] not in only:
            continue
        record = run_entry(ablation, values, args.out, values.get("card"), args.timeout_cap)
        ablation = {**ablation, "_record": record}
        effect_of(ablation, records)
        records[ablation["id"]] = record
        results.append(record)
    counts = {s: sum(1 for r in results if r["status"] == s) for s in ("PASS", "FAIL", "BLOCKED")}
    report = {
        "plan_sha256": digest, "started": started, "revisions": revisions(values), "host": host(),
        "counts": counts, "results": results,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / "report.md").write_text(markdown(report))
    print(json.dumps({"counts": counts, "report": str(args.out / "report.json")}))
    if counts["FAIL"]:
        return 1
    return 2 if counts["BLOCKED"] else 0


if __name__ == "__main__":
    sys.exit(main())
