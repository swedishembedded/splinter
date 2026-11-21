#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements test suites that are proven to catch the
# defects they exist for, for its clients. If your team needs expertise in
# mutation testing of learning-system arithmetic, you can procure our services
# by sending an email to info@swedishembedded.com.
"""Mutation testing of the learning algorithms: each mutant in mutants.tsv is
one deliberate defect in production code (a reversed reward, a dropped
shrinkage, a two-sided test where the spec is one-sided, a group kept that
carries no signal). The runner applies it, runs the tests named for it, and
counts it killed when they fail. The source is restored after every mutant,
also on an error or an interrupt; a mutant whose search text is not in its
file exactly once is an error (the table is stale), never a survivor.

Survivors are printed with their id: a survivor is a hole in the suite.

Run: python3 -I scripts/conformance/mutants.py [--cargo CARGO] [--only ID ...]
(make test/conformance/integration). Exit 0 when every mutant is killed, 1 when
any survives, 2 when the table or a build is broken.
"""
import argparse
import csv
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def load(path):
    """The table: id, area, file, search, replace, package, test target
    (`lib`, `all` or an integration test's name), filter. Search and replace
    are written with Python string escapes (`\\n`, `\\t`)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(
            (line for line in f if line.strip() and not line.startswith("#")),
            delimiter="\t",
        ):
            for key in ("search", "replace"):
                row[key] = row[key].encode().decode("unicode_escape")
            rows.append(row)
    return rows


def cargo_test(cargo, row):
    target = row["target"]
    selector = {"lib": ["--lib"], "all": []}.get(target, ["--test", target])
    command = [cargo, "test", "--release", "--offline", "-p", row["package"], *selector,
               "--", "--test-threads=16"]
    if row["filter"]:
        command.insert(command.index("--") + 1, row["filter"])
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True)


def run(cargo, row):
    path = os.path.join(ROOT, row["file"])
    with open(path, encoding="utf-8") as f:
        original = f.read()
    if original.count(row["search"]) != 1:
        raise SystemExit(
            f"mutant {row['id']}: its search text occurs {original.count(row['search'])} times "
            f"in {row['file']}, not once: the table is stale"
        )
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(original.replace(row["search"], row["replace"]))
        result = cargo_test(cargo, row)
    finally:
        with open(path, "w", encoding="utf-8") as f:
            f.write(original)
    output = result.stdout + result.stderr
    if result.returncode != 0 and "could not compile" in output:
        return "unbuildable", output
    return ("killed" if result.returncode != 0 else "survived"), output


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cargo", default="cargo")
    parser.add_argument("--table", default=os.path.join(HERE, "mutants.tsv"))
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    rows = [r for r in load(args.table) if args.only is None or r["id"] in args.only]
    tally = {}
    broken = False
    for row in rows:
        status, output = run(args.cargo, row)
        area = tally.setdefault(row["area"], [0, 0])
        area[1] += 1
        if status == "killed":
            area[0] += 1
        elif status == "unbuildable":
            broken = True
            sys.stderr.write(output[-3000:])
        print(f"{row['id']}\t{row['area']}\t{status}", flush=True)
    total = [sum(v[0] for v in tally.values()), sum(v[1] for v in tally.values())]
    for area, (killed, count) in sorted(tally.items()):
        print(f"kill rate {area}: {killed}/{count}")
    print(f"kill rate all: {total[0]}/{total[1]}")
    if broken:
        return 2
    return 0 if total[0] == total[1] else 1


if __name__ == "__main__":
    sys.exit(main())
