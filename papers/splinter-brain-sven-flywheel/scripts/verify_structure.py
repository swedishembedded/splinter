#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reproducible verification of research claims
# against the code they describe, for its clients. If your team needs
# expertise in keeping a paper's statements about a system true as the system
# changes, you can procure our services by sending an email to
# info@swedishembedded.com.

"""Re-checks every STRUCTURE claim of the paper's ledger against the code.

A STRUCTURE claim is a fact about the repositories (a type exists, a constant
has a value, a crate does or does not depend on another, a CLI flag is
declared). Each is backed by one or more checks in data/structure_checks.jsonl
that read the file AT THE PINNED COMMIT (data/commits.json) with `git show`, so
the result does not depend on the state of any working tree.

    verify_structure.py              check every claim; exit 1 on any failure
    verify_structure.py --list       print the checks and their claims
    verify_structure.py --latest     check against each repository's HEAD instead
                                     of the pinned commit (to see what drifted)
    --splinter DIR --sven DIR --brain DIR
                                     where each repository is checked out (default:
                                     the repository holding this script, and
                                     ~/code/sven, ~/code/brain)

Every pin must be contained in a remote-tracking ref of its repository, so a
fresh clone of the remotes can reproduce the result; a pin that exists only as
a local object is a failure (a paper once pinned such a commit).

A check is a JSON object: {"id", "repo", "path", "kind", ...} where kind is
  regex   {"pattern", "min": 1, "max": null}  matches of the pattern in the file
  toml    {"key": "a.b.c", "equals": value}   a value in a TOML file ("contains" for lists)
  absent  {"pattern"}                         the pattern occurs nowhere in the file
  tree    {"pattern", "min", "max"}           matches of a regex over `git ls-tree -r` paths
                                              (path is a directory prefix)
  files   {"glob_regex", "min"}               same as tree, spelled for file counts

A check may carry "pin": a key of data/commits.json other than its repository's
(for example "brain_throughput", a later brain commit that holds a knowledge
record the main pin predates); the key's prefix before "_" names the repository.
"""

import csv
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPOS = {
    "splinter": ROOT.parent.parent,
    "sven": Path.home() / "code" / "sven",
    "brain": Path.home() / "code" / "brain",
}
_cache = {}


def git(repo, *args):
    done = subprocess.run(["git", "-C", str(REPOS[repo]), *args], capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} in {repo}: {done.stderr.strip()}")
    return done.stdout


def read(repo, rev, path):
    key = (repo, rev, path)
    if key not in _cache:
        _cache[key] = git(repo, "show", f"{rev}:{path}")
    return _cache[key]


def tree(repo, rev, prefix):
    key = (repo, rev, "tree:" + prefix)
    if key not in _cache:
        _cache[key] = git(repo, "ls-tree", "-r", "--name-only", rev, "--", prefix).splitlines()
    return _cache[key]


def pin_is_published(repo, rev):
    """True when a remote-tracking ref of the repository contains the commit."""
    return bool(git(repo, "branch", "-r", "--contains", rev).strip())


def lookup(data, dotted):
    for part in dotted.split("."):
        data = data[part]
    return data


def run_check(check, rev):
    kind = check["kind"]
    repo = check["repo"]
    if kind in ("tree", "files"):
        pattern = re.compile(check["pattern"])
        n = sum(1 for p in tree(repo, rev, check["path"]) if pattern.search(p))
        return in_range(n, check), f"{n} paths match"
    text = read(repo, rev, check["path"])
    if kind == "regex":
        n = len(re.findall(check["pattern"], text, flags=re.M | re.S if check.get("dotall") else re.M))
        return in_range(n, check), f"{n} matches"
    if kind == "absent":
        n = len(re.findall(check["pattern"], text, flags=re.M))
        return n == 0, f"{n} matches"
    if kind == "toml":
        value = lookup(tomllib.loads(text), check["key"])
        if "equals" in check:
            return value == check["equals"], f"value {value!r}"
        if "contains" in check:
            return check["contains"] in value, f"value {value!r}"
        if "not_contains" in check:
            return check["not_contains"] not in value, f"value {value!r}"
    raise ValueError(f"unknown check {check}")


def in_range(n, check):
    lo = check.get("min", 1)
    hi = check.get("max")
    return n >= lo and (hi is None or n <= hi)


def main():
    latest = "--latest" in sys.argv
    for repo in REPOS:
        flag = f"--{repo}"
        if flag in sys.argv:
            REPOS[repo] = Path(sys.argv[sys.argv.index(flag) + 1]).resolve()
    commits = json.loads((ROOT / "data" / "commits.json").read_text())
    checks = [json.loads(line) for line in (ROOT / "data" / "structure_checks.jsonl").read_text().splitlines()
              if line.strip() and not line.startswith("#")]
    by_id = {}
    for c in checks:
        if c["id"] in by_id:
            sys.exit(f"duplicate check id {c['id']}")
        by_id[c["id"]] = c
    with open(ROOT / "data" / "claims.csv", newline="", encoding="utf-8") as f:
        claims = list(csv.DictReader(f))
    if "--list" in sys.argv:
        for cl in claims:
            if cl["kind"] == "STRUCTURE":
                print(cl["id"], cl["evidence"])
        return
    failed = 0
    used = set()
    n_claims = 0
    if not latest:
        for key, rev in commits.items():
            repo = key.split("_")[0]
            try:
                published = pin_is_published(repo, rev)
            except RuntimeError as e:
                published = False
                print(f"FAIL pin {key}: {e}")
            if not published:
                print(f"FAIL pin {key} {rev[:8]} is contained in no remote-tracking ref of {REPOS[repo]}")
                failed += 1
    for cl in claims:
        if cl["kind"] != "STRUCTURE":
            continue
        n_claims += 1
        ids = [i for i in cl["evidence"].split(";") if i]
        if not ids:
            print(f"FAIL {cl['id']}: no checks")
            failed += 1
        for i in ids:
            used.add(i)
            check = by_id.get(i)
            if check is None:
                print(f"FAIL {cl['id']}: unknown check {i}")
                failed += 1
                continue
            rev = "HEAD" if latest else commits[check.get("pin", check["repo"])]
            try:
                ok, detail = run_check(check, rev)
            except Exception as e:  # a missing file at the pinned commit is a failure
                ok, detail = False, f"error: {e}"
            if not ok:
                failed += 1
                print(f"FAIL {cl['id']} {i} ({check['repo']}@{rev[:8]} {check['path']}): {detail}")
    unused = set(by_id) - used
    if unused:
        print(f"FAIL checks no claim uses: {sorted(unused)}")
        failed += len(unused)
    print(f"{n_claims} STRUCTURE claims, {len(used)} checks, {failed} failures")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
