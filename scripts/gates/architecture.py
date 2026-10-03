#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements architecture gates that keep large Rust
# workspaces layered as they grow. If your team needs expertise in
# dependency architecture or build-time design rules, you can procure our
# services by sending an email to info@swedishembedded.com.

"""Enforces architecture.toml against the real dependency graph.

  1. Every workspace package is listed with a tier, and a normal or build
     dependency points only at a STRICTLY LOWER tier, unless the ordered pair
     is a reviewed `[[same_layer]]` exception. Dev-dependencies are exempt:
     a test may reach for what production code must not.
  2. A `[[forbidden]]` pair is a design invariant: `to` must not be reachable
     from `from` over normal dependencies, at any depth. `to` is a workspace
     crate or an external one (named by a direct dependency of any crate on
     the way).
  3. Only the crates in `[brain].allowed` depend on brain, and only on its
     public SDK, `brain`: an internal `brain-*` crate is refused.
  4. Only the crates in `[sven].allowed` depend (normally) on sven, and only
     on its public SDK, `sven-sdk`: an internal `sven-*` crate is refused.
     A dev-dependency may drive sven directly, so a test can script an agent.
  5. A declared internal dependency must be named by the crate's sources.
  6. When `[surface]` is set, the crates in its tiers may depend on the one
     workspace crate it names and on nothing else of the workspace's.
  7. No tracked .rs file exceeds `max_file_lines`, except a file recorded in
     `[[allow.large_file]]`, whose recorded size may only shrink; an entry
     for a file back under the limit is itself a failure.

Every exception carries a `why`, and an exception that no longer applies is
a failure, so the list cannot rot. The graph comes from `cargo metadata
--no-deps`, the manifests cargo itself resolves, not from a text scan that a
renamed dependency would slip past.

Usage: python3 scripts/gates/architecture.py   (from scripts/gates/check-architecture.sh)
"""
import json
import os
import re
import subprocess
import sys
import tomllib

NORMAL = (None, "build")


def check(arch, packages, file_lines, mentions):
    """Return the list of violations; empty when the workspace obeys the rules.

    `packages` is `[{"name", "deps": [{"name", "kind"}]}]`, `file_lines` maps
    each source file to its line count, and `mentions(package, dependency,
    kind)` says whether the package's sources name the dependency.
    """
    failures = []
    crates = arch.get("crates", {})
    order = arch["tiers"]["order"]
    rank = {tier: i for i, tier in enumerate(order)}
    members = {p["name"] for p in packages}

    for pkg in packages:
        if pkg["name"] not in crates:
            failures.append(f"{pkg['name']}: not listed in architecture.toml")
    for listed, tier in crates.items():
        if listed not in members:
            failures.append(f"{listed}: listed in architecture.toml but not a workspace package")
        if tier not in rank:
            failures.append(f"{listed}: unknown tier '{tier}'")

    internal = {p["name"]: [d["name"] for d in p["deps"] if d["kind"] in NORMAL and d["name"] in members] for p in packages}
    external = {p["name"]: {d["name"] for d in p["deps"] if d["kind"] in NORMAL and d["name"] not in members} for p in packages}

    failures += _tiers(arch, crates, rank, internal)
    failures += _forbidden(arch, internal, external)
    failures += _brain(arch, packages)
    failures += _sven(arch, packages)
    failures += _unused(packages, members, mentions)
    failures += _surface(arch, crates, internal)
    failures += _large_files(arch, file_lines)
    return failures


def _tiers(arch, crates, rank, internal):
    failures = []
    same = {(e["from"], e["to"]) for e in arch.get("same_layer", [])}
    for a, deps in internal.items():
        for b in deps:
            ta, tb = crates.get(a), crates.get(b)
            if ta not in rank or tb not in rank:
                continue
            if rank[tb] > rank[ta]:
                failures.append(f"{a} -> {b}: depends on a higher tier ({tb} is above {ta})")
            elif rank[tb] == rank[ta] and (a, b) not in same:
                failures.append(f"{a} -> {b}: a same-tier ({ta}) dependency needs a [[same_layer]] exception with a why")
    for e in arch.get("same_layer", []):
        a, b = e["from"], e["to"]
        label = f"same_layer {a} -> {b}"
        if not e.get("why"):
            failures.append(f"{label}: an exception needs a why")
        if crates.get(a) != crates.get(b):
            failures.append(f"{label}: the crates are in different tiers; delete the exception")
        elif b not in internal.get(a, []):
            failures.append(f"{label}: the edge no longer exists; delete the exception")
    return failures


def _reach(start, internal, external):
    """Map every crate reachable from `start` to the path that reaches it,
    and every external crate to the path to the crate depending on it."""
    paths = {start: [start]}
    queue = [start]
    while queue:
        node = queue.pop(0)
        for nxt in internal.get(node, []):
            if nxt not in paths:
                paths[nxt] = paths[node] + [nxt]
                queue.append(nxt)
    outside = {}
    for node, path in paths.items():
        for dep in sorted(external.get(node, ())):
            outside.setdefault(dep, path + [dep])
    return paths, outside


def _forbidden(arch, internal, external):
    failures = []
    for rule in arch.get("forbidden", []):
        origin, target = rule["from"], rule["to"]
        if origin not in internal:
            failures.append(f"forbidden {origin} -> {target}: {origin} is not a workspace package")
            continue
        inside, outside = _reach(origin, internal, external)
        path = inside.get(target) if target != origin else None
        path = path or outside.get(target)
        if path:
            failures.append(f"{origin} reaches {target} ({' -> '.join(path)}): {rule.get('why', 'forbidden')}")
    return failures


def _brain(arch, packages):
    failures = []
    allowed = set(arch.get("brain", {}).get("allowed", []))
    for pkg in packages:
        for dep in pkg["deps"]:
            name = dep["name"]
            if name.startswith("brain-"):
                failures.append(f"{pkg['name']} -> {name}: a brain crate beyond the public SDK; use `brain`, and publish what it lacks in the brain SDK")
            elif name == "brain" and pkg["name"] not in allowed:
                failures.append(f"{pkg['name']} -> brain: only {', '.join(sorted(allowed))} may depend on brain")
    return failures


def _sven(arch, packages):
    rule = arch.get("sven")
    if rule is None:
        return []
    failures = []
    allowed = set(rule.get("allowed", []))
    for pkg in packages:
        for dep in pkg["deps"]:
            name = dep["name"]
            if dep["kind"] not in NORMAL:
                continue
            if name.startswith("sven-") and name != "sven-sdk":
                failures.append(f"{pkg['name']} -> {name}: a sven crate beyond the public SDK; use `sven-sdk`, and publish what it lacks in the sven SDK")
            elif name == "sven-sdk" and pkg["name"] not in allowed:
                failures.append(f"{pkg['name']} -> sven-sdk: only {', '.join(sorted(allowed))} may depend on sven")
    return failures


def _unused(packages, members, mentions):
    failures = []
    for pkg in packages:
        for dep in pkg["deps"]:
            if dep["name"] in members and not mentions(pkg["name"], dep["name"], dep["kind"]):
                failures.append(f"{pkg['name']} -> {dep['name']}: declared but never named in its sources; remove the dependency")
    return failures


def _surface(arch, crates, internal):
    rule = arch.get("surface")
    if not rule:
        return []
    failures = []
    only = rule["only"]
    for name, deps in internal.items():
        tier = crates.get(name)
        if tier in rule["tiers"]:
            for dep in deps:
                if dep != only:
                    failures.append(f"{name} -> {dep}: a {tier} crate may depend only on {only}; if it needs more, {only} is missing something")
    return failures


def _large_files(arch, file_lines):
    failures = []
    limit = arch["limits"]["max_file_lines"]
    recorded = {e["path"]: e for e in arch.get("allow", {}).get("large_file", [])}
    for path, lines in sorted(file_lines.items()):
        if lines <= limit:
            continue
        entry = recorded.get(path)
        if entry is None:
            failures.append(f"{path}: {lines} lines, over the {limit}-line limit - split it")
        elif lines > entry["lines"]:
            failures.append(f"{path}: {lines} lines, grew past its recorded {entry['lines']} lines (an allow.large_file entry only shrinks)")
    for path in sorted(recorded):
        if file_lines.get(path, 0) <= limit:
            failures.append(f"{path}: is back under the limit; delete its allow.large_file entry")
    return failures


def _workspace():
    """The real workspace: packages from cargo, source facts from the tree."""
    out = subprocess.run(["cargo", "metadata", "--no-deps", "--offline", "--format-version", "1"],
                         capture_output=True, text=True, check=True).stdout
    meta = json.loads(out)
    packages, dirs = [], {}
    for p in meta["packages"]:
        packages.append({"name": p["name"], "deps": [{"name": d["name"], "kind": d["kind"]} for d in p["dependencies"]]})
        dirs[p["name"]] = os.path.dirname(p["manifest_path"])

    cache = {}

    def sources(package):
        if package not in cache:
            text = []
            for root, _, names in os.walk(dirs[package]):
                if os.sep + "target" in root:
                    continue
                for n in names:
                    if n.endswith(".rs"):
                        with open(os.path.join(root, n), encoding="utf-8") as fh:
                            text += [ln for ln in fh if not ln.lstrip().startswith("//")]
            cache[package] = "".join(text)
        return cache[package]

    def mentions(package, dep, kind):
        return re.search(r"\b" + re.escape(dep.replace("-", "_")) + r"\b", sources(package)) is not None

    listed = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.rs"],
                            capture_output=True, text=True, check=True).stdout.split()
    file_lines = {}
    for path in listed:
        # A tracked file deleted in the working tree is no longer a source file.
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                file_lines[path] = sum(1 for _ in fh)
    return packages, file_lines, mentions


def main():
    os.chdir(subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip())
    with open("architecture.toml", "rb") as fh:
        arch = tomllib.load(fh)
    packages, file_lines, mentions = _workspace()
    failures = check(arch, packages, file_lines, mentions)
    if failures:
        print("check-architecture: FAILED")
        print("\n".join("  " + f for f in failures))
        return 1
    print("check-architecture: OK (%d packages)" % len(packages))
    return 0


if __name__ == "__main__":
    sys.exit(main())
