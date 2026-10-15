#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for reproducible randomness in
# machine-learning pipelines for its clients. If your team needs expertise in
# seeding, experiment provenance or statistically honest evaluation, you can
# procure our services by sending an email to info@swedishembedded.com.

"""The seed-literal ratchet: a fixed seed in production code needs a reason.

A default seed that is a literal makes every unseeded run the same run. The
rule is that a default seed is drawn, printed, and recorded; a literal seed in
production code is therefore either one of the kinds that must stay fixed, or
a site still to be fixed. Every such site is listed in
scripts/gates/seed-literals.allow.d/*.allow as

    path<TAB>symbol<TAB>reason

where symbol is `<enclosing items>::<matched name>` (so an entry survives a
line moving) and the reason starts with one of

    salt:         a constant mixed into a caller's seed to separate streams
    world:        part of a synthetic world or a protocol definition
    frozen:       the seed of an artifact that is pinned by content
    greedy:       a seed that reaches no random choice (greedy decoding)
    test-double:  a fake that must be deterministic
    bench-input:  a benchmark input that must be identical across runs
    todo:         a site a later change draws instead; counted, never grows

The gate fails on a production hit with no entry, on an entry that matches no
hit (a stale entry: delete it, the ratchet only tightens), and on a reason
without one of those prefixes. Test code is reported, not enforced: a seed
literal in a test should carry `// seed: <why determinism is needed>`.

Detected, in Rust: `seed<...>: <int>`, `seed<...> = <int>`, `seed: Some(<int>)`,
`_or("--<...>seed<...>", <int>)`, `map_or(Ok(<int>)` / `unwrap_or(<int>)` on a
line naming a seed, `.seed(<int>)`, a clap `default_value_t = <int>` on a field
named like a seed, `ParamSpec` seed parameters with an integer `.default`,
`const <...>SEED<...>: T = <int>` and literal constructors of `*Rng`,
`SplitMix64`, `Lcg`. In Python and Makefiles: `add_argument("--<...>seed<...>",
default=<int>)`, `SEED = <int>`, `SEED ?= <int>`.

usage: seed-literals.py [--root DIR] [--allow-dir DIR] [-v]
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

REASON_PREFIXES = ("salt:", "world:", "frozen:", "greedy:", "test-double:", "bench-input:", "todo:")
TEST_MARK = re.compile(r"//\s*seed:")

INT = r"(?:0[xX][0-9A-Fa-f_]+|\d[\d_]*)(?:_?[ui](?:8|16|32|64|128|size))?"
INT_EXPR = INT + r"(?:\s*<<\s*\d+)?"
FIELD_DEFAULT = re.compile(r"\b(\w*[sS]eed\w*)\s*:\s*(?:\w+\s*=\s*|Some\(\s*)?" + INT + r"\b")
ASSIGN_DEFAULT = re.compile(r"\b(\w*[sS]eed\w*)\s*=(?!=)\s*" + INT + r"\b")
OR_FLAG = re.compile(r'_or\(\s*"--([\w-]*seed[\w-]*)"\s*,\s*' + INT)
OR_LITERAL = re.compile(r"\b(map_or|unwrap_or)\(\s*(?:Ok\(\s*)?" + INT + r"\b")
METHOD_SEED = re.compile(r"\.seed\(\s*" + INT + r"\s*\)")
CLAP_DEFAULT = re.compile(r'default_value(?:_t)?\s*=\s*"?' + INT + r'"?')
PARAM_DEFAULT = re.compile(r"\.default\(\s*json!\(\s*" + INT + r"\s*\)\s*\)")
CONST_SEED = re.compile(r"\bconst\s+(\w*SEED\w*)\s*:\s*[\w\[\]; ]+=\s*" + INT_EXPR)
RNG_NEW = re.compile(r"\b(\w*Rng|SplitMix64|Lcg)::(new|seed_from_u64|from_seed)\(\s*" + INT + r"\s*\)")
FIELD_LINE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(\w+)\s*:\s*[A-Za-z&\[(]")

PY_ARG_DEFAULT = re.compile(r"add_argument\(\s*[\"']--([\w-]*seed[\w-]*)[\"'][^)]*\bdefault\s*=\s*" + INT)
PY_ASSIGN = re.compile(r"^\s*([A-Z_]*SEED[A-Z_]*)\s*=\s*" + INT + r"\s*(?:#.*)?$")
MAKE_ASSIGN = re.compile(r"^\s*([A-Z_]*SEED[A-Z_]*)\s*[?:]?=\s*" + INT + r"\s*$")

ITEM = re.compile(
    r"\b(fn)\s+(\w+)"
    r"|^\s*(?:unsafe\s+)?(impl)\b(?:<[^>]*>)?\s+(?:[\w:<>, &']+\s+for\s+)?(\w+)"
    r"|\b(struct|enum|trait|mod)\s+(\w+)"
)
TEST_ATTR = re.compile(r"^\s*#\[(?:cfg\(test\)|test|tokio::test|cfg\(all\(test)")
TEST_MOD_FILE = re.compile(r"#\[cfg\(test\)\]\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)\s*;")


class Hit:
    """One seed literal: where, in which item, what kind, in test code or not."""

    def __init__(self, path, symbol, kind, line, test, marked):
        self.path, self.symbol, self.kind = path, symbol, kind
        self.line, self.test, self.marked = line, test, marked

    def key(self):
        return (self.path, self.symbol)


def split_code(line):
    """Return (code, skeleton): the line without its comment, and the same with
    string and char literals blanked so brace counting ignores their content."""
    code, skel = [], []
    in_str = False
    i = 0
    while i < len(line):
        c = line[i]
        if in_str:
            if c == "\\":
                code.append(line[i : i + 2])
                skel.append("  ")
                i += 2
                continue
            if c == '"':
                in_str = False
                skel.append(c)
            else:
                skel.append(" ")
            code.append(c)
        elif c == '"':
            in_str = True
            code.append(c)
            skel.append(c)
        elif c == "/" and line[i : i + 2] == "//":
            break
        else:
            code.append(c)
            skel.append(c)
        i += 1
    skeleton = re.sub(r"'(?:\\.|[^\\'])'", "' '", "".join(skel))
    return "".join(code), skeleton


def scan_rust(path, text, test_file=False):
    """Hits in one Rust file. `test_file` marks a whole file as test code."""
    raw = text.split("\n")
    hits = []
    depth = paren = 0
    stack = []  # (kind, name, depth the item's body opened at)
    pending = None  # an item header seen, its body not yet opened
    pending_test = False
    test_until = None
    param_context = 0  # lines since a line naming the "seed" parameter

    for idx, line in enumerate(raw):
        code, skel = split_code(line)
        in_test = test_file or test_until is not None
        if TEST_ATTR.match(line):
            pending_test = True

        header = ITEM.search(skel)
        if header and not skel.lstrip().startswith(("use ", "pub use ")):
            groups = [g for g in header.groups() if g]
            pending = (groups[0], groups[1])

        for name_hit in find_hits(code, skel, raw, idx):
            opening = [(*pending, depth)] if pending is not None and "{" in skel else []
            symbol = scope_name(stack + opening) + name_hit[0]
            marked = bool(TEST_MARK.search(raw[idx])) or (idx > 0 and bool(TEST_MARK.search(raw[idx - 1])))
            hits.append(Hit(path, symbol, name_hit[1], idx + 1, in_test or (pending_test and "{" in skel), marked))

        if '"seed"' in code:
            param_context = 5
        elif param_context:
            param_context -= 1

        for ch in skel:
            if ch in "([":
                paren += 1
            elif ch in ")]":
                paren = max(0, paren - 1)
            elif ch == ";" and paren == 0:
                pending = None
                pending_test = False
            elif ch == "{":
                if pending is not None:
                    stack.append((pending[0], pending[1], depth))
                    pending = None
                if pending_test and test_until is None:
                    test_until = depth
                    pending_test = False
                depth += 1
            elif ch == "}":
                depth -= 1
                while stack and stack[-1][2] >= depth:
                    stack.pop()
                if test_until is not None and depth <= test_until:
                    test_until = None
    return hits


def scope_name(stack):
    """`impl::fn::` style prefix of the items enclosing a line (modules omitted)."""
    names = []
    for kind, name, _ in stack:
        if kind != "mod" and (not names or names[-1] != name):
            names.append(name)
    return "".join(n + "::" for n in names)


def find_hits(code, skel, raw, idx):
    """Yield (matched name, kind) for the seed literals on one line. String
    contents are blanked in `skel`, so text that merely says `seed: 1` is not a hit."""
    for m in FIELD_DEFAULT.finditer(skel):
        yield m.group(1), "default"
    for m in ASSIGN_DEFAULT.finditer(skel):
        yield m.group(1), "default"
    for m in OR_FLAG.finditer(code):
        yield "--" + m.group(1), "default"
    if "seed" in code.lower():
        for m in OR_LITERAL.finditer(skel):
            yield m.group(1), "default"
    for m in METHOD_SEED.finditer(skel):
        yield ".seed()", "default"
    for m in CONST_SEED.finditer(skel):
        yield m.group(1), "const"
    for m in RNG_NEW.finditer(skel):
        yield m.group(1) + "::" + m.group(2), "rng"
    if CLAP_DEFAULT.search(code):
        field = clap_field(raw, idx)
        if field and "seed" in field.lower():
            yield field, "default"
    if PARAM_DEFAULT.search(code) and any('"seed"' in raw[j] for j in range(max(0, idx - 5), idx + 1)):
        yield "param:seed", "default"


def clap_field(raw, idx):
    """The field a clap attribute on line idx belongs to: the first following
    line (this one included, after the attribute) that declares `name: Type`."""
    after = re.sub(r"^.*?\]\s*", "", raw[idx]) if "]" in raw[idx] else ""
    candidates = ([after] if after else []) + raw[idx + 1 : idx + 9]
    for line in candidates:
        m = FIELD_LINE.match(line)
        if m and not line.lstrip().startswith("#"):
            return m.group(1)
    return None


def scan_other(path, text):
    """Hits in a Python or Makefile source; these have no test context."""
    hits = []
    for n, line in enumerate(text.split("\n"), 1):
        for m in PY_ARG_DEFAULT.finditer(line):
            hits.append(Hit(path, "--" + m.group(1), "default", n, False, False))
        m = (PY_ASSIGN if path.endswith(".py") else MAKE_ASSIGN).match(line)
        if m:
            hits.append(Hit(path, m.group(1), "const", n, False, False))
    return hits


def is_test_path(path):
    parts = path.split("/")
    return "tests" in parts or "benches" in parts


def test_module_files(path, text):
    """Files that `#[cfg(test)] mod name;` in this file pulls in."""
    base = Path(path)
    owner_dir = base.parent if base.name in ("lib.rs", "main.rs", "mod.rs") else base.parent / base.stem
    for name in TEST_MOD_FILE.findall(text):
        yield (owner_dir / (name + ".rs")).as_posix()
        yield (owner_dir / name / "mod.rs").as_posix()


def scan_tree(root, files):
    """Scan the given repository-relative files; returns all hits."""
    texts = {}
    for rel in files:
        if rel.endswith((".rs", ".py", "Makefile", ".mk")):
            try:
                texts[rel] = (root / rel).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
    test_files = {rel for rel in texts if is_test_path(rel)}
    for rel, text in texts.items():
        if rel.endswith(".rs"):
            test_files.update(test_module_files(rel, text))
    hits = []
    for rel, text in texts.items():
        if rel.endswith(".rs"):
            hits.extend(scan_rust(rel, text, rel in test_files))
        else:
            hits.extend(scan_other(rel, text))
    return hits


def read_allow(allow_dir):
    """Return ({(path, symbol): reason}, [problems]) from every *.allow file."""
    entries, problems = {}, []
    for file in sorted(Path(allow_dir).glob("*.allow")) if Path(allow_dir).is_dir() else []:
        for n, line in enumerate(file.read_text(encoding="utf-8").split("\n"), 1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split("\t")
            where = f"{file.name}:{n}"
            if len(fields) != 3 or not all(f.strip() for f in fields):
                problems.append(f"{where}: expected path<TAB>symbol<TAB>reason")
            elif not fields[2].startswith(REASON_PREFIXES):
                problems.append(f"{where}: reason must start with one of {', '.join(REASON_PREFIXES)}")
            elif (fields[0], fields[1]) in entries:
                problems.append(f"{where}: {fields[0]} {fields[1]} is listed twice")
            else:
                entries[(fields[0], fields[1])] = fields[2]
    return entries, problems


def check(hits, entries):
    """Failures for production hits without an entry and entries without a hit."""
    failures = []
    production = {h.key(): h for h in hits if not h.test}
    for key, hit in sorted(production.items()):
        if key not in entries:
            failures.append(
                f"{hit.path}:{hit.line}: seed literal {hit.symbol} ({hit.kind}) is not allowlisted: "
                "draw the seed and print it, or add an entry with a salt:/world:/frozen:/greedy:/"
                "test-double:/bench-input: reason"
            )
    for key in sorted(entries):
        if key not in production:
            failures.append(f"{key[0]}: allowlist entry {key[1]} matches no seed literal: delete it")
    return failures


def tracked_files(root):
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        capture_output=True,
        check=True,
    )
    return [p for p in out.stdout.decode().split("\0") if p]


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(here.parents[1]))
    ap.add_argument("--allow-dir", default=str(here / "seed-literals.allow.d"))
    ap.add_argument("-v", "--verbose", action="store_true", help="list test seeds without a reason")
    args = ap.parse_args()
    root = Path(args.root)

    hits = scan_tree(root, tracked_files(root))
    entries, problems = read_allow(args.allow_dir)
    failures = problems + check(hits, entries)

    todo = sum(1 for r in entries.values() if r.startswith("todo:"))
    unmarked = [h for h in hits if h.test and not h.marked]
    print(f"seed-literals: {len(entries)} allowlisted production sites, {todo} still todo")
    print(f"seed-literals: {len(unmarked)} test seed literals without a `// seed:` reason (reported, not enforced)")
    if args.verbose:
        for h in unmarked:
            print(f"  {h.path}:{h.line}: {h.symbol}")
    for f in failures:
        print(f"seed-literals: {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
