#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements benchmark suites for code-editing agents that
# stay trustworthy as they grow, for its clients. If your team needs expertise
# in executable task catalogs or mutation-checked test suites, you can procure
# our services by sending an email to info@swedishembedded.com.

"""Checks of the repository task families under tasks/.

A repository family is a small self-contained project that `splinter practice`
learns from its own tests. This script is the family author's pre-flight and
the catalog's keeper; it does not replace the generator's admission funnel.

  check [TREE...]      baseline suite passes and is identical over five runs,
                       fast, no skips, conventions hold (headers, ASCII,
                       sizes, stdlib only, test layout), identifiers are
                       disjoint across families, and the catalog entry (when
                       there is one) matches the tree
  mutate [TREE...]     a few faults per file by line edits; the suite must
                       catch nearly every fault that still runs (pre-flight
                       only: the generator's own operators decide admission)
  catalog [--write]    recompute the catalog's measured fields (size, tests,
                       digest) and print or write tasks/catalog.toml

Without TREE the families of the catalog are used. A TREE is a family
directory such as tasks/ttlcache. Fault drawing takes `--seed N`; without
one a seed is drawn and printed so the run can be repeated.
"""
import argparse
import ast
import hashlib
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import tomllib
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "tasks"
CATALOG = TASKS / "catalog.toml"

RUNS = 5
MAX_SUITE_SECONDS = 5.0
FAULT_TIMEOUT_SECONDS = 15.0  # a fault that hangs the suite is caught
# The generator keeps 64 KiB of a suite's report; a fault that fails tests adds
# a traceback each, so the passing report has to leave most of it free.
MAX_REPORT_BYTES = 32000
MAX_FILE_LINES = 800
MIN_FUNCTIONS = 12
MIN_KILL_RATE = 0.85
HEADER = (
    "SPDX-License-Identifier: Apache-2.0",
    "Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>",
)
TEST_COMMAND = ["python3", "-m", "unittest"]
RESULT_LINE = re.compile(r"^(\S+) \((\S+)\) \.\.\. (.+)$")
# Characters the generator's runner reads as a pattern in a test id, and a
# stem long enough to be taken for a quotation of the hidden tests.
PATTERN_CHARS = "*?["
MAX_STEM_WORDS = 7
MIN_STEM_WORDS = 4


class Family:
    """One family directory and what is read from it."""

    def __init__(self, path: Path):
        self.path = path.resolve()
        self.id = self.path.name
        self.lua = (self.path / "src").is_dir() and any((self.path / "src").glob("*.lua"))
        self.language = "lua" if self.lua else "python"

    def sources(self) -> list[Path]:
        """The files tasks may change, in path order."""
        if self.lua:
            return sorted((self.path / "src").glob("*.lua"))
        packages = [
            d for d in sorted(self.path.iterdir())
            if d.is_dir() and (d / "__init__.py").is_file() and d.name != "tests"
        ]
        return sorted(p for d in packages for p in d.glob("*.py") if p.name != "__init__.py")

    def tests(self) -> list[Path]:
        return sorted((self.path / "tests").glob("test_*.py"))

    def files(self) -> list[Path]:
        return sorted(
            p for p in self.path.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
        )

    def rel(self, path: Path) -> str:
        return path.relative_to(self.path).as_posix()


def digest(family: Family) -> str:
    h = hashlib.sha256()
    for path in family.files():
        h.update(family.rel(path).encode() + b"\0")
        h.update(hashlib.sha256(path.read_bytes()).digest())
    return "sha256:" + h.hexdigest()


def run_suite(root: Path, extra=(), timeout: float = 120.0) -> tuple[int, str, float]:
    """Exit status, the runner's report and seconds; 124 when stopped by the timeout."""
    started = time.monotonic()
    try:
        done = subprocess.run(
            TEST_COMMAND + ["-v", *extra], cwd=root, capture_output=True, text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return 124, "", time.monotonic() - started
    return done.returncode, done.stderr, time.monotonic() - started


def outcomes(report: str) -> dict[str, str]:
    found = {}
    for line in report.splitlines():
        m = RESULT_LINE.match(line)
        if m:
            found[m.group(2)] = m.group(3)
    return found


# ----------------------------------------------------------------- conventions

def defined_names(family: Family) -> set[str]:
    """Top-level function and class names (module functions for Lua)."""
    names: set[str] = set()
    for path in family.sources():
        text = path.read_text(encoding="utf-8")
        if family.lua:
            for m in re.finditer(r"function\s+(?:[\w]+[.:])*([A-Za-z_]\w*)\s*\(", text):
                names.add(m.group(1))
        else:
            for node in ast.parse(text).body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    names.add(node.name)
    return {n for n in names if not (n.startswith("__") and n.endswith("__"))}


def count_functions(family: Family) -> int:
    total = 0
    for path in family.sources():
        text = path.read_text(encoding="utf-8")
        if family.lua:
            total += len(re.findall(r"\bfunction\b\s*[\w.:]*\s*\(", text))
        else:
            total += sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                         for n in ast.walk(ast.parse(text)))
    return total


def lines_of_code(family: Family) -> int:
    return sum(
        1 for path in family.sources() for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith(("#", "--"))
    )


def stdlib_only(family: Family) -> list[str]:
    problems = []
    own = {p.name for p in family.path.iterdir() if p.is_dir()} | {"tests"}
    for path in family.sources() + family.tests() + [family.path / "tests" / "__init__.py"]:
        if path.suffix != ".py" or not path.exists():
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods = [node.module.split(".")[0]]
            for mod in mods:
                if mod not in sys.stdlib_module_names and mod not in own:
                    problems.append(f"{family.rel(path)}: imports {mod}, not the standard library")
    return problems


def hygiene(family: Family) -> list[str]:
    problems = []
    for path in family.files():
        rel = family.rel(path)
        if path.suffix in (".png", ".bin"):
            problems.append(f"{rel}: binary file")
            continue
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        if len(lines) > MAX_FILE_LINES:
            problems.append(f"{rel}: {len(lines)} lines, over {MAX_FILE_LINES}")
        for n, line in enumerate(lines, 1):
            stray = [c for c in line if ord(c) > 127 and c != "\u00f6"]
            if stray or ("\u00f6" in line and "Schr\u00f6der" not in line):
                problems.append(f"{rel}:{n}: non-ASCII character")
                break
        if path.suffix in (".py", ".lua", ".toml", ".md") and not any(
            HEADER[0] in line for line in lines[:3]
        ):
            problems.append(f"{rel}: no SPDX line in the first three lines")
        elif path.suffix in (".py", ".lua", ".toml", ".md") and not any(
            HEADER[1] in line for line in lines[:4]
        ):
            problems.append(f"{rel}: no copyright line next to the SPDX line")
    return problems


def layout(family: Family) -> list[str]:
    problems = []
    for needed in ("README.md", "tests/__init__.py"):
        if not (family.path / needed).is_file():
            problems.append(f"missing {needed}")
    if not family.tests():
        problems.append("no tests/test_*.py")
    stems = {p.stem for p in family.sources()}
    covered = {t.stem.removeprefix("test_") for t in family.tests()}
    for stem in sorted(stems - covered):
        problems.append(f"{stem}: no tests/test_{stem}.py of its own")
    if len(family.sources()) < 6:
        problems.append(f"{len(family.sources())} source files, the family needs at least 6")
    if count_functions(family) < MIN_FUNCTIONS:
        problems.append(f"fewer than {MIN_FUNCTIONS} functions")
    return problems


def test_names(report: dict[str, str]) -> list[str]:
    problems = []
    for tid in report:
        name = tid.rsplit(".", 1)[-1]
        if any(c in name for c in PATTERN_CHARS):
            problems.append(f"{tid}: a pattern character in the test name")
        stem = name.removeprefix("test_").split("_")
        if not MIN_STEM_WORDS <= len(stem) <= MAX_STEM_WORDS:
            problems.append(
                f"{tid}: {len(stem)} words after test_, want {MIN_STEM_WORDS} to {MAX_STEM_WORDS}"
            )
    return problems


def check_one(family: Family) -> list[str]:
    problems = layout(family) + hygiene(family) + stdlib_only(family)
    first: dict[str, str] | None = None
    report_bytes = 0
    slowest = 0.0
    for run in range(RUNS):
        code, report, took = run_suite(family.path)
        slowest = max(slowest, took)
        got = outcomes(report)
        report_bytes = len(report.encode())
        if code != 0:
            problems.append(f"run {run + 1}: the baseline suite fails (exit {code})")
            problems += [f"  {t}: {s}" for t, s in got.items() if s != "ok"][:10]
            break
        if first is None:
            first = got
        elif got != first:
            problems.append(f"run {run + 1}: outcomes differ from run 1 (not deterministic)")
    if first is not None:
        bad = {t: s for t, s in first.items() if s != "ok"}
        problems += [f"{t}: {s}, a test must pass without skipping" for t, s in bad.items()]
        problems += test_names(first)
        if report_bytes > MAX_REPORT_BYTES:
            problems.append(
                f"the verbose report is {report_bytes} bytes, over {MAX_REPORT_BYTES}: "
                "fewer or more compact tests (the generator truncates at 64 KiB)"
            )
        if slowest > MAX_SUITE_SECONDS:
            problems.append(f"slowest run {slowest:.1f}s, over {MAX_SUITE_SECONDS}s")
        print(f"{family.id}: {len(first)} tests, slowest of {RUNS} runs {slowest:.2f}s, "
              f"{len(family.sources())} source files, {count_functions(family)} functions")
    return problems


def disjoint(families: list[Family]) -> list[str]:
    owners: dict[str, list[str]] = {}
    for fam in families:
        for name in defined_names(fam):
            owners.setdefault(name, []).append(fam.id)
    return [f"identifier {n} defined by {', '.join(o)}" for n, o in sorted(owners.items()) if len(o) > 1]


# -------------------------------------------------------------------- catalog

def load_catalog() -> list[dict]:
    if not CATALOG.exists():
        return []
    return tomllib.loads(CATALOG.read_text(encoding="utf-8")).get("family", [])


def measured(family: Family) -> dict:
    code, report, _ = run_suite(family.path)
    if code != 0:
        raise SystemExit(f"{family.id}: baseline fails; refusing to describe it")
    return {
        "files_under_test": [family.rel(p) for p in family.sources()],
        "test_files": len(family.tests()),
        "tests": len(outcomes(report)),
        "functions": count_functions(family),
        "lines_of_code": lines_of_code(family),
        "digest": digest(family),
    }


HAND_FIELDS = ("id", "kind", "language", "domain", "runner", "summary")
MEASURED_FIELDS = ("lines_of_code", "functions", "tests", "test_files", "digest", "files_under_test")
EXTRA_FIELDS = ("generation_seed", "generation_flags", "admitted", "funnel")


def toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, list):
        if all(isinstance(v, str) for v in value):
            return "[" + ", ".join(toml_value(v) for v in value) + "]"
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{k} = {toml_value(v)}" for k, v in value.items()) + " }"
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_catalog(entries: list[dict]) -> str:
    head = (
        "# SPDX-License-Identifier: Apache-2.0\n"
        "# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>\n#\n"
        "# The repository task families: small original projects that `splinter\n"
        "# practice` learns from their own tests. Measured fields are written by\n"
        "# `python3 scripts/tasks/families.py catalog --write`; do not edit them.\n"
    )
    out = [head]
    for entry in entries:
        out.append("\n[[family]]\n")
        for key in (*HAND_FIELDS, *MEASURED_FIELDS, *EXTRA_FIELDS):
            if key in entry:
                out.append(f"{key} = {toml_value(entry[key])}\n")
    return "".join(out)


def catalog_command(write: bool) -> int:
    entries = load_catalog()
    if not entries:
        print("tasks/catalog.toml has no [[family]] entries to describe", file=sys.stderr)
        return 1
    fresh = []
    for entry in entries:
        updated = dict(entry)
        updated.update(measured(Family(TASKS / entry["id"])))
        fresh.append(updated)
    text = render_catalog(fresh)
    if write:
        CATALOG.write_text(text, encoding="utf-8")
        print(f"wrote {CATALOG.relative_to(ROOT)} ({len(fresh)} families)")
    else:
        sys.stdout.write(text)
    return 0


def check_catalog(families: list[Family]) -> list[str]:
    problems = []
    entries = {e["id"]: e for e in load_catalog()}
    for fam in families:
        entry = entries.get(fam.id)
        if entry is None:
            problems.append(f"{fam.id}: not in tasks/catalog.toml")
            continue
        if entry.get("language") != fam.language:
            problems.append(f"{fam.id}: catalog says {entry.get('language')}, the tree is {fam.language}")
        if entry.get("digest") != digest(fam):
            problems.append(f"{fam.id}: the catalog digest is stale; run catalog --write")
    return problems


# -------------------------------------------------------------------- mutation

COMPARISONS = [("==", "!="), ("!=", "=="), ("~=", "=="), ("<=", "<"), (">=", ">"),
               (" < ", " <= "), (" > ", " >= ")]
ARITHMETIC = [(" + ", " - "), (" - ", " + "), (" and ", " or "), (" or ", " and ")]
WORDS = [("True", "False"), ("False", "True"), ("true", "false"), ("false", "true")]


def docstring_lines(text: str) -> set[int]:
    lines: set[int] = set()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return lines
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) \
                    and isinstance(first.value.value, str):
                lines.update(range(first.lineno, first.end_lineno + 1))
    return lines


def candidates(family: Family, path: Path) -> list[tuple[int, str]]:
    """(line number, edit name) pairs; edits are applied by `edit`."""
    text = path.read_text(encoding="utf-8")
    skip = set() if family.lua else docstring_lines(text)
    found = []
    for n, line in enumerate(text.splitlines(), 1):
        code = line.strip()
        if not code or code.startswith(("#", "--")) or n in skip:
            continue
        found.append((n, "delete"))
        for tag, table in (("cmp", COMPARISONS), ("arith", ARITHMETIC), ("bool", WORDS)):
            for i, (old, _) in enumerate(table):
                if re.search(r"\b" + re.escape(old) + r"\b" if old.isalpha() else re.escape(old), line):
                    found.append((n, f"{tag}{i}"))
        if re.search(r"(?<![\w.\"'])\d+(?![\w.\"'])", line):
            found.append((n, "int+1"))
            found.append((n, "int-1"))
    return found


def edit(line: str, name: str) -> str:
    if name == "delete":
        return ""
    if name in ("int+1", "int-1"):
        step = 1 if name == "int+1" else -1
        return re.sub(r"(?<![\w.\"'])\d+(?![\w.\"'])", lambda m: str(max(0, int(m.group()) + step)),
                      line, count=1)
    for tag, table in (("cmp", COMPARISONS), ("arith", ARITHMETIC), ("bool", WORDS)):
        if name.startswith(tag):
            old, new = table[int(name[len(tag):])]
            pattern = r"\b" + re.escape(old) + r"\b" if old.isalpha() else re.escape(old)
            return re.sub(pattern, new, line, count=1)
    raise ValueError(name)


def runs_as_code(family: Family, path: Path) -> bool:
    if family.lua:
        done = subprocess.run(["luac", "-p", str(path)], capture_output=True)
        return done.returncode == 0
    try:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
        return True
    except SyntaxError:
        return False


def mutate_one(family: Family, per_file: int, rng: random.Random) -> list[str]:
    survivors, tried, killed, broken = [], 0, 0, 0
    with tempfile.TemporaryDirectory() as scratch:
        copy = Path(scratch) / family.id
        shutil.copytree(family.path, copy, ignore=shutil.ignore_patterns("__pycache__"))
        twin = Family(copy)
        for source in twin.sources():
            original = source.read_text(encoding="utf-8")
            pool = candidates(twin, source)
            for n, name in rng.sample(pool, min(per_file, len(pool))):
                lines = original.splitlines(keepends=True)
                changed = edit(lines[n - 1].rstrip("\n"), name)
                if changed == lines[n - 1].rstrip("\n"):
                    continue
                lines[n - 1] = changed + ("\n" if lines[n - 1].endswith("\n") else "")
                source.write_text("".join(lines), encoding="utf-8")
                try:
                    if not runs_as_code(twin, source):
                        broken += 1
                        continue
                    tried += 1
                    code, _, _ = run_suite(copy, extra=("-f",), timeout=FAULT_TIMEOUT_SECONDS)
                    if code != 0:
                        killed += 1
                    else:
                        survivors.append(f"{twin.rel(source)}:{n} {name}: {lines[n - 1].strip()!r}")
                finally:
                    source.write_text(original, encoding="utf-8")
    rate = killed / tried if tried else 0.0
    print(f"{family.id}: {tried} runnable faults, {killed} caught ({rate:.0%}), {broken} unrunnable")
    problems = [f"survived: {s}" for s in survivors]
    if rate < MIN_KILL_RATE:
        problems.insert(0, f"kill rate {rate:.0%} is under {MIN_KILL_RATE:.0%}")
        return problems
    for s in survivors:
        print(f"  note: {s}")
    return []


# ------------------------------------------------------------------------ main

def resolve(trees: list[str]) -> list[Family]:
    if trees:
        return [Family(Path(t)) for t in trees]
    return [Family(TASKS / e["id"]) for e in load_catalog()]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check")
    check.add_argument("trees", nargs="*")
    mutate = sub.add_parser("mutate")
    mutate.add_argument("trees", nargs="*")
    mutate.add_argument("--per-file", type=int, default=6)
    mutate.add_argument("--seed", type=int, default=None)
    cat = sub.add_parser("catalog")
    cat.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "catalog":
        return catalog_command(args.write)
    families = resolve(args.trees)
    if not families:
        print("no families: name a tree or fill tasks/catalog.toml", file=sys.stderr)
        return 1
    problems: list[str] = []
    if args.command == "check":
        for fam in families:
            found = check_one(fam)
            problems += [f"{fam.id}: {p}" for p in found]
        if not args.trees:
            problems += check_catalog(families)
        problems += disjoint(families) if len(families) > 1 else []
    else:
        seed = args.seed if args.seed is not None else secrets.randbits(32)
        print(f"fault seed {seed}")
        rng = random.Random(seed)
        for fam in families:
            problems += [f"{fam.id}: {p}" for p in mutate_one(fam, args.per_file, rng)]
    for p in problems:
        print("FAIL " + p, file=sys.stderr)
    print("OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
