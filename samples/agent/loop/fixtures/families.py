# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements self-improving coding agents whose every
# step is auditable, for its clients. If your team needs expertise in agent
# evaluation or locally operated coding agents, you can procure our
# services by sending an email to info@swedishembedded.com.
"""Families of disposable repositories with one seeded bug each.

Each family is a small Python function with a docstring that says what it
does, a buggy implementation that violates the docstring in a way the
repository's own tests do not catch, and acceptance cases that do. A family
is rendered under a seeded choice of package and function names, so the same
bug appears in repositories that look different.

    python3 families.py render <out-dir> --family NAME [--seed N]
    python3 families.py list
    python3 families.py check            # buggy fails, fixed passes, repo tests pass on both

`render` writes <out-dir>/repo (a git repository with the buggy code
committed) and <out-dir>/accept.py (kept outside the repository). The fixed
implementation is used only by `check`; it is never written next to the
repository.
"""
import argparse
import os
import random
import subprocess
import sys
import tempfile

NAMES = ["alpha", "bravo", "cargo", "delta", "ember", "fjord", "gamma", "harbor", "ivory", "jungle",
         "kernel", "lantern", "meadow", "nectar", "orchid", "pebble", "quartz", "ripple", "summit", "tundra"]

FAMILIES = {}


def family(name, fn, doc, buggy, fixed, repo_tests, accept, task=None):
    FAMILIES[name] = dict(fn=fn, doc=doc, buggy=buggy, fixed=fixed, repo_tests=repo_tests, accept=accept,
                          task=task or "The function {pkg}.{fn} does not behave as its docstring says in some cases. Find the bug in the library code and fix it.")


family(
    "slugify", "slugify",
    "Lower-case the text and join its alphanumeric runs with single hyphens; no leading or trailing hyphen.",
    '''import re


def slugify(text):
    """Lower-case the text and join its alphanumeric runs with single hyphens; no leading or trailing hyphen."""
    return re.sub(r"[^a-z0-9]", "-", text.lower())
''',
    '''import re


def slugify(text):
    """Lower-case the text and join its alphanumeric runs with single hyphens; no leading or trailing hyphen."""
    return "-".join(re.findall(r"[a-z0-9]+", text.lower()))
''',
    'self.assertEqual(slugify("Hello World"), "hello-world")\n        self.assertEqual(slugify("a"), "a")',
    [('slugify("Hello,  World!")', '"hello-world"'), ('slugify("  -- A b--c  ")', '"a-b-c"'), ('slugify("!!!")', '""'), ('slugify("x1 y2")', '"x1-y2"')],
)

family(
    "flatten", "flatten",
    "Return a flat list of the items of an arbitrarily nested list or tuple, in order; strings are items, not sequences.",
    '''def flatten(items):
    """Return a flat list of the items of an arbitrarily nested list or tuple, in order; strings are items, not sequences."""
    out = []
    for item in items:
        if isinstance(item, (list, tuple)):
            out.extend(item)
        else:
            out.append(item)
    return out
''',
    '''def flatten(items):
    """Return a flat list of the items of an arbitrarily nested list or tuple, in order; strings are items, not sequences."""
    out = []
    for item in items:
        if isinstance(item, (list, tuple)):
            out.extend(flatten(item))
        else:
            out.append(item)
    return out
''',
    'self.assertEqual(flatten([1, [2, 3]]), [1, 2, 3])\n        self.assertEqual(flatten([]), [])',
    [('flatten([1, [2, [3, [4]]]])', '[1, 2, 3, 4]'), ('flatten(["ab", ["cd", ("e",)]])', '["ab", "cd", "e"]'), ('flatten([[], [[]], 5])', '[5]')],
)

family(
    "chunk", "chunk",
    "Split a sequence into consecutive lists of n items; the last list holds the remainder and is never empty. n must be positive or ValueError is raised.",
    '''def chunk(seq, n):
    """Split a sequence into consecutive lists of n items; the last list holds the remainder and is never empty. n must be positive or ValueError is raised."""
    if n <= 0:
        raise ValueError("n must be positive")
    return [list(seq[i:i + n]) for i in range(0, len(seq) - n + 1, n)]
''',
    '''def chunk(seq, n):
    """Split a sequence into consecutive lists of n items; the last list holds the remainder and is never empty. n must be positive or ValueError is raised."""
    if n <= 0:
        raise ValueError("n must be positive")
    return [list(seq[i:i + n]) for i in range(0, len(seq), n)]
''',
    'self.assertEqual(chunk([1, 2, 3, 4], 2), [[1, 2], [3, 4]])\n        with self.assertRaises(ValueError):\n            chunk([1], 0)',
    [('chunk([1, 2, 3, 4, 5], 2)', '[[1, 2], [3, 4], [5]]'), ('chunk("abcdefg", 3)', '[["a", "b", "c"], ["d", "e", "f"], ["g"]]'), ('chunk([1], 5)', '[[1]]'), ('chunk([], 3)', '[]')],
)

family(
    "rle", "encode",
    "Run-length encode a string as <count><char> pairs, for example 'aaabcc' becomes '3a1b2c'; the empty string encodes to the empty string.",
    '''def encode(text):
    """Run-length encode a string as <count><char> pairs, for example 'aaabcc' becomes '3a1b2c'; the empty string encodes to the empty string."""
    out = []
    count = 1
    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            count += 1
        else:
            out.append(f"{count}{text[i - 1]}")
            count = 1
    return "".join(out)
''',
    '''def encode(text):
    """Run-length encode a string as <count><char> pairs, for example 'aaabcc' becomes '3a1b2c'; the empty string encodes to the empty string."""
    out = []
    count = 1
    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            count += 1
        else:
            out.append(f"{count}{text[i - 1]}")
            count = 1
    if text:
        out.append(f"{count}{text[-1]}")
    return "".join(out)
''',
    'self.assertEqual(encode(""), "")\n        self.assertEqual(encode("aab"), "2a1b") if False else None',
    [('encode("aaabcc")', '"3a1b2c"'), ('encode("z")', '"1z"'), ('encode("aabb")', '"2a2b"'), ('encode("")', '""')],
)

family(
    "median", "median",
    "Return the median of a non-empty list of numbers in any order; for an even count, the mean of the two middle values.",
    '''def median(values):
    """Return the median of a non-empty list of numbers in any order; for an even count, the mean of the two middle values."""
    n = len(values)
    return values[n // 2]
''',
    '''def median(values):
    """Return the median of a non-empty list of numbers in any order; for an even count, the mean of the two middle values."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2
''',
    'self.assertEqual(median([1, 2, 3]), 2)\n        self.assertEqual(median([5]), 5)',
    [('median([3, 1, 2])', '2'), ('median([4, 1, 3, 2])', '2.5'), ('median([10, 0])', '5.0'), ('median([7, 7, 1, 9, 5])', '7')],
)

family(
    "intervals", "merge",
    "Merge overlapping or touching closed intervals given as (start, end) pairs in any order; return a sorted list of tuples.",
    '''def merge(intervals):
    """Merge overlapping or touching closed intervals given as (start, end) pairs in any order; return a sorted list of tuples."""
    out = []
    for start, end in intervals:
        if out and start < out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out
''',
    '''def merge(intervals):
    """Merge overlapping or touching closed intervals given as (start, end) pairs in any order; return a sorted list of tuples."""
    out = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out
''',
    'self.assertEqual(merge([(1, 3), (2, 4)]), [(1, 4)])\n        self.assertEqual(merge([(1, 2), (5, 6)]), [(1, 2), (5, 6)])',
    [('merge([(5, 6), (1, 3), (2, 4)])', '[(1, 4), (5, 6)]'), ('merge([(1, 2), (2, 3)])', '[(1, 3)]'), ('merge([(1, 10), (2, 3)])', '[(1, 10)]'), ('merge([])', '[]')],
)

family(
    "roman", "from_roman",
    "Convert a Roman numeral (I V X L C D M, with subtractive pairs such as IV and CM) to an integer.",
    '''VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def from_roman(text):
    """Convert a Roman numeral (I V X L C D M, with subtractive pairs such as IV and CM) to an integer."""
    return sum(VALUES[c] for c in text)
''',
    '''VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def from_roman(text):
    """Convert a Roman numeral (I V X L C D M, with subtractive pairs such as IV and CM) to an integer."""
    total = 0
    for i, c in enumerate(text):
        value = VALUES[c]
        if i + 1 < len(text) and VALUES[text[i + 1]] > value:
            total -= value
        else:
            total += value
    return total
''',
    'self.assertEqual(from_roman("VIII"), 8)\n        self.assertEqual(from_roman("XXX"), 30)',
    [('from_roman("IV")', '4'), ('from_roman("MCMXCIV")', '1994'), ('from_roman("XLII")', '42'), ('from_roman("CDXLIV")', '444')],
)

family(
    "duration", "parse_duration",
    "Parse a duration such as '1h30m', '45s' or '2d4h' into seconds. Units are d, h, m, s; each at most once; an empty string or an unknown unit raises ValueError.",
    '''import re

UNITS = {"d": 86400, "h": 3600, "m": 60, "s": 1}


def parse_duration(text):
    """Parse a duration such as '1h30m', '45s' or '2d4h' into seconds. Units are d, h, m, s; each at most once; an empty string or an unknown unit raises ValueError."""
    if not text:
        raise ValueError("empty duration")
    total = 0
    for number, unit in re.findall(r"(\\d+)([a-z])", text):
        total += int(number) * UNITS[unit]
    return total
''',
    '''import re

UNITS = {"d": 86400, "h": 3600, "m": 60, "s": 1}


def parse_duration(text):
    """Parse a duration such as '1h30m', '45s' or '2d4h' into seconds. Units are d, h, m, s; each at most once; an empty string or an unknown unit raises ValueError."""
    if not re.fullmatch(r"(\\d+[dhms])+", text or ""):
        raise ValueError(f"not a duration: {text!r}")
    seen = set()
    total = 0
    for number, unit in re.findall(r"(\\d+)([dhms])", text):
        if unit in seen:
            raise ValueError(f"unit {unit} twice")
        seen.add(unit)
        total += int(number) * UNITS[unit]
    return total
''',
    'self.assertEqual(parse_duration("1h30m"), 5400)\n        self.assertEqual(parse_duration("45s"), 45)',
    [('parse_duration("2d4h")', '187200'), ('parse_duration("1m1s")', '61')],
)
FAMILIES["duration"]["raises"] = [('parse_duration("")'), ('parse_duration("5x")'), ('parse_duration("1h1h")'), ('parse_duration("h")')]

family(
    "leap", "is_leap_year",
    "Return True for a Gregorian leap year: divisible by 4, except century years, which must also be divisible by 400.",
    '''def is_leap_year(year):
    """Return True for a Gregorian leap year: divisible by 4, except century years, which must also be divisible by 400."""
    return year % 4 == 0
''',
    '''def is_leap_year(year):
    """Return True for a Gregorian leap year: divisible by 4, except century years, which must also be divisible by 400."""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
''',
    'self.assertTrue(is_leap_year(2024))\n        self.assertFalse(is_leap_year(2023))',
    [('is_leap_year(1900)', 'False'), ('is_leap_year(2000)', 'True'), ('is_leap_year(2100)', 'False'), ('is_leap_year(1996)', 'True')],
)

family(
    "bisect", "index_of",
    "Return the index of the first occurrence of target in the sorted list, or -1 when it is absent. Runs in logarithmic time.",
    '''def index_of(items, target):
    """Return the index of the first occurrence of target in the sorted list, or -1 when it is absent. Runs in logarithmic time."""
    lo, hi = 0, len(items) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if items[mid] == target:
            return mid
        if items[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
''',
    '''def index_of(items, target):
    """Return the index of the first occurrence of target in the sorted list, or -1 when it is absent. Runs in logarithmic time."""
    lo, hi = 0, len(items)
    while lo < hi:
        mid = (lo + hi) // 2
        if items[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo if lo < len(items) and items[lo] == target else -1
''',
    'self.assertEqual(index_of([1, 3, 5], 3), 1)\n        self.assertEqual(index_of([1, 3, 5], 4), -1)',
    [('index_of([1, 3, 5, 7], 7)', '3'), ('index_of([1, 2, 2, 2, 3], 2)', '1'), ('index_of([4], 4)', '0'), ('index_of([], 1)', '-1'), ('index_of([1, 3], 9)', '-1')],
)

family(
    "anagrams", "group_anagrams",
    "Group words that are anagrams of each other, ignoring case. Each group lists its words in input order; groups are ordered by the first appearance of their first word; words keep their original case.",
    '''def group_anagrams(words):
    """Group words that are anagrams of each other, ignoring case. Each group lists its words in input order; groups are ordered by the first appearance of their first word; words keep their original case."""
    groups = {}
    for word in words:
        groups.setdefault("".join(sorted(word)), []).append(word)
    return sorted(groups.values())
''',
    '''def group_anagrams(words):
    """Group words that are anagrams of each other, ignoring case. Each group lists its words in input order; groups are ordered by the first appearance of their first word; words keep their original case."""
    groups = {}
    for word in words:
        groups.setdefault("".join(sorted(word.lower())), []).append(word)
    return list(groups.values())
''',
    'self.assertEqual(group_anagrams(["ab", "ba"]), [["ab", "ba"]])',
    [('group_anagrams(["Tea", "eat", "tan", "ATE", "nat", "bat"])', '[["Tea", "eat", "ATE"], ["tan", "nat"], ["bat"]]'), ('group_anagrams(["zz", "a", "b"])', '[["zz"], ["a"], ["b"]]')],
)

family(
    "titlecase", "title_case",
    "Capitalise each word of the text, except the small words a, an, the, of, in, on, and, or when they are not the first or last word; the rest of each word is lower-cased.",
    '''SMALL = {"a", "an", "the", "of", "in", "on", "and", "or"}


def title_case(text):
    """Capitalise each word of the text, except the small words a, an, the, of, in, on, and, or when they are not the first or last word; the rest of each word is lower-cased."""
    return " ".join(w.capitalize() for w in text.split())
''',
    '''SMALL = {"a", "an", "the", "of", "in", "on", "and", "or"}


def title_case(text):
    """Capitalise each word of the text, except the small words a, an, the, of, in, on, and, or when they are not the first or last word; the rest of each word is lower-cased."""
    words = text.split()
    out = []
    for i, w in enumerate(words):
        lower = w.lower()
        edge = i == 0 or i == len(words) - 1
        out.append(lower if lower in SMALL and not edge else lower.capitalize())
    return " ".join(out)
''',
    'self.assertEqual(title_case("hello world"), "Hello World")',
    [('title_case("the lord OF the rings")', '"The Lord of the Rings"'), ('title_case("war and peace")', '"War and Peace"'), ('title_case("a tale of two")', '"A Tale of Two"'), ('title_case("walk in")', '"Walk In"')],
)

family(
    "rotate", "rotate",
    "Rotate a list to the right by k places and return a new list; k may be negative (left) or larger than the list; an empty list stays empty.",
    '''def rotate(items, k):
    """Rotate a list to the right by k places and return a new list; k may be negative (left) or larger than the list; an empty list stays empty."""
    return items[-k:] + items[:-k]
''',
    '''def rotate(items, k):
    """Rotate a list to the right by k places and return a new list; k may be negative (left) or larger than the list; an empty list stays empty."""
    if not items:
        return []
    k %= len(items)
    return items[len(items) - k:] + items[:len(items) - k]
''',
    'self.assertEqual(rotate([1, 2, 3, 4], 1), [4, 1, 2, 3])',
    [('rotate([1, 2, 3], 0)', '[1, 2, 3]'), ('rotate([1, 2, 3], -1)', '[2, 3, 1]'), ('rotate([1, 2, 3], 5)', '[2, 3, 1]'), ('rotate([], 3)', '[]')],
)


def render_names(name, seed):
    rng = random.Random(f"{name}:{seed}")
    return rng.choice(NAMES)


def write_repo(fam, out, pkg, source):
    repo = os.path.join(out, "repo")
    os.makedirs(os.path.join(repo, pkg), exist_ok=True)
    os.makedirs(os.path.join(repo, "tests"), exist_ok=True)
    with open(os.path.join(repo, pkg, "__init__.py"), "w") as f:
        f.write(f"from .core import *  # noqa: F401,F403\n")
    with open(os.path.join(repo, pkg, "core.py"), "w") as f:
        f.write(source)
    fn = fam["fn"]
    with open(os.path.join(repo, "tests", "test_core.py"), "w") as f:
        f.write(f"import unittest\n\nfrom {pkg}.core import {fn}\n\n\nclass Core(unittest.TestCase):\n    def test_basic(self):\n        {fam['repo_tests']}\n\n\nif __name__ == \"__main__\":\n    unittest.main()\n")
    with open(os.path.join(repo, ".gitignore"), "w") as f:
        f.write("__pycache__/\n*.pyc\n")
    with open(os.path.join(repo, "README.md"), "w") as f:
        f.write(f"# {pkg}\n\n`{pkg}.{fn}`: {fam['doc']}\n\nTests: `python3 -m unittest discover -s tests`.\n")
    return repo


def accept_script(fam, pkg):
    fn = fam["fn"]
    lines = ['"""Acceptance for a seeded-bug fixture; kept outside the repository."""', "import sys", "", 'sys.path.insert(0, ".")',
             f"from {pkg}.core import *  # noqa: E402,F401,F403", "", "bad = 0", ""]
    for call, want in fam["accept"]:
        lines += ["try:", f"    got = {call}", "except Exception as e:", "    got = repr(e)",
                  f"if got != {want}:", "    bad += 1", f"    print({call!r}, '->', got, 'expected', {want})"]
    for call in fam.get("raises", []):
        lines += ["try:", f"    {call}", "    bad += 1", f"    print({call!r}, 'did not raise')", "except ValueError:", "    pass"]
    lines += ["sys.exit(1 if bad else 0)"]
    return "\n".join(lines) + "\n"


def render(name, out, seed=1, fixed=False):
    fam = FAMILIES[name]
    pkg = render_names(name, seed)
    os.makedirs(out, exist_ok=True)
    if os.listdir(out):
        raise SystemExit(f"{out} is not empty")
    repo = write_repo(fam, out, pkg, fam["fixed"] if fixed else fam["buggy"])
    with open(os.path.join(out, "accept.py"), "w") as f:
        f.write(accept_script(fam, pkg))
    for args in (["init", "-q", "-b", "main"], ["config", "user.email", "fixture@example.com"], ["config", "user.name", "fixture"], ["add", "-A"], ["commit", "-q", "-m", f"{pkg}: {fam['fn']}"]):
        subprocess.run(["git", "-C", repo] + args, check=True)
    task = fam["task"].format(pkg=pkg, fn=fam["fn"])
    with open(os.path.join(out, "task.txt"), "w") as f:
        f.write(task + "\n")
    return repo, os.path.join(out, "accept.py"), task


def run(cmd, cwd):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True).returncode


def check():
    bad = 0
    for name in FAMILIES:
        for fixed in (False, True):
            with tempfile.TemporaryDirectory() as d:
                repo, accept, _ = render(name, os.path.join(d, "x"), seed=1, fixed=fixed)
                tests = run([sys.executable, "-m", "unittest", "discover", "-s", "tests"], repo)
                acc = run([sys.executable, accept], repo)
                ok = tests == 0 and (acc == 0) == fixed
                print(f"{name:10} {'fixed' if fixed else 'buggy'}: repo tests {'pass' if tests == 0 else 'FAIL'}, acceptance {'pass' if acc == 0 else 'fail'} {'ok' if ok else 'WRONG'}")
                bad += not ok
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("check")
    r = sub.add_parser("render")
    r.add_argument("out")
    r.add_argument("--family", required=True, choices=sorted(FAMILIES))
    r.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "list":
        print("\n".join(sorted(FAMILIES)))
    elif a.cmd == "check":
        sys.exit(check())
    else:
        repo, accept, task = render(a.family, a.out, a.seed)
        print(f"repo={repo}\naccept={accept}\ntask={task}")


if __name__ == "__main__":
    main()
