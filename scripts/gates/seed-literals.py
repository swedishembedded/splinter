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
    todo:         a site a later change draws instead; counted, never grows,
                  and refused outright under --no-todo

The gate fails on a production hit with no entry, on an entry that matches no
hit (a stale entry: delete it, the ratchet only tightens), and on a reason
without one of those prefixes. Test code is enforced too: a default seed is
random, so a test fixes a seed only where determinism is necessary (a golden
or replay assertion, a parity or reproducibility claim, a fixture whose
outcome was measured at one seed) and says why in a `// seed: <reason>` comment
(`# seed:` in Python) on the line or the line above; where it is not necessary
the literal is dropped and the test passes for any seed. A reason that
says any value will do (it starts with `any`) is no reason: the test draws.

Detected, in Rust (beyond the seed written beside the word seed): a literal
seed constructor (`Seed::new(<int>)`, `Mt19937`, `ChaCha*`, `Philox*`), a
function with no arguments that returns a fixed seed (named or typed like a
seed, or named by `#[serde(default = "...")]` on a seed field), a loop or range
over literal seeds (`for seed in 1..`, `(1..=N).map(|seed| ...)`), and a literal
or a constant of the seed's type passed where a function of the same crate
takes its seed (`name#<index>`; a name defined twice with different parameters
is not resolved; test code only with `--positional-tests`, and a range in test
code is a sweep, not a default). `seed<...>: <int>`, `seed<...> = <int>`, `seed: Some(<int>)`,
`_or("--<...>seed<...>", <int>)`, `map_or(Ok(<int>)` / `unwrap_or(<int>)` on a
line naming a seed, `.seed(<int>)`, a clap `default_value_t = <int>` on a field
named like a seed, `ParamSpec` seed parameters with an integer `.default`,
`const <...>SEED<...>: T = <int>` and literal constructors of `*Rng`,
`SplitMix64`, `Lcg`. In Python and Makefiles: `add_argument("--<...>seed<...>",
default=<int>)`, `SEED = <int>`, `SEED ?= <int>`, and in Python the calls
`np.random.default_rng(<int>)`, `RandomState(<int>)`, `torch.manual_seed(<int>)`,
`random.seed(<int>)` (symbol `<enclosing def, class or ->:<call>`); in Python
also `seed: int = <int>`, `"seed": <int>`, `environ.get("..SEED..", "<int>")` and
`seed ... or|else <int>`. In shell and `.bats`: `--...seed...[ =]<int>`,
`${..SEED..:-<int>}`, `SEED=<int>` and a `seq <int>` feeding a seed flag; in
Makefile recipes `--...seed... <int>` (symbol `<target>:<flag>`). In Markdown, a
fenced command with `--...seed... <int>` unless the block, the line before it
or the line after it says it reproduces a result (symbol `doc:--seed <int>`).
Python test files (`test_*.py`, `*_test.py`, under `tests/`) and `.bats` files
are test code and take `# seed: <reason>`. A `seed:` reason covers its line and
the run of lines below it up to the next blank line.

usage: seed-literals.py [--root DIR] [--allow-dir DIR] [--positional-tests]
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

REASON_PREFIXES = ("salt:", "world:", "frozen:", "greedy:", "test-double:", "bench-input:", "todo:")
TEST_MARK = re.compile(r"(?://|#)\s*seed:\s*(.*)$")
NO_REASON = re.compile(r"any\b", re.I)  # "any value will do" says determinism is not needed

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
RNG_NEW = re.compile(
    r"\b(\w*Rng|SplitMix64|Lcg|Seed|Mt19937|ChaCha\w*|Philox\w*)::(new|from|seed_from_u64|from_seed)\(\s*" + INT + r"\s*\)"
)
SEED_RANGE_FOR = re.compile(r"\bfor\s+\(?\s*\w*seed\w*\s+in\s+" + INT + r"\s*\.\.")
SEED_RANGE_MAP = re.compile(
    r"\(\s*" + INT + r"\s*\.\.=?[^)]*\)\s*\.(?:map|for_each|flat_map|filter_map)\(\s*\|\s*\(?\s*\w*seed\w*"
)
FN_NO_ARGS = re.compile(r"\bfn\s+(\w+)\s*\(\s*\)\s*->\s*([\w:]+)")
FIXED_BODY = re.compile(r"^\s*\{\s*" + INT + r"\s*\}")
SERDE_SEED_DEFAULT = re.compile(
    r'#\[serde\([^\]]*\bdefault\s*=\s*"(\w+)"[^\]]*\]\s*(?:#\[[^\]]*\]\s*)*(?:pub(?:\([^)]*\))?\s+)?\w*seed\w*\s*:',
    re.I,
)
FN_DEF = re.compile(r"\bfn\s+(\w+)\s*(?:<[^>{(]*>)?\s*\(")
CALL = re.compile(r"(?<![\w!])(\w+)\s*\(")
LITERAL_ARG = re.compile(r"^" + INT_EXPR + r"$")
CONST_ARG = re.compile(r"^[A-Z][A-Z0-9_]*$")
CALL_LOOKAHEAD = 10  # lines the arguments of one call may span
CONST_TYPE = re.compile(r"\bconst\s+(\w+)\s*:\s*([\w:]+)\s*=")
SELF_PARAM = re.compile(r"&?\s*(?:'\w+\s+)?(?:mut\s+)?self$")
RANGE_START = re.compile(r"\(\s*" + INT + r"\s*\.\.")
FIELD_LINE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(\w+)\s*:\s*[A-Za-z&\[(]")

PY_ARG_DEFAULT = re.compile(r"add_argument\(\s*[\"']--([\w-]*seed[\w-]*)[\"'][^)]*\bdefault\s*=\s*" + INT)
PY_ASSIGN = re.compile(r"^\s*([A-Z_]*SEED[A-Z_]*)\s*=\s*" + INT + r"\s*(?:#.*)?$")
PY_SEED_CALL = re.compile(
    r"\b(default_rng|RandomState|manual_seed|random\.seed|np\.random\.seed|numpy\.random\.seed)\(\s*" + INT + r"\s*\)"
)
PY_SEED_NAME = re.compile(r"\b([a-z_0-9]*seed[a-z_0-9]*)\s*=(?!=)\s*" + INT + r"\b")
PY_ANNOTATED = re.compile(r"\b(\w*seed\w*)\s*:\s*[A-Za-z_][\w\[\], .|]*?\s*=(?!=)\s*" + INT + r"\b", re.I)
PY_DICT = re.compile(r"[\"'](\w*seed\w*)[\"']\s*:\s*" + INT + r"\b", re.I)
PY_ENV = re.compile(r"(?:environ\.get|getenv)\(\s*[\"'](\w*SEED\w*)[\"']\s*,\s*[\"']?" + INT)
PY_FALLBACK = re.compile(r"\b(?:or|else)\s+" + INT + r"\b")
PY_DEF = re.compile(r"^(\s*)(?:async\s+)?(?:def|class)\s+(\w+)")
SH_FLAG = re.compile(r"(--[\w-]*seed[\w-]*)[ =]" + INT + r"(?![\w.])")
SH_DEFAULT = re.compile(r"\$\{(\w*SEED\w*):-" + INT + r"\}")
SH_ASSIGN = re.compile(r"(?<![\w-])([A-Z_]*SEED[A-Z_]*)=" + INT + r"(?![\w.])")
SH_SEQ = re.compile(r"\bseq\s+" + INT + r"\b")
SH_FUNC = re.compile(r"^(?:function\s+)?(\w+)\s*\(\)\s*\{")
SH_COMMENT = re.compile(r"(^|\s)#.*$")
MAKE_TARGET = re.compile(r"^([\w./%$()-]+)\s*:(?!=)")
DOC_FLAG = re.compile(r"--([\w-]*seed[\w-]*)[ =](" + INT + r")(?![\w.])")
DOC_FENCE = re.compile(r"^\s*(```|~~~)")
SHELL_LOOKAHEAD = 2  # lines after a `seq <int>` that may carry the seed flag it feeds
MAKE_ASSIGN = re.compile(r"^\s*([A-Z_]*SEED[A-Z_]*)\s*[?:]?=\s*" + INT + r"\s*$")

ITEM = re.compile(
    r"\b(fn)\s+(\w+)"
    r"|^\s*(?:unsafe\s+)?(impl)\b(?:<[^>]*>)?\s+(?:[\w:<>, &']+\s+for\s+)?(\w+)"
    r"|\b(struct|enum|trait|mod)\s+(\w+)"
)
TEST_ATTR = re.compile(r"^\s*#\[(?:cfg\(test\)|test|tokio::test|cfg\(all\(test)")
TEST_MOD_FILE = re.compile(r"#\[cfg\(test\)\]\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)\s*;")


# Its sources spell out the literals the gate must find; they are not seeds.
SELF_SPEC = "scripts/gates/test-seed-literals.py"


class Hit:
    """One seed literal: where, in which item, what kind, in test code or not."""

    def __init__(self, path, symbol, kind, line, test, reason, const=None):
        self.path, self.symbol, self.kind = path, symbol, kind
        self.line, self.test, self.reason = line, test, reason  # reason: the `seed:` text covering it, or None
        self.marked = reason is not None
        self.const = const  # for a positional hit, the constant passed as the seed

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


def scan_rust(path, text, test_file=False, seed_params=None):
    """Hits in one Rust file. `test_file` marks a whole file as test code;
    `seed_params` maps a function name to the index of its seed parameter (see
    `seed_param_index`), which is what makes a literal passed there visible."""
    raw = text.split("\n")
    seed_params = seed_params or SeedParams()
    seed_fns = set(SERDE_SEED_DEFAULT.findall(text))
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

        for name, kind, const in find_hits(code, skel, raw, idx, seed_params, seed_fns):
            opening = [(*pending, depth)] if pending is not None and "{" in skel else []
            symbol = scope_name(stack + opening) + name
            reason = seed_reason(raw, idx + 1)
            hits.append(Hit(path, symbol, kind, idx + 1, in_test or (pending_test and "{" in skel), reason, const))

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


def find_hits(code, skel, raw, idx, seed_params, seed_fns):
    """Yield (matched name, kind, constant) for the seed literals on one line:
    the ones written next to the word seed, then the ones behind a function."""
    for name, kind in literal_hits(code, skel, raw, idx):
        yield name, kind, None
    yield from hidden_hits(skel, raw, idx, seed_params, seed_fns)


def hidden_hits(skel, raw, idx, seed_params, seed_fns):
    """Seeds that no line names: a loop or range over literal seeds, a function
    that returns one, and a literal passed where a function takes its seed."""
    ahead = "\n".join(split_code(line)[1] for line in raw[idx : idx + 3])
    if SEED_RANGE_FOR.search(skel) or (RANGE_START.search(skel) and SEED_RANGE_MAP.search(ahead)):
        yield "seed-range", "range", None
    fixed = FN_NO_ARGS.search(skel)
    if fixed and ("seed" in (fixed.group(1) + fixed.group(2)).lower() or fixed.group(1) in seed_fns):
        if FIXED_BODY.match(ahead[fixed.end() :]):
            yield "returns", "fixed-return", None
    for call in CALL.finditer(skel):
        known = seed_params.params.get(call.group(1))
        if known is None or re.search(r"\bfn\s+$", skel[: call.start()]):
            continue
        position, arity, ty = known
        args = call_arguments(skel, raw, idx, call.end())
        if args is None or len(args) != arity:
            continue  # not a call of the function indexed under that name
        arg = args[position]
        name = f"{call.group(1)}#{position}"
        if LITERAL_ARG.match(arg):
            yield name, "positional", None
        elif CONST_ARG.match(arg) and seed_params.const_types.get(arg) == ty:
            yield name, "positional-const", arg


def call_arguments(skel, raw, idx, start):
    """The stripped arguments of the call whose `(` ends at skel[start-1],
    reading on into the following lines while the call stays open; None if it
    does not close within the lookahead."""
    text = skel[start:] + "\n" + "\n".join(split_code(line)[1] for line in raw[idx + 1 : idx + 1 + CALL_LOOKAHEAD])
    args, closed = split_arguments(text, 0)
    return [a.strip() for a in args] if closed else None


def literal_hits(code, skel, raw, idx):
    """Yield (matched name, kind) for the seed literals written next to the word
    seed on one line. String contents are blanked in `skel`, so text that merely
    says `seed: 1` is not a hit."""
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
    """Hits in a Python, shell, Makefile or Markdown source. A Python test file,
    a `.bats` file and a file under `tests/` are test code."""
    if path.endswith(".py"):
        return scan_python(path, text)
    if path.endswith((".sh", ".bats")):
        return scan_shell(path, text)
    if path.endswith(".md"):
        return scan_markdown(path, text)
    return scan_makefile(path, text)


def seed_reason(lines, n):
    """The `seed:` reason covering line n (1-based), or None: one on the line
    itself, or in the run of non-blank lines above it, so a comment of several
    lines (or a seed constant beside another) covers what follows it up to the
    next blank line. Its text is that of the comment's first line."""
    for i in range(n - 1, -1, -1):
        if not lines[i].strip():
            return None
        found = TEST_MARK.search(lines[i])
        if found:
            return found.group(1).strip()
    return None


def scan_python(path, text):
    """A seeding call or default is named `<enclosing def or class or ->:<name>`."""
    hits = []
    test = is_python_test(path)
    lines = text.split("\n")
    defs = []  # (indent, name) of the open `def`s and classes
    for n, line in enumerate(lines, 1):
        reason = seed_reason(lines, n)
        code = line.split("#", 1)[0]
        if code.strip():
            indent = len(code) - len(code.lstrip())
            while defs and defs[-1][0] >= indent:
                defs.pop()

        def add(symbol, kind):
            hits.append(Hit(path, symbol, kind, n, test, reason))

        for m in PY_ARG_DEFAULT.finditer(line):
            add("--" + m.group(1), "default")
        m = PY_ASSIGN.match(line)
        if m:
            add(m.group(1), "const")
        d = PY_DEF.match(code)
        if d:  # the parameters of a def belong to it
            defs.append((len(d.group(1)), d.group(2)))
        owner = defs[-1][1] if defs else "-"
        for call in PY_SEED_CALL.finditer(code):
            add(f"{owner}:{call.group(1)}", "python-rng")
        for pattern in (PY_SEED_NAME, PY_ANNOTATED, PY_DICT, PY_ENV):
            for named in pattern.finditer(code):
                add(f"{owner}:{named.group(1)}", "python-default")
        if "seed" in code.lower():
            for _ in PY_FALLBACK.finditer(code):
                add(f"{owner}:seed-fallback", "python-default")
    return hits


def scan_shell(path, text):
    """A seed flag or variable with a literal value, named `<function or ->:<flag>`;
    a `seq <int>` that feeds a seed flag is `seq-seed`."""
    hits = []
    test = path.endswith(".bats") or is_test_path(path)
    lines = text.split("\n")
    owner = "-"
    for n, line in enumerate(lines, 1):
        code = SH_COMMENT.sub("", line)
        func = SH_FUNC.match(code)
        if func:
            owner = func.group(1)
        elif code.startswith("}"):
            owner = "-"
        reason = seed_reason(lines, n)
        found = [m.group(1) for pattern in (SH_FLAG, SH_DEFAULT, SH_ASSIGN) for m in pattern.finditer(code)]
        if SH_SEQ.search(code) and any("seed" in following.lower() for following in lines[n - 1 : n + SHELL_LOOKAHEAD]):
            found.append("seq-seed")
        hits.extend(Hit(path, f"{owner}:{name}", "shell", n, test, reason) for name in found)
    return hits


def scan_makefile(path, text):
    """A seed assignment, and a seed flag with a literal value in a recipe line
    (named `<target>:<flag>`)."""
    hits = []
    lines = text.split("\n")
    target = "-"
    for n, line in enumerate(lines, 1):
        reason = seed_reason(lines, n)
        m = MAKE_ASSIGN.match(line)
        if m:
            hits.append(Hit(path, m.group(1), "const", n, False, reason))
        t = MAKE_TARGET.match(line)
        if t:
            target = t.group(1)
        if line.startswith("\t"):
            hits.extend(Hit(path, f"{target}:{f.group(1)}", "shell", n, False, reason) for f in SH_FLAG.finditer(line))
    return hits


def scan_markdown(path, text):
    """A fenced command that passes a literal seed, unless the block or the
    paragraph beside it says it is reproducing a result: documentation shows a
    fixed seed as the way to reproduce, never as the way to run."""
    hits = []
    lines = text.split("\n")
    fence = None  # (first line index of the block) while inside one
    for i, line in enumerate(lines):
        if not DOC_FENCE.match(line):
            continue
        if fence is None:
            fence = i
            continue
        block = lines[fence : i + 1]
        before = next((x for x in reversed(lines[:fence]) if x.strip()), "")
        after = lines[i + 1] if i + 1 < len(lines) else ""
        if not re.search(r"reproduc", "\n".join([before, *block, after]), re.I):
            for offset, content in enumerate(block):
                for m in DOC_FLAG.finditer(content):
                    hits.append(Hit(path, f"doc:--{m.group(1)} {m.group(2)}", "doc", fence + offset + 1, False, False))
        fence = None
    return hits


class SeedParams:
    """What the functions of one crate say about their seed parameter: for a
    name, (index among the non-self parameters, parameter count, type), and the
    declared type of each constant."""

    def __init__(self, params=None, const_types=None):
        self.params = params or {}
        self.const_types = const_types or {}


def seed_param_index(texts):
    """Index the functions defined in `texts` (the sources of one crate) by the
    position of their seed parameter. A name defined more than once with
    different parameter lists, or with no seed parameter, is left out: the call
    being read cannot be resolved to one definition."""
    found, const_types = {}, {}
    for text in texts:
        stripped = "\n".join(split_code(line)[1] for line in text.split("\n"))
        const_types.update(CONST_TYPE.findall(stripped))
        for m in FN_DEF.finditer(stripped):
            params = [p for p in split_arguments(stripped, m.end())[0] if p.strip()]
            params = [p for p in params if not SELF_PARAM.match(p.split(":")[0].strip())]
            names = [p.split(":")[0].strip().removeprefix("mut ").strip() for p in params]
            seed = next((i for i, n in enumerate(names) if re.fullmatch(r"\w*seed\w*", n, re.I)), None)
            ty = params[seed].split(":", 1)[1].strip() if seed is not None and ":" in params[seed] else None
            found.setdefault(m.group(1), set()).add((seed, len(params), ty))
    unique = {name: next(iter(v)) for name, v in found.items() if len(v) == 1 and next(iter(v))[0] is not None}
    return SeedParams(unique, const_types)


def split_arguments(text, start):
    """The top-level comma-separated items of the bracketed list that opened just
    before text[start:], and whether its closing bracket was found. `->` does
    not close a `<`."""
    items, depth, current, i, closed = [], 0, [], start, False
    while i < len(text):
        ch = text[i]
        if ch == "-" and text[i + 1 : i + 2] == ">":
            current.append("->")
            i += 2
            continue
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            if depth == 0:
                closed = True
                break
            depth -= 1
        if ch == "," and depth == 0:
            items.append("".join(current))
            current = []
        else:
            current.append(ch)
        i += 1
    items.append("".join(current))
    return items, closed


def is_python_test(path):
    name = Path(path).name
    return name.startswith("test_") or name.endswith("_test.py") or is_test_path(path)


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


def crate_of(rel, crate_dirs):
    """The nearest enclosing directory that holds a Cargo.toml ("" outside any)."""
    parent = Path(rel).parent
    while parent.as_posix() != ".":
        if parent.as_posix() in crate_dirs:
            return parent.as_posix()
        parent = parent.parent
    return "" if "" in crate_dirs else None


def left_unread(hit, positional_tests):
    """Whether a hit in test code is not asked for a reason. A loop over a range
    of seeds is a property checked over many worlds, not a default seed; a
    literal passed as a seed argument is read only on request."""
    return hit.kind == "range" or (hit.kind.startswith("positional") and not positional_tests)


def scan_tree(root, files, positional_tests=False):
    """Scan the given repository-relative files; returns all hits. A literal
    passed as a seed argument in test code is reported only with
    `positional_tests`: tests pass fixed arguments for many reasons, and each
    needs reading before it can be called necessary or not."""
    texts = {}
    for rel in files:
        if rel == SELF_SPEC:
            continue
        if rel.endswith((".rs", ".py", ".sh", ".bats", ".md", "Makefile", ".mk")):
            try:
                texts[rel] = (root / rel).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
    test_files = {rel for rel in texts if is_test_path(rel)}
    for rel, text in texts.items():
        if rel.endswith(".rs"):
            test_files.update(test_module_files(rel, text))
    crate_dirs = {Path(rel).parent.as_posix() for rel in files if Path(rel).name == "Cargo.toml"}
    sources = {}
    for rel, text in texts.items():
        if rel.endswith(".rs"):
            sources.setdefault(crate_of(rel, crate_dirs), []).append(text)
    seed_params = {crate: seed_param_index(group) for crate, group in sources.items()}
    hits = []
    for rel, text in texts.items():
        if rel.endswith(".rs"):
            found = scan_rust(rel, text, rel in test_files, seed_params[crate_of(rel, crate_dirs)])
            hits.extend(h for h in found if not (h.test and left_unread(h, positional_tests)))
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


def check(hits, entries, no_todo=False):
    """Failures for production hits without an entry, entries without a hit
    and test hits without a `seed:` reason; with `no_todo`, also an entry whose
    reason is still `todo:`."""
    failures = []
    if no_todo:
        failures.extend(
            f"{path}: {symbol} is still a todo ({reason}): draw the seed and delete the entry"
            for (path, symbol), reason in sorted(entries.items())
            if reason.startswith("todo:")
        )
    for hit in hits:
        marker = "#" if hit.path.endswith(".py") else "//"
        if hit.test and not hit.marked:
            failures.append(
                f"{hit.path}:{hit.line}: test seed literal {hit.symbol} ({hit.kind}) has no reason: "
                f"drop the literal so the test passes for any seed, or say why determinism is "
                f"necessary with `{marker} seed: <reason>`"
            )
        elif hit.test and NO_REASON.match(hit.reason):
            failures.append(
                f"{hit.path}:{hit.line}: test seed literal {hit.symbol} ({hit.kind}) says any seed "
                f"will do (`{marker} seed: {hit.reason}`): drop the literal so the test draws its seed"
            )
    named = {key[1].rsplit("::", 1)[-1] for key in entries} | {h.symbol.rsplit("::", 1)[-1] for h in hits if h.kind == "const"}
    production = {h.key(): h for h in hits if not h.test and not (h.const and h.const in named)}
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
    ap.add_argument(
        "--positional-tests",
        action="store_true",
        help="also report literals passed as a seed argument in test code",
    )
    ap.add_argument(
        "--no-todo",
        action="store_true",
        help="fail on any allowlist entry whose reason is still `todo:`",
    )
    args = ap.parse_args()
    root = Path(args.root)

    hits = scan_tree(root, tracked_files(root), args.positional_tests)
    entries, problems = read_allow(args.allow_dir)
    failures = problems + check(hits, entries, args.no_todo)

    todo = sum(1 for r in entries.values() if r.startswith("todo:"))
    print(f"seed-literals: {len(entries)} allowlisted production sites, {todo} still todo")
    for f in failures:
        print(f"seed-literals: {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
