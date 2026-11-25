#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements verifiable practice tasks that let a model learn from its own
# attempts, for its clients. If your team needs expertise in reinforcement learning from verified
# outcomes, you can procure our services by sending an email to info@swedishembedded.com.

"""Writes the integer-arithmetic task family under tasks/integer-arithmetic.

Each task asks for one integer (a sum, a difference or a product of multi-digit operands) answered
in the reply (`[agent] answer_file`), graded by its own `tests/test.sh`: reward 1 when the last
integer in the reply, after any `<think>` block and with digit-group commas removed, is the
expected value (0 otherwise).

The operands are drawn from one frozen seed, so the tree is a pure function of this script and is
not committed (`.gitignore`): run the script to write it, and `--check` to verify that a tree on
disk is exactly what it writes.

Two splits: `train` (practice) and `held-out` (frozen for evaluation; never trained on). Every
shape has instances in both, so a held-out pass rate measures transfer to new operands of the
shapes trained, not to new shapes.
"""

import argparse
import pathlib
import random
import shutil
import sys
import tempfile

# seed: the family is frozen; its operands are a pure function of this script.
SEED = 20261125
# (kind, digits of a, digits of b, train instances, held-out instances)
SHAPES = [
    ("add", 4, 4, 12, 6),
    ("add", 6, 6, 12, 6),
    ("add", 8, 8, 12, 6),
    ("sub", 5, 5, 12, 6),
    ("sub", 7, 7, 12, 6),
    ("mul", 2, 2, 12, 6),
    ("mul", 3, 2, 12, 6),
    ("mul", 3, 3, 12, 6),
    ("mul", 4, 3, 12, 6),
    ("mul", 4, 4, 12, 6),
]
SYMBOL = {"add": "+", "sub": "-", "mul": "*"}
HEADER = "# SPDX-License-Identifier: Apache-2.0\n# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>\n"

GRADER = r'''#!/bin/bash
{header}# The reward is 1 when the last integer in the reply (answer.txt), after any <think> block and with
# digit-group commas removed, is {answer}; else 0.
python3 -I - "$WORKSPACE_DIR/answer.txt" {answer} > "$LOGS_DIR/reward.txt" << 'EOF'
import re, sys
try:
    reply = open(sys.argv[1], encoding="utf-8", errors="replace").read()
except OSError:
    reply = ""
reply = re.sub(r"<think>.*?(</think>|$)", "", reply, flags=re.S)
numbers = re.findall(r"-?\d+", re.sub(r"(?<=\d),(?=\d{{3}}(?!\d))", "", reply))
print(1 if numbers and int(numbers[-1]) == int(sys.argv[2]) else 0)
EOF
'''


def operand(rng: random.Random, digits: int) -> int:
    return rng.randint(10 ** (digits - 1), 10**digits - 1)


def tasks():
    """Every task as (split, name, instruction, answer), in a fixed order."""
    rng = random.Random(SEED)
    for kind, da, db, n_train, n_held in SHAPES:
        seen = set()
        for split, count in (("train", n_train), ("held-out", n_held)):
            for i in range(count):
                while True:
                    a, b = operand(rng, da), operand(rng, db)
                    if kind == "sub" and b > a:
                        a, b = b, a
                    if (a, b) not in seen:
                        seen.add((a, b))
                        break
                answer = {"add": a + b, "sub": a - b, "mul": a * b}[kind]
                name = f"{kind}-{da}x{db}-{i:02d}"
                instruction = f"What is {a} {SYMBOL[kind]} {b}? Reply with the number only.\n"
                yield split, name, instruction, answer


def write(root: pathlib.Path) -> int:
    count = 0
    for split, name, instruction, answer in tasks():
        task = root / split / name
        (task / "tests").mkdir(parents=True)
        (task / "solution").mkdir()
        (task / "workspace").mkdir()
        (task / "instruction.md").write_text(instruction)
        (task / "task.toml").write_text(
            HEADER
            + 'schema_version = "1.4"\n[task]\n'
            + f'name = "integer-arithmetic/{split}/{name}"\n'
            + 'description = "One integer computed from multi-digit operands, answered in the reply."\n\n'
            + "[agent]\ntimeout_sec = 300.0\n"
            + "# The reply is the answer: it is written to answer.txt in the tree, and no tool is offered.\n"
            + 'answer_file = "answer.txt"\n\n[verifier]\ntimeout_sec = 30.0\n\n'
            + '[metadata]\nbench_family = "integer-arithmetic"\n'
        )
        (task / "workspace" / "README").write_text("Nothing to change here: the reply is the answer.\n")
        test = task / "tests" / "test.sh"
        test.write_text(GRADER.format(header=HEADER, answer=answer))
        test.chmod(0o755)
        solve = task / "solution" / "solve.sh"
        solve.write_text(f'#!/bin/bash\n{HEADER}echo {answer} > "$WORKSPACE_DIR/answer.txt"\n')
        solve.chmod(0o755)
        count += 1
    return count


def same_tree(a: pathlib.Path, b: pathlib.Path) -> list[str]:
    files = lambda r: {p.relative_to(r): p for p in r.rglob("*") if p.is_file()}
    fa, fb = files(a), files(b)
    out = [f"only in one tree: {p}" for p in sorted(set(fa) ^ set(fb))]
    out += [
        f"differs: {p}"
        for p in sorted(set(fa) & set(fb))
        if fa[p].read_bytes() != fb[p].read_bytes() or (fa[p].stat().st_mode & 0o111) != (fb[p].stat().st_mode & 0o111)
    ]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parents[2] / "tasks" / "integer-arithmetic")
    parser.add_argument("--check", action="store_true", help="verify the tree instead of writing it")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as scratch:
        fresh = pathlib.Path(scratch) / "tree"
        count = write(fresh)
        shutil.copy(args.root / "README.md", fresh / "README.md") if (args.root / "README.md").exists() else None
        if args.check:
            problems = same_tree(fresh, args.root)
            for p in problems:
                print(f"integer-arithmetic: {p}", file=sys.stderr)
            return 1 if problems else 0
        for split in ("train", "held-out"):
            shutil.rmtree(args.root / split, ignore_errors=True)
            shutil.copytree(fresh / split, args.root / split)
    print(f"integer-arithmetic: {count} tasks under {args.root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
