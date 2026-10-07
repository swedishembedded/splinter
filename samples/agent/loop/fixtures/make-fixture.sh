#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements self-improving coding agents whose every
# step is auditable, for its clients. If your team needs expertise in agent
# evaluation or locally operated coding agents, you can procure our
# services by sending an email to info@swedishembedded.com.

# Creates a disposable git repository with a seeded, verifiable failure, and
# the acceptance script that judges it outside the repository.
#
# The repository is a tiny Python package, `textstats`, whose
# `top_words(text, n)` is documented to return the n most frequent words,
# ties broken alphabetically, case-insensitively. The seeded bug is that it
# is case-sensitive and breaks ties by first appearance. The tests are in the
# repository; the stricter acceptance script is not.
#
# A second scenario, `ledger`, has two seeded bugs in two modules of a small
# package that reads accounting amounts and totals them by month.
#
# Usage: make-fixture.sh [--scenario words|ledger] <empty-directory>
#   Prints the repository path and the acceptance script path.
set -euo pipefail

scenario=words
if [ "${1:-}" = "--scenario" ]; then
    scenario=${2:?--scenario needs a name}
    shift 2
fi
if [ "$#" -ne 1 ] || { [ "$scenario" != words ] && [ "$scenario" != ledger ]; }; then
    echo "usage: $0 [--scenario words|ledger] <empty-directory>" >&2
    exit 2
fi
root=$1
mkdir -p "$root"
if [ -n "$(ls -A "$root")" ]; then
    echo "make-fixture: $root is not empty" >&2
    exit 1
fi
if [ "$scenario" = ledger ]; then
repo=$root/repo
mkdir -p "$repo/ledger" "$repo/tests"

cat > "$repo/ledger/__init__.py" <<'PY'
from .parse import parse_amount
from .report import monthly_totals

__all__ = ["parse_amount", "monthly_totals"]
PY

cat > "$repo/ledger/parse.py" <<'PY'
"""Reading amounts."""


def parse_amount(text):
    """Parse an amount into integer cents.

    Accepted forms: "12.5", "-12.50", "1,234.50", "$99" and the accounting
    negative "(12.00)". Two decimals at most; a missing fraction is zero.
    """
    text = text.strip().replace("$", "")
    negative = text.startswith("-")
    text = text.lstrip("-")
    whole, _, frac = text.partition(".")
    cents = int(whole) * 100 + int((frac + "00")[:2])
    return -cents if negative else cents
PY

cat > "$repo/ledger/report.py" <<'PY'
"""Monthly totals."""

from .parse import parse_amount


def monthly_totals(rows):
    """Total the (iso_date, amount_text) rows by month.

    Returns a dict from "YYYY-MM" to integer cents whose keys are in
    chronological order, whatever the order of the rows.
    """
    totals = {}
    for date, amount in rows:
        month = date[:7]
        totals[month] = totals.get(month, 0) + parse_amount(amount)
    return totals
PY

cat > "$repo/tests/test_ledger.py" <<'PY'
import unittest

from ledger import monthly_totals, parse_amount


class Ledger(unittest.TestCase):
    def test_plain_amounts(self):
        self.assertEqual(parse_amount("12.5"), 1250)
        self.assertEqual(parse_amount("-3.07"), -307)

    def test_totals_a_month(self):
        rows = [("2026-01-05", "10.00"), ("2026-01-20", "5.50")]
        self.assertEqual(monthly_totals(rows), {"2026-01": 1550})


if __name__ == "__main__":
    unittest.main()
PY

printf '__pycache__/\n*.pyc\n' > "$repo/.gitignore"
printf '# ledger\n\n`parse_amount` and `monthly_totals`. Tests: `python3 -m unittest discover -s tests`.\n' > "$repo/README.md"

git -C "$repo" init -q -b main
git -C "$repo" config user.email "fixture@example.com"
git -C "$repo" config user.name "fixture"
git -C "$repo" add -A
git -C "$repo" commit -q -m "ledger: amounts and monthly totals"

cat > "$root/accept.py" <<'PY'
"""Acceptance for the ledger fixture; kept outside the repository."""
import sys

sys.path.insert(0, ".")
from ledger import monthly_totals, parse_amount  # noqa: E402

bad = 0


def check(got, want, what):
    global bad
    if got != want:
        bad += 1
        print(f"{what}: got {got!r}, expected {want!r}")


check(parse_amount("1,234.50"), 123450, "thousands separator")
check(parse_amount("(12.00)"), -1200, "accounting negative")
check(parse_amount("$1,000"), 100000, "dollar sign and separator")
check(parse_amount("-0.05"), -5, "small negative")
check(parse_amount("7"), 700, "no fraction")
rows = [("2026-03-01", "1.00"), ("2026-01-15", "(2.00)"), ("2026-02-10", "3,000.00"), ("2026-01-02", "5.00")]
got = monthly_totals(rows)
check(list(got), ["2026-01", "2026-02", "2026-03"], "chronological month order")
check(got, {"2026-01": 300, "2026-02": 300000, "2026-03": 100}, "totals")
sys.exit(1 if bad else 0)
PY

else
repo=$root/repo
mkdir -p "$repo/textstats" "$repo/tests"

cat > "$repo/textstats/__init__.py" <<'PY'
from .words import top_words

__all__ = ["top_words"]
PY

cat > "$repo/textstats/words.py" <<'PY'
"""Word statistics."""


def top_words(text, n):
    """Return the n most frequent words of text as (word, count) pairs.

    Words are compared case-insensitively and returned in lower case. Words
    with equal counts are ordered alphabetically.
    """
    counts = {}
    for word in text.split():
        counts[word] = counts.get(word, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: -item[1])
    return ranked[:n]
PY

cat > "$repo/tests/test_words.py" <<'PY'
import unittest

from textstats import top_words


class TopWords(unittest.TestCase):
    def test_counts_words(self):
        self.assertEqual(top_words("a b a", 1), [("a", 2)])

    def test_limits_to_n(self):
        self.assertEqual(len(top_words("a b c d", 2)), 2)


if __name__ == "__main__":
    unittest.main()
PY

printf '__pycache__/\n*.pyc\n' > "$repo/.gitignore"

cat > "$repo/README.md" <<'MD'
# textstats

`top_words(text, n)` returns the n most frequent words as (word, count) pairs.
Run the tests with `python3 -m unittest discover -s tests`.
MD

git -C "$repo" init -q -b main
git -C "$repo" config user.email "fixture@example.com"
git -C "$repo" config user.name "fixture"
git -C "$repo" add -A
git -C "$repo" commit -q -m "textstats: top_words"

cat > "$root/accept.py" <<'PY'
"""Acceptance for the textstats fixture; kept outside the repository."""
import sys

sys.path.insert(0, ".")
from textstats import top_words  # noqa: E402

cases = [
    (("The the THE cat Cat dog", 2), [("the", 3), ("cat", 2)]),
    (("b a b a c", 3), [("a", 2), ("b", 2), ("c", 1)]),
    (("Zed zed apple apple Apple", 2), [("apple", 3), ("zed", 2)]),
    (("one", 5), [("one", 1)]),
    (("", 3), []),
]
bad = 0
for args, expected in cases:
    got = top_words(*args)
    if got != expected:
        bad += 1
        print(f"top_words{args!r} -> {got!r}, expected {expected!r}")
sys.exit(1 if bad else 0)
PY

fi

echo "repo=$repo"
echo "accept=$root/accept.py"
