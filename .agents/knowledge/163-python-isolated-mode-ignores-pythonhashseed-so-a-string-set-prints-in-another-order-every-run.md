<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 163. Python's isolated mode ignores `PYTHONHASHSEED`, so a string set prints in another order every run

**Lesson:** a Python output-prediction task is not deterministic just because the program has no clock or
random draw: `python3 -I` (what the `python3` runtime runs) ignores every `PYTHON*` variable, so the hash
seed cannot be pinned, and a program that prints a set or dict-view of strings changes between processes. The
task generator's two runs catch it only half the time for a two-element set; the Python family therefore
reruns every proposed program four more times (`Admission::reruns`) and refuses an output that differs.

Found 2026-11-16 building the Python 3 family of `samples/campaign-lua`: three `python3.14 -I -c
"print(set('abcdefghijklmnop'))"` runs printed three different lines, and with `PYTHONHASHSEED=0` in the
environment two runs still differed. Integer sets are unaffected. Also confirmed there: a distribution
`python3.14` run through a symbolic link finds its standard library (it resolves the link), so the runtime's
search-path lookup can be pinned to the interpreter's digest with a link named `python3` in a directory of
the family's own (`Language::runtime_dir`).
