<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 139. A verifier that imports from its working directory is beaten by a planted module

A verifier written as `cd /app && python3 -c "import sqlite3, json ..."` earned a full reward from the factory's red team in every sampled instance: the `tamper:hijack-imports` attempt writes `sqlite3.py` and `json.py` into the workspace, and the interpreter imports them from the working directory before the standard library's, so the planted code writes `{"reward": 1.0}` itself. The test `a_verifier_that_imports_from_the_working_directory_is_quarantined` pins it: the environment is quarantined with that attack on record. The same attack fails against the database workflow's verifier, which runs `python3 -I /tests/verify.py` (isolated mode: no working directory, no `PYTHON*` variables) and adds its own trusted directory to the path explicitly.

A second form of the same mistake: a verifier whose answer is the same for every instance passes the red team's "another instance's answer" attempt only if that attempt is allowed to use an answer equal to the instance's own. An early version skipped equal answers as "not an attack"; the constant-7 verifier then looked clean.

Rule: a verifier never runs with the solver's directory on its import or search path, and the attempt that replays other instances' answers keeps the ones that equal the instance's own. Evidence: `crates/knowledge/tests/it/factory_admission.rs`, `crates/knowledge/src/factory/attack.rs`.
