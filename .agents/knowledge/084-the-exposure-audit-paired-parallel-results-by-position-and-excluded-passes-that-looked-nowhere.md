<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 084. The exposure audit paired parallel results by position and excluded passes that looked nowhere

A trace audit of the 27B teacher sets (372 attempts) found 35 hidden-check passes excluded by the exposure audit;
by hand 23 had looked at nothing hidden. Causes, all in `exposure_paths`:

* tool results arrive in the order the calls finished, not the order they were made (893 of 1880 multi-call steps),
  and `outputs_by_call` paired them by position, so a guessed hidden test file that did not exist was credited with
  its sibling's file text. Results are now paired by `source_call_id` (position only when the ids are shared or
  unknown; call ids are unique per step, not per run);
* any line that was not a refusal counted as "reached": a Python or Lua traceback, a module-not-found list. A
  failure report is not what a path returned;
* a write to `/tmp` that the sandbox refused, and a `cd` into an invented directory (`cd /home/user/project
  2>/dev/null; python3 -m unittest`, whose output is the solver's own tree), were counted as looking outside;
  a path in a here-document's data (`{"root": "/srv"}`) and `'$(pwd)'/src` (text glued to a closing quote) were
  taken for absolute paths.

Writes and directory changes read nothing; hidden material that a later command prints is still caught by the
hidden-path and content checks. The opposite error was left in on purpose: a search of `/` for another copy of the
code is the behaviour the audit is for, so it fails whatever came back (an empty `find /` had become a harmless
probe, which let four passes that hunted for the answer into the training data).

The check that would have caught it: the audit judged on the stored trajectories of a real run, not only on
hand-built one-call steps. The test helper numbered every parallel call `call_0`, which is why the position pairing
looked fine.
