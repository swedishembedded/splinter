<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 081. The repo-replay gate called honest work forbidden or unread in five ways seen in real traces

The first thirty verdicts of the v2 replay of a real repository (27B, harness v2) were read
trace by trace. Fourteen of the failures were `diff_reference`, and those were right: the
grounding prompt asked the agent to compare its `git diff` with "the change above", so the agent
wrote "the diff matches the intended change exactly" (the review now checks against the task, which
the training record holds). The other rules lied five times:

* `forbidden_shell` "an interpreter writing a file" fired on `python3 -c "...yaml.safe_load(open('board.yaml'))..."`: the
  pattern took any `open(` as a write. Now only a write mode, `write` or `mode='w'` counts.
* `forbidden_shell` "a here-document" fired on `grep -n 'x << 9\|y' file`, and `patch`, `tee`, `curl` fired on a
  grep for those words. These patterns are now read on the command with quoted text removed and
  anchored where a command starts, unless the command hands a quoted script to a shell or interpreter.
* `leaked_target` fired on `acme_core_m` in a grep because the parent's manifest spells the crate
  `acme-core-m`; identifier words split at hyphens, so the underscore spelling looked new. A dashed
  name now also counts as its underscore spelling, in the parent vocabulary, the plan and the justification support.
* `unread_edit` fired after `grep` with `context_lines: 25` in the very file that was then edited (the file had
  been shown). A search in that file with context lines now shows it.
* Pass rates counted every ledger line, so a redone attempt (`--redo-class`) was counted beside the
  attempt it replaced. Rates now count the latest line of each (commit, style).

Not defects of the gate but of the environment, now said in the replay prompts: builds have no network
(the model set `CARGO_HOME=/tmp/...` and redirected output to `/tmp`, both outside the tree), and
`git stash` made the model believe its edits were lost (they were not) and loop on one todo call.
Specs: `samples/repo-replay/tests/gate_spec.rs`, `samples/repo-replay/src/ledger.rs`.
