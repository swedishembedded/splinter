<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 192. An edited driver script breaks running bash jobs

**Lesson:** bash reads a script as it runs it, not all at once. Editing a driver script (`practice.sh`) while jobs still run
it makes each of them resume at its old byte offset in the new text once its current command returns, and run whatever
fragment is there. Never edit a script that running jobs execute: copy it to a new name for each new launch.

Seen 2026-11-20 (item 21 S1b): the lane's `practice.sh` was rewritten to add cell (h) while three long practice runs were
inside its `splinter tasks harbor practice` command. When splinter returned, each run executed a continuation-line
fragment (`--sven-build ...: command not found`), which also truncated its log through the line's redirection, and
recorded rc 127. The practice itself had finished: each run's `report.json` was written at its end time and holds every
trial (128, 24 and 16), so the results were harvested from the state and only the logs were lost. A run that ends nonzero
after its main command has returned can be complete; check the state before rerunning.
