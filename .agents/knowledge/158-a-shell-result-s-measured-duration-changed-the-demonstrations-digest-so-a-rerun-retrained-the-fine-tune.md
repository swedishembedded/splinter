<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 158. A shell result's measured duration changed the demonstrations' digest, so a rerun retrained the fine-tune

**Lesson:** a dataset whose digest keys training must be a function of what the policy saw and did, never
of when or how fast. Anything a tool reports about time (a shell's `Duration: 0.1s`), a server-assigned
id (a tool call's `call_c7dc..._0`) or an attempt number makes two collections of the same episodes
different bytes, and every stage keyed on the dataset reruns. Canonicalise those fields where records are
written (`crates/data/src/wire.rs`), digest an episode's decisions (`outcome.json`, records) rather than
its timestamped trajectory, and pin it with a spec that writes the same episode twice.

Found 2026-11-16 in the minimind pass-2 run ledger (`~/project/shared/minimind-runs/gpu3`): the second run
re-collected the 1600 expert demonstrations (the whole-tree code revision in every key had changed), the
records differed only in shell durations and attempt-numbered episode names, and `ouro/sft` trained
again for 13 minutes (`ouro/sft#2`) on what was the same data. With the fields made canonical and the
stage keys holding per-stage recipe versions instead of the tree revision, a rerun after a report-only
change starts no unit but `init`, and a new sven re-collects the episodes yet trains nothing
(acceptance spec `a_change_recomputes_what_consumes_it_and_nothing_else`).
