<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 182. An archive ends at a checkpoint, but adopting it walked past

**Lesson:** when one walk of a history stops at a checkpoint (what a snapshot needs) and another walks through it
(what has been absorbed), anything copied by the first must be readable by the second. Test a copy, an archive or a
restore on a history long enough to hold a checkpoint, not only on a few commits.

Found 2026-11-19 (lane H) on the first real-Hub bundle. `publish` of the 550-experience gen-0 set (from an old
loose-layout state) succeeded, and every `ingest-hf` of it failed with `manifests 07f73dc7... not found`; a plain
`state restore` of the bundle's `experiences.tar.zst` failed the same way. The staged database had passed a
checkpoint (the writer squashes every 64 commits): manifest `ac454a6a` is a checkpoint whose parent is `07f73dc7`.
`Snapshot::files`, which `state archive` copies, follows `resolve`, which stops at a checkpoint since it carries the
whole state, so `07f73dc7` was rightly not archived. But `Database::adopt` (the restore's last step) folds the head
into the catalog and trims the catalog with `ancestry_all`, which walked through checkpoints and read every
manifest below them. Every archive of a state whose history held a checkpoint failed to restore; the specs had
archived histories of a few commits. The layout of the source state played no part.

Fix (`crates/expdb/src/manifest/resolve.rs`): `ancestry_all` still walks through a checkpoint, but a manifest missing
below one ends that branch (it is counted as absorbed, since the checkpoint names it); one missing above every
checkpoint is still an error. Archives already written, including the published bundle, restore unchanged. Specs:
`recovery::a_state_whose_history_passed_a_checkpoint_restores_from_its_archive` (70 commits, and the restored state
takes a write) and `hub_bundle::a_large_set_from_a_loose_state_round_trips_past_a_checkpoint` (80 trials from a
loose-layout source); both failed with the same `NotFound` before the fix.
