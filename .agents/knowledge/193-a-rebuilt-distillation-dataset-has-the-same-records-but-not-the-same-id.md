<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 193. A rebuilt distillation dataset has the same records but not the same id

**Lesson:** a dataset id covers the projection's code as well as the experiences, so rebuilding a
published dataset with a later splinter gives another id even when the training content is the same.
Compare the records, not the digest. Rebuild with the original flags (here `--repository-history keep`,
since the reference set predates `drop-step`), count tokens with the trainer (`--brain-bin` or an
endpoint: without one the floor count kept 421 records instead of 420), and diff the records keyed by
their experience.

Found 2026-11-20 while documenting the distillation recipe (`docs/guides/distillation.md`). The reference SFT
set `blake3:e2eb1efb...` (420 packed records, splinter 3fc94481f) was ingested from the private bundle at
`splinter-b0f65b12fffa370c40fa` and rebuilt from its 550 experiences with
`dataset build --view sft-step --pack --max-block 16384 --from local:Qwen/Qwen3-4B --repository-history keep`.
The result was `blake3:0efa3688...` (the same id in two fresh state roots). It had the same 420 experiences and
the same exclusions (failed 105, repeated_action 7, over_block 25). Brain's `check_dataset` read the same
2,894,518 tokens, 1,544,541 of them supervised, with the longest at 15,901. The messages and tools of all 420
records are identical once two deliberate changes since then are undone:
- a long listing's `agent` owner is now kept instead of renamed `user` (`crates/data/src/scrub.rs`,
  `SHARED_OWNERS`);
- every record now carries `group` and `split_draw`, and the manifest carries the seeded, stratified split
  (knowledge 169).

The frozen 3fc94481f binary cannot read today's packed store, so a byte-exact rebuild is not possible
from the bundle.
