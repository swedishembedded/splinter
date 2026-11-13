<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 141. A seed drawn over the whole u64 range does not fit a TOML integer

Seeds are drawn over all of `u64` and TOML integers are signed 64-bit. `task.toml` recorded the seed as an integer and every admission failed on the first instance whose seed exceeded 2^63 ("out-of-range value for u64 type"); in the first run all twelve admission tests failed at the first generated instance, since the seeds are drawn that way. The seed is now written as decimal text. The same lesson holds for any record a Harbor-format file shares with other tools: write the full-range value as text.

The same session found the cost of file-per-instance storage: a campaign draws thousands of instances of a few files each, so the store keeps one `instances.json` per environment with the files shared between instances stored once, and a task folder exists only while a run has it staged.

Rule: a value that can use all 64 bits is written to TOML and to any JSON a JavaScript reader may see as decimal text. Evidence: `crates/knowledge/src/factory/artifact.rs` (`seed_text`), `crates/knowledge/src/factory/store.rs`.
