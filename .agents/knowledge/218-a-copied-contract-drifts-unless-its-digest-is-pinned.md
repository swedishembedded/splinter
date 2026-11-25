<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 218. A copied contract drifts unless its digest is pinned and the server reports its own

**Lesson:** Splinter's copy of brain's training-record example set (`crates/data/tests/fixtures/training-records/v2`,
"copied unchanged") had fallen behind brain's: it lacked the `behavior_top` and `behavior_top_count` refusal files,
the rollout record with `policy_version`, `sampler` and the sampler's top-K, and the `loss_weight` rule, while the
view test that reads the copy kept passing, since it checks Splinter's records against whatever the copy says. A
copy is held to its source only by a digest both sides compute: brain reports the digest of its own set
(`GET /v1/capabilities` `server.training_records.digest`, `data::chat_rollout::contract_digest`), Splinter pins the
digest of its copy (`splinter_core::revision::RECORD_CONTRACT_DIGEST`, held to the copy by a unit test) and refuses a
server that reports another (`ServedRevision::check_contract`, before any stage of a run that trains runs).

## Evidence (2026-11-25, lane agrpo-prov)

`diff -r` of the two directories at brain `b0c6e5aaf` and Splinter `fd263c909`: two files only in brain's set,
`chat_rollout.jsonl` one record shorter, `manifest.json` and `README.md` behind. The digest of brain's set is
`sha256:fc3d95ec85ab3278fd93a04aab623ad1768038aea152178177d6901667b0ff75` (computed by a Python script and by
`contract_digest`, which agree); the copy now equals it, and the refusal test
(`crates/factory/tests/it/revisions.rs`) shows a server reporting any other digest refused before a stage runs.

## Resync (2026-11-25, lane agrpo-support)

brain's set gained the per-token sampler support of a cut sampler (`behavior_support_offsets`,
`behavior_support_ids`, `behavior_support_logprobs`): `chat_rollout.jsonl` line 9, `invalid/behavior_support.jsonl`,
`invalid/behavior_support_count.jsonl`, the manifest and README. The copy was replaced with brain's set file for file
(`diff -r` empty) and `RECORD_CONTRACT_DIGEST` moved with it to
`sha256:4fb13e2f610fb2d3d4ede4731c5da509d2ccfae1071143c796015e824a4bc20b` (`data::chat_rollout::CONTRACT_DIGEST` of
brain f00e2186a), so a brain built before that commit is refused, as is this build by one after a later change.
