<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 254. An agent session merges per segment, because the history is sent back as sampled

**Lesson:** with brain sending each reply's `sampled` record and sven sending every answer back verbatim with it, an
agent session is append-only between the points where its history is rewritten (a compaction, a changed system
prompt or tool list, a deferred tool's load). Each such run of calls is one merged training record, every segment
carrying the rollout's one reward; a session stays one record per call only when a call carries no traced ids or
its calls were drawn under several samplers. This supersedes knowledge 252: neither the chat template nor the
runtime re-renders an earlier answer any more.

- **What the record names.** brain's `sampled` record holds the generated ids (stop token included), the generation
  header the prompt ends with (`prefill_token_ids`), the prompt's length (`context_len`) and its digest
  (`context_hash`, sha256 of the ids as little-endian u32), the ids LEB128 varints, base64. It does not hold the
  whole prompt: the prompt ids come from the policy trace, and a call trains them only when the record agrees on all
  four (`ProxyReason::SampledDiffers` otherwise).
- **Tool schemas keep their key order by convention.** brain renders a request's tool list as sent; sven and
  splinter both write JSON objects with sorted keys, so a record's tools render as the request did. A client that
  sends tools with other key orders (a script, another runtime) makes brain refuse the record as
  `prompt_token_ids_mismatch` at the tool block.
- **A turn that reasoned and only called** carries its reasoning on its call in the request log; the history
  comparison must read it there, or every such session diverges at that turn.

**Evidence (2026-11-28, CPU brain, Qwen3-0.6B, f32 KV, temperature 1, policy trace on):** two two-call sessions (a
shell call, its result, a text answer), the first answer sent back with its record. brain reported
`history.append_only: true` on every call; the second prompt starts with the first prompt and generated ids (196 of
218 ids). splinter projected both sessions exactly, merged each into one record (`SessionRecord::segments`), and
brain's `check_dataset` accepted both as packed rollout records (2 valid, 86 supervised ids). brain's `score` of
each call's ids, teacher-forced at temperature 1, reproduced the behaviour log-probabilities to a worst per-id
difference of 5.6e-5 over 4 calls (sums -0.83817/-0.83815, -10.43450/-10.43430, -2.39840/-2.39838). The first
session is the fixture `crates/data/tests/fixtures/brain-session.json`.
