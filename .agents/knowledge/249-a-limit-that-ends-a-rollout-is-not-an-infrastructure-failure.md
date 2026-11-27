<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 249. A limit that ends a rollout is not an infrastructure failure

**Lesson:** an online rollout whose conversation outgrew the served context was an `Err` the controller retried as an
infrastructure failure, so the same runaway ran again until its retries were spent and was then abandoned, never
scored. In the profile of a short Qwen3-4B run (async-grpo ledger) one such rollout, three attempts at the 2048-token
context, took 129 s of a 175 s run while the learner waited.

**Fix:** one truncation classifier (`splinter_core::factory::truncation`) reads a run in this process (its conclusion,
or a server's `context_length_exceeded`) and a trial on the execution worker (its grade and evidence) onto the same
causes (context, output length, output budget, turns, deadline). A rollout a limit ended is a scored failure (reward 0)
recorded with its cause and never retried; its completed calls stay training material unless a reply was cut at its own
token limit (brain trains none of such a call). Specs:
`a_conversation_that_outgrew_the_context_is_a_scored_failure_never_retried` (splinter-pipelines `it::online_episodes`),
`a_rollout_a_limit_ended_is_a_scored_failure_with_its_cause` (splinter-factory `online::rollout`).
