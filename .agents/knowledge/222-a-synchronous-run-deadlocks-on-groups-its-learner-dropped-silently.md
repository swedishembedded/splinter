<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 222. A synchronous run deadlocks on groups its learner dropped silently

**Lesson:** a learner that drops stale groups when it looks at its queue, but reports the drop only with its next
step, holds less than a step while its producer believes it holds the groups. With an allowance of one step (a
synchronous run, `max_staleness: 0`) that is a deadlock: the producer admits nothing until it hears of the drop,
and the learner steps only once it has a step's worth. brain's streaming job does exactly this (its feed pops stale
heads while it waits; their `dropped` frame goes out with the next step). The controller now stops counting a sent
group against the allowance as soon as a publication makes it stale by the learner's own rule (judged against that
publication's version), and keeps it sent until the learner applies or drops it. A run with a staleness bound of two
steps or more always had a step's room beside such groups. That is why the discrete-event simulation, whose learner
reports drops the same way, never hit it: a deadlock that needs the tightest setting needs a test at that setting.
Spec: `crates/factory/tests/it/online/mod.rs` `a_sent_group_a_publication_makes_stale_leaves_the_allowance_at_once`.

## Evidence (2026-11-25, lane agrpo-real)

Qwen3-0.6B on a GH200, 128 groups of 8, `step_tokens: 1024`, `max_staleness: 0`. After step 2's publication
(version 3) the controller held four sent groups drawn at version 2: 1216 tokens by its count and by brain's, and
the two counts agree to the token on applied groups. That was over the 1024-token allowance. brain's trainer
waited in its feed's `take_step` (gdb) with its input thread idle, and nothing moved for 6 minutes until the run
was stopped. With the change the same setting trained all 128 groups to the end (run cmp-S-1, brain 760c04d3b).
