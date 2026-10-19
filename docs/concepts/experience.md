<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Experience

An **experience** is one recorded attempt at a task: the whole trajectory of what an agent
did, in the ATIF format that [sven](https://swedishembedded.com/docs/sven) records, together
with the environment it ran in and the model that produced it. Experience is the raw material of
learning, and the rule that governs it is that it is never rewritten.

## Immutable, with facts derived about it

An experience is content-addressed and immutable. What is later learned about it is added beside
it, never into it:

- **Verdicts** from verifiers, ranked by strength (executable, formal, consistency, judged),
  each recording the verifier that gave it.
- **Rewards and decisions** derived from the verdicts by one rule that turns an experience's
  verdicts into a single decision.
- **Step labels** and **relations** to other experiences (a critique of it, a retry of it, a
  revision, a preferred-over pair, a variant of its task).

Experiences are grouped in experience sets. Every experience is also an attempt in the
experience database's graph, so `splinter lineage` can trace it. Inspect them with `splinter
experiences list` and `splinter experiences show ID --graph`; compare two sets of the same
tasks with `splinter experiences compare`.

## Which experience qualifies as training data

Splinter, not the agent runtime, decides what an experience is worth:

- A task is worth training on when the policy fails it at least sometimes and a verified answer
  exists, either the policy's own (the frontier) or a teacher's (taught). A task every attempt
  solved carries no signal; a task with no verified answer has nothing to learn from; a task
  with no graded attempt has no rate, and unmeasured is never zero.
- A round's training set is deduplicated with the task generator's own
  near-duplicate rule, strongest verdict first, then capped: a concept may
  take at most a quarter of it, a task kind half, a verification strength
  three quarters - never less than an equal split among the groups the
  pool actually has.
- A teacher's open-book solve is training data and never a measurement of the policy: it counts
  toward neither a task's pass rate nor concept mastery.
- The student sees only the instruction: the material a teacher was shown is not in the training
  record (`dataset build --strip`).
- Experiences whose trajectory shows the hidden tests, and attempts the harness could not carry
  out, are set aside and counted, and are in no denominator.

The curriculum behind these rules is described in [the learning loop](learning-loop.md) and the
projections in the [datasets reference](../reference/datasets.md).
