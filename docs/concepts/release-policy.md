<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Release policy

A trained candidate is not the policy. It becomes the policy only when a gate, run by Splinter
and not by the training, measures that it improved and did not break anything else. This page
explains the rules and why they are shaped as they are; the checks, thresholds and report
fields are in the [evaluation reference](../reference/evaluation.md#the-release-gate), and
the commands in [Release, serve and roll back](../guides/releases.md).

## A candidate must earn its place

The gate compares the candidate with the **champion**, the release the alias points at (the
base model before any release), on the same suites, graded closed-book by each task's own
verifiers. It releases only if all four checks pass:

- **Improvement.** The candidate learned what it was trained on, measured on questions it was
  not trained on: the held-out records and differently worded variants of trained tasks.
- **Retention.** It still does what earlier releases learned and what the frozen suite of
  general tasks holds.
- **Anchor.** It stays within a tighter floor on that frozen suite of general tasks.
- **Serve.** It runs on plain `brain serve` and answers as it did in-process.

A check that could not be measured fails. A number that was not measured is absent, never zero.

## One promotion rule

Improvement and retention are decided by the one promotion rule every promotion in Splinter goes
through: the paired gain on the improvement suite
must have its two-sided 95% interval above zero over at least six families, and on the retention
suites (every earlier release's suite and the anchor suite together) the lower bound of the
one-sided 95% interval of the paired change must be above -0.05, both intervals resampling whole
families. The family minimum is there because a resampled interval says nothing about how many
families agree: two families, both won, resample to the interval [1, 1]. Six is the fewest whose
unanimous agreement the exact sign test finds significant at the interval's tail (2.5%: 1/64, where
five give 1/32), so below it no exam outcome could show a gain at the stated level; a gain over
fewer families is refused as `too_few_families`. The other
checks (every assigned task counted for each arm, the anchor floor, the served candidate, the
canaries) refuse beside the rule, never in its place.

When the caller holds an examiner's signed verdicts on the candidate, the rule decides on them and
a fresh sealed exam slice is part of the decision: the release is **exam-verified**. A release
decided on Splinter's own suites, a release from `release` among them, whose exam the learner made
itself, has no sealed slice: the same two rules decide, and the release record (and the `release`
output) says **not exam-verified**, with why. It is never a silent bypass.

## Why the tests are paired and counted by family

Records about one family of source text (two prints of a letter, several questions about one
document) are one piece of evidence: a model that knows the letter gets all its questions right.
The promotion rule's intervals therefore resample whole families, and a gain resting on a few
families that disagree shows as a wide interval (with a single family there is no interval, and no
promotion); a gain over fewer than six families is never counted, since unanimous families
resample to an interval of no width.
The report adds the sign test that counts a family once and the exact sign test over tasks; they
are read beside the decision and do not make it. A gain not shown on an exam too small to see the
lead you asked for (`--target-effect`) is reported inconclusive and not as a failure; either way
nothing is released. The rule decides at its fixed level; no setting spends the level across a
lineage.

## What stops a release from lying

- **Leakage.** A held-out task the candidate trained on, whether the same question, a near
  duplicate or its words inside a longer prompt, is left out of the measurement and counted.
- **Canaries.** `release --canaries` hands the gate tasks the candidate must fail; passing one
  refuses it whatever else it passed.
- **The champion is fixed.** An alias moves only from the champion the gate measured against, and
  moves by compare-and-set; a run keeps the release it started with however the alias moves.
- **Immutability.** A release is written once, with every number the gate measured, and is
  never edited. `rollback` repoints the alias at the release the current one was trained from.
- **Terms.** A release records the terms of everything it was made from and stays local unless
  every one allows distribution and commercial use (`release --unrestricted`).

## Absorbing a person's corrections

The nightly `absorb` loop is gated by a declared rule on counts instead of a significance test:
a claim is answered when every one of its held-out paraphrases passes, improvement means answering
a claim the champion did not, and retention means every claim the champion answered is still
answered, with no tolerance. See [Learn from your own agent sessions](../guides/agent-sessions.md).
