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
- **Retention.** It still does what earlier releases learned.
- **Anchor.** It still behaves like a general model on a frozen suite of general tasks.
- **Serve.** It runs on plain `brain serve` and answers as it did in-process.

A check that could not be measured fails. A number that was not measured is absent, never zero.

## Why the tests are paired and counted by family

Records about one family of source text (two prints of a letter, several questions about one
document) are one piece of evidence: a model that knows the letter gets all its questions right.
The gate's sign test therefore counts a family once, won or lost by which model got more of its
tasks right. The report adds an interval that resamples families and the exact sign test over tasks,
so a result on a few families is read by its spread. An exam too small to see the lead you asked
for (`--target-effect`) is reported inconclusive and not as a failure. Over a lineage of releases,
`--alpha-spending` spends the significance level on a fixed schedule so that a long run of
candidates no better than their champions does not eventually release one by chance.

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
