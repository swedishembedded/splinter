<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Release policy

A trained candidate is not the policy. It becomes the policy only when a gate, run by Splinter and
not by the training, measures that it improved and did not break anything else. This page explains
the rules and why they are shaped as they are; the checks are in the
[evaluation reference](../reference/evaluation.md#where-a-release-is-decided), and the commands in
[Release, serve and roll back](../guides/releases.md).

## A candidate must earn its place

A factory run's `promotion` stage compares each candidate with its baseline on evaluations of
stated purposes, and promotes one only when all of them agree:

- **Target.** It improved on what it was trained for.
- **Transfer.** It improved on tasks it was not trained on, so the gain is not the verifier's alone.
- **Retention.** It did not regress on what the baseline could already do: neither a loss the
  plan's sign test finds significant nor a pass rate more than 0.05 below the baseline's.

A recipe without a transfer or retention suite has to say so (`requires: [target]`); a promotion
never quietly decides on fewer purposes than it was asked to. A check that could not be measured
fails. A number that was not measured is absent, never zero.

## Why transfer is asked for

A candidate can learn to satisfy its verifier without learning the task. Each release records a
lineage, one point per generation of the champion with its target and transfer pass rates, and a
candidate whose target keeps climbing over the last generations while transfer stays flat trips
the divergence alarm and is not promoted.

## Why the tests are paired and counted by family

Tasks of one family (several questions about one document, several faults in one file) are one
piece of evidence: a model that knows the document gets all its questions right. The comparisons
therefore pair the candidate's and the baseline's attempts at the same tasks and count a family
once, so a gain resting on a few families that disagree is not mistaken for many independent wins.

## What stops a release from lying

- **Leakage.** A task the candidate trained on, whether the same task, a near copy or its words
  inside a longer prompt, never reaches an evaluation that decides about it, and examination
  material never reaches a dataset.
- **Signed verdicts.** Where a plan is protected, the examiner's verdict counts only when its
  signature verifies under the examiner's public key and it names this candidate; Splinter holds no
  signing key, so it cannot make a verdict it would accept.
- **The champion is fixed.** A run names the champion it trains from (`champion:RUN`) and records
  the one it leaves in use; an alias of the release store moves only by compare-and-set.
- **Immutability.** A release is written once, with every number its gate measured, and is
  never edited. `rollback` repoints an alias at the release the current one was trained from.
- **Terms.** A release records the terms of everything it was made from.

## Absorbing a person's corrections

The nightly `absorb` loop is gated by a declared rule on counts instead of a significance test:
a claim is answered when every one of its held-out paraphrases passes, improvement means answering
a claim the champion did not, and retention means every claim the champion answered is still
answered, with no tolerance. See [Learn from your own agent sessions](../guides/agent-sessions.md).
