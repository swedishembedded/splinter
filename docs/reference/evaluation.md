<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Evaluation reference

Where a release is decided and what is checked there, the anchor suite that `eval` freezes and shows,
and the file format of the anchor suite. How to run an evaluation is in the [evaluation guide](../guides/evaluation.md).

## Where a release is decided

A candidate becomes a release in one of two places, each with its own rule:

- **A factory run's `promotion` stage** decides on the run's evaluations, each of a stated
  purpose: the target and transfer suites must be improved and the retention suite must not
  regress, and a candidate whose lineage shows its target climbing while transfer stays flat trips
  the divergence alarm. The rules are in [the learning factory](../guides/learning-factory.md#what-each-stage-guarantees).
  An evaluation of a protected plan counts an examiner's verdict only when its Ed25519 signature
  verifies under the examiner's public key and it is about this candidate at the plan's level.
- **Absorption** (`absorb`, `consolidate`) decides on counts, with no significance test: a claim
  is answered when every one of its stopping paraphrases passes, the candidate must answer a claim
  the champion did not and keep every claim the champion answered, and, when an anchor suite is
  frozen, may not get fewer of its items right. The rules are in
  [Learn from your own agent sessions](../guides/agent-sessions.md#absorb-from-sessions-to-a-release).

Both hold the candidate to the serve check before it is released:

| Check | Passes when |
|---|---|
| serve | `brain serve --adapter <candidate>` (the `brain` on `PATH`, or `--brain-bin`), with the base checkpoint the candidate was trained on as its `--checkpoint`, in a runtime directory of its own, starts - brain binds the adapter only to the base whose digest training recorded on it - reports the candidate's adapter digest, and re-answers a sample of the held-out items through its OpenAI-compatible endpoint, both sides decoding greedily and without a reasoning block, with the same answers as in-process on at least three quarters of them (the same verdict, and either the same text over its first nine tenths, runs of whitespace aside, or - worded otherwise, as a sampling server or another summation order will - the same meaning: embedded by Qwen3-Embedding, an answer must be nearer its own in-process answer than to the in-process answer of any other task, so the comparison cannot call everything alike); each task answered differently is reported with both answers |

A release made before these rules carries the report of the earlier graded-answer gate
(improvement, retention, anchor and serve checks, and its promotion record); such releases still
load, and `release list` and their reports show those numbers as recorded.

## Training warnings

The warnings a training run reports about its curve (a large generalisation gap at the step
carried, a monitoring loss that rose after it or was still falling at the end of the budget) are
carried by the exam of that candidate (`training_warnings`) and printed with its verdict.

## Suites

`splinter eval` with nothing after it shows the anchor suite in force. The anchor suite is frozen
from one or more files with `eval --freeze FILE...` (the
files' tasks in order are one version; different tasks are the next version;
the same tasks are the same version). An
anchor file is JSON Lines, `{"instruction", "reference", "kind"?}`, `kind`
(default `recall`) a closed-book kind graded against its reference by code:
`recall` (the answer states the reference), `arithmetic` (the last number the
answer states is the reference) or `format` (the answer has exactly the
non-empty lines the reference counts). The last two are what a persona
fine-tune of a reasoning model erodes and trivia cannot show. A file in this format is also
what `eval --suite FILE` grades a model on ([evaluation guide](../guides/evaluation.md#grade-a-model-on-a-file-of-tasks)).
