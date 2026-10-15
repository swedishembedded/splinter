<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->
<!--
Swedish Embedded AB implements controlled validation of learning systems
against ground truth that cannot have been seen in pretraining, for its
clients. If your team needs expertise in proving that a trained model has
learned what a corpus implies, you can procure our services by sending an
email to info@swedishembedded.com.
-->

# synthetic-author - can Splinter learn beliefs it was never told?

A real author's texts cannot show whether a model learned the author's
judgment, because the model may have met the author in pretraining and the
author's decisions are open to interpretation. This sample removes both
problems. It draws an author at random whose beliefs are few, formal and
known, has the author write about fixed topics so that the beliefs show only in
what is decided, and measures a model trained on those texts against what the
beliefs determine.

Nothing generated here is checked in. An author is a pure function of a seed;
the corpus and every reference answer are written to a cache outside the
repository and can be thrown away and drawn again.

## What exists

The author: six tenets, each read one of two opposed ways, four held at a time
in a hidden order of priority. Given a situation (facts and three options),
code decides what the author chooses and which belief decided it. Every belief
an author holds changes some decision, and the situations are the same for
every author, so a situation alone says nothing about who decides it.

```bash
synthetic-author persona --seed 7 --show 3
```

## What is measured (as it is built)

A trained model, unprompted, against the author's own decisions on situations
and tools it never saw; against the same base given the beliefs in a prompt;
against the base with the corpus in its context; and against an author who
holds the opposite beliefs.

## Seeds and manifests

Every default seed is random, is printed when it is drawn, and is recorded
beside what it made.

* `corpus` without `--seed` draws the author, prints the seed and writes
  `manifest.json` into the corpus directory: the author seed, the writer and
  reader references, their decoding, the run seed the per-request writer and
  reader seeds are derived from (pass `--run-seed` to reproduce, or read it
  from the manifest), the options and the code revision. A corpus is only
  continued with the recipe it was begun with.
* Every later command that is given the corpus directory (`recheck`,
  `dataset`, `exam`, `items`, `audit`) reads the author from it and refuses a
  `--seed` that names another author. `persona`, `plan` and `show` have no
  corpus and still take `--seed`.
* `exam` writes `exam/manifest.json`, one record per arm: the digest of the
  items, `--articles`, `--per-cell`, the base and adapter digests, the
  decoding (greedy, which draws no seed), the reply limit and the arm's
  system turn. Each outcome carries the digest of its item and of its model.
  Outcomes are never appended to under another manifest: a rerun with other
  flags, another adapter, or an adapter retrained at the same path is refused
  with both manifest digests named. `report` and `cross` join outcomes only
  when their digests agree.
* Outcome files written before manifests existed are taken over with
  `exam --adopt`, which records the flags given as their manifest and says it
  did; the old outcomes are not checked against it.
* `train` without `--train-seed` draws a seed and prints it. `summary.json`
  records the seed, steps, rank, alpha, learning rate (`null` is brain's
  default), records per step, bf16 and the digest of the records.
* A cell with no items reports `not measured`, and an arm asked for in
  `report` that has no outcomes is named in it.
