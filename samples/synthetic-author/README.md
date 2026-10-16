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
synthetic-author persona --show 3
```

The command draws an author, prints `no --seed given, using seed N`, and shows
the beliefs and decisions. To reproduce the author, pass the printed seed back:
`synthetic-author persona --seed N --show 3` prints the same text.

## What is measured (as it is built)

A trained model (`tuned`), unprompted, against the author's own decisions on
situations and tools it never saw; against the same base alone (`base`), given
the author's name (`named`), given the beliefs in a prompt (`principles`) and
given the paragraphs of the corpus that share the most with the situation
(`retrieval`); and, where an author who holds the opposite beliefs would
decide otherwise, whose decision it took (`cross`). The opposite author is an
oracle for every author's `cross` rows, and `corpus --opposite-of SEED` writes
a corpus for it that a model can be trained on (see below). The base is not
given the whole corpus in its context: the retrieval arm hands it the
paragraphs that share the most with the situation, and nothing longer exists.

## Seeds and manifests

Every default seed is random, is printed when it is drawn, and is recorded
beside what it made.

* `corpus` without `--seed` draws the author, prints the seed and writes
  `manifest.json` into the corpus directory: the author seed, the writer and
  reader references, their decoding, the run seed the per-request writer and
  reader seeds are derived from (pass `--run-seed` to reproduce, or read it
  from the manifest), the options and the code revision. A corpus is only
  continued with the recipe it was begun with.
* `mine` without `--run-seed` draws the seed its reader's per-request seeds
  are derived from, prints it and records it with the reader in `mine.json`
  (a reader reached over a network is not seeded). A later `mine` of the
  corpus continues with both and refuses another reader or another seed.
* Every later command that is given the corpus directory (`recheck`,
  `dataset`, `exam`, `items`, `audit`, and `persona`, `plan` and `show` with
  `--cache`) reads the author from it and refuses a `--seed` that names
  another author. `persona`, `plan`, `show` and `items` without a corpus take
  `--seed`, else draw one and print it.
* `exam` writes `exam/manifest.json`, one record per arm: the digest of the
  items, `--articles`, `--per-cell`, the base and adapter digests, the
  decoding (greedy, which draws no seed; or sampled, with its temperature, run
  seed and number of draws), the reply limit and the arm's
  system turn and which examination it is (`v1` or `v2`). Each outcome carries
  the digest of its item and of its model.
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

## The examination, in two versions

`exam --exam v1` is the examination as first reported: every item lists its
options in one canonical order, which puts the right answer first in about
half the items, so a policy that always takes the first listed option beats a
guess. `exam --exam v2` (the default) draws the same situations and is a
different examination, with a different digest; outcomes of one are refused
under the other (use another corpus directory, or pass `--exam v1`):

* each item lists its options in an order drawn from the seed, and the right
  answer sits at each position equally often within every kind of difficulty,
  to within one item (a third of 20 items is 6 or 7); `items` shows the listed
  order and the right answer's position;
* each item says whether some decision kept from the corpus has its facts and
  its options (`combination_trained`), which is not the same as
  `combination_seen`: that one says only that the combination was not
  withheld from the corpus's planner. The flag is read from `decisions.jsonl`,
  so v2 needs `mine` first and refuses to be drawn without it;
* `report` scores two constant policies on the arms' own items and lists them
  beside the arms, each compared with the tuned arm by the same paired sign
  test: always the option listed first, and always the option the beliefs
  determine most often (`majority-option`). A result is worth something only
  above them. `report` draws the items again from the manifest and does not
  score the policies if they no longer match the digest the arms were asked,
  as when the corpus was mined again after the examination.

## Failures

A request to the writer or the reader that fails in transit says nothing about
the article. `corpus` repeats it up to `--transport-retries` times (default
four), pausing `--transport-backoff-secs` (default five) before the first
repeat and twice as long before each next, never more than a minute. A repeat
uses no attempt at the article and is counted in the budget of calls. Past the
bound the run stops and says which model could not be reached, and the article
is reported as `unreachable`, apart from those given up on after their
attempts. A reader whose answer is in no readable shape turns the draft away as
a content rejection, which the writer is not told of.

Each line of `rejected.jsonl` carries a `kind`: `content` (a gate or an
unreadable answer turned the attempt away; only these reach the writer as the
reason for a retry), `writer-unavailable`, `reader-unavailable` (the draft is
kept, unjudged) or `budget` (the run stopped before the draft was read).

`mine` fails when a line cannot be appended to `decisions.jsonl`, and reads an
article again when some of its cases have no line, appending only those.
`recheck` fails naming the line of `rejected.jsonl` that has no article index.
An unreadable `usage.json` fails `corpus` before any request, naming the file.

## What the gates took out

`survivors --cache DIR` reads the plan, `articles/` and `rejected.jsonl`, asks
no model, and prints JSON: the decisions planned against the ones the kept
articles tell, by the option chosen (`by_option`), by the belief that decided
(`by_belief`) and by the facts and options of the case (`by_combination`), and
the planned articles that were not kept, grouped by why (`given_up`): `content`
when a gate turned some attempt away, `writer-unavailable`,
`reader-unavailable` or `budget` when only requests failed, `unattempted` when
nothing was logged. A decision counts as kept when the kept article tells its
case.

## Sampled decoding beside greedy

`exam --decoding sample[:T] [--draws K] [--decoding-seed S]` asks every item K
times (default eight) at temperature T and keeps the outcomes beside the greedy
ones, in `exam/<arm>-sampled.jsonl` under the record `<arm>-sampled`. Each
draw has its own seed, derived from one run seed and the item's place and
written with the draw; the run seed is the `--decoding-seed`, drawn and
printed when absent and read back from the record afterwards. The record names
the decoding, the run seed and the number of draws, and outcomes are never
appended to under another one. `report` lists each sampled arm beside its
greedy result: the share of an item's draws that were right, averaged over
items, with a bootstrap interval over items, and for each pair of arms the
mean difference of their per-item shares with its interval. The bootstrap seed
is `--bootstrap-seed`, drawn and printed when absent and written into the
report.

## Reasons

A reason is part of what a record teaches, so `dataset` makes no record for a
decision without one: a decision whose quoted reasons are all blank, or repeat
a phrase of the brief the essay was written from, is counted in `without_reason`
of its summary and leaves no record. A reason is taken from the essay's own
words; the brief-copy gate refuses an essay that repeats a six-word run of what
the author did in the brief's voice or in the author's own ("settled it
myself, at once, without waiting"), and the dataset refuses a reason that does.

`report` grades the reasons of each arm with code and no judge, beside the
accuracy table: how many name the fact the deciding belief turns on (a fixed
list of word stems per fact, so a paraphrase can be missed), how many repeat a
phrase of the brief, and how many different reasons there are. An arm that gave
no reason is `not measured`.

## The opposite author

`corpus --opposite-of SEED` writes the corpus of the author who holds every
belief of the author SEED draws the other way, in the same world (the same
situations and the same withheld combinations). The manifest records
`persona_source: opposite-of` and names SEED, which is also the corpus's author
seed, so every later command given the directory (`recheck`, `mine`, `dataset`,
`exam`, `items`, `audit`, `survivors`, and `persona`, `plan` and `show` with
`--cache`) reads the opposite back as that author. The opposite must pass the
rule any drawn author passes (every belief changes some decision, and the
beliefs settle most situations); one that does not is refused with the belief
or the share that fails, and nothing is written. A corpus is only continued as
the kind of author it was begun for.

`cross --exams DIR` reads `DIR/author-<seed>` for each author and, where it
exists, `DIR/opposite-of-<seed>` for the opposite trained on its own corpus, and
lists for each trained model whose decision it took on the situations where
the two authors are determined and decide differently. The two adapters are
examined on the situations of their own authors, so the rows set them side by
side; a paired test between the two on identical situations needs an
examination both are asked, which does not exist.

## Mechanism ablations

Switches for asking which part of the method accounts for the result. None of
them trains or asks a model by itself; each is a name or a flag.

* `dataset --briefings both|reader-only|fixed-wording-only` makes the records
  with both briefings of a situation (the default), with the reader's alone, or
  with the fixed-wording one alone (the facts in the words the examination
  puts them in, which exists only for a decision whose essay stated every
  fact). Each setting is written to a file of its own, and
  `train --briefings` trains on that file in an attempt directory of its own
  and records the setting in `summary.json`.
* `exam --arm` takes, besides the arms above, `retrieval-2` and `retrieval-12`
  (paragraphs of whole articles, the original `retrieval` being six) and
  `retrieval-records-2`, `retrieval-records-6` and `retrieval-records-12`
  (passages drawn from the training records themselves: each decision kept
  that has a reason to teach, as the situation the reader briefed, the
  decision in the tools' label and the reason). Retrieval over the records
  needs `mine` first. The arm's record in `exam/manifest.json` names what the
  passages were drawn from and how many were handed over (the original arm's
  record is as it was), and the digest of the corpus they were drawn from;
  outcomes are not appended to after the articles or the decisions changed.
* `principles-verdicts` is the principles arm that also states, for each
  conviction, what it makes of each way to act (which ways it rules out,
  disfavours or favours, and where), read off the same table the oracle
  decides by, and how the convictions are applied in order.

## A reader other than the writer

One model writes the essays, one reads each draft for the gate, and one reads
the decisions out of the kept articles (`corpus --writer`, `corpus --reader`,
`mine --miner`). If two of them are the same family, a mistake that family
makes consistently is written, passed and read back without anything noticing,
so the protocol wants three families. A model's family is the letters its name
begins with, in lower case (`aster-xl-4` and `Aster-3.5` are `aster`,
`Cedar3-8B` and a checkpoint directory `Cedar2.5-7B` are `cedar`); a policy alias
is its own family.

* `corpus` refuses a reader of the writer's family, naming both and the family.
* `mine` refuses a miner of the family of the writer or the reader the corpus
  manifest records. A corpus from before manifests records neither, so nothing
  is compared.
* `--allow-same-family` sets the check aside; `manifest.json` (for `corpus`)
  or `mine.json` (for `mine`) then says `allow_same_family: true`, and a corpus
  is only continued the way it was begun.
* The gate reader's reading of every draft is kept in `readings.jsonl` with its
  attempt and whether the attempt was kept. `audit` sets it beside the miner's:
  the gate reader's agreement with the beliefs over every case it read, kept or
  turned away; the miner's over the decisions it kept; and, on the kept cases
  both read, how many agree on the decision, on the ways open and on the facts,
  with the cases that differ listed. A corpus without readings (judged by marker
  words, or written before they were kept) reports `not measured`.
