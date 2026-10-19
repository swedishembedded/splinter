<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Datasets reference

A dataset is training data projected from experience sets by one view, stored with a manifest
that names where every record came from. `dataset build` makes one, `dataset export` copies
it with its manifest into a directory, and `train` reads it. The guide to using them is
[Build datasets and train a candidate](../guides/training.md).

## Views


`dataset build` views: `sft-final`, `sft-step`, `critic`, `preference`,
`verifier`, `decision`, `retrieval`, `outcome`, `denoise`, `cpt`, `voice`
(the writer's own text as answers, from every text part of the sources the
experience sets cite; see [Learn how a person thinks](../guides/personas.md)), `rehearsal` (a model's own answers as it
gave them, every experience with a final answer that no verifier decided
wrong, under the default prompt). `--limit N` keeps at most N records of any
view, evenly spread over the projection, the rest counted as `over_limit`. The
default `--min-strength` is `consistency`; `--strip` defaults to `all`
(the student sees only the instruction - also for a teacher's solve,
whatever it was shown). Every conversation starts with the system turn
every solve, `ask` and probe runs under: one short system prompt of
Splinter's own, in place of sven's coding-agent prompt, so the policy is
trained under the prompt it answers under. Chat views are written as
brain's `generic-messages-v2`, `preference` as brain's
`generic-preference-v1` (one pair per line: the student's prompt, the
chosen and the rejected answer), each checked by brain's own parser.
Objectives brain cannot train (contrastive, reward, raw text) need
`--export-only`.

## Manifest

Every dataset is stored under the state root by content address, beside a manifest that
names where each record came from: the experience and task it was projected from, and the
source content under them. The manifest also keeps the seeds the build drew (for example
`--mix-seed`) and the build's controls. A record of the writer's own text (the `voice` view)
names the part it prints and no experience, so a dataset shows what is the writer's and what a
model wrote. The manifest is what lets `splinter lineage` walk from a release back to the
bytes it learned from.
