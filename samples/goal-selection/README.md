<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->

# goal-selection

A participant with a goal is given a pile of raw material, most of it no use
to the goal, and has to find what to learn. This sample generates such a
pile with every section labelled, runs the goal-directed loop on it
(`splinter_sdk::goal`), and compares what the loop learned with reading
nothing, everything, the same number of words at random, and only what the
labels call relevant.

```text
goal-selection --form property --corpora 10 [--ablate] [--seed N] [--out result.json]
goal-selection --form behaviour --goal-topics 1 --other-topics 7 --hard --signal fused --corpora 10
goal-selection --hard --signal fused --embedder served --reader model --corpora 5 \
    --brain-endpoint http://127.0.0.1:8801/v1 --brain-key-file KEY --export DIR --out result.json
```

Corpus `i` is drawn from seed `base + i` (the base is drawn and printed when not
given); every number is a mean over the corpora with a bootstrap interval of 0.95 coverage,
and the last column is the held-out difference to the selective arm, corpus by
corpus. `--export DIR` writes each arm's material and the held-out questions for
training a model on each arm; `--diagnose` prints how relevant words alone and
meaning alone find each kind of section.

## The corpus

A world has topics (equipment lines, or client types) made of units (one
item's attributes, or a few situations a client type meets), each fact of a
unit having a true value. The goal names some topics; dev probes ask about
a sample of their facts and the held-out questions about the rest. Around the
goal's topics the corpus holds, with labels kept apart from the documents:

| Label | What it is |
|---|---|
| relevant | current, authoritative documents on a goal topic |
| known | the same, for units the participant knows already (scripted reader only) |
| outdated | an older version of the same publisher, other values |
| misleading | a newer, low-authority document with wrong values |
| irrelevant | other topics built from the same kinds and attributes, and prose that states nothing |

`--form property` is facts about equipment ("The service interval of alder
pump 3 is 400 hours."); `--form behaviour` is how a type of client behaves
("When the rent is late, a hesitant client asks for a payment plan."), the
material a counterpart role is built from. With `--goal-topics 1` the
behaviour form also reports the fidelity of the persona each arm's material
makes: the share of held-out situations of that client type a reader acting
from the persona alone gets right.

## Hard corpora

`--hard` adds what lexical selection is weak against: some current documents
paraphrase the probes' words (`maintenance period` for `service interval`,
`reluctant` for `hesitant`), header-less near-copies with a wrong value in every
section, a current manual that covers only part of a topic while the older one
is the only, and right, source for the rest, and prose that names the goal's
topics and states nothing. Facts and the rulings between disagreeing statements
come from admitted knowledge objects (`splinter_knowledge::objects`): a key stays in
the words of its quote, so statements that paraphrase each other are not seen to
disagree, and a section no object can be made of is left out. `--signal` chooses
what decides relevance and how the reader's memory is searched: the gaps' words
(`lexical`), closeness in meaning (`semantic`), or either (`fused`); `--embedder
lexicon` is a stand-in that knows only the corpus's synonym table, `--embedder
served` an embedding model served by brain.

## What the loop does

1. Probe the participant with the goal's probes; cluster the failures into
   gaps (missing capabilities).
2. For every section decide, with the reason kept: not relevant to an open
   gap, contradicted by a more authoritative section, superseded by a newer
   one, already known (the participant answers the questions its facts
   answer), a duplicate, over the word budget, or chosen.
3. Learn the chosen sections, probe again, select again, until the goal is
   met, nothing is worth learning, or the budget or rounds are spent.

`--ablate` adds the loop with the conflict check, the novelty check and the
goal's objective each switched off.

## What it measures, and what it does not

Per arm: sections and words learned, questions put to the participant to
decide, the dev and held-out pass rates, the precision and recall of the
chosen sections against the labels, and the share of them that are outdated
or misleading. The held-out questions are never used to select.

The scripted reader answers from the best-ranked passage it is shown and
does not weigh dates or authority, so it proves the selection logic, not
what a language model makes of the material. `--reader model` asks a model
served by brain one query at a time; a model knows none of an invented
corpus, so no section is labelled known there. A failed model call voids the
result. Multi-hop questions whose answer lies in two documents are not generated.
