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
goal-selection --form property --reader scripted [--ablate] [--seed N] [--out result.json]
goal-selection --form behaviour --goal-topics 1 --other-topics 7 --reader scripted
goal-selection --reader model --model Qwen/Qwen3.8-27B-FP8 \
    --brain-endpoint http://127.0.0.1:8803/v1 --brain-key-file KEY
```

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
result. Selection here is lexical; a corpus whose relevant material shares
no words with the goal or its probes is not covered.
