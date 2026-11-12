<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 137. Selecting the material before training raised generated exact match on held-out questions and did not damage reading

The 27B comparison of finding 136 re-scored on generated answers through bench (environment `recall`: a closed-book question per task, `answer.json`, a deterministic whole-word check; the exam is the held-out questions of each corpus, 24 per property corpus and 18 per client-behaviour corpus, plus 8 reading items per corpus from topics no arm was taught; `bench run` of the base and of each arm's adapter through sven on the one training card, one trial each, the same load). Five corpora per form (property seeds 8100-8104, client-behaviour 8200-8204); 150 optimizer steps for every arm. Pass rate, mean over corpora with a 95% bootstrap interval:

| recall (from memory) | base | selective | oracle-relevant | random, same words | everything |
|---|---|---|---|---|---|
| property | 0.000 | 0.158 [0.100,0.217] | 0.092 [0.042,0.158] | 0.008 [0.000,0.025] | 0.008 [0.000,0.025] |
| client behaviour | 0.000 | 0.400 [0.211,0.589] | - | 0.133 [0.078,0.189] | 0.133 [0.089,0.189] |

Paired by corpus, selective minus random and minus everything is +0.150 [0.100,0.200] (property) and +0.267 [0.111,0.444] / +0.267 [0.067,0.500] (behaviour); selective minus oracle +0.067 [-0.033,0.167] (property), level. Reading a passage (the no-regression check): base, selective, oracle and random 1.000 on every corpus; training on everything dropped one property corpus to 0.62 (mean 0.925 [0.775,1.000], paired -0.075 [-0.225,0.000]). The reading items are nearly saturated for the base (behaviour base 0.975), so the check can show damage but not gain.

What this establishes: at equal compute the loop's material is learned and recalled by generation, not only scored by teacher forcing, and training on all of it can hurt reading where the selected material did not. What it does not: absolute recall is low (about 0.16 and 0.40 after 150 steps of rank 16); one seed per adapter; the held-out questions concern facts the relevant arms trained on; reading is saturated. Evidence: bench `environments/recall`, `tasks/recall-*`, `~/project/shared/goal-selection/weights2/exam-analysis.txt`.
