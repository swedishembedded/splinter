<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 097. Supervising every action taught the exploration, and the fine-tune fell off the hard tail

Two adapters, the same fifteen development sets, the same recipe (rank 16, alpha 32, lr 5e-5,
`--split-over-block --max-block 16384`, 120 steps), the same evaluation build, on the
46-family benchmark D46 at greedy decoding, one task per family:

| supervision | corpus | pass@1 | exposed |
|---|---|---|---|
| every admitted action | `blake3:646692b2` (769 records) | **12/46** | 3 |
| edits and the conclusion only | `blake3:92252209` (the same 769 records) | **23/46** | 4 |
| none (the base) | - | 21/46 | 9 |

The second corpus is the first one put through `--only-actions edit_file,write_file`: the
same records with the same context, but 5,822 actions - the exploration and the verification -
are no longer supervised. That single change is the difference between a model that loses
nine tasks the base solved and a model that solves two more than the base.

The failure it removes is behavioural and concentrated: on the thirteen tasks the
full-supervision adapter lost, the base edits and says "Fixed. The bug was in X" (10-25
steps) while the adapter repeats a probe until the repetition stop aborts it, or diagnoses
the bug in prose and never applies the patch. The aggregate is not a general loss of
editing - both adapters edit 24 of 46 attempts - so what the full supervision cost is
convergence on the hard tail, where most of a trajectory is exploration.

The behaviour moves with the outcome, measured over all 46 attempts of each arm: the base
edits a file in 24 attempts (52 %) and ends 13 (28 %) in a repetition abort; the
full-supervision adapter on the pinned binary edits in 24 (52 %) and aborts in 9 (20 %); the
same all-action corpus on the rebuilt binary edits in 22 (48 %) and aborts in 10 (22 %); the
decisive-action adapter on the rebuilt binary edits in **35 (76 %)** and aborts in **1
(2 %)**. So the rebuild that is worth six tasks changes the outcomes and not the behaviour,
and the decisive-action supervision is what moves the behaviour - two independent lines
pointing at supervision for the five tasks it adds.

# The control that was run afterwards changes the attribution, and this entry is corrected
# rather than left flattering. The same all-action corpus trained with the same rebuilt
# binary (`allnew`) passes 18/46 - not 12 - so the rebuild accounts for six of the fifteen
# tasks between the two original arms, and the decisive-action corpus adds five on top of
# that (23 against 18, both with 4 exposed). On S1 the picture is weaker still: the decisive
# corpus passed 16 at one seed and 9 at a second, and the all-action corpus with the rebuilt
# binary passed 9 - so run-to-run spread there is as large as either effect.

Caveats, stated so the number is not over-read: the decisive run trained with a splinter
built from current main rather than the pinned revision, because a dataset carrying the new
exclusion reason cannot be decoded by the older binary - and that rebuild is worth six
tasks of the result above; 23 against the base's 21 is not significant on its own (34 paired
families, 9W-8L-17T, p = 0.5); and with one seed per arm, and a seven-task spread between two
seeds of one recipe on S1, the claim is that neither corpus regresses with the rebuilt binary,
not that decisive supervision is worth five tasks.

Consequence: a coding-agent corpus should weight the decisive actions. `--only-actions` makes
that available; the next run should test it at a second seed and against the frozen training
binary rebuilt with the same sources.
