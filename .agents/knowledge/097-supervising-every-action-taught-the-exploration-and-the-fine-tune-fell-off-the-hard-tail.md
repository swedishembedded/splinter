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

Caveats, stated so the number is not over-read: the decisive run trained with a splinter
built from current main rather than the pinned revision, because a dataset carrying the new
exclusion reason cannot be decoded by the older binary; 23 against the base's 21 is not
significant on its own (34 paired families, 9W-8L-17T, p = 0.5); and with one seed, the
claim is that the regression is gone, not that the gain is real.

Consequence: a coding-agent corpus should weight the decisive actions. `--only-actions` makes
that available; the next run should test it at a second seed and against the frozen training
binary rebuilt with the same sources.
