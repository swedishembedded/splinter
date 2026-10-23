<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 095. A packed view removed every trajectory longer than its block, and the survivors taught the adapter to stop early

The packed `sft-step` build of the CE8-T experiences reported 132 records and `over_block: 130`: every
successful trajectory longer than the 16384-token block was dropped before training, which is half of the
262 successful attempts of the eight development sets. The identical build for the decision-level view
kept 2689 and dropped 1148 (43%). The surviving records therefore carry a median of about 1650 supervised
tokens, and the long-horizon episodes, the ones whose solutions take many steps, are structurally absent.
The trainer then saw 113 records in one pass (57 steps at two records per step), roughly 210k supervised
tokens.

What that produced was measured on the pilot subset S1: the median episode is 32 steps for the base and
13 for the CE8-T adapter, 12 for its shuffled-instruction control, 16 for the decision-level candidate, with
782/474/392/632 tool calls and 11/10/9/3 successes; the longest episodes still run to 58-61 steps. The
control matches the trained adapter, so the update moved when a turn ends, not what a trajectory solves.
And the base against itself on a second protocol seed pairs 1W-3L-9T (p = 0.9375), so the pilot cannot
separate any of this from sampling noise.

Fix, for the next candidate: raise or remove the block for a packed view (or window the long trajectories
with supervision preserved) and check the `over_block` count before a run is sized; treat the dropped share
as part of the dataset's description, not a build detail. A packed view that cannot hold the teacher's
longest successes is not a sample of its behaviour.
