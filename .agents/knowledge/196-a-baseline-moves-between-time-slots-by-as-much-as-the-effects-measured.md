<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 196. A baseline moves between time slots by as much as the effects measured

**Lesson:** the same cell (same weights, tasks, builds, caps and run seeds) can shift by several points between two time
slots on the same node, as load and timeouts change with what else runs. Compare cells only when they run interleaved in
the same slot, with each card running every cell in rotation. Never compare against a baseline from an earlier slot, and
count model-caused timeouts as failures in both arms.

Seen 2026-11-20: A2s1's no-knowledge cell (48 tasks x 8) gave pass@1 0.529 in item 21 S1 and 0.487 in S1b, a difference of
-0.041 [-0.090, +0.008]. That is about the size of every contrast those stages measured. S1b's cell (h) ran after its
baseline in a quieter slot (wall time per trial 238 s against 298 s), so its +0.044 could not be separated from load. S1c
interleaves every cell for this reason. Related: bench knowledge 065 (a pass-rate row holds only for its serving build).
