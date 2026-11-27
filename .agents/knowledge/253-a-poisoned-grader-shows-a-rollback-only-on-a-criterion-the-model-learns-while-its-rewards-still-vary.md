<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 253. A poisoned grader shows a rollback only on a criterion the model learns, poisoned while its rewards still vary

**Lesson:** a fault-injection run proves a watchdog only if the poisoned training actually makes the policy worse on the
honest held-out tasks. On the CPU toy stack (a randomly drawn qwen3 toy, lr 0.02, groups of 4), three fixtures did not:

- "the reply starts with a letter": the honest pass rate stayed near 0.2 in every phase (warm-up, poisoned window,
  after) at lr 0.002 and at lr 0.02. The toy does not learn which characters it writes, so the flipped rewards
  changed nothing measurable on the monitor pool (comparisons 6:12, 9:10, 11:10 discordant, all inconclusive).
- "the reply is not blank": the toy almost never stops at once (pass rate 0.97 to 1.0), so 115 of 120 groups had
  zero reward variance and trained nothing, and the flipped window had zero variance too (every reward 0).
- "a brief reply (at most 400 characters)", poisoned after 40 groups: the honest rate rose from 0.57 to 0.75 in the
  warm-up, but by the window the replies were all brief. Every flipped reward was 0, the groups were filtered as
  zero variance, and the policy did not move.

A brief reply, poisoned after 12 groups while about half the replies still pass, does regress the policy. Run: alarm
(entropy drift), then comparisons 15:34, 15:34, 10:34 discordant pairs against the anchor, the last regressed
(e 948, threshold 822); rollback to the anchor, and the run finished in epoch 1.

The test's fixed alternative also sets what is detectable. With `delta` 0.3 (the anchor wins 80% of discordant
pairs) and `alpha` 0.1, a clean run rolled back on 3:12 at the first comparison. A real 9:23 regression at the fifth
comparison stayed inconclusive (e 13 against 411), because the index-based spending raises the bar with the square of
the comparison's place.

**Fix:** the fixture (scripts/online/cpu-watchdog.sh) grades a brief reply (whose length the stop logit governs), so
about half the toy's replies pass at the start. It poisons gradings 49..328 of the training tasks, and evaluates 96
monitor pairs with `delta` 0.25, `alpha` 0.05 and `evaluate_every` 10. It logs each training grading's honest result,
so recovery is measured by the honest pass rate, not by the poisoned rewards. A fault fixture for another model needs
the same three checks: the criterion moves under training, rewards still vary when the poison starts, and the
monitor budget can reach the threshold of the comparison that will see the regression.
