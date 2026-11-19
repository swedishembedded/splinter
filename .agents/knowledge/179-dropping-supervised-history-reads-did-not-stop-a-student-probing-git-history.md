<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 179. Dropping supervised history reads did not stop a student probing git history

**Lesson:** `--repository-history drop-step` takes the supervision off an action that reads the repository's
history but leaves it in the context of the actions after it, and its detector misses reads the examiner's leak
audit flags: a read_file of `.git/refs/heads/main`, `ls .git/objects`, `git cat-file`, `git log --stat`. A student
retrained on such data probes history as often as one trained with the reads supervised. Measure the examiner's
security_violation rate on the arm, not the curation's drop count; when it must fall, use `truncate` (end the
demonstration before the read) and align the detector with the examiner's audit rules.

Found 2026-11-19 (lane R, gen 1): gen-1 SFT data rebuilt with drop-step (17 steps dropped; 420 gen-0 + 265 own
records), trained from base at three seeds. On the dev eval (129 items x 2) and retention (91 x 2) the examiner flagged
security_violation in 10 of 1320 gen-1 trials (0.8 %) against 2 of 440 for A2 seed 1 (0.5 %), trained on the same
gen-0 data with the reads supervised: no reduction. Most flagged gen-1 calls are `git log --all`, `git reflog`,
`git log --stat` and reads under `.git/refs`; the RL arms trained on the rollouts with history kept (the
policy-gradient view does not curate) were flagged as often (A6 s1 6 of 440, DPO 4 of 440).
