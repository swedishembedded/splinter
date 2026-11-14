<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 150. A generation whose only signal is a teacher's verified answers was rolled back untrained

**Lesson:** The campaign controller decided whether an arm makes a candidate from `policy_gradient + distilled` records only.
With `all_fail = teacher`, a pool where every group failed produced taught records (the teacher's verified answers to the
failed tasks), yet the arm was rolled back with "no group carried a reward spread or a teacher signal" and nothing was
trained: the rescue path the strategy exists for never ran. Repair records were outside the sum too (they are only built from
informative groups, which already carry advantages, so that gap could not show). `RecordCounts::fresh_signal` is now the one
definition of an arm's own training signal (advantages, repairs, distillation targets, taught answers); replay alone still makes
no candidate.

## What this does and does not establish

Verified with the scripted campaign world: two tasks the policy never solves, a teacher that answers each with one record; the
arm now trains once on the taught records (`groups_that_all_failed_are_trained_on_the_teachers_verified_answers`, red before
the fix with the rollback reason above). No earlier registered run used `all_fail = teacher`, so no recorded outcome changes.
Whether teacher rescue helps a real model is not measured here.
