<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 121. A library of verified sub-solutions solves compound tasks with fewer tokens

Practice on single sub-tasks, keep only what the verifier passed (`pipelines::skill_library`), then solve
compound tasks that are new combinations of those sub-tasks by taking a held sub-solution from storage and
asking the model for the rest. On the toy checkpoint with 64 draws per practised sub-task, 47 of 90 had a
verified solution; on 100 two-part compound tasks the share solved end to end went from 0.09 (model alone)
to 0.29, and the tokens the model generated per task from 7.3 to 3.3. With 16 draws the library held 37
of 90 and the gain was 0.09 to 0.15: the gain follows the library's coverage of the sub-tasks.

Caveat: the gain is reuse of checked answers, not a better model; held-out compounds whose parts were never
practised get nothing.
