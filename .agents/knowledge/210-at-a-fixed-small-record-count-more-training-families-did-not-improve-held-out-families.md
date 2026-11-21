<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 210. At a fixed small record count, more training families did not improve held-out families

**Lesson:** with the number of verified teacher records fixed at 122, students trained on 4, 8 or 12 vim difficulty
rungs scored the same on two held-out rungs: 0.287, 0.276 and 0.281. Seven folds cover all 168 tasks and 14 rungs,
x 4 attempts x 2 seeds. 12 minus 4 is -0.005, family interval [-0.041, +0.028], so a +0.05 breadth effect is ruled out
at this size (generic-learning E3, 2026-11-21). All three doses sat about +0.05 above the base, against +0.118 for
students trained on 343-403 records of 9-10 rungs (knowledge 209). In this regime the amount of verified data moved
transfer, the spread of families did not. Selecting the cheapest passes within the budget (E2) did not shorten the
students much either: exhaustion fell only from 58% to 50%, and the gain over v1, +0.037 [-0.021, +0.105], was not
supported.

**How to apply:** do not fund breadth for its own sake at a few hundred records. Compare data volume against breadth at
the record counts you can actually afford, and treat the students' reasoning length (half of all trials exhaust the
budget) as the open failure mode. Curation alone does not fix it.
