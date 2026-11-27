<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 245. A champion claims completion on most of the attempts it fails

**Lesson:** read from the trajectory's structure (it ends with the agent's own answer and no call, and no limit ended
it), the Qwen3-8B generation-1 champion claimed completion on 296 of its 377 failed exam attempts (eval-exam-g2's
baseline arm: 23 passed, undetected-failure rate 0.928 of its claims, claim and grade disagree on 0.740 of the
attempts). On the generation-2 new tasks (champion-attempts-g2) it claimed 4119 of 4576 attempts and 2866 of the
claims failed (0.696); the teacher (Qwen3.8-27B) failed 280 of its 1712 claims on the same tasks (0.164). Its passes
are almost all claimed (1253 of 1261), so a claim carries little information about the outcome.

Of the champion's 3315 failures on those tasks, the rules attribute 262 to the budget, 1030 to a call repeated with
the same arguments and the same result (failure to recover), and 2023 stay unattributed: no rule decides them from
what the run records. The repeat rule discriminates but is not specific: the same repeat occurs in 13% of the
champion's passes and 38% of its failures (166 of 1261, 1246 of 3315).

**How to apply:** a loop that trains on the champion's own passes teaches it to end its turn; nothing in the records
teaches it to read the final state back and catch the 70-93% of claims that are wrong. Count undetected failures per
generation (`splinter observe`, conduct) beside the pass rate, and decide the unattributed failures with records the
run does not keep yet (which calls write and which read, the final state against the agent's account).
