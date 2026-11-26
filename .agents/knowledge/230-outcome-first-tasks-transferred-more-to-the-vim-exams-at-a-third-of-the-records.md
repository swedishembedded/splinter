<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 230. Outcome-first tasks transferred more to the Vim exams than section-by-section tasks, at a third of the records

**Lesson:** in the source-to-transfer experiment (`.agents/roadmap/source-to-transfer.md`, 2026-11-27), Qwen3-4B adapters
trained on verified trajectories of tasks authored *outcome first* (arm G: the 27B author proposes start and end files of
a multi-change job, the teacher must reach the end state with the vim tool, the request is written last) passed 42 of
1200 attempts on the 100 held-out course exams and solved 25 exams at least once. The same selection rule on tasks
authored section by section from the manual (arm B) passed 22 and solved 10. G against B: 21 exams won, 5 lost,
two-sided sign test p 0.0025, pass-rate difference +0.017 [+0.003, +0.031] (task bootstrap). G trained on 1100 records
of 348 tasks, B on 3359 records of 857 tasks.

What this does not isolate (secondary comparison, not a registered claim):
- **Budget per record.** Both arms had the same 800-step budget with early stopping on validation loss. G kept step 520
  (about 1.9 passes over its records), B ran all 800 (about 0.95 passes).
- **Difficulty.** The outcome-first tasks are harder: student pass rate 0.031 against 0.100, teacher 0.722 against
  0.811. On the exams the teacher's pass@1 is 0.63, so the outcome-first tasks are closer to them.
- **Teacher share.** G's records are 92% teacher (1014 of 1100), B's 80% (2699 of 3359).

**How to apply:** when authoring practice tasks from a source document for transfer to composed, multi-change exams,
prefer outcome-first authoring (`recipes/vim-manual/backward.yaml` in bench) over per-section step tasks; more records
of the per-section kind did not make up the difference. To learn whether the format or the difficulty carries the
effect, compare at matched difficulty and matched passes per record. Evidence: private HF
`swedishembedded/source-to-transfer-evidence` (`decision/secondary-eval-exam.json`, `evaluation/aggregates.json`).
