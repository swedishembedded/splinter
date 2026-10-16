<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 069. A run report keeps only the wrong answers, so undecided ones cannot be regraded

Found 2026-10-16 re-grading the anchor-200 runs of knowledge 068 with the new short-answer
grading (`crates/eval/examples/regrade_anchor.rs`).

`eval --json` lists in `missed` the answers of tasks decided wrong, and nothing for tasks that
passed or stayed undecided. Re-grading the stored reports therefore reaches 10 answers (5, 4 and
1 for Qwen3-1.7B, Qwen3-4B and the 27B): all stay wrong (for example "Twelve months have 31
days" against 7), and the 31, 21 and 81 undecided answers cannot be re-graded without a model
run. The grading change itself is specified on constructed answers (verbose, emphasised,
thousands-separated, hedged) in `crates/eval/tests/short_answer_grading.rs`.

How to avoid it: a run that will be re-graded must keep every answer (an eval report option),
or be repeated with the new grading and its counts read from the fresh report.
