<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 212. The vim tool's observations carry error dynamics, not the changed buffer

**Lesson:** a transition (world-model) dataset learns only what the tool's answers contain. Measure that before you
train on observations. The vim headless-mastery tool answers an edit with the exit status, Vim's messages and the
failing script line, and shows buffer text only when the model itself prints it.
- On E1's 1,344 base trials (4,114 observations), 13.6% show buffer lines (an `output:` section): 13.4% after an
  edit-like call.
- 58% are Vim errors (E-codes with the failing line), and 28% only file messages such as `"main.txt" 1L, 7B written`.
- Every error observation names a random scratch path (`/tmp/vim-tool-XXXX`). These tokens make up 8.1-8.5% of the
  supervised observation tokens.
- Asking for it does not change this. With a task-free play instruction that says to print the buffer after edits, the
  base 4B showed buffer lines in 11.6% of 544 observations (168 trials).

So on vim, E6's `transition` records teach which commands fail and why, not what an edit does to the text. Read a null
result there as weak evidence against early-experience training in general. A tool version that returns the changed
buffer in every observation is the better test bed.

## Evidence (2026-11-22, lane E6)
- Counts from bench exports of E1's base exam jobs (`fa/fb/fc-base-*`). Lane VIM's `train-k1/k4` give 11.9% over
  2,730 observations. The play job is `play-base-r1` (brain-87a32d2fe, sven-fe9cda90, bench 0d8f13de).
- Script `lane-e6/scripts/obs_stats.py` counts token lengths and the presence of an `output:` section. It is a
  diagnostic, never a filter.
- Supervised observation tokens per fold's transition dataset: 222k / 215k / 254k, against 672k / 580k / 794k policy
  tokens in v1's data. That is 0.33-0.37:1, every de-duplicated transition of the base's exam-mode trials.
