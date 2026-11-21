<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 199. The 27B teacher solved 163 of 168 vim practice tasks and called an absent tool in three of four passes

**Lesson:** on the vim benchmark's train side, Qwen3.8-27B-FP8 is a strong enough teacher: pass@1 0.713 and pass@4
0.899 over 672 uniform trials, and 163 of the 168 tasks have a verified trajectory after up to 28 attempts. Before
building an SFT dataset, check which tools the trajectories call. In 433 of 576 passing trajectories the teacher first
called `shell` or `read_file`, which the session did not offer: sven's system prompt named them (sven 0d94638e fixes
the prompt). Build with `--only-actions vim-vim` so that those turns stay in the context and are not supervised (441
actions).

## Run (lane VSFT, 2026-11-21, one GH200)

- Teacher `Qwen/Qwen3.8-27B-FP8` on brain-5e25d5b82 at `--max-batch 128`, which keeps a 340k-token KV pool. At 160
  the pool drops to 143k, at 192 to 4k, and at 256 the model is not placed (brain 539).
- Sven 1db749d9, default profile, sampling 0.6 / 0.95 / 20, agent limit 600 s; plain `harbor run` of
  `vim-headless-mastery` train/ (revision c875538865ad).
- Trials reached splinter through `bench adopt` (bench 0b969a03), `bench export` and `experiences ingest-bench`.

| attempts | trials graded | passes |
|---|---:|---:|
| 4 per task (uniform) | 672 | 479 (pass@1 0.713, pass@4 0.899) |
| +8 on 28 tasks with at most one pass | 224 | 30 |
| +16 on the 10 still unsolved | 160 | 10 |
| a 224-wide job cut short by the gateway pool (bench 070) | 136 counted | 60 |

- Never solved: 098, 100, 102 (macros), 129 (quickfix across files) and 139 (an escape keystroke in a macro).
- The dataset `sft-step --pack --only-actions vim-vim --max-block 16384` has 576 records and 1.81M tokens for
  Qwen3-4B, 1.05M of them supervised. Excluded: 613 failed trials, 3 over the block, 4 repeated actions and 7 reads of
  repository history. It is published PUBLIC as `swedishembedded/vim-headless-mastery-sft` tag v1, split by whole task
  into train (516 records, 147 tasks) and eval (60 records, 16 tasks), the latter being splinter's held-out set.
- 106 of the 576 records are passes the 600 s agent limit stopped before a final answer (status `timeout`, graded on
  the files). They are verified, but they end on an action rather than an answer.
- Held-out loss of Qwen3-4B on the 60 eval records (seed 1, A2 recipe, 230 steps): 2.153 for the base. A first build that
  supervised the absent-tool calls reached 0.792, and the `--only-actions` build's result is in the lane's
  experience-to-competence ledger row.
