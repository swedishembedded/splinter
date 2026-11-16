<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# minimind and arXiv 2611.10719 ("No Second Run: Does RSI Stop at Frontier Pre-training?")

What the paper's practices became in `samples/minimind`, a 2-5M-parameter model trained from random
weights to a workspace agent. Measured outcomes are in `.agents/roadmap/minimind.md`.

| Paper practice | In minimind | Where |
|---|---|---|
| Verified agentic trajectories as first-class pre-training data (its final stage holds 15% of them) | a mid-training stage after general pretraining: raw text with the verified demonstrations rendered in the template as `midtrain.agentic_share` (0.15) of the bytes; the fine-tune proper stays role-masked (`full_train`, tool results unsupervised) | `src/sequence.rs` (`midtrain_data`), stage `<model>/midtrain` |
| Checkpoints and recipes chosen by post-fine-tune headroom (Tiny-SFT), not base score | after every pretraining segment, a fixed small fine-tune (`headroom_steps`) and the probes; the table shows base retrieval and post-fine-tune passes per segment | `<model>/headroom-<i>`, `probe/<model>/headroom-<i>` |
| Learning aptitude `s(N) = b + k log10(N/N0)` over fine-tune steps, slope `k` and `N*` to a target, to compare models | the fine-tune in `sft_segments` segments with the probe after each; the fit and `N*` to a 0.75 development pass rate are printed per model label, so an `ouro` and a `qwen3` label under one sequence compare by `k` | `src/panel.rs` (`aptitude`) |
| A data-debugging loop over generated data | detectors at the data stages: one request template's share, one final reply's share, repeated records and documents, each bounded (`data_checks`); over a bound fails the stage naming what dominated | `src/datacheck.rs` |
| Anti-hacking flags before trajectories are admitted | a verified episode is admitted only if it did not rewrite the task's own files, look for verifier material, or write the answer before reading its sources; flagged ones are counted, never trained on | `src/admission.rs` |

The capability ceiling as a maximum over evaluation protocols, attention-cost-balanced packing and the
data-parallel resume bug were not taken up: one protocol (sven's compact profile) is examined, packing is
brain's, and runs use one card.
