<!--
SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->

# Papers

`make papers/<name>/pdf` (or `make papers/pdf` for all) builds a paper into
`target/papers/<name>.pdf`.

Each paper has one question, stated as its scope in the introduction. A new
verified finding goes into the paper whose scope it falls in, in the same change
that produces it (see "Research papers" in `AGENTS.md`). A paper is a draft until it
holds validated results only; `splinter-brain-sven-flywheel` is a draft (`status.txt`), and
`make -C papers/splinter-brain-sven-flywheel final` fails until it is final. A finding that fits no
scope below starts a new paper directory with its own scope; it is not squeezed into
the nearest one.

| Directory | Scope: a finding belongs here if it changes what is claimed about, or how one would measure... |
|---|---|
| `persona-training/` | an adapter fine-tuned on one author's writings so that it decides and acts as the author would: the controlled-author benchmark (oracle, held-out topics, combinations and tools, mechanical grading), the training pipeline from writings to adapter, paired evaluation against prompted and retrieval baselines, extraction validity, and what the data looks like at each stage |
| `learning-voice-agent/` | a voice agent that learns from being corrected: how spoken or typed corrections of facts, rules, procedures and manner of speech are read into typed teachings, confirmed, applied, taken back, and tested, with the pre-registered results that passed their gates and those that did not, and the reproducible record of what has been measured |
| `agent-driven-residual-survival/` | whether a locally operated, supervised research agent finds learnable structure beyond an additive survival model on single-visit cohort data |
| `splinter-brain-sven-flywheel/` | the self-learning loop as a whole, from source to release, and its application to repository tasks: how Splinter, Brain and Sven divide the work and keep the boundary, how tasks are admitted and verified, how the corpus is expanded and trained on, how the release gate, releases and rollback behave, what Brain's training substrate contains for a large model on one card, the requirements that make the loop work, and the registered coding-task experiments (repository tasks, task validity and leakage, trajectory data, paired exams, retention across rounds, replication on a second repository) with the claims ledger that ties every statement to a check or a measurement |

Not yet covered by any paper, so a verified finding there starts one: timelines
and longitudinal learning, predictive gates, learning from usage policy, retrieval
of source passages at answer time, serving performance.
