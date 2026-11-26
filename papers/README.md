<!--
SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->

# Papers

`make papers/<name>/pdf` (or `make papers/pdf` for all) builds a paper into
`target/papers/<name>.pdf`.

A paper whose Makefile has an `arxiv/check` target also writes its arXiv source archive,
`target/papers/<name>-arxiv.tar.gz`, in the same build, and checks it by compiling it in an empty
directory (`make papers/<name>/arxiv/check` reruns that check). Upload the archive to arXiv as it is;
it holds only what pdflatex reads, with comments removed.

Each paper has one question, stated as its scope in the introduction. A new
verified finding goes into the paper whose scope it falls in, in the same change
that produces it (see `.agents/rules/papers.md`). A paper is a draft until it
holds validated results only; `splinter-brain-sven-flywheel` is final (`status.txt`), and
`make -C papers/splinter-brain-sven-flywheel final` checks that it holds no registered or planned
claim. A finding that fits no
scope below starts a new paper directory with its own scope; it is not squeezed into
the nearest one.

| Directory | Scope: a finding belongs here if it changes what is claimed about, or how one would measure... |
|---|---|
| `persona-training/` | an adapter fine-tuned on one author's writings so that it decides and acts as the author would: the controlled-author benchmark (oracle, held-out topics, combinations and tools, mechanical grading), the training pipeline from writings to adapter, paired evaluation against prompted and retrieval baselines, extraction validity, and what the data looks like at each stage |
| `learning-voice-agent/` | a voice agent that learns from being corrected: how spoken or typed corrections of facts, rules, procedures and manner of speech are read into typed teachings, confirmed, applied, taken back, and tested, with the pre-registered results that passed their gates and those that did not, and the reproducible record of what has been measured |
| `agent-driven-residual-survival/` | whether a locally operated, supervised research agent finds learnable structure beyond an additive survival model on single-visit cohort data |
| `splinter-brain-sven-flywheel/` | the self-learning loop as a whole, from source to release, and its application to repository tasks: how Splinter, Brain and Sven divide the work and keep the boundary, how tasks are admitted and verified, how the corpus is expanded and trained on, how the release gate, releases and rollback behave, what Brain's training substrate contains for a large model on one card, the requirements that make the loop work, and the registered coding-task experiments (repository tasks, task validity and leakage, trajectory data, paired exams, retention across rounds, replication on a second repository), the generations of a model trained on its own experience, and the registered recursive-self-improvement series judged by an independent examiner, with the claims ledger that ties every statement to a check or a measurement |
| `generalized-continuous-learning/` | continuous improvement across successive generations of a model taught by an executable environment: whether every generation beats its predecessor on held-out exams judged by a separate examiner, how the verified supervision (reasoning traces, interpreter labels, agent trajectories on tasks authored from a source document) is collected and selected, and whether it extends to task families beyond the first (Lua output prediction, Vim file editing) |
| `baseline-to-competence/` | taking a small model (0.6B to 8B) from failing an agentic file-editing task to solving it: the verified teacher trajectories, the student's own verified solutions and a teacher's continuations of its failures as training data, their admission as new solutions only, supervised training, a policy-gradient reward that penalises looping, and the measurement on held-out tasks (one attempt, reruns of the failed, a paired test over tasks) |

Not yet covered by any paper, so a verified finding there starts one: timelines
and longitudinal learning, predictive gates, learning from usage policy, retrieval
of source passages at answer time, serving performance.
