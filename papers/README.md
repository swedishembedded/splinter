<!--
SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->

# Papers

Each paper has one question, stated as its scope in the introduction. A new
verified finding goes into the paper whose scope it falls in, in the same change
that produces it (see "Research papers" in `AGENTS.md`). A finding that fits no
scope below starts a new paper directory with its own scope; it is not squeezed into
the nearest one.

| Directory | Scope: a finding belongs here if it changes what is claimed about, or how one would measure... |
|---|---|
| `persona-training/` | an adapter fine-tuned on one author's writings: its training data (views, tasks, abstention, replay), training mechanics, evaluation design (held-out source families, exam, judge, statistics, retention), release-gate checks as measurements, and defects in any of these |
| `agent-driven-residual-survival/` | whether a locally operated, supervised research agent finds learnable structure beyond an additive survival model on single-visit cohort data |

Not yet covered by any paper, so a verified finding there starts one: timelines
and longitudinal learning, predictive gates, learning from usage policy, retrieval
of source passages at answer time, serving performance.
