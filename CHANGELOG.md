<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`splinter-expdb`.** A new standalone crate for a versioned experience graph
  database: immutable files, manifests as the transaction layer, snapshots,
  modality-neutral episodes of streams, queries and training views, with its
  storage protocol model-checked in TLA+. It is not yet used by any other crate.

## [0.1.0] - 2026-10-01

First release. Splinter is a learning agent with its own model: it is told
in plain language what to learn, learns it, measures that it did, and
releases a better version of itself. It is built on sven 3.0.0 (the agent)
and brain 2.0.0 (the model), pinned in `Cargo.lock`.

### Added
- **Command line.** `splinter "<sentence>"` and a REPL, plus one verb per
  stage: `source`, `tasks`, `solve`, `verify`, `critique`, `dataset`,
  `train`, `eval`, `release`, `runs`, `experiences`, `lineage`, `ask` and
  `status`. `learn` runs every stage as one recorded run. Every command
  takes `--json`, and a model is named the same way everywhere
  (`policy:`, `release:`, `local:`; `remote:` only with `--allow-remote`).
- **Sources.** Documents, repositories and the captured output of a
  command, split into sections that tasks cite by position and byte range.
- **Tasks that name their subject.** A generator model writes tasks from
  the sections, shown what the source is called. A question must name the
  product, tool or version it is about, so it has one answer without the
  source in front of the student; two questions about one subject with
  different answers are both left out. Several task kinds ship (recall,
  explain, predict, construct, debug, and more), and `denoise` needs no
  model.
- **Learning where it counts.** Each task is attempted closed-book several
  times. Tasks the policy always solves are dropped, and a task it never
  solves is solved by a teacher shown the source passage. The teacher is
  the policy itself unless `--teacher` names another model. The training
  record keeps the question and the verified answer, never the passage.
- **Verifiers by strength.** Executable checks (code seen to run to its
  end, with generated tests validated against mutants), a recall check that
  accepts an answer stating the reference, a consistency check, and a
  calibrated judge. A judge alone never makes a training record.
- **Critique and retry.** A failed attempt becomes a critique and a
  revision, kept as relations between experiences, so preference data comes
  from real attempts.
- **Reworded questions.** The generator rewrites each kept question in
  other words, keeping its subject. These are never trained on and
  measure whether a fact was learned rather than a sentence memorised.
- **Training.** A candidate continues the current release as a LoRA
  adapter, by supervised fine-tuning with a replay of what earlier
  releases learned, or by preference training on verified against failed
  answers. Experiences project into ten views, with the passage and other
  privileged material stripped where a student must not see it.
- **Release gate.** A candidate is released only when all four checks
  pass: it improved on reworded held-out questions (sign test, after
  leakage exclusion), it kept what earlier releases learned, it held a
  frozen anchor suite of general questions, and it gives the same answers
  on plain `brain serve --adapter`.
- **Releases.** Immutable: an adapter plus a manifest recording the base
  model and its digest, the parent release, the datasets and every number
  the gate measured. `release:` and `policy:default` name them, and
  `splinter rollback default` returns to the previous one.
- **Lineage.** `splinter lineage <ID>` walks from an answer, release or
  dataset up to the source, part and byte range it came from, and down to
  everything that came from it.
- **Curriculum and integrity.** Concept mastery is tracked over releases
  and forgotten concepts are queued for new tasks. Held-out questions
  that near-duplicate a training record are excluded, and an experience's
  code calls can be replayed in their recorded environment.
- **One resident model.** The policy, a candidate and the champion are one
  base with different adapters, so one copy of the base is loaded.
- **Default policy.** Qwen3-0.6B from brain's model store;
  `BRAIN_QWEN_WEIGHTS` names another checkpoint.
- **Building.** Builds from the pinned `Cargo.lock` with `make build`;
  `make local` builds against local sven and brain checkouts instead.
