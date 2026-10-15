<!--
SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->

# Persona program: learn to think, act and write like a person

Goal: `splinter "Learn to think and act like Thomas Jefferson from ./jefferson"`
(optionally with a base model and an adapter name) produces an adapter that
decides, acts through tools and writes as that person would on situations and
tools it has never seen, and a paper whose every number comes from the final
method run end to end. The same command, with no per-person code, must work
for any profession with a body of materials.

Update this file, and the file that holds the item, in the commit that
finishes or changes an item. The roadmap is split:

| File | Holds |
|---|---|
| `persona-program.md` | goal, decisions, what is built, the index of work items, their dependencies and order |
| `persona-seeds-and-health.md` | the seed rule; PP-01 to PP-09 (build health, brain test skips, seeds) |
| `persona-study.md` | PP-10 to PP-27 (training records, the examination, replication and the study's gaps, real authors, the paper) |
| `persona-platform.md` | PP-28 to PP-39 (named adapters, orchestration, sven seams, persona chat, trace and speed, sample code moved into the SDK) |
| `persona-findings.md` | the findings ledger: confirmed defects (PF-nn), unconfirmed suspicions (PU-nn), checks that came back clean |

Every item states its problem, the evidence (path:line in the code as it
stands), acceptance as tests or gates, the design, what it depends on, its
risk and its size (S, M, L).

## Decisions taken

- Skills proven: write, judge (what the person would decide, with reasons),
  act (tool calls on unseen tools, including refusals). Headline claims are
  on judge and act.
- A win is a task the prompted base cannot solve and the tuned model can.
  Arms: base, base with the persona prompt, tuned, tuned with the persona
  prompt, and base with a long prompt or retrieval over the same corpus. The
  persona prompt carries style and role only, checked mechanically for leaks
  against gold answers.
- Mechanical verifiers first: a captured tool-call argument compared as a
  string; a judge model only where no mechanical check exists.
- Exams are frozen and hashed before the final training run, and each is
  reported once.
- Every draw has a seed that defaults to a fresh value, is printed and is
  recorded, and reproduces the draw when passed (`persona-seeds-and-health.md`).
- Training: augmented rewriting of every source (paraphrase, question,
  reverse question, consequence), principle extraction, positive and opposing
  claims, self-play against corpus-checked answers. Corpus size is set by a
  saturation curve of held-out gain against expanded size, grounded in the
  literature, not by a time budget.
- Opposing persona: user-supplied opposing materials when given, otherwise
  generated diametrically opposed claims. Whether that works is itself a
  measured result (each model decides differently on the same dilemma, in the
  direction of its author).
- 8B policy, LoRA only. Full fine-tuning and larger orchestrators are out of
  the paper until validated on suitable hardware.
- Selection grammar `base+adapter` everywhere (brain, sven, splinter); an
  existing adapter name continues training.
- Orchestration: a sven orchestrator agent drives the campaign through tools;
  sibling agents (separate engines, separate providers) do preprocessing,
  step checks and judging. Sven wraps the model, brain runs it, splinter
  owns the training machinery.

## What is built

- `base+name` selects an adapter in brain's model references and
  `local:<base>+<name>` in Splinter's; brain's LoRA fine-tune command
  continues an adapter that exists (`--fresh` starts over).
- `learn --adapter NAME [--base CHECKPOINT]` (and the same two fields of the
  sentence's intent) trains on the checkpoint, continues the adapter when the
  name exists, and publishes the candidate into the model store instead of
  gating it.
- `samples/persona-chat`: a chat and a batch over tools the caller defines
  (`decide(choice, motivation)` by default), each call checked and answered,
  a trace line per exchange.
- `samples/synthetic-author`: an author drawn from a seed (four of six formal
  beliefs in a hidden order, a name), situations shared by every author, the
  code oracle, the opposite author's oracle, the corpus written under gates by
  a hosted writer, the miner, the dataset, training, the examination with its
  cells, the arms (base, named, principles, retrieval, tuned), the paired
  report, the cross-author divergence count and the extraction audit, run for
  three authors (seeds 101, 202, 303).
- `papers/persona-training`: the controlled study on those three authors,
  REVIEW-READY by its review record.
- Shared code already moved out of samples: `core::rng`, `agent::blocking`,
  `store::jsonl`, `agent::tool_turn`, `eval` `contains_words`.

What a pass of the synthetic study proves is that the pipeline can recover
beliefs that were only projected through decisions and apply them to unseen
situations. It does not prove that a real author's judgment was learned; the
Jefferson and Adams runs, scored on held-out documented decisions, are kept
apart and untouched by any synthetic data.

## Work items

| Id | Item | Size | Depends on |
|---|---|---|---|
| PP-01 | Re-pin brain in `Cargo.lock` to a commit on brain's remote main | S | user push of brain |
| PP-02 | Port the adapter-manifest spec to brain's current catalog | S | PP-01 |
| PP-03 | No silent skips in brain; a tighter skip gate here | S / L | - |
| PP-04 | Sign test that holds for large suites (brain) | S | - |
| PP-05 | The seed mechanism (draw, print, record) (done) | M | - |
| PP-06 | Training seeds always explicit and recorded (done) | S | PP-05 |
| PP-07 | Synthetic-author seeds and run manifests (done) | M | PP-05 |
| PP-08 | Pipeline seeds of `learn`, `exam` and `practice` (done) | M | PP-05 |
| PP-09 | Seeds of the other persona samples (done) | S | PP-05 |
| PP-10 | Training records list the options the case offered (done; new corpora under PP-16) | M | - |
| PP-11 | Every exam outcome bound to its item and its model (done) | S | PP-07 |
| PP-12 | Failures accounted for, not mistaken for content | M | - |
| PP-13 | Examination v2: balanced positions, trivial baselines, trained-on axis | M | PP-07, PP-11 |
| PP-14 | Records that teach stopping after the tool answers | M | brain loss mask for multi-step tool records |
| PP-15 | Motivations worth learning, and a measure of them | M | PP-10 |
| PP-16 | Replication: training seeds and author worlds | L | PP-05, PP-06, PP-07, PP-10 to PP-15, PP-23 |
| PP-17 | The opposite author, trained on its own corpus | M | PP-07, PP-16 |
| PP-18 | The long-context arm | M | PP-11, hardware |
| PP-19 | Mechanism ablations | M | PP-16 |
| PP-20 | Sampled decoding beside greedy | S | PP-05, PP-11 |
| PP-21 | Survivorship analysis | S | PP-12 |
| PP-22 | Enough items to settle tuned against retrieval | S | PP-13, PP-16 |
| PP-23 | A reader other than the writer | M | - |
| PP-24 | Decisions and tool calls in `learn` | L | PP-10, PP-14, PP-15, PP-36, PP-37, PP-38 |
| PP-25 | Documented-decision exams for Jefferson and Adams | L | PP-24, PP-34 |
| PP-26 | Real-author runs through the one sentence | L | PP-08, PP-24, PP-25, PP-28 |
| PP-27 | Paper: REVIEW-READY to SUBMISSION-READY | M | disclosure: none; submission: PP-16, PP-19 |
| PP-28 | Named adapters that cannot be lost | M | - |
| PP-29 | The orchestrator agent and its sibling agents | L | sven SDK |
| PP-30 | The retrieval store seam in sven | L | sven memory |
| PP-31 | Persona chat, complete | S | PP-05 |
| PP-32 | Trace and speed | L | brain batching in the local provider |
| PP-33 | The last private generators | S | PP-09 |
| PP-34 | One shared-run check | S | - |
| PP-35 | The held-out share and family naming in `data` | S | - |
| PP-36 | Principle verification in `knowledge` | M | - |
| PP-37 | Scenario planning and the giveaway gate in `knowledge` | M | - |
| PP-38 | Judge calibration in `eval`, document bundling in `knowledge` | M | - |
| PP-39 | Recall at k, the embedding adapter, title case | S | PP-30 for the embedder |

## Dependencies

```text
PP-01 --> PP-02
PP-05 --> PP-06, PP-07, PP-08, PP-09, PP-20, PP-31
PP-07 --> PP-11 --> PP-13 --> PP-22
PP-07 --> PP-17
PP-11 --> PP-18 (and hardware), PP-20
PP-12 --> PP-21
PP-10 --> PP-15
brain loss mask --> PP-14
PP-06, PP-07, PP-10..PP-15, PP-23 --> PP-16 --> PP-17, PP-19, PP-22, PP-27 (submission)
PP-36, PP-37, PP-38, PP-10, PP-14, PP-15 --> PP-24 --> PP-25 --> PP-26
PP-34 --> PP-25;  PP-08, PP-28 --> PP-26
sven SDK (sibling agents) --> PP-29
sven memory seams --> PP-30 --> PP-39 (embedder)
PP-09 --> PP-33
independent: PP-03, PP-04, PP-32, PP-35
```

## Recommended order

1. Make the tree build from a clean checkout and stop silent damage: PP-01,
   PP-02, PP-12, PP-11, PP-28, and the disclosure part of PP-27 (the paper
   states PF-01, PF-04, PF-05 and PF-15 now).
2. Seeds: PP-05, then PP-06, PP-07, PP-08, PP-09, PP-31.
3. Correct what the next corpora and exams will be made of, before any new
   training run: PP-10, PP-15, PP-13, PP-23 (the reader choice), PP-14 once
   brain supervises a turn after a tool result.
4. The study's runs under one frozen protocol: PP-16, with PP-19, PP-20 and
   PP-22 in the protocol; PP-17; PP-21 on the new corpora; PP-18 when the
   hardware holds the corpus.
5. Real authors: PP-34, PP-36, PP-37, PP-38, then PP-24, PP-25, PP-26.
6. The paper to submission: PP-27.

In parallel, owned elsewhere or independent: PP-03 and PP-04 (brain), PP-29
and PP-30 (sven first), PP-32, PP-33, PP-35, PP-39.

## The three findings that matter most

- **PF-01** (high): training records of offer cases list an option the case
  did not offer and label a decision that contradicts the oracle on the
  prompt shown (78, 105 and 96 records of the three authors), the same
  option pair as the adapters' dominant errors for two authors. Fixed by
  PP-10.
- **PF-02** (high): exam outcomes are joined on an item id that names
  different items under different flags, with nothing tying an answer to the
  adapter that gave it; a rerun silently keeps stale answers. Fixed by PP-11.
- **PF-03** (high): every synthetic-author subcommand re-derives the author
  from `--seed` and never checks the corpus's recorded author, so a wrong
  seed grades against the wrong oracle without a word. Fixed by PP-07.

## Persona task menu (replaces the document-oriented kinds for persona runs)

Each kind has a mechanical verifier where one exists. None of these kinds is
in `learn`'s catalogue yet (PP-24).

| Kind | What the student is asked | Verified by |
|---|---|---|
| principle | state the principle the author applies in a passage, in own words | code: terms and quote from the passage; stance pair |
| decision | a documented situation with the decision withheld: what does the author decide, and why | string comparison on a captured `decide` argument; reasons by quote overlap |
| application | a new situation: apply a stated principle | rubric from the extracted principles; stance against the opposing claim |
| tool-decision | choose among options through a caller-defined tool | string comparison on the captured argument |
| polarity | agree or disagree with a claim and its negation | the claim and its generated opposite; both must be answered consistently |
| rebuttal | answer an opponent's argument as the author would | quote overlap with the author's own rebuttals |
| correspondence | reply to a letter | held-out letter, judged only where nothing mechanical fits |
| anachronism | a thing the author could not know | declines or asks, never invents |
| attribution | who held a position, when, and why it changed | exact fields from the source |
| interlocutor (self-play) | the model plays opponent and author in turn; kept only when the author side stays consistent with extracted claims | consistency against the claim ledger |

## What task generation showed on the Jefferson texts

A pinned run of the 8B policy on four Jefferson texts with the kinds recall,
explain and advise admitted 101 of about 240 proposals (advise 51, explain
49, recall 1). Rejections: 82 do not name their subject, 28 quote nothing
from the section, 14 ungrounded, 13 not self-contained, 2 duplicates. The
admitted tasks are single-passage questions; some come from Gutenberg
boilerplate and OCR tables, advise tasks put a modern question beside an
eighteenth-century quotation, and almost none ask what the author decided or
why.

Consequences: the gates do their job but the kinds are the wrong frame for a
person's judgment (PP-24); boilerplate and tables must be removed from sources
before tasks are written; recall needs its subject taken from the text and not
from the source's name; and the requests are the cost, so a corpus of
millions of tokens is out of reach until the local provider lets brain batch
them (PP-32).

## Paper rewrite rules

One arXiv-style paper in `papers/persona-training/`. It states the method and
the measured results of the final system only: no development history, no
defect catalogue, no superseded exams or version names, no hardware section
beyond a setup line, no pilots, no "does not claim" section (the evidence
must carry the claims). Figures for the pipeline, the exam arms and the
results. Lessons from the old defects move to `.agents/knowledge/`. Drafted by
one agent and grounded by an independent second review after the writer
skill's checks.

## Open prerequisites

- A key for a hosted provider to play the orchestrator remotely (none
  configured; a local model plays the role until then).
- A GPU that holds the whole corpus in the policy's context (PP-18) and the
  larger orchestrators; whether one will be available for any run in the
  paper.
