<!--
SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->

# Persona program: named adapters, orchestration, tooling and shared code

Scope: persona work on named adapters, orchestration, tooling and shared code.

Work items PP-28 to PP-39 of the persona program (index and order in
`persona-program.md`, findings in `persona-findings.md`).

## PP-28 Named adapters that cannot be lost

- **Problem.** `learn --adapter NAME` overwrites `NAME:latest` with no
  earlier version kept, does not serialise two runs of one name, publishes
  whatever the exam found (PF-09); a dry run of the sentence path hides the
  exam stages the run will take (PF-10); continuing an adapter at another
  rank is unverified (PU-03).
- **Evidence.** `crates/pipelines/src/learn/publish.rs:55-60`, `:64-74`,
  `:76-93`; `crates/pipelines/src/learn/mod.rs:303-335`, `:399-431`;
  `crates/orchestrator/src/model_ref.rs:123-131` (`name:tag` resolves a tag;
  publish writes only `latest`); `crates/pipelines/src/learn/exam_stages.rs:46-55`;
  `crates/pipelines/src/train/candidate.rs` (`rank` is the rank asked for).
- **Acceptance.** Each publish writes `NAME:<n>` (the next free number) and
  then moves `latest` to it by compare-and-set, recording the previous digest;
  a second run of the same name and base while one holds it is refused with
  the holder's run id (a lock file in the adapter's directory, released on
  every exit path, stale locks named by their run id); the learn report shows
  the exam verdict beside the publish and the CLI prints it; continuing at a
  rank other than the adapter's own is refused before training; the dry run
  lists `reserve` and `exam-set` as conditional on the plan's persona. Tests:
  two sequential publishes leave both versions loadable; a concurrent second
  run is refused; a rank mismatch is refused (CPU, tiny model); the dry-run
  list equals the run's stage list for a fixture plan naming a persona.
- **Design.** The alias machinery the release store already has
  (`aliases moved by compare-and-set`, AGENTS.md, orchestrator row) applied to
  the model store's named adapters.
- **Depends on.** Nothing.
- **Risk.** The lock is process-local to one machine's file system; two
  machines sharing a model store over a network file system are not covered,
  and the docs say so.
- **Size.** M.

## PP-29 The orchestrator agent and its sibling agents

- **Problem.** The decision is that a sven orchestrator agent drives the
  campaign through tools, with sibling agents (separate engines and
  providers) for preprocessing, step checks and judging. No orchestrator role
  exists, and sven's SDK offers no way to start or coordinate sibling agents.
- **Evidence.** sven `crates/team` (task list, spawn and lifecycle tools,
  orchestrator prompt fragments) exists; sven's public SDK
  (`crates/sdk/src/lib.rs:56-198`) re-exports the engine, agent, toolset,
  harness, chain, workspace and model surfaces and nothing of the team crate.
  Splinter's roles are `policy`, `teacher`, `generator`, `planner`, `judge`,
  `critic`, `router` (AGENTS.md, Laws).
- **Acceptance.** In sven (its own sample proving it): the SDK starts sibling
  agents with their own engine and provider, gives them a shared task list
  and returns each one's trajectory. In Splinter: an `orchestrator` role with
  its fallback rule in `core::role`; the `learn` stages offered to it as tools
  by the agent adapter, the stage engine still recording every stage; each
  sibling's trajectory linked from the run record; a scripted integration
  spec (mock models) in which the orchestrator runs a learn campaign with a
  judging sibling and the run record equals that of the plain pipeline on
  the same inputs.
- **Design.** Splinter's side lives in `splinter-agent` (sven calls) and
  `splinter-orchestrator` (the role); the pipeline order stays code, the
  orchestrator chooses only what the stages expose.
- **Depends on.** The sven SDK change.
- **Risk.** A model choosing stages can skip the exam; the exam and release
  stages are not offered as optional tools.
- **Size.** L.

## PP-30 The retrieval store seam in sven

- **Problem.** Retrieval over a person's corpus is a measured competitor and
  a way the model is meant to be used; Splinter keeps its own BM25 while sven
  has a memory store without the seams retrieval needs.
- **Evidence.** sven `crates/memory/src/store.rs:56-80` (`VectorStore`:
  insert, `search(query, limit)`, delete, `list(tag)`, get), SQLite with FTS5
  and optional stored vectors (`crates/memory/src/sqlite.rs:348-352`), no
  embedder trait, no bulk ingest or chunking, no typed metadata filter beyond
  one tag, no score threshold or rerank, no citation spans, no corpus
  snapshot id, and not re-exported by `sven-sdk`. Splinter:
  `crates/knowledge/src/retrieve` (`Bm25`, `Passage`) used by
  `samples/synthetic-author/src/retrieval.rs:14` and `learn --passages`.
- **Acceptance.** In sven: an embedder seam; bulk ingest with chunking; a
  per-host store injected through the engine; typed metadata filters, a
  score threshold and rerank; citation spans with byte offsets; a read-only
  retrieval tool with fixed top-k and a token budget; corpus snapshots with a
  version id; exposed through `sven-sdk`; sven's own sample. In Splinter: the
  retrieval arm and the retrieved-passage records go through it, the run
  record names the snapshot id, and `knowledge::retrieve` is removed or
  becomes the lexical backend sven's store is given.
- **Depends on.** The sven change.
- **Risk.** Retrieval results must not move silently: the synthetic-author
  retrieval arm is re-run and compared item by item with the BM25 one.
- **Size.** L.

## PP-31 Persona chat, complete

- **Problem.** A reply in prose with no call ends the exchange (the forced
  call is a listed to-do); a failed turn corrupts the history and a batch
  cannot resume (PF-21); the trace does not say which model, decoding or seed
  answered (PP-09).
- **Evidence.** `samples/persona-chat/src/session.rs:124`, `:140`, `:169`;
  `src/main.rs:150-171`; `src/trace.rs` (rows hold id and exchange only).
- **Acceptance.** When a round ends in prose and a final tool exists, the
  model is asked once more, told to decide through the tool, and the round
  is recorded as forced; a failed provider call removes the user turn and
  reports the error; `batch` skips ids already in the trace, refuses a
  repeated id in the prompts, and records model reference, adapter digest,
  decoding and seed on every line. Tests with a scripted provider for each.
- **Design.** Whether sven's request can require a tool call is to be read
  from sven's model API before choosing between a required-tool request and
  a follow-up turn.
- **Depends on.** PP-05 for the seed.
- **Risk.** Low.
- **Size.** S.

## PP-32 Trace and speed

- **Problem.** Task generation is serial and its per-call record is partial;
  brain's serving costs are not profiled.
- **Evidence.** `crates/pipelines/src/tasks/mod.rs:527-575` (one window at a
  time, each a blocking call; dead kinds are dropped by
  `still_worth_asking`); the per-call note (`:160-177`) keeps kind, prompt,
  prompt size, elapsed time, candidate count and a decline, not the raw
  reply, tokens, repairs or the window and sections; judge rationales are
  not stored with verdicts.
- **Acceptance.** Every model call of a run appends one line to the run's
  call log (prompt, raw reply, tokens in and out, latency, repairs, window
  and section ids, verdict rationale for a judge); task generation submits
  windows and kinds concurrently up to a bound, sharing the prefix across
  kinds, with the same admitted set as the serial path on a fixed seed and a
  scripted generator; a brain profile of a pinned small run (prefill against
  decode, prefix-cache hits, KV pool use, adapter folding per engine)
  recorded before any kernel work; a reviewer reads the whole trace of one
  small real run and files what it finds here.
- **Depends on.** Brain's batching of concurrent requests in the local
  provider.
- **Risk.** Concurrency changes the order of admissions; dedup must not
  depend on arrival order (test it).
- **Size.** L.

## Generalizing sample code into the SDK

Rule: whatever two samples need, or a sample re-implements, moves to the crate
that owns it, and the samples keep only their apparatus. Done: the seeded
generator (`core::rng`); the blocking typed call (`agent::blocking`); the
resumable job log (`store::jsonl`); one tool-calling turn (`agent::tool_turn`);
the consecutive-word occurrence check (`contains_words`). Each item below is
one commit, behaviour pinned by the tests that exist. Verified against the
code as it stands.

### PP-33 The last private generators

- **Still duplicated.** Production: `samples/voice-loop/src/ingress.rs:195-201`
  (an LCG choosing which training examples are transcribed). Test-only:
  `crates/eval/src/speech_judge.rs:211-214` (tests from `:206`) and the
  lifecourse tests (`intervals.rs:135`, `lifeexp.rs:268`,
  `cause_metrics.rs:326`, `contributing.rs:437-439`, `pooled.rs:291`). No
  copy remains in the speech features of the model crate. `splinter-expdb`
  keeps its own generator by design (no internal dependencies).
- **Acceptance.** The voice-loop draw uses `core::rng` with its seed from
  PP-09; the test streams are re-baselined in the same commit.
- **Size.** S.

### PP-34 One shared-run check

- **Still duplicated.** "Do two texts share a run of k words": absorb
  `samples/absorb/src/probes.rs:208-217` (`shares_a_run`, 8 words);
  Jefferson `samples/jefferson/src/corpus.rs:290-300` (hashed 8-word runs),
  `src/scenarios.rs:150-157` (6 words), `src/application.rs:59-62` (8 words);
  synthetic-author `src/gates.rs:225-253` (belief and brief runs). Eval has
  `contains_words` (a needle inside a haystack,
  `crates/eval/src/verifiers/quotation.rs:124`) and `overlap::RUN`, not the
  shared-run test.
- **Acceptance.** `eval::verifiers::quotation::shares_run(a, b, k)` (and the
  run that matched), the five call sites on it, their tests unchanged.
- **Size.** S.

### PP-35 The held-out share and family naming in `data`

- **Still duplicated.** `samples/jefferson/src/corpus.rs:225-232`
  (`is_exam_family`) and `samples/adams/src/split.rs:124-131`
  (`family_hash`) both take blake3 of the seed's little-endian bytes then the
  key, and reduce the first eight bytes modulo 100. Family grouping already
  uses `eval::overlap::overlap_groups` in both; Adams adds a union of
  related documents named by the smallest member id
  (`samples/adams/src/split.rs:96-121`).
- **Acceptance.** `data::holdout::share_of(seed, key) -> u64` with the input
  byte-identical, fixed test vectors taken from the current freezes, both
  samples on it, their frozen exams unchanged; the naming rule in `data`.
- **Size.** S.

### PP-36 Principle verification in `knowledge`

- **Still in a sample.** `samples/adams/src/principles.rs:37-140`
  (proposals with quotations, quotes looked up by code, status computed from
  what was found, `verify` at `:120`).
- **Acceptance.** In knowledge over a document trait, beside the claim gates
  it resembles; Adams calls it; PP-24's `principle` kind uses it.
- **Size.** M.

### PP-37 Scenario planning and the giveaway gate in `knowledge`

- **Still in samples.** Planning with deliberate non-fits:
  `samples/adams/src/scenario.rs` (plan, `gate` at `:121`). Generated
  questions that must not give the source away:
  `samples/adams/src/reconstruct.rs:78-110` (`gate`, `gives_away`) and
  `samples/jefferson/src/scenarios.rs:123-160` (`admit`, `GivesItAway`,
  `Copies`).
- **Acceptance.** One planner and one giveaway gate in knowledge, both
  samples on them, tests unchanged; PP-24's `decision` kind uses the gate.
- **Size.** M.

### PP-38 Judge calibration in `eval`, document bundling in `knowledge`

- **Still duplicated.** `samples/adams/src/judge.rs:143-200` (`Calibration`,
  `Control`, `calibrate`) beside `crates/eval/src/verifiers/calibration.rs:33-190`
  (`Calibration`, `measure`, `CalibratedJudge`); bundling documents for
  mining in `samples/adams/src/miner.rs:47-125` (`Bundle`, `bundles`).
- **Acceptance.** Adams' reconstruction judge is an eval `CalibratedJudge`;
  bundling is in knowledge and used by PP-24's miner.
- **Size.** M.

### PP-39 Recall at k, the embedding adapter, title case

- **Still in samples.** Recall at k and the embedder adapter:
  `samples/jefferson/src/retrieval.rs:46`, `:88`. Title case, three copies:
  `samples/adams/src/corpus.rs:407`, `samples/adams/src/materials.rs:128`,
  `samples/jefferson/src/corpus.rs:149`.
- **Acceptance.** Recall at k in eval, the embedder behind PP-30's seam;
  one title-case function in knowledge (beside `sections::title`).
- **Size.** S.

Not moved: the Cushing OCR parser, the authorship rules, the century and name
gates, the training apparatus of the Adams sample, the Gutenberg letter
parsing and all prompt wording.
