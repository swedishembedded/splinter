<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# splinter-expdb

A versioned **experience graph** database for what agents and robots do: tasks,
worlds, observations, decisions, actions, outcomes and the interpretations made
of them. It stores plain immutable files and is built for many writers at once.

Swedish Embedded AB implements storage engines for machine-learning experience
data for its clients. If your team needs expertise in versioned,
content-addressed databases or parallel-filesystem I/O, you can procure our
services by sending an email to info@swedishembedded.com.

The crate stands alone. Nothing else in the workspace uses it yet.

## The one rule

> Experience is the source of truth. Training data is a reproducible projection
> of it.

Raw experience is written once and never changed. Evaluations, credit
assignments, skills, transcripts, embeddings and datasets are records that say
who produced them and how, so a better algorithm can reinterpret old
experience and a withdrawn evaluator stops counting without anything being
deleted.

## What it stores

| Concept | What it is |
|---|---|
| Task, task instance, state | What was to be achieved, one concrete instance of it, and a reproducible point in the world (with how reproducible it is). |
| Attempt, decision, transition | One try at an instance; the moment an agent chose; what the choice caused. Attempts at one instance from one world form an **episode family**. |
| Fork, counterfactual set | An alternative taken from an earlier decision. It shares everything before that decision. |
| Evaluation, retraction | A judgement of a target by a named evaluator version, and the withdrawal of an evaluator version. |
| Skill and its evidence | A hypothesis, believed in proportion to the evidence for and against it. |
| Derivation, lineage | How a derived record was made. Links run from a model back to the experiences it rests on and forward again. |
| Episode, stream, chunk, event, action segment | Modality-neutral experience: streams of any registered modality on explicit clock domains, with events, actions over intervals, and cross-modal correspondences that carry a confidence. |

Nothing is specific to language. A modality is a registered schema, time is an
interval on a clock, and a stream's samples are blob chunks. Raw streams stay
canonical; a transcript or an embedding is another stream that names the
derivation and the source it came from.

## How it is stored

```text
<root>/
  FORMAT                 marker and format version
  refs/                  tiny pointers: one chain per writer, the catalog
  manifests/             immutable manifests: the only transaction layer
  segments/              immutable records and edges in column blocks
  blobs/                 immutable packs of content-addressed chunks
  indexes/               immutable index runs, vector and text shards
  pins/                  snapshots that must be kept
  overlay/  cache/       mutable replay state and derived data, kept apart
```

- **Immutable files, no locks.** Published files are never changed. A file is
  part of the database only when a manifest names it, so a crashed writer
  leaves an orphan nobody reads.
- **No coordination.** Each writer derives its own id; records are
  `(writer, sequence)`. Manifests only add and remove files and a removal is
  permanent, so merging histories is a set union in any order.
- **Snapshots.** A snapshot is a fixed set of files. It reads the same records
  for as long as it is pinned, whatever is published or compacted after.
- **Content addressing.** Large objects are split by content-defined chunking
  and stored in packs, so a small edit stores only the chunks it touched and
  identical content is stored once.
- **Indexes are projections.** A reader assembles an index from persisted runs
  and scans only the segments no run covers. Queries use zone maps and Bloom
  filters to skip blocks, or an index when one covers the snapshot.
- **Compaction** merges small segments, blob packs and index runs without
  changing any record id.

## Using it

```rust
let db = Database::open(root, Config::default())?;
let mut collector = db.collector(&WriterIdentity::new("exp", "job-1", "host-a", 0))?;
let mut run = collector.start_attempt(&definition, &instance, &initial_state, &policy, Some(7))?;
let decision = run.decision().commit(Action::new("grep", args))?;
run.transition(&decision, &next_state, Some(0.2), None)?;
run.finish(Outcome::Fail)?;
let mut branch = collector.fork(&decision, &policy)?;   // an alternative; shares the prefix
collector.flush()?;                                      // nothing is visible before this

let snapshot = db.snapshot()?;
snapshot.pin("training-run-1")?;
let plan = snapshot.compile(&Recipe::dpo().min_gap(0.5))?;   // the dataset is recipe + snapshot
```

`cargo run --release -p splinter-expdb --example quickstart` runs a complete
version of this. The specs under `tests/` are the reference for everything
else, one file per concern:

| Concern | Spec |
|---|---|
| Blobs, trees, segments, manifests, snapshots | `blobs`, `segments`, `snapshots` |
| Writers, spool, concurrency | `collector`, `concurrency` |
| Graph, families, indexes, queries | `graph`, `families`, `indexes`, `queries` |
| Evaluations, skills, lineage, experiments | `evaluations`, `skills`, `derivations`, `experiments`, `analysis` |
| Training views for text and decisions | `training_views`, `loader`, `overlay` |
| Streams, clocks, windows, multimodal views | `streams`, `clocks`, `windows`, `mm_views` |
| Compaction and properties | `compaction`, `properties` |

## Training views

A [`Recipe`] compiled against a snapshot is a plan of references, not a copy:

- imitation (steps or whole episodes), preference pairs at one state and
  context, group-relative learning over families whose rewards differ,
  single transitions, step labels;
- world-model windows (past and an action to the future), contrastive pairs
  with hard negatives from nearby in time, masked prediction, action chunks
  conditioned on observations and an instruction.

Flow-matching noise, time, noisy action and velocity are derived when a sample
is materialised, from the clean action and a seed. Nothing pre-noised is
stored. An export to JSON lines is a projection; the plan and its snapshot
remain the dataset.

## Limits

- One storage backend, plain POSIX files. The backend trait keeps record
  references independent of paths; other backends are not built.
- The read model (index, adjacency, timeline) is held in memory per snapshot.
  Persisted index runs avoid scanning but do not avoid loading.
- Record bodies are canonical JSON inside compressed column blocks. Envelope
  fields are columns; bodies are parsed only for rows that survive a filter.
- Several writers sharing one job name can overwrite each other's head, so
  each writer gets its own ref and maintenance tasks publish on unique refs.
- Concurrency is tested with independent database handles in threads, not with
  separate processes.
- Embeddings are supplied by the caller and searched exactly. Computing them,
  credit assignment and skill extraction belong to the model runtime.
