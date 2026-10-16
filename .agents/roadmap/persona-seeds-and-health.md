<!--
SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->

# Persona program: seed policy and build health

Work items PP-01 to PP-09 of the persona program (index and order in
`persona-program.md`, findings in `persona-findings.md`).

## The seed rule

Every draw a run makes (an author world, a writer's or a policy's sampled
decoding, an adapter's initialisation and batch order, an exam's choice of
families or items, a replay or rehearsal sample, a bootstrap) takes a seed
that:

1. defaults to a fresh random value when the caller names none;
2. is printed when it is drawn (`no --seed given, using seed N (pass --seed N
   to reproduce)`, the form brain's command line already uses in
   `crates/cli/src/qwen_cli.rs:910-912`);
3. is written into the run's manifest (the run record for a pipeline, the
   corpus or exam directory's manifest for a sample, the candidate record and
   the adapter card for training);
4. reproduces the draw exactly when passed explicitly.

Two kinds of constant are not draws and stay fixed: salts that separate one
derived stream from another (`EXAM_SALT`, `CORPUS_SALT`, `OPPOSITE_SALT`), and
constants that define the world itself (`REFERENCE_SEED`, the pool of
situations a drawn author is checked against). A seed that decides a frozen
artifact (an exam split, a reserved family set) is drawn once when the artifact
is created, recorded in it, and read back from it afterwards; it is never drawn
again for the same artifact. Fixed literal seeds are allowed in tests only.

## PP-01 Re-pin brain to a reachable commit

- **Problem.** `Cargo.lock` pins brain at `75ae7f6a`, which no local branch,
  `main` or `origin/main` contains (PF-17). A build that does not use the
  local override depends on the remote still serving an unreferenced object
  (PU-04).
- **Evidence.** `Cargo.lock:1005` and the other brain `source` lines; the
  rebased equivalent `1dae29b7b` is an ancestor of brain's `main` and
  `origin/main`. Brain's local `main` is three commits ahead of `origin/main`
  (including the change that lets its LoRA fine-tune command continue an
  existing adapter); a pin may only name what the remote holds.
- **Acceptance.** `make deps/lock` against a brain checkout whose `HEAD` is on
  its remote main; `make build test` from a clean clone with an empty cargo
  home (no local override) succeeds; `check-lock-sources` refuses a pinned
  revision that is not an ancestor of the dependency's remote-tracking main.
- **Design.** Extend `scripts/gates/check-lock-sources.sh` with the ancestry
  check over the local remote-tracking ref (no fetch); re-pin after the user
  has pushed brain.
- **Depends on.** The user pushing brain's local commits, if the pin needs them.
- **Risk.** The re-pin pulls in brain's catalog change (PP-02) and any other
  API drift; build the whole workspace and run the file-size and architecture
  gates after it.
- **Size.** S.

## PP-02 Port the adapter-manifest spec to brain's current catalog

- **Problem.** `crates/model/tests/adapter_manifest_is_followed.rs` compiles
  only at the stale pin (PF-18).
- **Evidence.** Test `:50-90` compares `ManifestPoll` values with `assert_eq!`
  and reads `release.path`/`release.digest`; brain
  `crates/catalog/src/adapter_release.rs:198-210` has no derives on
  `ManifestPoll` and `Release(Box<LoadedAdapter>)` with the release under
  `.release` (`:115-120`).
- **Acceptance.** The test builds and passes at the new pin, still asserting
  the four behaviours it names (followed, replaced whole, digest mismatch
  rejected and named, withdrawn is missing).
- **Design.** Match on `ManifestPoll` variants instead of comparing them; read
  `release.release.path` and `.digest`. Do not ask brain to restore derives
  for a test's convenience.
- **Depends on.** PP-01.
- **Risk.** Low.
- **Size.** S.

## PP-03 No silent skips in brain, and a tighter gate here

- **Problem.** The standing rule (AGENTS.md, "No test skips itself
  silently") is enforced for Splinter only; brain, whose tests Splinter's
  results rest on, skips widely, and Splinter's own gate misses several shapes
  (PF-16).
- **Evidence (brain, tracked `.rs` files, read only).**
  - `brain_testutil::skip(reason)`: 765 call sites in 358 files. It prints
    `SKIP:` and returns, and panics only when `BRAIN_REQUIRE_FIXTURES` is set
    (`crates/testutil/src/lib.rs:49-54`). By the words of the reason, about
    526 name a missing model, fixture or environment variable, 29 an external
    tool, 21 a device, 13 the network and 176 something else (a keyword
    classification, approximate).
  - `brain_testutil::skip_unavailable(reason)`: 457 call sites; never fatal
    (`crates/testutil/src/lib.rs:87-89`); the reasons are mostly a missing
    device or backend.
  - 70 file-local `fn skip() -> bool` helpers with 214 call sites, 63 of them
    reading `MOE_SKIP_GPU_TESTS` and printing nothing (most `if skip() {
    return; }`), concentrated in the gradient-check, benchmark and model
    crates; 17 `fn skip_gpu()` helpers with 49 call sites.
  - The count of 546 used in earlier notes does not match any of these
    measures; the numbers above are the ones to plan from.
- **Acceptance.** In brain (brain-owned commits): every skip either fails
  with its reason or is `#[ignore = "reason"]`; an environment variable that
  turns a GPU test off becomes an ignore attribute or a cargo feature; a gate
  equivalent to `check-test-skips` runs in brain's `make check`. In Splinter:
  the gate's fixture with `eprintln!("SKIP: ...")`, `skip_unavailable(`,
  `if skip() { return; }` and a `mod gpu_tests {` module fails it.
- **Design.** Brain: convert per crate, fixture-backed skips to
  `#[ignore]` plus a `BRAIN_REQUIRE_FIXTURES` run in CI, device skips to
  ignores; Splinter: widen the regex and the scope to every `#[cfg(test)]`
  module and every `tests/` file.
- **Depends on.** Nothing here; brain work is scheduled by brain.
- **Risk.** Converting hundreds of brain tests surfaces real failures that
  skips hid; plan for them.
- **Size.** S (Splinter gate), L (brain).

## PP-04 Sign test that holds for large suites

- **Problem.** brain's exact sign test returns an infinite or NaN p-value past
  about 1030 discordant pairs (PF-11).
- **Evidence.** brain `crates/promote/src/stats.rs:61-85`.
- **Acceptance.** brain test: 2000 discordant pairs at every split give a
  finite p in [0, 1] equal to a log-space reference within 1e-12 relative;
  the existing small-n cases unchanged.
- **Design.** Sum the tail in log space (log-gamma or a running log
  coefficient) in brain; Splinter keeps calling it.
- **Depends on.** Nothing.
- **Risk.** Low.
- **Size.** S.

## PP-05 The seed mechanism

- **Status.** Done. `Seed` in `splinter-core`, `splinter_model::seed::draw`, `Recorder::seed` and the run record's `seeds`, `splinter_sdk::seed::{resolve, frozen}`; brain's matching mechanism is the `brain-seed` crate. Both repositories gate fixed seeds with `seed-literals` (production and test mode).

- **Problem.** No shared way exists to draw, print and record a seed; each
  site invents its own default (inventory under PP-06 to PP-09).
- **Evidence.** `crates/core/src/rng.rs` holds the generator only.
- **Acceptance.** A `Seed` value in the vocabulary that parses and prints,
  drawn fresh from the operating system by one function in the orchestrator
  (the layer that owns the run record) and by one SDK helper for samples;
  the run record has a `seeds` map (name to value) every stage writes into;
  a test asserts that a run given every seed explicitly reproduces the run
  record's draws and that a run given none records them all.
- **Design.** `core::rng::Seed(u64)` with `Display`/`FromStr`;
  `orchestrator::runs::Recorder::seed(name, Option<Seed>) -> Seed` draws,
  prints through the progress receiver and records; samples use the same
  through the SDK. No environment variable.
- **Depends on.** Nothing.
- **Risk.** Low; the run record format gains a field (old records read
  without it).
- **Size.** M.

## PP-06 Training seeds always explicit and recorded

- **Status.** Done in splinter (every training path draws, records on the run and the candidate) and in brain (builders draw, read the seed back on resume, record it in `training.json`; the full fine-tune now writes a record). Brain's LoRA video and image fine-tunes now seed their adapter's initial weights with the run seed.

- **Problem.** Every training path leaves the seed to brain's fixed default
  1337 and records nothing (PF-04).
- **Evidence.** `crates/model/src/train.rs:109-112`, `:406-407`;
  `crates/model/src/consolidate.rs:46-47`, `:104-105`;
  `crates/pipelines/src/train/mod.rs:185-188`, `:389`;
  `crates/pipelines/src/train/full.rs:51`; CLI `crates/splinter/src/cli/learn.rs:177-181`,
  `cli/train.rs:163-166`, `cli/practice.rs:104-106` ("brain's default if not
  given"); `samples/synthetic-author/src/training.rs` (`train_seed`);
  brain `crates/sdk/src/finetune.rs:151`, `full_finetune.rs:102`,
  `preference.rs:108`. The three study adapters carry seed 1337 in their cards.
- **Acceptance.** `FineTune.seed`, `FullFineTune.seed` and the preference
  request take a `Seed`, not an `Option`; the pipeline draws one through
  PP-05 when the command names none; `Candidate` records it beside
  `training_record`; synthetic-author `summary.json` records seed, steps,
  rank, alpha, learning rate, records per step, bf16 and the dataset's
  digest; a CPU test on the tiny model shows the same seed giving the same
  adapter digest and different seeds different ones.
- **Design.** Splinter side as above. Brain side (brain-owned): the SDK
  builders draw and report a seed when none is set, as its command line does,
  and `training.json` records it.
- **Depends on.** PP-05.
- **Risk.** Changing the default changes what an unseeded rerun of an old
  command produces; that is the point, and the record says which seed.
- **Size.** S.

## PP-07 Synthetic-author seeds and run manifests

- **Status.** Done. `corpus` draws and prints the author seed when none is
  given and writes `manifest.json` (author seed and whether it was drawn,
  writer and reader references, decoding, the run seed every per-request
  writer and reader seed is derived from, options, code revision); a corpus
  is only continued with the recipe it was begun with. `recheck`, `dataset`,
  `exam`, `items`, `audit` and `cross` read the author from the corpus and
  refuse a differing `--seed` (a corpus from before manifests is known by its
  `persona.json`). `exam` writes `exam/manifest.json`, refuses to append under
  another manifest naming both digests, and takes over old outcomes only with
  `--adopt`. `train --train-seed` draws and prints when absent and
  `summary.json` records the seed and every setting. `mine` draws the seed
  its reader's per-request seeds derive from when `--run-seed` is absent,
  prints it and records it with the reader in `mine.json`; a later `mine`
  continues with both and refuses another reader or seed.
- **Problem.** The author seed is a required argument repeated on every
  subcommand and never checked against the corpus (PF-03); the writer's
  draws are unseeded and unrecorded; exam and corpus directories have no
  manifest of what produced them (PF-02).
- **Evidence.** `samples/synthetic-author/src/main.rs` and the same
  `seed: u64` on `plan`, `recheck`, `dataset`, `exam`, `show`, `items`,
  `audit`, `cross`, `corpus`; local writer sampling (`src/models.rs`) inherits
  `seed: None` from `crates/model/src/local/mod.rs:61`, so the provider draws a
  seed for each sampled request and a recorded run keeps them
  (`crates/model/src/local/request.rs:37-43`, `Run.sampled`);
  a remote writer runs at its provider's defaults with no seed.
- **Acceptance.** `corpus` without `--seed` draws an author seed, prints it
  and writes `manifest.json` (author seed, writer and reader references,
  decoding, writer seed per request or a run seed it derives them from,
  options, code revision); every later subcommand reads the author seed from
  the cache and refuses a differing `--seed`; `exam` writes
  `exam/manifest.json` (items digest, `--articles`, `--per-cell`, arm, base
  and adapter digests, decoding and its seed) and refuses to append to
  outcomes made under another manifest; a test per refusal.
- **Design.** One `Manifest` type in the sample; the seed helper of PP-05.
- **Depends on.** PP-05; shares the outcome binding with PP-11.
- **Risk.** Existing study directories have no manifest: `exam` accepts them
  only with an explicit `--adopt` that writes one from the flags given and
  says so.
- **Size.** M.

## PP-08 Pipeline seeds of `learn`, `exam` and `practice`

- **Status.** Done. Every site in the table takes its seed from the run and keeps it in the artifact it decides; the release gate's intervals and the report intervals do too.

- **Problem.** The persona `learn` run reserves its exam, samples its replay
  and rehearsal and bootstraps its intervals from literal seeds.
- **Evidence.**
  | Site | Seed | Decides |
  |---|---|---|
  | `crates/pipelines/src/learn/exam_stages.rs:75` | `seed: 0` | which families are the exam and the dev suite |
  | `crates/pipelines/src/rehearsal.rs:69` (used `learn/stages.rs:635`) | `REHEARSAL_SEED = 0` | the rehearsal tasks |
  | `crates/pipelines/src/train/mod.rs:129` (used `:720`) | `REPLAY_SEED = 0` | the replayed records |
  | `crates/pipelines/src/datasets.rs:52` | `MIX_SEED = 0` | which subjects keep privileged context |
  | `crates/pipelines/src/tasks/mod.rs:64` | `DENOISE_SEED = 0` | denoise tasks |
  | `crates/pipelines/src/powered/analysis.rs:207` | `SEED = 0` | the family bootstrap |
  | `crates/pipelines/src/exam.rs:529` | `INTERVAL_SEED = 0` | the per-family gain interval |
  | `crates/pipelines/src/practice/exam.rs:41` | `EXAM_SEED = 0` | the practice exam |
  | `crates/splinter/src/exam_session.rs:192` | `seed: 0` | the power simulation |
  | `crates/splinter/src/cli/exam.rs:132-134`, `:200-202` | default 0 | `exam-set create` families; `labels-export` sample |
- **Acceptance.** Each site takes its seed from the run (PP-05), recorded in
  the run record and in the artifact it decides (the reservation, the
  dataset manifest, the powered report); a frozen exam set reads its seed
  back; a test per artifact shows a rerun with the recorded seed reproduces
  it. Reserving with a fresh seed must still honour `touched_by` (knowledge
  note 017): a test reserves twice over a lineage and finds no trained
  family examined.
- **Design.** Replace each constant by a named field of the request filled by
  the stage from the recorder.
- **Depends on.** PP-05.
- **Risk.** A fresh reservation seed per run means two runs on the same
  sources examine different families; comparisons across runs must use a
  recorded seed, which the report states.
- **Size.** M.

## PP-09 Seeds of the other persona samples, and the rest

- **Status.** Done for every sample and script in the repository (the persona samples, the lifecourse and health programs, the Python tools and paper scripts). Frozen Jefferson and Adams exams did not move.

- **Problem.** The persona samples fix their own seeds by default; the
  remaining samples do too and belong to their own programs.
- **Evidence.** Persona program: `samples/absorb/src/main.rs:70-72` (default
  1, "seeds the choice of families and every later hash");
  `samples/jefferson/src/main.rs:74`, `:124`, `:364` and
  `samples/adams/src/main.rs:239` (split and task seeds default 1, which
  decide the frozen exam); `samples/adams/src/run.rs:723` (`seed: 0` for the
  reinforcement loop); `samples/voice-loop/src/main.rs:61-62` (default 1) and
  its private LCG `samples/voice-loop/src/ingress.rs:195-201`;
  persona-chat decodes with the default sampling and records no seed
  (`samples/persona-chat/src/main.rs:102-124`, the model loaded at `:112`). Other programs, for their
  owners: `samples/lifecourse/src/recipe.rs:39`, `report.rs:41`,
  `contributing.rs:55`, `pooled.rs:57`, `prevalence.rs:68`;
  `samples/health/src/main.rs:81`. Test-only literal seeds (absorb fixtures,
  synthetic-author tests) comply.
- **Acceptance.** Persona samples: a split seed is drawn when absent and
  written into the freeze (`FROZEN.json` and the held-out list), and a later
  freeze reads it back; the reinforcement loop seed is drawn and logged;
  persona-chat records model, decoding and seed in every trace line.
- **Design.** PP-05's helper through the SDK.
- **Depends on.** PP-05.
- **Risk.** A frozen Jefferson or Adams exam must not move: the existing
  freezes record seed 1 explicitly before the default changes.
- **Size.** S (persona samples); the others are tracked by their programs.
