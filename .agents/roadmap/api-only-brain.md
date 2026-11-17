# Brain through its API only

Scope: retiring every in-process use of brain from Splinter, so that Splinter reaches brain only through brain's API (HTTP OpenAI-compatible endpoints with `policy_trace`, `/v1/jobs` training jobs, adapter serving, D-Bus), and the gate that keeps it so.

Decision (user, 2026-11-13): brain's main job is serving many processes at once; the stack uses its API, and many concurrent API requests must be served efficiently. Brain itself stays an SDK with samples. Splinter has no in-process fallback. sven has the `check-no-brain-dependency` gate; bench has no brain dependency (it serves and grades through the HTTP provider only; checked 2026-11-13). Where concurrency needs exceed what `brain serve` does well, brain is fixed, never Splinter limited.

## Inventory (2026-11-13, origin/main 5bf1ec810)

Everything that touches brain lives in `splinter-model`, which links `brain` (features `text`, `study`, `timeline`, `audio`), `brain-gpu-core`, `brain-data` (in tests it linked `brain-catalog` and `brain-testutil` until 2026-11-17). `splinter-model` is the only crate allowed to by `architecture.toml` `[brain].allowed`.

| Area | Where | In-process use | API replacement |
|---|---|---|---|
| Generation | `local/` (`LocalQwen`), `answer.rs` (`Answerer`), `exam.rs`, `selection.rs` `ModelSelection::load`, `residency/`, `orchestrator::context` | `brain::ChatPipeline` on resident bases; `LocalQwen::complete` cancels the previous generation when a second request arrives | `POST /v1/chat/completions` (exists; `selection/served.rs` already builds it) against a server Splinter starts or is pointed at |
| Training | `train.rs` (`fine_tune`, `train_preference`, `score_chat`), `policy_gradient.rs`, `consolidate.rs` (`full_fine_tune`, `merge`), `timeline/training.rs` | `brain::ChatFineTune`, `PreferenceFineTune`, `ChatFullFineTune`, `merge_adapter` | `brain/train` job: `sft` and `policy_gradient` exist (`served_train/`); DPO, full-parameter tuning, adapter merge and timeline training have no job yet (brain gaps below) |
| Held-out scoring | `train::score_chat`, `timeline/scoring` | `brain::score_chat` teacher-forced | job report carries base and tuned held-out scores for `sft`; standalone teacher-forced scoring has no endpoint (gap) |
| Dataset checks | `capabilities.rs`, `train.rs::validate_*`, `measure.rs`, `tokens.rs`, `pictures.rs` | `brain-data` parsers, brain tokenizer | a job refuses a bad record by line at admission (400); `POST /v1/messages/count_tokens` counts; a dry-run validation endpoint is a gap |
| Embeddings | `embed.rs` | `brain` embedding pipeline | `POST /v1/embeddings` |
| Speech | `speech/` | `TranscribePipeline`, `TtsPipeline` | `POST /v1/audio/transcriptions`; synthesis endpoint to verify |
| Timeline | `timeline/` | `brain::timeline` model, survival arithmetic | `/v1/jobs` timeline model class (gap); the survival arithmetic is pure statistics |
| Statistics | `stats.rs`, `timeline/survival` | `brain::promote` sign test, bootstrap, survival metrics | pure arithmetic, no model: kept as a reviewed exception until brain serves it |
| Hardware | `device.rs`, `store.rs` | `brain-gpu-core` inventory, `--device` grammar, store scan | `GET /v1/capabilities`, `GET /v1/models`; the `--device` text is passed to the server Splinter starts |
| Campaign controller | `pipelines::campaign::brain` (`ChatCollector`, `ChatValidator`, `BrainTrainer`) | in-process `Answerer`, `train_policy_gradient` | the served collector and validator of `campaign::served`, and `served_train` |
| Release gate | `pipelines::release::serve` (`compare.rs`, `meaning.rs`, `gating.rs`) | candidate answered in-process, then compared with the served run | served against served, or the comparison is dropped |
| Surfaces | CLI `--brain-endpoint`, `--brain-key-file`, `--brain-artifact-store`, `--brain-adapter-manifest`, `--brain-bin`, `--device`, `--backend`, `--vram-budget`; samples (`campaign-lua`, `synthetic-author`, `goal-selection`, `tool-syntax`, `adams`, `jefferson`, health, lifecourse); docs `getting-started/configuration.md`, `concepts/*`, `reference/*` | all of the above | one configuration: a brain server given (`--brain-endpoint`) or started by Splinter (`--brain-bin`, `--models-dir`) |
| Tests | `pipelines/tests/it/residency.rs`, `orchestrator/tests/it/sampled_seeds.rs`, `model/src/fixture.rs`, `residency/scripted.rs`, `model/tests/it/*` | scripted in-process loaders, toy checkpoints | `scripted_endpoint` / `scripted_jobs` (an HTTP double of the contract), `fake_brain` |

## Plan (each a self-contained gated commit)

1. This ledger.
2. Generation: a `local:` model is always requested from a brain server; with no endpoint configured the runtime starts one (`brain serve --models-dir ... --serve-adapters DIR --training --artifact-store DIR`) and owns it. Delete `local/`, `residency/`, `Answerer`'s in-process path and `ModelSelection::load` for local weights; the in-process exam path goes with them.
3. Training: always a `brain/train` job; delete the in-process trainers; the artifact store is the managed server's.
4. Campaign: delete `ChatCollector`, `ChatValidator`, `BrainTrainer` once `campaign::served` has landed; release gate compares served against served.
5. Brain gaps closed in brain (each with a regression test): DPO objective, full-parameter job and adapter merge, dataset dry-run validation, standalone teacher-forced scoring, timeline job.
6. Manifests: drop `brain`, `brain-gpu-core`, `brain-data`, `brain-catalog`; `architecture.toml` lists the only brain crates the model adapter may keep (pure statistics) and `scripts/gates/check-no-brain-linkage.sh` refuses any other.
7. Docs, samples, `architecture.md`, `AGENTS.md`, regenerated `docs/reference/cli.md`.

## Status

- 2026-11-13: inventory and plan written.
- 2026-11-13: generation is API-only (commit "generation"). Removed: `LocalQwen` (whose `complete` cancelled the previous generation when a second request arrived: the concurrency defect the RSI2 lane hit, gone with it), `Residency`, `Answerer`'s in-process loader, the in-process exam path, in-process embeddings, `release_bases`, the drawn-seed recorder (the store still reads old records), the in-process `BrainLoader` of the synthetic-author study. Added: `ManagedBrain` (a `brain serve` over the model store with `--serve-adapters`, `--training` and an artifact store, started on first use and stopped on drop), `AdapterOffer::Directory`, `Embeddings` over `/v1/embeddings`, `DeviceFlags` carrying `--brain-endpoint/--brain-key-file/--brain-bin` for every sample. The method smokes (`make test/methods`) now generate through a `brain` on `PATH` serving the toy checkpoint on the CPU backend.
- Still in-process after this commit: training (`train`, `policy_gradient`, `consolidate`, `timeline`), dataset parsers and the tokenizer (`capabilities`, `measure`, `tokens`), speech, the timeline model, `device` inventory, statistics.
- 2026-11-17: `splinter-model` no longer links `brain-catalog` or `brain-testutil`. brain moved the manifest follower into its serving internals; the manifest spec now holds a written manifest to `brain::AdapterManifest`, the public type the server parses it with, and leaves the digest and base checks to brain's own tests.

## Open (what remains before `splinter-model` drops its brain crates)

Each item is brain API to add first (in brain, with a regression test), then the Splinter call site moves and the in-process code goes.

| Gap | Replaces | Note |
|---|---|---|
| `brain/train` objective `dpo` | `train::train_preference` | `served_train` refuses DPO by name today |
| Full-parameter job and adapter merge | `consolidate::{full_fine_tune, merge}` | refused by name with endpoints today |
| Dataset dry-run (validate a chat or preference file and report rows and token lengths without a job) | `capabilities`, `train::validate_*`, `measure::RecordMeter`, `tokens::TokenCounter`, `pictures` | job admission already refuses a bad record by line, but the pipelines check before publishing |
| Teacher-forced held-out scoring of a base or adapter on a dataset | `train::score_chat`, `timeline/scoring` | the `sft` job report carries base and tuned scores only |
| Device inventory (`GET /v1/devices`: card, total and free bytes) | `device::gpu_cards`, `Devices`, `device::select*` (used by `roles::discovered_assistant`, `doctor`) | |
| Timeline (survival) model as a served class | `timeline/` | its survival arithmetic (`brain::timeline::survival`) is pure statistics |
| Speech synthesis route; transcription exists (`/v1/audio/transcriptions`) | `speech/` | |
| Statistics: paired sign test, bootstrap interval | `stats.rs` | pure arithmetic: a reviewed exception, or a brain route |
| A toy chat checkpoint writer in the brain CLI | `fixture::toy_chat_checkpoint` | the method smokes and `dataset_split` tests build checkpoints with `brain::toy` |
| Delete `campaign::brain::{ChatCollector, ChatValidator, BrainTrainer}` | | `campaign-lua`'s `main` still uses them; `rsi2` already uses `campaign::served` |
| Pass endpoints explicitly instead of `BrainDefaults` | `orchestrator::config::set_brain_defaults` | samples that build a `Config` in several places state the flags once |
| ~~`scripts/gates/` refusing a brain crate outside the allowed client~~ | | landed: `[[brain.exception]]` in `architecture.toml` names each (package, brain crate) pair still linked, with a why; any other brain dependency fails `check/gates/architecture`, and an exception whose dependency is gone fails as stale, so the list only shrinks |

Brain landed: `/v1/embeddings` takes `instruction` and `max_tokens` (the client sends both; a brain without them drops the instruction silently, so Splinter needs a brain with ab4379f00).

Verified: `make check`, and `make test` with `--skip timeline_` (the in-process timeline specs create a wgpu adapter and fail when the cards are busy: they passed alone, and leave with the timeline retirement). `sandbox::process_runner::nothing_outlives_a_supervisor_that_is_killed_mid_run` fails on the shared node with the sandbox crate untouched. Not verified: a real `brain serve` started by `ManagedBrain` (the scripted contract and a stand-in binary only; no brain binary was available and the cards were busy).

## Plan for the served-only training family (brain now offers every API it needs)

Brain origin/main has `brain/train` objectives `dpo`, `full_train`, `merge`, `timeline_train`, the `check_dataset` action, teacher-forced `score` through `/v1/run`, `GET /v1/devices`, `POST /v1/audio/speech` and `brain qwen3 toy`. The order that keeps every commit green:

1. `served_train::job::run` takes the action; add `check_dataset` (typed report), `score`, and the request builders and delivery for `dpo`, `full_train` (a checkpoint leaves as a manifest of per-file digests) and `merge`; scripted-job doubles for each.
2. Replace `train_checks`, `capabilities::overlong_records` and `pictures::validate_dataset_on` with the `check_dataset` report (row length is its `tokens.max`; overlong lines are `over_max_block_lines`; refused records are `invalid`); `BrainDatasetCheck` becomes `splinter_data::Unchecked` at write time, since job admission names a refused line.
3. Pipelines always train through `Runtime::brain()`: delete `fine_tune`, `train_preference`, `train_policy_gradient`, `full_fine_tune`, `merge`, `score_chat` and `refuse_unserved`; `powered::voice` scores through `/v1/run`.
4. Per-record token counts (`TokenCounter`, `RecordMeter`, used by `block.rs`, `datasets.rs`, `voice`, repo-replay) need a per-line count from `check_dataset` (brain gap: an option returning each example's tokens); until then they keep the `brain` dependency.
5. Hardware inventory (`device`, `roles::discovered_assistant`, `doctor`) onto `/v1/devices`; speech recognition and synthesis onto `/v1/audio/*` (speech ingress, the TTS policy trainer and the listener have no route and need one or retirement); the timeline model onto `timeline_train` and its forecast action; toy checkpoints onto `brain qwen3 toy`.
6. Paired statistics stay in splinter as pure arithmetic: the sign test and bootstrap are reimplemented here; `brain-survival` (no dependencies) is the one brain crate that may stay as a reviewed exception.
7. Delete `campaign::brain::{ChatCollector, ChatValidator, BrainTrainer}` and `campaign-lua`'s `main` (the generic `lineage` driver supersedes it), then `BrainDefaults`; drop the exceptions one by one.
- 2026-11-17 (step 1): `served_train` runs every action it needs: `check_dataset` (`check_dataset`, a typed `DatasetReport` with every record's `[line, tokens, prompt, trained, examples]`, paged), teacher-forced `score_chat` through `POST /v1/run` (the model is the id `locate` finds, an adapter offered first), `train_preference_served` (`dpo`, pairs checked and the row sized by the service), `full_fine_tune_served` and `merge_served` (installed as `local/<name>` in the model store; files fetched from the artifact store by the manifest when the server's store is not this machine's). The scripted job double answers each.
- 2026-11-17 (steps 2-3 and the campaign half of 7): nothing trains, checks or counts in Splinter's process. Deleted: `fine_tune`, `train_preference`, `train_policy_gradient`, `full_fine_tune`, `merge`, `score_chat` (in-process), `validate_dataset*`, `check_train_precision`, `pictures`, `tokens::TokenCounter`, `BrainDatasetCheck` (datasets are written `Unchecked`; the trainer reads every one before a run), `refuse_unserved`, `campaign::brain::{ChatCollector, ChatValidator, BrainTrainer}` and `campaign-lua`'s controller binary (`lineage` is the sample's driver; `campaign::served` the controller's served path). `RecordMeter` measures many lines in one `check_dataset`; `dataset build --max-block` falls back to the character floor only when no service can be asked. Chunking (the voice view, the voice score) measures in characters: there is no per-text count short of a check job, and the row a run trains is sized by the trainer's count anyway. The policy-gradient settings a plan records are the ones named: brain's capabilities declare no defaults for the objective's params, so the defaults are stated by the run's training record (brain gap, minor). A training precision is no longer parsed up front: the `full_train` job refuses a bad one at admission.
- 2026-11-13: model calls of the typed calls and the solves are asked again, bounded, on a transient connection failure only (`splinter_agent::retry`, finding 149); the `lineage` binary's behaviour is otherwise unchanged and a running experiment keeps its pinned build.
