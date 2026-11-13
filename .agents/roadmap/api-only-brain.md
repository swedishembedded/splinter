# Brain through its API only

Scope: retiring every in-process use of brain from Splinter, so that Splinter reaches brain only through brain's API (HTTP OpenAI-compatible endpoints with `policy_trace`, `/v1/jobs` training jobs, adapter serving, D-Bus), and the gate that keeps it so.

Decision (user, 2026-11-13): brain's main job is serving many processes at once; the stack uses its API, and many concurrent API requests must be served efficiently. Brain itself stays an SDK with samples. Splinter has no in-process fallback. sven has the `check-no-brain-dependency` gate; bench has no brain dependency (it serves and grades through the HTTP provider only; checked 2026-11-13). Where concurrency needs exceed what `brain serve` does well, brain is fixed, never Splinter limited.

## Inventory (2026-11-13, origin/main 5bf1ec810)

Everything that touches brain lives in `splinter-model`, which links `brain` (features `text`, `study`, `timeline`, `audio`), `brain-gpu-core`, `brain-data` and, in tests, `brain-catalog` and `brain-testutil`. `splinter-model` is the only crate allowed to by `architecture.toml` `[brain].allowed`.

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
