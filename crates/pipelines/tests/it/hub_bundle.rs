// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements distribution of curated training data,
// adapters and evaluation verdicts as immutable, verifiable versions, for its
// clients. If your team needs expertise in sharing learning artifacts between
// machines without losing their provenance, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec: a bundle published to a private Hugging Face dataset repository and
//! ingested back at a pinned revision, against a local emulator of the Hub.
//!
//! * Publish then ingest round-trips: the set, its experiences, the dataset
//!   (its records byte for byte), the candidate and its adapter, and the
//!   verdict arrive under the ids they had.
//! * Publishing the same bundle again commits nothing.
//! * A file whose bytes are not the manifest's is refused, and nothing of
//!   the bundle is stored.
//! * Sealed or item-level material is refused before anything is sent.
//! * A repository the Hub does not state private is refused before any
//!   upload.
//! * A branch (or tag) revision is resolved to the commit it names.
//! * Ingesting again stores nothing new.
//! * A set from a state in the former loose layout, large enough that the
//!   staged database passes a checkpoint, round-trips too.

#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use ed25519_dalek::{Signer as _, SigningKey};
use serde_json::{json, Value};
use splinter_agent::CancelToken;
use splinter_core::digest::Digest;
use splinter_core::training::TrainingCurve;
use splinter_eval::promotion::canonical::canonical_bytes;
use splinter_model::train::{Trained, TrainedPreference};
use splinter_orchestrator::runs::record;
use splinter_orchestrator::{Context, OrchestratorError};
use splinter_pipelines::bench::ingest_bench_trials;
use splinter_pipelines::datasets::build;
use splinter_pipelines::hub::{ingest, publish, Hub, PublishRequest};
use splinter_pipelines::train::{
    candidate_record, load_candidate, train_recorded, TrainPlan, TrainRequest, Trainer, Tuning,
    DEFAULT_REPLAY_FRACTION,
};
use splinter_store::experiences::SetId;

use crate::common::bench::{context, export, sft_request, Trial, NO_OPTIONS};
use crate::common::gate::{dataset, gate_context, policy, unscored, Brain, BASE_BYTES};
use crate::common::hub::{Emulator, Quirks};
use crate::common::Scratch;

const REPO: &str = "lab/evidence";

/// A trainer whose adapter is a function of the seed, and large enough to go
/// to the Hub in parts.
struct SeedOnly;

impl Trainer for SeedOnly {
    fn train(
        &self,
        _: &Context,
        plan: &TrainPlan,
        _: &CancelToken,
    ) -> Result<Trained, OrchestratorError> {
        let adapter = plan.dir.join("adapter.safetensors");
        let bytes = format!("adapter of seed {}\n", plan.seed).repeat(1000);
        std::fs::write(&adapter, &bytes).unwrap();
        let record = plan.dir.join("training.json");
        std::fs::write(
            &record,
            r#"{"trainer":"seed-only","tokenizer_digest":"sha256:feed"}"#,
        )
        .unwrap();
        Ok(Trained {
            adapter,
            adapter_digest: Digest::sha256_of(bytes.as_bytes()).to_string(),
            base_digest: Digest::sha256_of(BASE_BYTES).to_string(),
            training_record: record,
            records: 0,
            block: 0,
            base: unscored(),
            tuned: unscored(),
            curve: TrainingCurve::unmonitored(plan.steps, plan.steps),
            evaluations: Vec::new(),
        })
    }

    fn train_preference(
        &self,
        _: &Context,
        _: &TrainPlan,
        _: &CancelToken,
    ) -> Result<TrainedPreference, OrchestratorError> {
        Err(OrchestratorError::Refused("supervised only".into()))
    }
}

/// The examiner stand-in's key.
fn examiner() -> SigningKey {
    SigningKey::from_bytes(&[42; 32])
}

/// `doc` signed as the examiner signs a verdict.
fn signed(mut doc: Value) -> Value {
    let bytes = canonical_bytes(&doc).unwrap();
    let value: String = examiner()
        .sign(&bytes)
        .to_bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    doc["signature"] = json!({"alg": "ed25519", "key_id": "test", "value": value});
    doc
}

fn write_json(path: &Path, value: &Value) -> PathBuf {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
    path.to_path_buf()
}

/// A source state root holding a bench train set and its eval set, an
/// sft-step dataset of the train set, a candidate trained on a dataset of
/// its own, a signed aggregate verdict and the examiner's key.
struct Source {
    scratch: Scratch,
    ctx: Context,
    train: SetId,
    eval: SetId,
    dataset: String,
    candidate: String,
    verdict: PathBuf,
    key: PathBuf,
}

fn source(test: &str) -> Source {
    let (scratch, ctx) = gate_context(test, Brain::Honest);
    let dir = export(
        &scratch.0.join("export"),
        &[
            Trial::new("t-pass", "org/a", Some("train"), true),
            Trial::new("t-fail", "org/a", Some("train"), false),
            Trial::new("e-pass", "org/b", Some("eval"), true),
        ],
    );
    let done = ingest_bench_trials(&ctx, &[dir], NO_OPTIONS).unwrap();
    let (train, eval) = (done.train_set.unwrap(), done.eval_set.unwrap());
    let built = build(&ctx, &sft_request(train.clone())).unwrap();
    let request = TrainRequest {
        datasets: vec![dataset(&ctx, "alpha", 20).to_string()],
        rehearsal: None,
        from: policy(),
        replay_fraction: DEFAULT_REPLAY_FRACTION,
        replay_policy: Default::default(),
        steps: Some(3),
        rank: 4,
        beta: None,
        tuning: Tuning::default(),
    };
    let candidate = record(&ctx, "train", &request, |run| {
        train_recorded(&ctx, run, &request, &SeedOnly)
    })
    .unwrap()
    .report
    .candidate;
    let verdict = write_json(
        &scratch.0.join("verdicts/verdict.json"),
        &signed(json!({
            "schema": "bench.verdict/3",
            "overall": {"candidate_pass_rate": 0.55, "base_pass_rate": 0.03},
        })),
    );
    let key = scratch.0.join("verdicts/verdict.key.pub");
    let hex: String = examiner()
        .verifying_key()
        .to_bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    std::fs::write(&key, hex).unwrap();
    Source {
        scratch,
        ctx,
        train,
        eval,
        dataset: built.dataset.to_string(),
        candidate,
        verdict,
        key,
    }
}

impl Source {
    fn request(&self) -> PublishRequest {
        PublishRequest {
            repo: REPO.into(),
            sets: vec![self.train.to_string()],
            datasets: vec![self.dataset.clone()],
            candidates: vec![self.candidate.clone()],
            verdicts: vec![self.verdict.clone()],
            examiner_key: Some(self.key.clone()),
        }
    }
}

fn hub(emulator: &Emulator) -> Hub {
    Hub::new(emulator.url(), Some("test-token".into())).unwrap()
}

#[test]
fn a_published_bundle_ingests_back_under_the_same_ids_and_once() {
    let src = source("hub-roundtrip-src");
    let emulator = Emulator::start(Quirks::default());
    let published = publish(&src.ctx, &hub(&emulator), &src.request()).unwrap();
    assert!(published.committed);
    assert!(published.lfs_uploaded >= 2, "{published:?}");
    assert_eq!(emulator.commits(REPO), 1);

    // Again: the tag names it, nothing is sent.
    let again = publish(&src.ctx, &hub(&emulator), &src.request()).unwrap();
    assert!(!again.committed, "republishing is a no-op");
    assert_eq!(again.commit, published.commit);
    assert_eq!(again.content, published.content);
    assert_eq!(emulator.commits(REPO), 1);

    // A branch is resolved to the commit it names.
    let (_dst_scratch, dst) = context("hub-roundtrip-dst");
    let ingested = ingest(&dst, &hub(&emulator), &format!("{REPO}@main")).unwrap();
    assert!(ingested.resolved);
    assert_eq!(ingested.commit, published.commit);
    assert_eq!(ingested.content, published.content);

    // The set, its members and every experience, under the same ids.
    assert_eq!(ingested.experiences.sets, vec![src.train.clone()]);
    let theirs = src.ctx.experiences();
    let ours = dst.experiences();
    let members = theirs.get_set(&src.train).unwrap().members;
    assert_eq!(ours.get_set(&src.train).unwrap().members, members);
    for id in &members {
        assert_eq!(ours.get(id).unwrap(), theirs.get(id).unwrap());
    }
    // The datasets byte for byte, under the same ids.
    let mut expected = vec![src.dataset.clone()];
    let candidate = load_candidate(&src.ctx, &src.candidate).unwrap();
    expected.extend(candidate.datasets.iter().map(|d| d.0.to_string()));
    for id in &expected {
        let a = splinter_pipelines::datasets::resolve_dataset(&src.ctx, id).unwrap();
        let b = splinter_pipelines::datasets::resolve_dataset(&dst, id).unwrap();
        assert_eq!(a.id, b.id);
        assert_eq!(
            std::fs::read(&a.path).unwrap(),
            std::fs::read(&b.path).unwrap()
        );
    }
    assert_eq!(ingested.datasets_new, expected.len());
    // The candidate, its record and its adapter.
    assert_eq!(ingested.candidates, vec![src.candidate.clone()]);
    assert_eq!(
        candidate_record(&dst, &src.candidate).unwrap(),
        candidate_record(&src.ctx, &src.candidate).unwrap()
    );
    let adopted = load_candidate(&dst, &src.candidate).unwrap();
    assert_eq!(
        std::fs::read(&adopted.adapter).unwrap(),
        std::fs::read(&candidate.adapter).unwrap()
    );
    assert_eq!(ingested.verdicts, 1);
    // An exam register comes along only for what was published.
    assert!(splinter_pipelines::bench::split::trials(&dst)
        .unwrap()
        .iter()
        .all(|r| members.contains(&r.experience)));

    // At the commit itself, and again: nothing new.
    let pinned = format!("{REPO}@{}", published.commit);
    let twice = ingest(&dst, &hub(&emulator), &pinned).unwrap();
    assert!(!twice.resolved);
    assert_eq!(twice.experiences.experiences_new, 0);
    assert_eq!(twice.experiences.annotations_new, 0);
    assert_eq!(twice.datasets_new, 0);
    assert_eq!(twice.artifacts_new, 0);
    assert_eq!(twice.record, ingested.record);

    // The tag names the same commit.
    let tagged = ingest(&dst, &hub(&emulator), &format!("{REPO}@{}", published.tag)).unwrap();
    assert!(tagged.resolved);
    assert_eq!(tagged.commit, published.commit);
    drop(src.scratch);
}

#[test]
fn a_tampered_file_is_refused_and_nothing_of_the_bundle_is_stored() {
    let src = source("hub-tamper-src");
    let emulator = Emulator::start(Quirks::default());
    let mut request = src.request();
    request.candidates.clear();
    request.verdicts.clear();
    publish(&src.ctx, &hub(&emulator), &request).unwrap();
    let id = splinter_pipelines::datasets::resolve_dataset(&src.ctx, &src.dataset)
        .unwrap()
        .id;
    emulator.tamper(REPO, &format!("datasets/{}.jsonl", id.0.hex()));
    let (_dst_scratch, dst) = context("hub-tamper-dst");
    let err = ingest(&dst, &hub(&emulator), &format!("{REPO}@main"))
        .unwrap_err()
        .to_string();
    assert!(err.contains("not what was expected"), "{err}");
    assert!(dst.datasets().list().unwrap().is_empty());
    assert!(dst.experiences().list_sets().unwrap().is_empty());
}

/// What the emulator saw that would have sent material: an upload or a
/// commit.
fn uploads(emulator: &Emulator) -> Vec<String> {
    emulator
        .log()
        .into_iter()
        .filter(|l| {
            l.contains("/preupload/")
                || l.contains("/lfs/")
                || l.contains("/commit/")
                || l.contains("lfs-")
        })
        .collect()
}

#[test]
fn sealed_and_item_level_material_is_refused_before_anything_is_sent() {
    let src = source("hub-sealed-src");
    let emulator = Emulator::start(Quirks::default());
    let refused = |request: &PublishRequest, expected: &str| {
        let err = publish(&src.ctx, &hub(&emulator), request)
            .unwrap_err()
            .to_string();
        assert!(err.starts_with("withheld:"), "{err}");
        assert!(err.contains(expected), "{expected}: {err}");
    };
    // Examination trials.
    let mut exam = src.request();
    exam.sets = vec![src.eval.to_string()];
    refused(&exam, "examination material");
    // A verdict from the examiner's private directory, never opened.
    let mut private = src.request();
    private.verdicts = vec![src.scratch.0.join("bench-private/verdict.json")];
    refused(&private, "private material");
    // An item-level file.
    let mut items = src.request();
    items.verdicts = vec![write_json(
        &src.scratch.0.join("verdicts/exam.items.json"),
        &json!({}),
    )];
    refused(&items, "item-level file");
    // A verdict carrying its items.
    let mut item_level = src.request();
    item_level.verdicts = vec![write_json(
        &src.scratch.0.join("verdicts/per-task.json"),
        &signed(json!({"schema": "bench.verdict/3", "overall": {"items": [{"task": "x"}]}})),
    )];
    refused(&item_level, "item-level field");
    // An unsigned one.
    let mut unsigned = src.request();
    unsigned.verdicts = vec![write_json(
        &src.scratch.0.join("verdicts/unsigned.json"),
        &json!({"schema": "bench.verdict/3", "overall": {}}),
    )];
    refused(&unsigned, "not a signed aggregate verdict");
    // Material beside a sealed export.
    let sealed_dir = src.scratch.0.join("sealed");
    write_json(&sealed_dir.join("sealed-export.json"), &json!({}));
    let mut sealed = src.request();
    sealed.verdicts = vec![write_json(&sealed_dir.join("verdict.json"), &json!({}))];
    refused(&sealed, "sealed export");
    assert!(emulator.log().is_empty(), "the hub was not even asked");
}

#[test]
fn a_repository_not_stated_private_is_refused_before_any_upload() {
    let src = source("hub-private-src");
    for quirks in [
        Quirks {
            no_private_flag: true,
            ..Quirks::default()
        },
        Quirks {
            public: true,
            ..Quirks::default()
        },
    ] {
        let emulator = Emulator::start(quirks);
        let err = publish(&src.ctx, &hub(&emulator), &src.request())
            .unwrap_err()
            .to_string();
        assert!(err.contains("nothing was uploaded"), "{err}");
        assert!(uploads(&emulator).is_empty(), "{:?}", emulator.log());
        assert!(
            emulator.log().iter().any(|l| l == "POST /api/repos/create"),
            "the repository was created, asking for privacy"
        );
    }
}

/// Trials enough that the staged database's history passes a checkpoint.
const MANY_TRIALS: usize = 80;

#[test]
fn a_large_set_from_a_loose_state_round_trips_past_a_checkpoint() {
    use splinter_expdb::backend::Layout;
    use splinter_expdb::{Config, Database};
    let scratch = Scratch::new("hub-loose-src");
    let root = scratch.state();
    std::fs::create_dir_all(root.expdb()).unwrap();
    drop(Database::open_as(root.expdb(), Layout::Loose, Config::default()).unwrap());
    let src = Context::new(crate::common::config(&scratch), false).unwrap();
    let trials: Vec<Trial> = (0..MANY_TRIALS)
        .map(|i| {
            // The fixture takes names for the life of the test binary.
            let name: &'static str = Box::leak(format!("t-{i}").into_boxed_str());
            let task: &'static str = Box::leak(format!("org/t{i}").into_boxed_str());
            Trial::new(name, task, Some("train"), i % 2 == 0)
        })
        .collect();
    let dir = export(&scratch.0.join("export"), &trials);
    let train = ingest_bench_trials(&src, &[dir], NO_OPTIONS)
        .unwrap()
        .train_set
        .unwrap();
    assert_eq!(Layout::of_root(&root.expdb()).unwrap(), Some(Layout::Loose));

    let emulator = Emulator::start(Quirks::default());
    let request = PublishRequest {
        repo: REPO.into(),
        sets: vec![train.to_string()],
        ..PublishRequest::default()
    };
    let published = publish(&src, &hub(&emulator), &request).unwrap();
    let (_dst_scratch, dst) = context("hub-loose-dst");
    let ingested = ingest(&dst, &hub(&emulator), &format!("{REPO}@{}", published.tag)).unwrap();
    assert_eq!(ingested.experiences.sets, vec![train.clone()]);
    let members = src.experiences().get_set(&train).unwrap().members;
    assert_eq!(members.len(), MANY_TRIALS);
    assert_eq!(dst.experiences().get_set(&train).unwrap().members, members);
    for id in &members {
        assert_eq!(
            dst.experiences().get(id).unwrap(),
            src.experiences().get(id).unwrap()
        );
    }
}
