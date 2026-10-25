// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible, auditable fine-tuning
// pipelines for its clients. If your team needs expertise in seeding and
// experiment provenance for model training, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec: a training run's seed is never implicit. A run given none draws one,
//! reports it, and records it in the run record and in the candidate; training
//! again with that seed writes the same adapter.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::sync::{Arc, Mutex};

use common::gate::{dataset, gate_context, policy, unscored, Brain, BASE_BYTES};
use splinter_agent::CancelToken;
use splinter_core::digest::Digest;
use splinter_core::training::TrainingCurve;
use splinter_model::train::{Trained, TrainedPreference};
use splinter_orchestrator::runs::record;
use splinter_orchestrator::{Context, OrchestratorError};
use splinter_pipelines::train::{
    train_recorded, Candidate, TrainPlan, TrainRequest, Trainer, Tuning, DEFAULT_REPLAY_FRACTION,
};
use splinter_store::runs::read_run;

/// A trainer whose adapter is a function of the seed it is handed, as a
/// deterministic fine-tune's is.
struct SeedOnly;

impl Trainer for SeedOnly {
    fn train(
        &self,
        _: &Context,
        plan: &TrainPlan,
        _: &CancelToken,
    ) -> Result<Trained, OrchestratorError> {
        let adapter = plan.dir.join("adapter.safetensors");
        let bytes = format!("adapter of seed {}", plan.seed);
        std::fs::write(&adapter, &bytes).unwrap();
        let record = plan.dir.join("training.json");
        std::fs::write(&record, r#"{"trainer":"seed-only"}"#).unwrap();
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

/// What one recorded `train` of `seed` left: the candidate, the run's
/// record of its seeds and the seed events reported.
fn trained(
    test: &str,
    seed: Option<u64>,
) -> (Candidate, Vec<(String, u64)>, usize, serde_json::Value) {
    let (_scratch, ctx) = gate_context(test, Brain::Honest);
    let reported = Arc::new(Mutex::new(0usize));
    let counter = Arc::clone(&reported);
    let ctx = ctx.with_progress(Box::new(move |stage, _| {
        if stage == "seed" {
            *counter.lock().unwrap() += 1;
        }
    }));
    let data = dataset(&ctx, "alpha", 20).to_string();
    let request = TrainRequest {
        datasets: vec![data],
        rehearsal: None,
        from: policy(),
        replay_fraction: DEFAULT_REPLAY_FRACTION,
        replay_policy: Default::default(),
        steps: Some(3),
        rank: 4,
        beta: None,
        tuning: Tuning {
            seed,
            ..Tuning::default()
        },
    };
    let recorded = record(&ctx, "train", &request, |run| {
        train_recorded(&ctx, run, &request, &SeedOnly)
    })
    .unwrap();
    let record = read_run(ctx.workspace(), &recorded.run).unwrap();
    let seeds = record.seeds.into_iter().collect();
    let outputs = record.outputs;
    let count = *reported.lock().unwrap();
    (recorded.report, seeds, count, outputs)
}

#[test]
fn an_unseeded_train_records_the_seed_it_drew_and_a_rerun_with_it_writes_the_same_adapter() {
    let (first, seeds, reports, _) = trained("train-seed-drawn", None);
    let drawn = first
        .seed
        .expect("the candidate records the seed it trained with");
    assert_eq!(
        seeds,
        [("seed".to_string(), drawn)],
        "the run record states it"
    );
    assert_eq!(reports, 1, "reported once, as it is drawn");
    assert_eq!(
        first.adapter_digest,
        Digest::sha256_of(format!("adapter of seed {drawn}").as_bytes()).to_string()
    );

    let (again, again_seeds, _, _) = trained("train-seed-given", Some(drawn));
    assert_eq!(again.seed, Some(drawn));
    assert_eq!(again_seeds, seeds, "a given seed is recorded the same way");
    assert_eq!(again.adapter_digest, first.adapter_digest);

    let (other, _, _, _) = trained("train-seed-other", None);
    assert_ne!(other.seed, first.seed, "an unseeded run draws afresh");
}

#[test]
fn a_training_run_records_the_executable_that_trained_it() {
    let (candidate, _, _, outputs) = trained("train-identity", Some(7));
    let exe = std::env::current_exe().unwrap();
    let expected = {
        let file = std::fs::File::open(&exe).unwrap();
        Digest::sha256_of_reader(std::io::BufReader::new(file)).unwrap()
    };
    assert_eq!(
        candidate.binary.path.as_deref(),
        Some(exe.display().to_string().as_str())
    );
    assert_eq!(candidate.binary.digest, Some(expected.clone()));
    assert!(candidate.binary.version.is_some());
    assert_eq!(
        outputs["binary"]["digest"],
        serde_json::json!(expected.to_string()),
        "the run's JSON names the same binary"
    );
}
