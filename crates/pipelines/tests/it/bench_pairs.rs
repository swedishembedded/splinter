// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that turn graded agent attempts
// into preference and feedback training data for its clients. If your team
// needs expertise in learning from verified successes and failures of agents
// then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec, end to end: the graded attempts of a benchmark's practice tasks,
//! exported and ingested as every trial is, are built into a preference
//! dataset (a verified pass over a verified fail of one task, from where they
//! part) and a feedback dataset (each verified attempt judged), with the
//! infrastructure failure left out of both and the pairing policy and class
//! weights in the manifests; building again gives the same datasets; `train`
//! trains each by its own objective, the feedback run at the weights its
//! manifest records, and refuses a setting of the other objective by name.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use crate::common;

use common::bench::{export, sft_request, Trial, NO_OPTIONS};
use common::gate::{gate_context, policy, Brain, FakeTrainer, ANCHOR};
use serde_json::{json, Value};
use splinter_agent::CancelToken;
use splinter_core::training::Regime;
use splinter_data::Exclusion;
use splinter_model::train::PairLoss;
use splinter_pipelines::bench::ingest_bench_trials;
use splinter_pipelines::datasets::{
    build, BuildRequest, RlBuild, ViewName, DEFAULT_PAIRS_PER_GROUP,
};
use splinter_pipelines::train::{
    train, PairSettings, TrainRequest, Tuning, DEFAULT_REPLAY_FRACTION,
};
use splinter_store::experiences::SetId;

/// Practice attempts at one task, all checked out at one commit: a pass, two
/// fails that went another way, and a trial the model server failed.
fn practice(dir: &std::path::Path) -> std::path::PathBuf {
    let trial = |name, pass, command| {
        Trial::new(name, "org/books", Some("train"), pass)
            .at_commit("c0ffee")
            .command(command)
            .scored()
    };
    let crashed = Trial::new("crashed", "org/books", Some("train"), false)
        .at_commit("c0ffee")
        .command("pwd")
        .score(json!({
            "schema": "bench.score/1", "status": "model_error", "counted": false,
            "score": null, "pass": null, "cause": {"code": "server_error", "message": "502"},
        }));
    export(
        dir,
        &[
            trial("p1", true, "make test"),
            trial("f1", false, "ls"),
            trial("f2", false, "cat main.c"),
            crashed,
        ],
    )
}

fn request(set: SetId, view: ViewName) -> BuildRequest {
    BuildRequest {
        view,
        rl: RlBuild::default(),
        ..sft_request(set)
    }
}

fn lines(path: &std::path::Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn train_request(dataset: &str, pairs: PairSettings) -> TrainRequest {
    TrainRequest {
        datasets: vec![dataset.to_string()],
        rehearsal: None,
        from: policy(),
        replay_fraction: DEFAULT_REPLAY_FRACTION,
        replay_policy: Default::default(),
        steps: Some(1),
        rank: 4,
        beta: None,
        tuning: Tuning {
            pairs,
            ..Tuning::default()
        },
    }
}

#[test]
fn practice_attempts_become_preference_and_feedback_datasets_that_train_by_their_objectives() {
    let (scratch, ctx) = gate_context("bench-pairs", Brain::Honest);
    let done =
        ingest_bench_trials(&ctx, &[practice(&scratch.0.join("export"))], NO_OPTIONS).unwrap();
    assert!(done.refused.is_empty(), "{:?}", done.refused);
    let set = done.train_set.unwrap();

    let pairs = build(&ctx, &request(set.clone(), ViewName::Preference)).unwrap();
    assert_eq!(pairs.view, "preference");
    assert_eq!(pairs.records, 2, "{:?}", pairs.excluded);
    assert_eq!(
        pairs.excluded.get(&Exclusion::InfrastructureFailure),
        Some(&1)
    );
    for line in lines(&pairs.path) {
        assert_eq!(line["metadata"]["rollout"]["rollouts"][0], "p1", "{line}");
        assert_ne!(line["metadata"]["rollout"]["rollouts"][1], "crashed");
        assert_eq!(
            line["prompt"].as_array().unwrap().last().unwrap()["role"],
            "user"
        );
    }
    let stored = ctx.datasets().get(&pairs.dataset).unwrap();
    let pairing = stored.manifest.pairing.clone().unwrap();
    assert_eq!(
        (pairing.attempts, pairing.per_group),
        (2, Some(DEFAULT_PAIRS_PER_GROUP))
    );
    assert!(pairing.seed.is_some(), "the seed the pairs were chosen by");
    let again = build(&ctx, &request(set.clone(), ViewName::Preference)).unwrap();
    assert_eq!(
        again.dataset, pairs.dataset,
        "the seed is derived from the experiences"
    );

    let judged = build(&ctx, &request(set.clone(), ViewName::Feedback)).unwrap();
    assert_eq!(judged.records, 3, "{:?}", judged.excluded);
    assert_eq!(
        judged.excluded.get(&Exclusion::InfrastructureFailure),
        Some(&1)
    );
    let desirable: Vec<bool> = lines(&judged.path)
        .iter()
        .map(|l| l["desirable"].as_bool().unwrap())
        .collect();
    assert_eq!(desirable, [true, false, false]);
    let balance = ctx
        .datasets()
        .get(&judged.dataset)
        .unwrap()
        .manifest
        .feedback
        .unwrap();
    assert_eq!((balance.desirable, balance.undesirable), (1, 2));
    assert!((balance.desirable_weight - 2.0).abs() < 1e-12);

    // Pairing settings belong to the preference view.
    let wrong = BuildRequest {
        rl: RlBuild {
            pairs_per_group: Some(1),
            ..RlBuild::default()
        },
        ..request(set, ViewName::Feedback)
    };
    assert!(build(&ctx, &wrong)
        .unwrap_err()
        .to_string()
        .contains("--pairs-per-group"));

    // Each trains by its own objective.
    let trainer = FakeTrainer::knowing(&[ANCHOR]);
    let feedback = train(
        &ctx,
        &train_request(&judged.dataset.to_string(), PairSettings::default()),
        &trainer,
        &CancelToken::new(),
    )
    .unwrap();
    assert_eq!(feedback.regime, Regime::Feedback);
    let summary = feedback.feedback.unwrap();
    assert_eq!(
        (summary.desirable_weight, summary.undesirable_weight),
        (2.0, 1.0)
    );
    let simpo = PairSettings {
        loss: Some(PairLoss::Simpo),
        gamma: Some(0.5),
        ..PairSettings::default()
    };
    let preference = train(
        &ctx,
        &train_request(&pairs.dataset.to_string(), simpo),
        &trainer,
        &CancelToken::new(),
    )
    .unwrap();
    assert_eq!(preference.regime, Regime::Dpo);
    assert_eq!(
        preference.preference.unwrap().loss.as_deref(),
        Some("simpo")
    );
    assert_eq!(
        *trainer.called.lock().unwrap(),
        [Regime::Feedback, Regime::Dpo]
    );

    let refused = |dataset: &str, pairs: PairSettings| {
        let error = train(
            &ctx,
            &train_request(dataset, pairs),
            &trainer,
            &CancelToken::new(),
        )
        .unwrap_err();
        assert!(error.is_refusal(), "{error}");
        error.to_string()
    };
    let loss = PairSettings {
        loss: Some(PairLoss::Ipo),
        ..PairSettings::default()
    };
    assert!(refused(&judged.dataset.to_string(), loss).contains("--loss"));
    let weight = PairSettings {
        desirable_weight: Some(1.5),
        ..PairSettings::default()
    };
    assert!(refused(&pairs.dataset.to_string(), weight).contains("--desirable-weight"));
    let margin = PairSettings {
        gamma: Some(0.5),
        ..PairSettings::default()
    };
    assert!(refused(&pairs.dataset.to_string(), margin).contains("--loss simpo"));
}
