// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that prove a training run
// used the cards it was given, for its clients. If your team needs expertise
// in verifying multi-card and multi-node training, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: a served fine-tune records the `parallel` object of the service's
//! report beside what it asked for, flags a run that trained on fewer ranks
//! than requested, and records a report without the object (an older brain)
//! as not reported. A full fine-tune is spread over the cards the same way.

#![allow(clippy::unwrap_used)]

use splinter_model::consolidate::FullTune;
use splinter_model::endpoint::TrainSpan;
use splinter_model::scripted_endpoint::ScriptedEndpoint;
use splinter_model::scripted_jobs::JobScript;
use splinter_model::served_train::{
    fine_tune_served, full_fine_tune_served, train_preference_served, Install,
};
use splinter_model::train::{PairLoss, PreferenceTune};

#[path = "support/served.rs"]
mod served;
use served::{endpoints, fine_tune_request, pairs, records, rig};

/// What brain's `parallel` object says for `world` ranks of this server's `[0, ranks)`.
fn parallel_report(world: u32, ranks: u32, collective: &str) -> serde_json::Value {
    serde_json::json!({
        "world_size": world, "ranks": [0, ranks],
        "devices": (0..ranks).map(|r| serde_json::json!({"rank": r, "device": format!("gpu{r}"), "name": "GH200"})).collect::<Vec<_>>(),
        "collective": collective, "transport": "in_process", "network": null, "sharding": "replicated"
    })
}

fn parallelism_of(
    parallel: Option<serde_json::Value>,
    span: TrainSpan,
) -> splinter_core::training::ParallelRecord {
    let rig = rig();
    let (train, held_out, attempt) = (
        rig.dir.path().join("train.jsonl"),
        rig.dir.path().join("held.jsonl"),
        rig.dir.path().join("attempt-1"),
    );
    records(&train, false);
    records(&held_out, false);
    let service = ScriptedEndpoint::start(vec![]).with_training(JobScript {
        parallel,
        ..JobScript::over(rig.store.clone())
    });
    let brain = endpoints(&service, &rig).with_train_span(span);
    let request = fine_tune_request(&rig, &attempt, &train, &held_out);
    fine_tune_served(&brain, &request)
        .unwrap()
        .curve
        .parallel
        .unwrap()
}

#[test]
fn a_runs_parallelism_is_recorded_and_checked_against_the_cards_asked_for() {
    let matching = parallelism_of(Some(parallel_report(2, 2, "nccl")), TrainSpan::Cards(2));
    assert!(matching.warnings().is_empty(), "{matching:?}");
    assert_eq!(matching.reported.unwrap().devices.len(), 2);

    // Two cards asked for; one trained on the host transport: a failed expectation.
    let short = parallelism_of(Some(parallel_report(1, 1, "host")), TrainSpan::Cards(2));
    assert!(short.mismatches.len() >= 2, "{short:?}");
    assert!(short
        .warnings()
        .iter()
        .any(|w| w.contains("asked for 2 rank(s) and the run trained on 1")));

    // An older brain: not reported, which is not one rank.
    let old = parallelism_of(None, TrainSpan::Cards(2));
    assert!(
        old.reported.is_none() && old.mismatches.is_empty(),
        "{old:?}"
    );
    assert!(old.warnings()[0].contains("not reported"));
}

/// A full fine-tune spreads over the cards the endpoints say, as a `train`
/// job does, and its checkpoint carries the parallelism checked against them.
#[test]
fn a_full_fine_tune_spreads_over_the_cards_asked_for_and_records_its_parallelism() {
    let rig = rig();
    let (train, attempt) = (
        rig.dir.path().join("train.jsonl"),
        rig.dir.path().join("attempt-full"),
    );
    records(&train, false);
    let models = rig.dir.path().join("models");
    let service = ScriptedEndpoint::start(vec![]).with_training(JobScript {
        models: Some(models.clone()),
        parallel: Some(parallel_report(4, 4, "nccl")),
        ..JobScript::over(rig.store.clone())
    });
    let brain = endpoints(&service, &rig).with_train_span(TrainSpan::Cards(4));
    let full = FullTune {
        model_dir: &rig.model_dir,
        train: &train,
        replay: &[],
        replay_share: None,
        weighted_replay: &[],
        grad_accum: 1,
        attempt_dir: &attempt,
        steps: 3,
        learning_rate: None,
        weight_decay: 0.0,
        train_precision: None,
        // seed: the job's request carries it.
        seed: 9,
        thinking: false,
        cancel: None,
        on_step: None,
    };
    let install = Install {
        models_dir: &models,
        name: "tuned-4",
    };
    let checkpoint = full_fine_tune_served(&brain, &full, &install).unwrap();
    let params = service.jobs().created().last().cloned().unwrap();
    assert_eq!(
        params["cards"], 4,
        "the run asks for the four cards: {params}"
    );
    let parallel = checkpoint.parallel.expect("the parallelism is recorded");
    assert!(parallel.warnings().is_empty(), "{parallel:?}");
    assert_eq!(parallel.reported.unwrap().devices.len(), 4);
}

/// Preference optimization trains as ranks too (brain shards the pairs): the
/// run is spread over the cards the endpoints say, as a supervised one is.
#[test]
fn a_preference_run_spreads_over_the_cards_asked_for() {
    let rig = rig();
    let (train, held_out, attempt) = (
        rig.dir.path().join("pairs.jsonl"),
        rig.dir.path().join("held-pairs.jsonl"),
        rig.dir.path().join("attempt-dpo"),
    );
    pairs(&train);
    pairs(&held_out);
    let service = ScriptedEndpoint::start(vec![]).with_training(JobScript::over(rig.store.clone()));
    let brain = endpoints(&service, &rig).with_train_span(TrainSpan::Cards(2));
    let request = PreferenceTune {
        model_dir: &rig.model_dir,
        train: &train,
        held_out: &held_out,
        attempt_dir: &attempt,
        steps: 3,
        rank: 4,
        alpha: 8.0,
        beta: 0.1,
        nll_weight: 0.0,
        loss: PairLoss::Dpo,
        gamma: None,
        grad_accum: 1,
        learning_rate: None,
        // seed: the job's request carries it.
        seed: 5,
        continue_from: None,
        cancel: None,
    };
    train_preference_served(&brain, &request).unwrap();
    let params = service.jobs().created().last().cloned().unwrap();
    assert_eq!(
        (&params["objective"], &params["cards"]),
        (&serde_json::json!("preference"), &serde_json::json!(2)),
        "the preference run asks for both cards: {params}"
    );
}
