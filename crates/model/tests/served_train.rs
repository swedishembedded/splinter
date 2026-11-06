// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems whose training runs as a
// job on a shared model service, for its clients. If your team needs
// expertise in moving fine-tuning off the application and onto a serving
// cluster, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: with brain endpoints a fine-tune and a policy-gradient run are jobs
//! of the service. The datasets go to the artifact store and are named by
//! digest; the job's frames are the steps; the adapter and the training
//! record come back by digest and are checked; a run the service has no room
//! for is asked again, a stream that breaks is resumed without losing or
//! repeating a step, a failure carries brain's reason, and a cancel stops the
//! job on the service.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::cell::{Cell, RefCell};
use std::path::{Path, PathBuf};
use std::time::Duration;

use splinter_core::digest::Digest;
use splinter_model::endpoint::BrainEndpoints;
use splinter_model::fixture::byte_level_checkpoint;
use splinter_model::policy_gradient::{PolicyGradientOptions, PolicyGradientTune, Reference};
use splinter_model::scripted_endpoint::{ScriptedEndpoint, SCRIPTED_KEY};
use splinter_model::scripted_jobs::JobScript;
use splinter_model::served_train::{fine_tune_served, train_policy_gradient_served, ArtifactStore};
use splinter_model::train::{BasePrecision, FineTune, StepHook, StepReport};
use splinter_model::PolicyError;

struct Rig {
    dir: tempfile::TempDir,
    model_dir: PathBuf,
    store: PathBuf,
}

/// A checkpoint directory brain's store would name `Qwen/Toy`, an artifact
/// store, and a training set of three conversations.
fn rig() -> Rig {
    let dir = tempfile::tempdir().unwrap();
    let model_dir = dir.path().join("models/Qwen/Toy");
    std::fs::create_dir_all(&model_dir).unwrap();
    byte_level_checkpoint(&model_dir);
    let store = dir.path().join("artifacts");
    std::fs::create_dir_all(&store).unwrap();
    Rig {
        dir,
        model_dir,
        store,
    }
}

fn records(path: &Path, advantage: bool) {
    let lines: Vec<String> = ["cab", "fed", "ace"]
        .iter()
        .enumerate()
        .map(|(i, prompt)| {
            let mut record = serde_json::json!({
                "messages": [
                    {"role": "user", "content": prompt, "train": false},
                    {"role": "assistant", "content": format!("answer {i}"), "train": true}
                ],
                "tools": []
            });
            if advantage {
                record["advantage"] = serde_json::json!(if i == 0 { 1.0 } else { -1.0 });
            }
            record.to_string()
        })
        .collect();
    std::fs::write(path, lines.join("\n")).unwrap();
}

fn endpoints(service: &ScriptedEndpoint, rig: &Rig) -> BrainEndpoints {
    BrainEndpoints::new(vec![service.url()], SCRIPTED_KEY)
        .unwrap()
        .with_artifact_store(ArtifactStore::open(&rig.store).unwrap())
        .with_refused_wait(Duration::from_millis(50))
}

fn fine_tune_request<'a>(
    rig: &'a Rig,
    attempt: &'a Path,
    train: &'a Path,
    held_out: &'a Path,
) -> FineTune<'a> {
    FineTune {
        model_dir: &rig.model_dir,
        train,
        held_out,
        held_out_text: None,
        monitor: None,
        eval_every: 0,
        patience: 0,
        attempt_dir: attempt,
        steps: 3,
        rank: 4,
        alpha: 8.0,
        replay: &[],
        replay_share: None,
        weighted_replay: &[],
        grad_accum: 2,
        continue_from: None,
        cancel: None,
        base: BasePrecision::Bf16,
        learning_rate: Some(2e-4),
        weight_decay: 0.01,
        // seed: the job's request carries it, and the spec reads it back.
        seed: 7,
        max_block: None,
        thinking: false,
        pack_answers: false,
        on_step: None,
        keep_evaluations: None,
    }
}

fn data_of(params: &serde_json::Value) -> serde_json::Value {
    serde_json::from_str(params["data"].as_str().unwrap()).unwrap()
}

#[test]
fn a_fine_tune_is_a_job_whose_data_goes_by_digest_and_whose_adapter_comes_back_checked() {
    let rig = rig();
    let (train, held_out, attempt) = (
        rig.dir.path().join("train.jsonl"),
        rig.dir.path().join("held.jsonl"),
        rig.dir.path().join("attempt-1"),
    );
    records(&train, false);
    records(&held_out, false);
    let service = ScriptedEndpoint::start(vec![]).with_training(JobScript::over(rig.store.clone()));
    let brain = endpoints(&service, &rig);
    let steps: RefCell<Vec<(u32, u32)>> = RefCell::new(Vec::new());
    let hook = |s: &StepReport| steps.borrow_mut().push((s.step, s.steps));
    let mut request = fine_tune_request(&rig, &attempt, &train, &held_out);
    request.on_step = Some(StepHook(&hook));

    let trained = fine_tune_served(&brain, &request).unwrap();

    // The service was asked for what the in-process trainer would have been.
    let jobs = service.jobs().created();
    assert_eq!(jobs.len(), 1);
    let params = &jobs[0];
    assert_eq!(params["base"], "Qwen/Toy");
    assert_eq!(params["objective"], "sft");
    assert_eq!(
        (
            params["steps"].as_u64(),
            params["rank"].as_u64(),
            params["grad_accum"].as_u64(),
            params["seed"].as_u64()
        ),
        (Some(3), Some(4), Some(2), Some(7))
    );
    assert_eq!(
        (params["base_format"].as_str(), params["thinking"].as_bool()),
        (Some("bf16"), Some(false))
    );
    assert!(params["max_block"].as_u64().unwrap() >= 64);
    assert_eq!(
        params["adapter_id"], "Qwen/Toy:splinter:candidate:attempt-1",
        "the candidate's card id is brain's adapter grammar over the base's id in its store"
    );
    // The datasets travelled by digest, and the store holds exactly them (and what the job published).
    let data = data_of(params);
    let train_digest = Digest::of(&std::fs::read(&train).unwrap());
    assert_eq!(data["train"]["digest"], train_digest.hex());
    assert!(rig.store.join(train_digest.hex()).is_file());
    assert!(data.get("path").is_none() && params.get("out_dir").is_none());

    // The steps were reported as they came, with the run's total.
    assert_eq!(*steps.borrow(), vec![(1, 3), (2, 3), (3, 3)]);
    // The adapter and the record are where the in-process trainer leaves them, and the adapter is the one published.
    assert_eq!(trained.adapter, attempt.join("adapter.safetensors"));
    assert_eq!(
        trained.adapter_digest,
        Digest::sha256_of_reader(std::fs::File::open(&trained.adapter).unwrap())
            .unwrap()
            .to_string()
    );
    assert!(trained.training_record.is_file());
    assert_eq!((trained.records, trained.block), (6, 128));
    assert_eq!(
        (trained.base.loss, trained.tuned.loss),
        (Some(2.0), Some(1.5))
    );
    assert_eq!(
        (
            trained.curve.steps,
            trained.curve.steps_completed,
            trained.curve.selected_step
        ),
        (3, 3, 3)
    );
}

#[test]
fn a_policy_gradient_run_sends_only_the_settings_named_and_reads_its_series_from_the_record() {
    let rig = rig();
    let (train, attempt) = (
        rig.dir.path().join("rollouts.jsonl"),
        rig.dir.path().join("attempt-pg"),
    );
    records(&train, true);
    let service = ScriptedEndpoint::start(vec![]).with_training(JobScript::over(rig.store.clone()));
    let brain = endpoints(&service, &rig);
    let request = PolicyGradientTune {
        model_dir: &rig.model_dir,
        train: &train,
        attempt_dir: &attempt,
        steps: 3,
        rank: 4,
        alpha: 8.0,
        weight_decay: 0.0,
        grad_accum: 1,
        continue_from: None,
        cancel: None,
        base: BasePrecision::Bf16,
        learning_rate: None,
        // seed: the job's request carries it.
        seed: 11,
        max_block: Some(256),
        thinking: true,
        options: PolicyGradientOptions {
            kl_beta: Some(0.1),
            reference: Some(Reference::StartOfRun),
            ..PolicyGradientOptions::default()
        },
    };

    let trained = train_policy_gradient_served(&brain, &request).unwrap();

    let params = &service.jobs().created()[0];
    assert_eq!(params["objective"], "policy_gradient");
    assert_eq!(
        (
            params["kl_beta"].as_f64().map(|v| (v * 10.0).round()),
            params["reference"].as_str()
        ),
        (Some(1.0), Some("start"))
    );
    assert!(
        params.get("clip_eps").is_none() && params.get("lr").is_none(),
        "brain's defaults are brain's to apply: {params}"
    );
    assert_eq!(params["max_block"], 256);
    assert_eq!(trained.records, 3);
    let kls: Vec<Option<f32>> = trained.statistics.steps.iter().map(|s| s.kl).collect();
    assert_eq!(kls.len(), 3, "every step of the record");
    assert_eq!(trained.statistics.first_kl, kls[0]);
    assert!(trained.adapter.is_file() && trained.training_record.is_file());
}

#[test]
fn a_run_the_service_has_no_room_for_is_asked_again_until_it_is_placed() {
    let rig = rig();
    let (train, held_out, attempt) = (
        rig.dir.path().join("t.jsonl"),
        rig.dir.path().join("h.jsonl"),
        rig.dir.path().join("a1"),
    );
    records(&train, false);
    records(&held_out, false);
    let mut script = JobScript::over(rig.store.clone());
    script.refuse_first = 2;
    let service = ScriptedEndpoint::start(vec![]).with_training(script);
    let trained = fine_tune_served(
        &endpoints(&service, &rig),
        &fine_tune_request(&rig, &attempt, &train, &held_out),
    );
    assert!(trained.is_ok(), "{trained:?}");
    assert_eq!(service.jobs().refusals(), 2);
    assert_eq!(service.jobs().created().len(), 1);
}

#[test]
fn a_stream_that_breaks_is_resumed_after_the_last_event_so_no_step_is_lost_or_repeated() {
    let rig = rig();
    let (train, held_out, attempt) = (
        rig.dir.path().join("t.jsonl"),
        rig.dir.path().join("h.jsonl"),
        rig.dir.path().join("a1"),
    );
    records(&train, false);
    records(&held_out, false);
    let mut script = JobScript::over(rig.store.clone());
    // Two stage frames and the first step, then the connection ends.
    script.break_stream_after = Some(3);
    let service = ScriptedEndpoint::start(vec![]).with_training(script);
    let seen: RefCell<Vec<u32>> = RefCell::new(Vec::new());
    let hook = |s: &StepReport| seen.borrow_mut().push(s.step);
    let mut request = fine_tune_request(&rig, &attempt, &train, &held_out);
    request.on_step = Some(StepHook(&hook));
    fine_tune_served(&endpoints(&service, &rig), &request).unwrap();
    assert_eq!(*seen.borrow(), vec![1, 2, 3]);
    assert_eq!(
        service.jobs().streams(),
        vec![None, Some(3)],
        "the second connection asked for what followed event 3"
    );
}

#[test]
fn a_job_that_fails_fails_the_run_with_brains_reason_and_a_request_it_refuses_is_not_retried() {
    let rig = rig();
    let (train, held_out, attempt) = (
        rig.dir.path().join("t.jsonl"),
        rig.dir.path().join("h.jsonl"),
        rig.dir.path().join("a1"),
    );
    records(&train, false);
    records(&held_out, false);
    let mut script = JobScript::over(rig.store.clone());
    script.fail_with = Some("train: the device failed during the run (see the server log)".into());
    let service = ScriptedEndpoint::start(vec![]).with_training(script);
    let error = fine_tune_served(
        &endpoints(&service, &rig),
        &fine_tune_request(&rig, &attempt, &train, &held_out),
    )
    .unwrap_err();
    assert!(
        matches!(&error, PolicyError::Train { reason, .. } if reason.contains("the device failed")),
        "{error}"
    );

    // A service that does not hold a published dataset refuses the job at the door (400): no retry.
    let blind = ScriptedEndpoint::start(vec![])
        .with_training(JobScript::over(rig.dir.path().join("elsewhere")));
    std::fs::create_dir_all(rig.dir.path().join("elsewhere")).unwrap();
    let error = fine_tune_served(
        &endpoints(&blind, &rig),
        &fine_tune_request(&rig, &attempt, &train, &held_out),
    )
    .unwrap_err();
    assert!(
        matches!(&error, PolicyError::Train { reason, .. } if reason.contains("not in the artifact store")),
        "{error}"
    );
    assert!(blind.jobs().created().is_empty());
}

#[test]
fn cancelling_stops_the_job_on_the_service() {
    let rig = rig();
    let (train, held_out, attempt) = (
        rig.dir.path().join("t.jsonl"),
        rig.dir.path().join("h.jsonl"),
        rig.dir.path().join("a1"),
    );
    records(&train, false);
    records(&held_out, false);
    let mut script = JobScript::over(rig.store.clone());
    script.hang = true;
    let service = ScriptedEndpoint::start(vec![]).with_training(script);
    let token = sven_sdk::CancelToken::default();
    let stepped = Cell::new(0u32);
    let hook = |s: &StepReport| {
        stepped.set(s.step);
        if s.step == 1 {
            token.cancel();
        }
    };
    let mut request = fine_tune_request(&rig, &attempt, &train, &held_out);
    request.on_step = Some(StepHook(&hook));
    request.cancel = Some(&token);
    let error = fine_tune_served(&endpoints(&service, &rig), &request).unwrap_err();
    assert!(matches!(error, PolicyError::Cancelled { .. }), "{error}");
    assert!(
        service.jobs().was_cancelled(),
        "the DELETE reached the service"
    );
    assert!(stepped.get() >= 1);
}

#[test]
fn training_on_an_endpoint_needs_the_artifact_store_and_says_which_flag() {
    let rig = rig();
    let (train, held_out, attempt) = (
        rig.dir.path().join("t.jsonl"),
        rig.dir.path().join("h.jsonl"),
        rig.dir.path().join("a1"),
    );
    records(&train, false);
    records(&held_out, false);
    let service = ScriptedEndpoint::start(vec![]).with_training(JobScript::over(rig.store.clone()));
    let bare = BrainEndpoints::new(vec![service.url()], SCRIPTED_KEY).unwrap();
    let error = fine_tune_served(&bare, &fine_tune_request(&rig, &attempt, &train, &held_out))
        .unwrap_err()
        .to_string();
    assert!(error.contains("--brain-artifact-store"), "{error}");
    assert!(service.jobs().created().is_empty());
}

#[test]
fn a_dataset_the_trainer_would_refuse_is_refused_here_before_anything_is_sent() {
    let rig = rig();
    let (train, held_out, attempt) = (
        rig.dir.path().join("t.jsonl"),
        rig.dir.path().join("h.jsonl"),
        rig.dir.path().join("a1"),
    );
    std::fs::write(&train, "{\"not\": \"a chat record\"}\n").unwrap();
    records(&held_out, false);
    let service = ScriptedEndpoint::start(vec![]).with_training(JobScript::over(rig.store.clone()));
    let error = fine_tune_served(
        &endpoints(&service, &rig),
        &fine_tune_request(&rig, &attempt, &train, &held_out),
    )
    .unwrap_err();
    assert!(matches!(error, PolicyError::Dataset { .. }), "{error}");
    assert!(service.jobs().created().is_empty());
    assert_eq!(
        std::fs::read_dir(&rig.store).unwrap().count(),
        0,
        "nothing was published for a run that cannot start"
    );
}
