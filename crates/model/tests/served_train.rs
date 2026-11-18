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
//! job on the service. A dataset check, a preference run, a full-parameter
//! run, a merge and a teacher-forced score go through the same service.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::cell::{Cell, RefCell};
use std::path::{Path, PathBuf};
use std::time::Duration;

use splinter_core::digest::Digest;
use splinter_model::consolidate::FullTune;
use splinter_model::endpoint::BrainEndpoints;
use splinter_model::policy_gradient::{PolicyGradientOptions, PolicyGradientTune, Reference};
use splinter_model::scripted_endpoint::{Card, ScriptedEndpoint, SCRIPTED_KEY};
use splinter_model::scripted_jobs::JobScript;
use splinter_model::served_train::{
    check_dataset, fine_tune_served, full_fine_tune_served, merge_served, score_chat,
    train_policy_gradient_served, train_preference_served, ArtifactStore, CheckSpec, Install,
};
use splinter_model::train::{BasePrecision, FineTune, PreferenceTune, StepHook, StepReport};
use splinter_model::LocalWeights;
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
    let store = dir.path().join("artifacts");
    std::fs::create_dir_all(&store).unwrap();
    Rig {
        dir,
        model_dir,
        store,
    }
}

fn records(path: &Path, rollouts: bool) {
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
            if rollouts {
                record["reward"] = serde_json::json!(if i == 0 { 1.0 } else { 0.0 });
                record["group"] = serde_json::json!("g");
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
fn a_dataset_the_trainer_refuses_is_named_by_line_before_a_run_is_asked_for() {
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
    assert!(error.to_string().contains("line 1"), "{error}");
    assert!(service.jobs().created().is_empty(), "no run was asked for");
    assert_eq!(service.jobs().actions(), vec!["check_dataset"]);
}

fn pairs(path: &Path) {
    let lines: Vec<String> = ["a", "bb", "ccc"]
        .iter()
        .map(|p| {
            serde_json::json!({
                "messages": [{"role": "user", "content": p}],
                "chosen": {"role": "assistant", "content": format!("{p} yes")},
                "rejected": {"role": "assistant", "content": "no"}
            })
            .to_string()
        })
        .collect();
    std::fs::write(path, lines.join("\n")).unwrap();
}

#[test]
fn a_dataset_check_is_the_services_report_with_every_records_length_by_line() {
    let rig = rig();
    let path = rig.dir.path().join("mixed.jsonl");
    records(&path, false);
    let mut text = std::fs::read_to_string(&path).unwrap();
    text.push_str("\nnot a record\n");
    std::fs::write(&path, text).unwrap();
    let service = ScriptedEndpoint::start(vec![]).with_training(JobScript::over(rig.store.clone()));

    let report = check_dataset(
        &endpoints(&service, &rig),
        &path,
        &CheckSpec::chat("Qwen/Toy", false, false),
    )
    .unwrap();

    let params = &service.jobs().checks()[0];
    assert_eq!(service.jobs().actions(), vec!["check_dataset"]);
    assert_eq!(
        (params["base"].as_str(), params["kind"].as_str()),
        (Some("Qwen/Toy"), Some("chat"))
    );
    assert_eq!(
        params["dataset_digest"],
        Digest::of(&std::fs::read(&path).unwrap()).hex(),
        "the dataset travelled by digest"
    );
    assert_eq!(
        (report.records, report.valid_records, report.invalid_total),
        (4, 3, 1)
    );
    // "cab" + "answer 0": every valid record's length, by its line.
    assert_eq!(report.per_record.len(), 3);
    assert_eq!(report.tokens_at(1), Some(11));
    assert_eq!(
        report.per_record[0].prompt + report.per_record[0].trained,
        11
    );
    let refused = report.refuse_invalid(&path).unwrap_err().to_string();
    assert!(refused.contains("line 4"), "{refused}");
}

#[test]
fn a_preference_run_is_a_dpo_job_sized_to_its_longest_pair() {
    let rig = rig();
    let (train, held_out, attempt) = (
        rig.dir.path().join("pairs.jsonl"),
        rig.dir.path().join("held_pairs.jsonl"),
        rig.dir.path().join("attempt-dpo"),
    );
    pairs(&train);
    pairs(&held_out);
    let service = ScriptedEndpoint::start(vec![]).with_training(JobScript::over(rig.store.clone()));
    let request = PreferenceTune {
        model_dir: &rig.model_dir,
        train: &train,
        held_out: &held_out,
        attempt_dir: &attempt,
        steps: 3,
        rank: 4,
        alpha: 8.0,
        beta: 0.2,
        nll_weight: 0.0,
        grad_accum: 1,
        learning_rate: None,
        // seed: the job's request carries it.
        seed: 5,
        continue_from: None,
        cancel: None,
    };

    let trained = train_preference_served(&endpoints(&service, &rig), &request).unwrap();

    let jobs = service.jobs();
    assert_eq!(
        jobs.actions(),
        vec!["check_dataset", "check_dataset", "train"]
    );
    assert_eq!(jobs.checks()[0]["kind"], "preference");
    let params = &jobs.created()[0];
    assert_eq!(params["objective"], "dpo");
    let longest = std::fs::read_to_string(&train)
        .unwrap()
        .lines()
        .map(|l| l.chars().count())
        .max()
        .unwrap() as u64;
    assert_eq!(params["max_block"].as_u64(), Some(longest.max(64)));
    assert!((params["beta"].as_f64().unwrap() - 0.2).abs() < 1e-6);
    assert_eq!(
        data_of(params)["held_out"]["digest"],
        Digest::of(&std::fs::read(&held_out).unwrap()).hex()
    );
    assert_eq!(trained.records, 6);
    assert_eq!(trained.held_out_score.map(|s| s.pairs), Some(4));
    assert!(trained.adapter.is_file() && trained.training_record.is_file());
}

#[test]
fn a_full_parameter_run_and_a_merge_are_installed_checkpoints_fetched_by_manifest_when_not_shared()
{
    let rig = rig();
    let (train, attempt) = (
        rig.dir.path().join("train.jsonl"),
        rig.dir.path().join("attempt-full"),
    );
    records(&train, false);
    let models = rig.dir.path().join("models");
    // A server sharing the model store installs the checkpoint where it is read.
    let shared = ScriptedEndpoint::start(vec![]).with_training(JobScript {
        models: Some(models.clone()),
        ..JobScript::over(rig.store.clone())
    });
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
        train_precision: Some("state=int8"),
        // seed: the job's request carries it.
        seed: 9,
        thinking: false,
        cancel: None,
        on_step: None,
    };
    let install = Install {
        models_dir: &models,
        name: "tuned-1",
    };
    let checkpoint = full_fine_tune_served(&endpoints(&shared, &rig), &full, &install).unwrap();
    let params = shared.jobs().created().last().cloned().unwrap();
    assert_eq!(
        (
            params["install_as"].as_str(),
            params["train_precision"].as_str()
        ),
        (Some("tuned-1"), Some("state=int8"))
    );
    assert_eq!(checkpoint.dir, models.join("local/tuned-1"));
    assert_eq!(
        checkpoint.weights,
        models.join("local/tuned-1/model.safetensors")
    );
    assert_eq!(checkpoint.steps_completed, 3);
    assert_eq!(
        checkpoint.weights_digest,
        Digest::sha256_of(&std::fs::read(&checkpoint.weights).unwrap()).to_string()
    );

    // A server with a store of its own: the files come through the artifact store.
    let apart = ScriptedEndpoint::start(vec![]).with_training(JobScript::over(rig.store.clone()));
    let adapter = rig.dir.path().join("adapter.safetensors");
    std::fs::write(&adapter, b"lora").unwrap();
    let merged = merge_served(
        &endpoints(&apart, &rig),
        &rig.model_dir,
        &adapter,
        &Install {
            models_dir: &models,
            name: "merged-1",
        },
        None,
    )
    .unwrap();
    let params = &apart.jobs().created()[0];
    assert_eq!(apart.jobs().actions(), vec!["merge"]);
    assert_eq!(params["adapter_digest"], Digest::of(b"lora").hex());
    assert_eq!(merged.dir, models.join("local/merged-1"));
    assert!(merged.dir.join("config.json").is_file() && merged.weights.is_file());
    assert_eq!((merged.steps_completed, merged.initial_loss), (0, None));
    assert!(Install {
        models_dir: &models,
        name: "../x"
    }
    .dir()
    .starts_with(&models));
    let refused = merge_served(
        &endpoints(&apart, &rig),
        &rig.model_dir,
        &adapter,
        &Install {
            models_dir: &models,
            name: "../x",
        },
        None,
    );
    assert!(
        refused.is_err(),
        "a name that leaves the store is refused before any job"
    );
}

#[test]
fn a_model_is_scored_on_records_through_the_servers_score_action() {
    let rig = rig();
    let path = rig.dir.path().join("held.jsonl");
    records(&path, false);
    let service = ScriptedEndpoint::start(vec![Card::base("Qwen/Toy")]);
    let weights = LocalWeights {
        base: rig.model_dir.clone(),
        adapter: None,
        context_tokens: None,
    };

    let score = score_chat(&endpoints(&service, &rig), &weights, &path, false).unwrap();

    let runs = service.runs_asked();
    assert_eq!(runs.len(), 1);
    assert_eq!(
        (runs[0]["model"].as_str(), runs[0]["action"].as_str()),
        (Some("Qwen/Toy"), Some("score"))
    );
    // "answer 0" .. "answer 2", eight characters each, at -0.5 a token.
    assert_eq!((score.positions, score.records), (24, 3));
    assert_eq!(score.loss, Some(0.5));
    assert_eq!(
        score.token_accuracy, None,
        "the action measures no accuracy"
    );
}

#[test]
fn the_cards_are_the_ones_the_server_lists_in_index_order() {
    let rig = rig();
    let service = ScriptedEndpoint::start(vec![]);
    let cards = splinter_model::device::devices(&endpoints(&service, &rig)).unwrap();
    let gib = 1u64 << 30;
    assert_eq!(
        cards
            .iter()
            .map(|c| (c.index, c.total_bytes / gib, c.free_bytes / gib))
            .collect::<Vec<_>>(),
        vec![(0, 80, 50), (1, 80, 80)],
        "the CPU is not a card, and the cards come in index order"
    );
}
