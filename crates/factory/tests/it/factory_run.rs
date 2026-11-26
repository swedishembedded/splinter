// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! A whole recipe through the controller, on a scripted execution backend
//! and a scripted trainer: every stage commits once and is reused after, a
//! later pass runs only what earlier passes left unsolved, the
//! infrastructure's failures are retried and never become training
//! negatives, a crash loses and repeats nothing, a corrupt trace is
//! quarantined, a changed variant retrains only itself, and the run can be
//! verified on its own.

use std::path::Path;

use splinter_core::factory::product::{Decision, TrialResult};
use splinter_core::factory::trial::TrialStatus;
use splinter_factory::resolve::{resolve, Resolved};
use splinter_factory::run::dataset::dataset_of;
use splinter_factory::run::eval::result_of;
use splinter_factory::run::rollout::trials_of;
use splinter_factory::run::{fork, status, verify, Factory, RunError, Services};
use splinter_store::factory::runstate::RunStateError;

use crate::fakes::{FakeBackend, FakeTrainer, Outcome};
use crate::repo::{connections, dataset, evaluation, Repo, DATASET, EVALUATION, RECIPE, TRAINING};

const OWNER: &str = "test-controller";

pub(crate) fn resolved(repo: &Repo, recipe: &str) -> Resolved {
    resolve(&repo.root, recipe, connections()).expect("the recipe resolves")
}

pub(crate) fn run(
    resolved: &Resolved,
    dir: &Path,
    backend: &FakeBackend,
    trainer: &FakeTrainer,
) -> Result<Vec<(String, bool)>, RunError> {
    let services = Services {
        backend,
        trainer: Some(trainer),
        models: None,
        variables: None,
        online: None,
        build: splinter_core::build::BuildInfo::this_executable(),
    };
    let mut factory = Factory::open(resolved, dir, services, OWNER)?;
    Ok(factory
        .run()?
        .stages
        .into_iter()
        .map(|s| (s.key, s.computed))
        .collect())
}

/// Eight attempts on the held-out task, so a candidate that wins every pair
/// is decided at the plan's level.
pub(crate) fn decisive(repo: &Repo) {
    // seed: the plan must state its seed; the fake backend draws nothing from it.
    repo.write(
        "eval/dev.yaml",
        "schema: splinter.evalplan/v1\ncollection: collections/vim.yaml\nsplit: eval\n\
         environment: environments/vim.yaml\nattempts: 8\nseed: 5\ndecision: { alpha: 0.05 }\n",
    );
}

fn tasks_run(backend: &FakeBackend, pass: &str) -> Vec<String> {
    let mut tasks: Vec<String> = backend
        .executed()
        .into_iter()
        .filter(|s| s.slot.pass == pass)
        .map(|s| s.task.id)
        .collect();
    tasks.dedup();
    tasks
}

#[test]
fn a_recipe_runs_every_stage_once_and_a_second_run_reuses_every_product() {
    let repo = Repo::new();
    decisive(&repo);
    let resolved = resolved(&repo, RECIPE);
    let backend = FakeBackend {
        candidate: Some(Outcome::Pass),
        ..FakeBackend::with(Outcome::Fail)
    };
    backend.script("train/001-first", &[Outcome::Pass, Outcome::Pass]);
    let trainer = FakeTrainer::default();
    let dir = tempfile::tempdir().unwrap();

    let first = run(&resolved, dir.path(), &backend, &trainer).unwrap();
    assert!(first.iter().all(|(_, computed)| *computed), "{first:?}");

    // The remote pass ran only the task the local pass left unsolved.
    assert_eq!(
        tasks_run(&backend, "local"),
        ["train/001-first", "train/002-second"]
    );
    assert_eq!(tasks_run(&backend, "remote"), ["train/002-second"]);
    // A reward-0 trial is kept as a graded failure and never retried.
    let local = trials_of(&dir.path().join("local")).unwrap();
    assert_eq!(local.len(), 4);
    // Each trial says which sandbox it ran in.
    assert!(local
        .iter()
        .all(|t| t.sandbox.as_ref().is_some_and(|s| s.provider == "proot")));
    let failed: Vec<&TrialResult> = local
        .iter()
        .filter(|t| t.task == "train/002-second")
        .collect();
    assert!(failed
        .iter()
        .all(|t| t.grade.status == TrialStatus::Graded && t.grade.passed == Some(false)));
    assert_eq!(
        backend
            .executed()
            .iter()
            .filter(|s| s.slot.pass == "local")
            .count(),
        4
    );

    // The dataset holds the two verified successes, under Sven's own prompt.
    let dataset = dataset_of(&dir.path().join("sft")).unwrap();
    let steps: Vec<(&str, u64)> = dataset
        .waterfall
        .iter()
        .map(|w| (w.step.as_str(), w.remaining))
        .collect();
    assert_eq!(
        steps,
        [
            ("raw_trials", 6),
            ("verified_successes", 2),
            ("projected", 2),
            ("accepted", 2)
        ]
    );
    assert_eq!(dataset.exclusions.get("not_a_success:graded"), Some(&4));
    let train = std::fs::read_to_string(dir.path().join("sft/train.jsonl")).unwrap();
    assert_eq!(train.lines().count(), 2);
    assert!(
        train.contains("You are Sven"),
        "the agent's own system prompt is kept"
    );

    // Both variants trained; each candidate beat the baseline on every pair.
    assert_eq!(trainer.trained.lock().unwrap().len(), 2);
    for candidate in ["lora-r8", "lora-r16"] {
        let result = result_of(&dir.path().join(format!("eval-{candidate}"))).unwrap();
        assert_eq!(
            result.decision,
            Decision::Improved,
            "{:?}",
            result.statistics
        );
        assert_eq!(
            (
                result.baseline.as_ref().unwrap().passed,
                result.candidate.as_ref().unwrap().passed
            ),
            (0, 8)
        );
    }
    // The baseline arm ran once for both candidates.
    assert_eq!(
        backend
            .executed()
            .iter()
            .filter(|s| s.slot.pass.starts_with("baseline-"))
            .count(),
        8
    );
    let release = std::fs::read_to_string(dir.path().join("release/release.json")).unwrap();
    assert!(
        release.contains("\"promoted\": null"),
        "an experiment promotes nothing: {release}"
    );

    // Run again: nothing is executed, trained or written again.
    let executed = backend.executed().len();
    let second = run(&resolved, dir.path(), &backend, &trainer).unwrap();
    assert!(second.iter().all(|(_, computed)| !*computed), "{second:?}");
    assert_eq!(backend.executed().len(), executed);
    assert_eq!(trainer.trained.lock().unwrap().len(), 2);
    let problems = verify(dir.path()).unwrap();
    assert!(problems.is_empty(), "{problems:?}");
    let stages = status(dir.path()).unwrap();
    assert!(stages.iter().all(|s| s.state == "completed"));
    // Every attempt stage and evaluation arm reports its slots, all finished.
    for key in [
        "local",
        "remote",
        "eval-lora-r8/_arms/baseline",
        "eval-lora-r8/_arms/candidate",
    ] {
        let stage = stages.iter().find(|s| s.key == key).unwrap();
        assert!(
            !stage.slots.is_empty() && stage.slots.keys().all(|k| k == "finished"),
            "{key}: {:?}",
            stage.slots
        );
    }
}

#[test]
fn the_infrastructures_failures_are_retried_and_never_become_training_negatives() {
    let repo = Repo::new();
    let recipe =
        RECIPE.split(DATASET).next().unwrap().to_owned() + &dataset(&["local", "remote"], "");
    let resolved = resolved(&repo, &recipe);
    let backend = FakeBackend::with(Outcome::Pass);
    // Slot 0 of the first task: the model fails on every attempt (three: one and two retries).
    backend.script(
        "train/001-first#0",
        &[
            Outcome::ModelError,
            Outcome::ModelError,
            Outcome::ModelError,
        ],
    );
    // The second task's first attempt is lost by the worker, then passes.
    backend.script("train/002-second#0", &[Outcome::Lost]);
    let trainer = FakeTrainer::default();
    let dir = tempfile::tempdir().unwrap();
    run(&resolved, dir.path(), &backend, &trainer).unwrap();

    let local = trials_of(&dir.path().join("local")).unwrap();
    let model_errors: Vec<_> = local
        .iter()
        .filter(|t| t.grade.status == TrialStatus::ModelError)
        .collect();
    assert_eq!(
        model_errors.len(),
        1,
        "the slot is kept as what it was after its retries"
    );
    assert_eq!(model_errors[0].attempt, 2);
    assert!(!model_errors[0].grade.counted);
    let attempts = std::fs::read_to_string(dir.path().join("local/attempts.jsonl")).unwrap();
    assert_eq!(
        attempts.lines().count(),
        6,
        "every reported attempt is kept, the retried ones too"
    );
    let dataset = dataset_of(&dir.path().join("sft")).unwrap();
    assert_eq!(
        dataset.exclusions.get("not_a_success:model_error"),
        Some(&1)
    );
    // The lost attempt was launched again and its slot passed.
    assert!(local
        .iter()
        .filter(|t| t.task == "train/002-second")
        .all(TrialResult::verified_success));
}

#[test]
fn a_corrupt_trace_keeps_its_evidence_and_is_quarantined_from_training() {
    let repo = Repo::new();
    let recipe = RECIPE.split(TRAINING).next().unwrap().to_owned();
    let resolved = resolved(&repo, &recipe);
    let backend = FakeBackend::with(Outcome::Pass);
    backend.script(
        "train/002-second",
        &[Outcome::CorruptTrace, Outcome::CorruptTrace],
    );
    let dir = tempfile::tempdir().unwrap();
    run(&resolved, dir.path(), &backend, &FakeTrainer::default()).unwrap();
    let dataset = dataset_of(&dir.path().join("sft")).unwrap();
    assert_eq!(dataset.exclusions.get("invalid_trace"), Some(&2));
    let excluded = std::fs::read_to_string(dir.path().join("sft/excluded.jsonl")).unwrap();
    assert!(excluded.contains("train/002-second"));
    let trial = &trials_of(&dir.path().join("local")).unwrap()[2];
    assert!(dir
        .path()
        .join("local")
        .join(&trial.native_dir)
        .join("agent/trajectory.json")
        .is_file());
}

#[test]
fn a_changed_variant_retrains_only_itself_and_reuses_the_rollouts_and_the_dataset() {
    let repo = Repo::new();
    decisive(&repo);
    let backend = FakeBackend {
        candidate: Some(Outcome::Pass),
        ..FakeBackend::with(Outcome::Pass)
    };
    let trainer = FakeTrainer::default();
    let first = tempfile::tempdir().unwrap();
    run(&resolved(&repo, RECIPE), first.path(), &backend, &trainer).unwrap();
    let executed = backend.executed().len();

    let changed = RECIPE.replace("params: { rank: 16 }", "params: { rank: 32 }");
    let second = tempfile::tempdir().unwrap();
    assert!(fork(first.path(), second.path()).unwrap() > 0);
    let stages = run(
        &resolved(&repo, &changed),
        second.path(),
        &backend,
        &trainer,
    )
    .unwrap();
    let computed: Vec<&str> = stages
        .iter()
        .filter(|(_, c)| *c)
        .map(|(k, _)| k.as_str())
        .collect();
    assert_eq!(computed, ["lora-r16", "eval-lora-r16", "release"]);
    assert_eq!(trainer.trained.lock().unwrap().len(), 3);
    // Only the changed candidate's arm ran; the shared baseline was reused.
    let new: Vec<_> = backend.executed()[executed..]
        .iter()
        .map(|s| s.slot.pass.clone())
        .collect();
    assert!(
        new.iter().all(|p| p == "eval-lora-r16.candidate") && new.len() == 8,
        "{new:?}"
    );
}

#[test]
fn a_run_refuses_changed_inputs_and_a_second_live_controller() {
    let repo = Repo::new();
    let recipe = RECIPE.split(DATASET).next().unwrap().to_owned();
    let backend = FakeBackend::with(Outcome::Pass);
    let trainer = FakeTrainer::default();
    let dir = tempfile::tempdir().unwrap();
    let resolved_now = resolved(&repo, &recipe);
    let services = Services {
        backend: &backend,
        trainer: Some(&trainer),
        models: None,
        variables: None,
        online: None,
        build: splinter_core::build::BuildInfo::this_executable(),
    };
    let _held = Factory::open(&resolved_now, dir.path(), services, OWNER).unwrap();
    let second = Factory::open(&resolved_now, dir.path(), services, "another-controller");
    assert!(
        matches!(
            second,
            Err(RunError::State(RunStateError::LeaseHeld { .. }))
        ),
        "{:?}",
        second.err()
    );

    repo.write("presets/sven.yaml", "agent: { max_turns: 99 }\n");
    let changed = resolved(&repo, &recipe);
    let refused = Factory::open(&changed, dir.path(), services, OWNER);
    match refused {
        Err(RunError::InputsChanged(changes)) => assert_eq!(changes, ["presets/sven.yaml changed"]),
        other => panic!("expected InputsChanged, got {:?}", other.err()),
    }
}

#[test]
fn a_finished_controller_gives_up_its_lease_and_a_dead_ones_is_taken_over() {
    let repo = Repo::new();
    let recipe = RECIPE.split(DATASET).next().unwrap();
    let resolved = resolved(&repo, recipe);
    let backend = FakeBackend::with(Outcome::Pass);
    let trainer = FakeTrainer::default();
    let dir = tempfile::tempdir().unwrap();
    let services = Services {
        backend: &backend,
        trainer: Some(&trainer),
        models: None,
        variables: None,
        online: None,
        build: splinter_core::build::BuildInfo::this_executable(),
    };
    // Owners are `host:pid@pidns:start`: this host and pid namespace, as a
    // controller of this process names itself.
    let me = splinter_factory::run::controller_owner("host");
    let namespace = me
        .split_once('@')
        .and_then(|(_, rest)| rest.rsplit_once(':'))
        .map(|(namespace, _)| namespace.to_owned())
        .unwrap();
    drop(Factory::open(&resolved, dir.path(), services, &me).unwrap());
    // Released on drop: another controller takes it at once. This one is
    // never dropped, as a killed process leaves its lease, and its pid names
    // no process.
    let dead = format!("host:4294967@{namespace}:1");
    std::mem::forget(Factory::open(&resolved, dir.path(), services, &dead).unwrap());
    // A controller of another host waits for the lease to lapse, and so does
    // one of another pid namespace, where the pid means another process...
    assert!(Factory::open(&resolved, dir.path(), services, "elsewhere:1@pid:[7]:1").is_err());
    assert!(Factory::open(&resolved, dir.path(), services, "host:3@pid:[1]:1").is_err());
    // ...one of the same host and namespace sees its process is gone and
    // takes over.
    assert!(Factory::open(&resolved, dir.path(), services, &me).is_ok());
}

#[test]
fn a_tampered_product_fails_verification() {
    let repo = Repo::new();
    let resolved = resolved(&repo, RECIPE.split(DATASET).next().unwrap());
    let backend = FakeBackend::with(Outcome::Pass);
    let dir = tempfile::tempdir().unwrap();
    run(&resolved, dir.path(), &backend, &FakeTrainer::default()).unwrap();
    let trials = dir.path().join("local/trials.jsonl");
    let mut text = std::fs::read_to_string(&trials).unwrap();
    text.push_str("{}\n");
    std::fs::remove_file(&trials).unwrap();
    std::fs::write(&trials, text).unwrap();
    let problems = verify(dir.path()).unwrap();
    assert!(
        problems.iter().any(|p| p.starts_with("local")),
        "{problems:?}"
    );
}

#[test]
fn a_cancelled_run_stops_at_a_boundary_and_resumes_from_what_it_recorded() {
    let repo = Repo::new();
    let recipe = RECIPE.split(DATASET).next().unwrap();
    let resolved_first = resolved(&repo, recipe);
    let backend = FakeBackend::with(Outcome::Pass);
    let trainer = FakeTrainer::default();
    let dir = tempfile::tempdir().unwrap();
    let services = Services {
        backend: &backend,
        trainer: Some(&trainer),
        models: None,
        variables: None,
        online: None,
        build: splinter_core::build::BuildInfo::this_executable(),
    };
    let stopped = {
        let mut factory = Factory::open(&resolved_first, dir.path(), services, OWNER).unwrap();
        splinter_factory::run::request_cancel(dir.path()).unwrap();
        factory.run()
    };
    assert!(matches!(stopped, Err(RunError::Cancelled)), "{stopped:?}");
    assert!(backend.executed().is_empty());

    let (repository, text) = splinter_factory::run::started_with(dir.path()).unwrap();
    assert_eq!(
        (repository.as_path(), text.as_str()),
        (repo.root.as_path(), recipe)
    );
    let again = resolve(&repository, &text, connections()).unwrap();
    let mut factory = Factory::open(&again, dir.path(), services, OWNER).unwrap();
    factory.run().unwrap();
    assert_eq!(trials_of(&dir.path().join("local")).unwrap().len(), 4);
}

#[test]
fn a_played_session_is_a_guided_trial_that_the_next_run_trains_on_with_the_rest_reused() {
    let repo = Repo::new();
    // seed: a recipe states every seed; nothing here draws from them.
    let recipe = format!(
        "schema: splinter.recipe/v2\nname: play\ninputs: {{ course: {{ kind: files, path: collections/vim.yaml }} }}\n\
         models: {{ student: \"brain:Qwen/Qwen3-4B\" }}\nstages:\n\
         \x20 - {{ id: tasks, uses: tasks.pin, inputs: {{ collection: $inputs.course }}, with: {{ split: train }} }}\n\
         \x20 - {{ id: local, uses: attempt.collect, inputs: {{ tasks: $stages.tasks.tasks }}, \
         with: {{ agent: sven, environment: environments/vim.yaml, seed: 1 }} }}\n\
         \x20 - {{ id: guided, uses: attempt.collect, inputs: {{ tasks: $stages.tasks.tasks }}, \
         with: {{ agent: sven, environment: environments/vim.yaml, interactive: true, seed: 2 }} }}\n{}",
        dataset(&["local", "guided"], "")
    );
    let resolved = resolved(&repo, &recipe);
    let backend = FakeBackend::with(Outcome::Fail);
    backend.script("train/001-first", &[Outcome::Pass]);
    let trainer = FakeTrainer::default();
    let dir = tempfile::tempdir().unwrap();
    run(&resolved, dir.path(), &backend, &trainer).unwrap();
    assert!(
        trials_of(&dir.path().join("guided")).unwrap().is_empty(),
        "nobody has played yet"
    );

    let services = Services {
        backend: &backend,
        trainer: Some(&trainer),
        models: None,
        variables: None,
        online: None,
        build: splinter_core::build::BuildInfo::this_executable(),
    };
    let first = {
        let mut factory = Factory::open(&resolved, dir.path(), services, OWNER).unwrap();
        let first = factory.play("guided", "train/002-second", None).unwrap();
        let second = factory.play("guided", "train/002-second", Some(0)).unwrap();
        assert_eq!(second.slot.slot, 1);
        assert!(
            factory.play("local", "train/002-second", None).is_err(),
            "only an interactive pass is played"
        );
        first
    };
    assert!(first.guided && first.verified_success());
    let played = backend.played.lock().unwrap().clone();
    assert!(
        played[0].1.is_none() && played[1].1.is_some(),
        "the second session continues the first"
    );

    let stages = run(&resolved, dir.path(), &backend, &trainer).unwrap();
    let computed: Vec<&str> = stages
        .iter()
        .filter(|(_, c)| *c)
        .map(|(k, _)| k.as_str())
        .collect();
    assert_eq!(computed, ["guided", "sft"]);
    let accepted = std::fs::read_to_string(dir.path().join("sft/accepted.jsonl")).unwrap();
    assert_eq!(
        accepted.lines().count(),
        3,
        "the unattended success and both played sessions"
    );
}

#[test]
fn a_portable_dataset_is_built_from_the_trajectories_alone() {
    let repo = Repo::new();
    let recipe = RECIPE.split(DATASET).next().unwrap().to_owned()
        + &dataset(&["local"], ", projection: portable-sft");
    let resolved = resolved(&repo, &recipe);
    let backend = FakeBackend::with(Outcome::Pass);
    let dir = tempfile::tempdir().unwrap();
    run(&resolved, dir.path(), &backend, &FakeTrainer::default()).unwrap();
    let dataset = dataset_of(&dir.path().join("sft")).unwrap();
    assert_eq!(dataset.projection, "portable-sft");
    assert_eq!(dataset.waterfall.last().map(|w| w.remaining), Some(4));
}

/// The verdict the examiner signs: about `candidate` against the base, at
/// `alpha`, with `outcome`.
pub(crate) fn signed_verdict(
    signer: &ed25519_dalek::SigningKey,
    candidate: &str,
    predecessor: Option<&str>,
    alpha: f64,
    outcome: &str,
) -> String {
    use ed25519_dalek::Signer as _;
    let mut doc = serde_json::json!({
        "schema": "bench.examination/2", "valid": true, "invalid_reasons": [], "integrity_flags": [],
        "verdict": outcome, "alpha": alpha,
        "candidate": {"adapter_digest": candidate}, "predecessor": {"adapter_digest": predecessor},
        "collection": {"name": "vim-sealed", "exam_sha256": "e".repeat(64), "sealed": true, "items": 40},
        "paired_gain": {"mean": 0.1, "interval": [0.02, 0.18], "one_sided_lower": 0.03, "families": 8},
    });
    let bytes = splinter_eval::promotion::canonical::canonical_bytes(&doc).unwrap();
    let value: String = signer
        .sign(&bytes)
        .to_bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    doc["signature"] = serde_json::json!({"alg": "ed25519", "key_id": "t", "value": value});
    doc.to_string()
}

#[test]
fn a_protected_plan_keeps_the_examiners_verdict_only_when_it_verifies() {
    let repo = Repo::new();
    let backend = FakeBackend {
        candidate: Some(Outcome::Pass),
        ..FakeBackend::with(Outcome::Pass)
    };
    let trainer = FakeTrainer::default();
    // Train first, to know the candidate the examiner will be handed.
    let trained = tempfile::tempdir().unwrap();
    let training_only = RECIPE.split(EVALUATION).next().unwrap();
    run(
        &resolved(&repo, training_only),
        trained.path(),
        &backend,
        &trainer,
    )
    .unwrap();
    let candidate =
        splinter_factory::run::training::candidate_of(&trained.path().join("lora-r8")).unwrap();
    let digest = candidate
        .artifacts
        .iter()
        .find(|a| a.path.ends_with(".safetensors"))
        .unwrap()
        .digest
        .to_string();

    // seed: the examiner's test key; the gate checks a signature under it.
    let signer = ed25519_dalek::SigningKey::from_bytes(&[3; 32]);
    let public: String = signer
        .verifying_key()
        .to_bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    repo.write("eval/examiner.pub", &public);
    repo.write(
        "eval/verdict.json",
        &signed_verdict(&signer, &digest, None, 0.05, "Promote"),
    );
    repo.write(
        "eval/examine.sh",
        "#!/bin/sh\n# --candidate FILE --out FILE\ncp eval/verdict.json \"$4\"\n",
    );
    // seed: the plan states its seed; the examiner draws nothing from it.
    repo.write(
        "eval/sealed.yaml",
        "schema: splinter.evalplan/v1\nattempts: 1\nseed: 5\ndecision: { alpha: 0.05, comparisons: 1 }\n\
         authority: { command: [sh, eval/examine.sh], public_key: eval/examiner.pub }\n",
    );
    let recipe = training_only.to_owned() + &evaluation("lora-r8", "eval/sealed.yaml", "student");
    let examined = tempfile::tempdir().unwrap();
    fork(trained.path(), examined.path()).unwrap();
    let stages = run(
        &resolved(&repo, &recipe),
        examined.path(),
        &backend,
        &trainer,
    )
    .unwrap();
    assert!(
        stages
            .iter()
            .all(|(k, computed)| !computed || k.starts_with("eval-")),
        "{stages:?}"
    );
    let result = result_of(&examined.path().join("eval-lora-r8")).unwrap();
    assert!(result.protected && result.signature.is_some() && result.baseline.is_none());
    assert_eq!(result.decision, Decision::Improved);
    assert!(examined.path().join("eval-lora-r8/verdict.json").is_file());

    // A verdict about another candidate is refused, whatever it says.
    repo.write(
        "eval/verdict.json",
        &signed_verdict(
            &signer,
            &format!("sha256:{}", "0".repeat(64)),
            None,
            0.05,
            "Promote",
        ),
    );
    let refused = tempfile::tempdir().unwrap();
    fork(trained.path(), refused.path()).unwrap();
    let error = run(
        &resolved(&repo, &recipe),
        refused.path(),
        &backend,
        &trainer,
    )
    .unwrap_err();
    assert!(error.to_string().contains("does not verify"), "{error}");
}

#[test]
fn a_dataset_for_a_model_that_does_not_reason_carries_no_reasoning_and_the_trainer_is_told() {
    // The live Sven 5.9 trial: its replies carry the model's reasoning.
    let repo = Repo::new();
    let recipe = RECIPE.split(EVALUATION).next().unwrap();
    let backend = FakeBackend::with(Outcome::Pass);
    let trainer = FakeTrainer::default();
    let dir = tempfile::tempdir().unwrap();
    run(&resolved(&repo, recipe), dir.path(), &backend, &trainer).unwrap();
    let train = std::fs::read_to_string(dir.path().join("sft/train.jsonl")).unwrap();
    assert!(
        !train.contains("reasoning_content"),
        "thinking: false keeps no reasoning"
    );
    let report = std::fs::read_to_string(dir.path().join("sft/report.md")).unwrap();
    assert!(report.contains("reasoning removed"), "{report}");
    assert!(
        trainer.thinking.lock().unwrap().iter().all(|t| !t),
        "the trainer trains for no-think mode"
    );

    let thinking = recipe.replace(
        "target: student, seed: 13",
        "target: student, thinking: true, seed: 13",
    );
    let other = tempfile::tempdir().unwrap();
    run(
        &resolved(&repo, &thinking),
        other.path(),
        &backend,
        &trainer,
    )
    .unwrap();
    let train = std::fs::read_to_string(other.path().join("sft/train.jsonl")).unwrap();
    assert!(
        train.contains("reasoning_content"),
        "thinking: true keeps the model's reasoning"
    );
    assert!(trainer.thinking.lock().unwrap().iter().any(|t| *t));
}

#[test]
fn a_fork_of_a_run_that_holds_two_products_of_a_stage_shows_the_one_it_reuses() {
    let repo = Repo::new();
    let before = RECIPE.split(TRAINING).next().unwrap().to_owned();
    let backend = FakeBackend::with(Outcome::Pass);
    let trainer = FakeTrainer::default();
    let first = tempfile::tempdir().unwrap();
    run(&resolved(&repo, &before), first.path(), &backend, &trainer).unwrap();
    // The same run directory gains a second dataset product (a fork into it
    // of a changed recipe), and its view now shows that one.
    let changed = before.replace(
        "target: student, seed: 13",
        "target: student, thinking: true, seed: 13",
    );
    let second = tempfile::tempdir().unwrap();
    fork(first.path(), second.path()).unwrap();
    run(
        &resolved(&repo, &changed),
        second.path(),
        &backend,
        &trainer,
    )
    .unwrap();
    fork(second.path(), first.path()).unwrap();
    // The first recipe forked again reuses its own dataset, and shows it.
    let third = tempfile::tempdir().unwrap();
    fork(first.path(), third.path()).unwrap();
    run(&resolved(&repo, &before), third.path(), &backend, &trainer).unwrap();
    let train = std::fs::read_to_string(third.path().join("sft/train.jsonl")).unwrap();
    assert!(
        !train.contains("reasoning_content"),
        "the view shows the reused product, not the other one"
    );
    assert!(verify(third.path()).unwrap().is_empty());
}

#[test]
fn a_directory_that_is_not_a_run_is_refused_and_left_untouched() {
    use splinter_factory::run::{request_cancel, started_with};
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("runs/none");
    let empty = dir.path().join("empty");
    std::fs::create_dir(&empty).unwrap();
    for target in [&missing, &empty] {
        let errors = [
            status(target).unwrap_err().to_string(),
            verify(target).unwrap_err().to_string(),
            started_with(target).unwrap_err().to_string(),
            request_cancel(target).unwrap_err().to_string(),
            fork(target, &dir.path().join("fork"))
                .unwrap_err()
                .to_string(),
        ];
        for error in errors {
            assert!(
                error.contains("not a factory run") || error.contains("does not exist"),
                "{error}"
            );
        }
    }
    assert!(
        !missing.exists(),
        "nothing was created for a run that is not there"
    );
    assert_eq!(
        std::fs::read_dir(&empty).unwrap().count(),
        0,
        "nor inside a directory that is not a run"
    );
}
