// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning experiments whose every result
// names the exact code, data and hardware that produced it, for its clients.
// If your team needs expertise in attributable, reproducible machine-learning
// runs, you can procure our services by sending an email to
// info@swedishembedded.com.

//! A run's revision manifest: a run pins the build it starts on; every
//! stage that trains or evaluates names the build, the servers it reached,
//! its inputs and its configuration in its product; a resume on a build of
//! other commits is refused by name unless the recipe allows mixed
//! revisions; a server whose training-record contract is not the one this
//! build writes is refused before any stage runs; and the brain trainer reads
//! what its server says it is.

use serde_json::{json, Value};
use splinter_core::build::{BuildInfo, Revisions};
use splinter_core::revision::{
    EngineBuild, RecordContract, ServedRevision, StageRevisions, RECORD_CONTRACT_DIGEST,
    REVISIONS_FILE,
};
use splinter_factory::run::{Factory, RunError, Services};
use splinter_factory::training::{BrainTrainer, Trainer};
use splinter_model::endpoint::BrainEndpoints;
use splinter_model::scripted_endpoint::{Card, ScriptedEndpoint, SCRIPTED_KEY};

use crate::factory_run::{decisive, resolved};
use crate::fakes::{FakeBackend, FakeTrainer, Outcome};
use crate::repo::{Repo, RECIPE};

const OWNER: &str = "test-controller";

fn build(splinter: &str) -> BuildInfo {
    BuildInfo {
        revisions: Revisions {
            splinter: Some(splinter.into()),
            sven: Some("5".repeat(40)),
            brain: Some("b".repeat(40)),
        },
        dirty: Some(false),
        ..BuildInfo::default()
    }
}

fn server(training_records: Option<RecordContract>) -> ServedRevision {
    ServedRevision {
        endpoint: "http://127.0.0.1:8990/v1".into(),
        engine: Some(EngineBuild {
            version: "6.0.0".into(),
            commit: Some("c".repeat(40)),
            dirty: Some(false),
        }),
        backend: Some("cuda".into()),
        training_records,
        hardware: json!({"object": "list", "host": {"backend": "cuda"}, "data": []}),
    }
}

fn open<'a>(
    resolved: &'a splinter_factory::resolve::Resolved,
    dir: &std::path::Path,
    backend: &'a FakeBackend,
    trainer: &'a FakeTrainer,
    build: &'a BuildInfo,
) -> Result<Factory<'a>, RunError> {
    let services = Services {
        backend,
        trainer: Some(trainer),
        models: None,
        variables: None,
        online: None,
        build,
    };
    Factory::open(resolved, dir, services, OWNER)
}

fn backend() -> FakeBackend {
    let backend = FakeBackend {
        candidate: Some(Outcome::Pass),
        ..FakeBackend::with(Outcome::Fail)
    };
    backend.script("train/001-first", &[Outcome::Pass, Outcome::Pass]);
    backend
}

#[test]
fn every_stage_that_trains_or_evaluates_names_its_code_server_data_and_configuration() {
    let repo = Repo::new();
    decisive(&repo);
    let resolved = resolved(&repo, RECIPE);
    let backend = backend();
    let trainer = FakeTrainer {
        served: vec![server(Some(RecordContract::pinned()))],
        ..FakeTrainer::default()
    };
    let dir = tempfile::tempdir().unwrap();
    let started = build(&"a".repeat(40));
    open(&resolved, dir.path(), &backend, &trainer, &started)
        .unwrap()
        .run()
        .unwrap();

    let pinned: Value =
        serde_json::from_slice(&std::fs::read(dir.path().join(REVISIONS_FILE)).unwrap()).unwrap();
    assert_eq!(pinned["revisions"]["splinter"], json!("a".repeat(40)));
    assert_eq!(pinned["training_records"]["digest"], RECORD_CONTRACT_DIGEST);
    for stage in ["lora-r8", "eval-lora-r8"] {
        let text = std::fs::read(dir.path().join(stage).join(REVISIONS_FILE))
            .unwrap_or_else(|e| panic!("{stage}: {e}"));
        let manifest: StageRevisions = serde_json::from_slice(&text).unwrap();
        assert_eq!(manifest.build, started, "{stage}");
        assert_eq!(manifest.training_records, RecordContract::pinned());
        assert_eq!(
            manifest.served, trainer.served,
            "{stage}: what the server said it is"
        );
        assert!(
            !manifest.inputs.is_empty(),
            "{stage}: its dataset or candidate, by digest"
        );
        assert!(manifest.configuration["parameters"].is_object(), "{stage}");
    }
    let train: StageRevisions = serde_json::from_slice(
        &std::fs::read(dir.path().join("lora-r8").join(REVISIONS_FILE)).unwrap(),
    )
    .unwrap();
    assert_eq!(
        train.inputs.len(),
        1,
        "the dataset it trained: {:?}",
        train.inputs
    );
    assert_eq!(train.configuration["parameters"]["params"]["rank"], 8);
    assert_eq!(
        train.models,
        [splinter_core::revision::ModelDigests {
            base_digest: Some(format!("sha256:{}", "e".repeat(64))),
            tokenizer_digest: None
        }],
        "the base the training job reported"
    );
    assert!(
        !dir.path().join("sft").join(REVISIONS_FILE).exists(),
        "a stage that neither trains nor evaluates carries none"
    );
}

#[test]
fn a_resume_on_other_commits_is_refused_by_name_unless_the_recipe_allows_it() {
    let repo = Repo::new();
    decisive(&repo);
    let backend = backend();
    let trainer = FakeTrainer::default();
    let (first, other) = (build(&"a".repeat(40)), build(&"d".repeat(40)));

    let resolved_pinned = resolved(&repo, RECIPE);
    let dir = tempfile::tempdir().unwrap();
    drop(open(&resolved_pinned, dir.path(), &backend, &trainer, &first).unwrap());
    let refused = open(&resolved_pinned, dir.path(), &backend, &trainer, &other)
        .err()
        .expect("another splinter commit");
    let RunError::RevisionsChanged(changes) = &refused else {
        panic!("{refused}")
    };
    assert_eq!(changes.len(), 1, "{changes:?}");
    assert!(changes[0].contains("splinter commit") && changes[0].contains(&"d".repeat(40)));
    assert!(refused.to_string().contains("revisions: mixed"));
    // The build it started on resumes it.
    drop(open(&resolved_pinned, dir.path(), &backend, &trainer, &first).unwrap());

    let mixed = RECIPE.replacen("stages:\n", "revisions: mixed\nstages:\n", 1);
    let resolved_mixed = resolved(&repo, &mixed);
    let dir = tempfile::tempdir().unwrap();
    drop(open(&resolved_mixed, dir.path(), &backend, &trainer, &first).unwrap());
    assert!(
        open(&resolved_mixed, dir.path(), &backend, &trainer, &other).is_ok(),
        "the recipe allows mixed revisions"
    );
}

#[test]
fn a_server_that_checks_another_record_contract_is_refused_before_any_stage() {
    let repo = Repo::new();
    decisive(&repo);
    let required = RECIPE.replacen("stages:\n", "contract: required\nstages:\n", 1);
    let other = RecordContract {
        digest: format!("sha256:{}", "0".repeat(64)),
        ..RecordContract::pinned()
    };
    // Another contract is refused however the recipe holds it; none, only when it is required.
    for (recipe, served) in [
        (RECIPE, Some(other.clone())),
        (required.as_str(), Some(other)),
        (required.as_str(), None),
    ] {
        let resolved = resolved(&repo, recipe);
        let backend = backend();
        let trainer = FakeTrainer {
            served: vec![server(served)],
            ..FakeTrainer::default()
        };
        let dir = tempfile::tempdir().unwrap();
        let refused = open(
            &resolved,
            dir.path(),
            &backend,
            &trainer,
            BuildInfo::this_executable(),
        )
        .err()
        .expect("refused");
        assert!(
            matches!(refused, RunError::ContractMismatch(_)),
            "{refused}"
        );
        assert!(
            refused.to_string().contains(RECORD_CONTRACT_DIGEST),
            "{refused}"
        );
        assert!(backend.executed().is_empty(), "nothing ran");
    }
}

/// A server that reports no record contract (a frozen older brain build)
/// trains unless the recipe requires the contract: the run proceeds and every
/// stage's manifest records its contract as `null`, unknown.
#[test]
fn a_server_that_reports_no_record_contract_trains_with_it_recorded_unknown() {
    let repo = Repo::new();
    decisive(&repo);
    let resolved = resolved(&repo, RECIPE);
    let backend = backend();
    let trainer = FakeTrainer {
        served: vec![server(None)],
        ..FakeTrainer::default()
    };
    let dir = tempfile::tempdir().unwrap();
    open(
        &resolved,
        dir.path(),
        &backend,
        &trainer,
        BuildInfo::this_executable(),
    )
    .expect("an unknown contract is not a mismatch")
    .run()
    .unwrap();
    let manifest: Value = serde_json::from_slice(
        &std::fs::read(dir.path().join("lora-r8").join(REVISIONS_FILE)).unwrap(),
    )
    .unwrap();
    let served = manifest["served"][0].as_object().unwrap();
    assert_eq!(
        served.get("training_records"),
        Some(&Value::Null),
        "written out as unknown: {manifest:#}"
    );
    assert_eq!(
        manifest["training_records"]["digest"], RECORD_CONTRACT_DIGEST,
        "what this build writes is known"
    );
}

#[test]
fn the_brain_trainer_reads_what_its_server_says_it_is() {
    let service = ScriptedEndpoint::start(vec![Card::base("Qwen/Toy")]);
    let endpoints = BrainEndpoints::new(vec![service.url()], SCRIPTED_KEY).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let trainer = BrainTrainer::new(endpoints, dir.path().join("models"));
    let served = trainer.served_revisions().unwrap();
    assert_eq!(served.len(), 1);
    assert_eq!(served[0].endpoint, service.url());
    assert_eq!(
        served[0].engine.as_ref().and_then(|e| e.commit.as_deref()),
        Some("0123456789abcdef0123456789abcdef01234567")
    );
    assert_eq!(served[0].training_records, Some(RecordContract::pinned()));
    assert_eq!(served[0].check_contract(true), Ok(()));
    assert_eq!(
        served[0].hardware["host"]["backend"], "cuda",
        "the device report as the server gave it"
    );

    // A brain build older than the report says nothing of itself: unknown, never guessed.
    let old = ScriptedEndpoint::start(vec![Card::base("Qwen/Toy")]).with_server(Value::Null);
    let endpoints = BrainEndpoints::new(vec![old.url()], SCRIPTED_KEY).unwrap();
    let unknown = BrainTrainer::new(endpoints, dir.path().join("models"))
        .served_revisions()
        .unwrap();
    assert_eq!(
        (
            unknown[0].engine.clone(),
            unknown[0].training_records.clone(),
            unknown[0].backend.clone()
        ),
        (None, None, None)
    );
    assert_eq!(unknown[0].hardware["host"]["backend"], "cuda");
    // One that says something this build cannot read is refused by name.
    let garbled =
        ScriptedEndpoint::start(vec![Card::base("Qwen/Toy")]).with_server(json!({"engine": 5}));
    let endpoints = BrainEndpoints::new(vec![garbled.url()], SCRIPTED_KEY).unwrap();
    let refused = BrainTrainer::new(endpoints, dir.path().join("models"))
        .served_revisions()
        .unwrap_err();
    assert!(refused.to_string().contains("\"server\""), "{refused}");
}
