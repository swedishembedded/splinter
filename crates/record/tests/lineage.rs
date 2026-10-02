// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements durable, content-addressed experience
// stores for learning agents, for its clients. If your team needs expertise
// in training-data lineage or crash-safe storage, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: a release can be traced back to the experience it was trained on. A
//! dataset records the experience it was projected from and pins the database
//! as it was, a training run records the datasets it read, a release records
//! the run that made it, each once however often it is reported.
#![allow(clippy::unwrap_used)]

use serde_json::json;
use splinter_expdb::model::RecordKind;
use splinter_expdb::{Config, Database};
use splinter_record::clock::FixedClock;
use splinter_record::digest::Digest;
use splinter_record::experience::{Environment, Experience, Provenance, Task};
use splinter_record::experiences::ExperienceStore;
use splinter_record::lineage::DatasetLineage;
use splinter_record::workspace::Workspace;
use splinter_record::StateRoot;
use sven_sdk::atif::{AgentProfile, StepOrigin, TraceStep, Trajectory};

fn experience(answer: &str) -> Experience {
    let task = Task::new(
        "recall",
        vec![],
        Environment {
            kind: "closed-book".into(),
            spec: json!({}),
            snapshot: None,
        },
        "What is six times seven?",
        vec![],
    )
    .unwrap();
    let mut trajectory = Trajectory::new(
        "ATIF-v1.7",
        AgentProfile {
            name: "test".into(),
            version: "1".into(),
            model_name: None,
            tool_definitions: None,
            extra: None,
        },
    );
    trajectory
        .steps
        .push(TraceStep::new(1, StepOrigin::Agent, answer));
    Experience::new(
        task,
        trajectory,
        Some(answer.into()),
        Provenance::new("solver", &FixedClock::new("2026-09-30T12:00:00.000Z")),
    )
    .unwrap()
}

#[test]
fn a_release_traces_back_through_its_run_and_dataset_to_the_experience() {
    let root = StateRoot::new(
        std::env::temp_dir().join(format!("splinter-lineage-{}", std::process::id())),
    );
    let _ = std::fs::remove_dir_all(root.path());
    let workspace = Workspace::at(&root);
    let store = ExperienceStore::new(&workspace);
    let used = store.put(&experience("42")).unwrap();

    let dataset = Digest::of(b"dataset one");
    let lineage = DatasetLineage {
        recipe: json!({ "view": "sft-final", "min_strength": "formal" }),
        records: 1,
    };
    workspace
        .record_dataset(&dataset, &lineage, std::slice::from_ref(&used))
        .unwrap();
    workspace
        .record_dataset(&dataset, &lineage, std::slice::from_ref(&used))
        .unwrap();
    workspace
        .record_training_run("candidate-1", std::slice::from_ref(&dataset), "sft", None)
        .unwrap();
    let release = Digest::of(b"release one");
    workspace
        .record_model(&release, "default", "candidate-1", None)
        .unwrap();
    workspace
        .record_model(&release, "default", "candidate-1", None)
        .unwrap();
    // A later release continues the first.
    let next = Digest::of(b"release two");
    workspace
        .record_training_run(
            "candidate-2",
            std::slice::from_ref(&dataset),
            "sft",
            Some(&release),
        )
        .unwrap();
    workspace
        .record_model(&next, "default", "candidate-2", Some(&release))
        .unwrap();

    let trace = workspace.trace_release(&next).unwrap().unwrap();
    assert_eq!(trace.candidates, vec!["candidate-2", "candidate-1"]);
    assert_eq!(trace.datasets, vec![dataset.clone()]);
    assert_eq!(trace.earlier_releases, vec![release.clone()]);
    assert_eq!(
        trace.attempts, 1,
        "the lineage ends at the attempt that was learned from"
    );
    assert_eq!(
        workspace.trace_release(&Digest::of(b"unknown")).unwrap(),
        None
    );

    let db = Database::open(root.expdb(), Config::default()).unwrap();
    let snapshot = db.snapshot().unwrap();
    let count = |kind| {
        snapshot
            .query(&splinter_expdb::query::Query::all().kind(kind))
            .unwrap()
            .records
            .len()
    };
    assert_eq!(
        count(RecordKind::Dataset),
        1,
        "reported twice, recorded once"
    );
    assert_eq!(count(RecordKind::TrainingRun), 2);
    assert_eq!(count(RecordKind::Model), 2);

    // The dataset kept the database as it was.
    let pins = db.pins().unwrap();
    assert!(pins
        .iter()
        .any(|(holder, _)| *holder == format!("dataset-{}", dataset.hex())));
    let _ = std::fs::remove_dir_all(root.path());
}

#[test]
fn a_run_cannot_read_a_dataset_that_was_never_recorded() {
    let root = StateRoot::new(
        std::env::temp_dir().join(format!("splinter-lineage-missing-{}", std::process::id())),
    );
    let _ = std::fs::remove_dir_all(root.path());
    let workspace = Workspace::at(&root);
    let missing = Digest::of(b"never recorded");
    assert!(workspace
        .record_training_run("candidate-x", &[missing], "sft", None)
        .is_err());
    let _ = std::fs::remove_dir_all(root.path());
}
