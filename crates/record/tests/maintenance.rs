// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements durable, content-addressed experience
// stores for learning agents, for its clients. If your team needs expertise
// in training-data lineage or crash-safe storage, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: maintenance makes the database smaller and quicker to open without
//! changing a thing that is stored, and collection never deletes what a
//! pinned snapshot or a young file still needs.
#![allow(clippy::unwrap_used)]

use serde_json::json;
use splinter_record::annotation::{Annotation, AnnotationBody, Outcome, Producer, Strength};
use splinter_record::clock::FixedClock;
use splinter_record::experience::{Environment, Experience, Provenance, Task};
use splinter_record::experiences::ExperienceStore;
use splinter_record::workspace::Workspace;
use splinter_record::StateRoot;
use sven_sdk::atif::{AgentProfile, StepOrigin, TraceStep, Trajectory};

fn experience(n: usize) -> Experience {
    let task = Task::new(
        "recall",
        vec![],
        Environment {
            kind: "closed-book".into(),
            spec: json!({}),
            snapshot: None,
        },
        format!("question {n}"),
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
        .push(TraceStep::new(1, StepOrigin::Agent, format!("answer {n}")));
    Experience::new(
        task,
        trajectory,
        Some(format!("answer {n}")),
        Provenance::new("solver", &FixedClock::new("2026-09-30T12:00:00.000Z")),
    )
    .unwrap()
}

#[test]
fn maintenance_merges_the_files_commits_leave_and_changes_nothing_stored() {
    let root =
        StateRoot::new(std::env::temp_dir().join(format!("splinter-maint-{}", std::process::id())));
    let _ = std::fs::remove_dir_all(root.path());
    let workspace = Workspace::at(&root);
    let store = ExperienceStore::new(&workspace);
    let mut ids = Vec::new();
    for n in 0..24 {
        // One commit each, as one-off writes are.
        let id = store.put(&experience(n)).unwrap();
        store
            .annotate(&Annotation {
                experience: id.clone(),
                producer: Producer {
                    name: "grader".into(),
                    version: "1".into(),
                },
                body: AnnotationBody::Verdict {
                    outcome: Outcome::Pass,
                    strength: Strength::Formal,
                    evidence: json!({ "n": n }),
                },
            })
            .unwrap();
        ids.push(id);
    }

    let report = workspace.maintain(false).unwrap();
    assert!(report.before.segments > report.after.segments, "{report:?}");
    assert!(report.after.segments >= 1);
    assert_eq!(report.removed, None, "nothing is deleted unless asked");

    // Read through a fresh handle, as the next command would.
    let after = ExperienceStore::new(&Workspace::at(&root));
    assert_eq!(after.list().unwrap().len(), 24);
    for (n, id) in ids.iter().enumerate() {
        assert_eq!(after.get(id).unwrap(), experience(n));
        assert_eq!(after.annotations(id).unwrap().annotations.len(), 1);
    }

    // Collecting right away deletes nothing: the files it would remove are
    // younger than the grace period a writer might still be publishing in.
    let collected = workspace.maintain(true).unwrap();
    assert_eq!(collected.removed, Some(0));
    assert_eq!(after.list().unwrap().len(), 24);
    let _ = std::fs::remove_dir_all(root.path());
}

#[test]
fn what_every_open_walks_stays_short_however_much_was_committed() {
    let root = StateRoot::new(
        std::env::temp_dir().join(format!("splinter-maint-history-{}", std::process::id())),
    );
    let _ = std::fs::remove_dir_all(root.path());
    let workspace = Workspace::at(&root);
    let store = ExperienceStore::new(&workspace);
    for n in 0..150 {
        store.put(&experience(n)).unwrap();
    }
    // The writer checkpoints its own chain as it goes.
    assert!(workspace.storage().unwrap().history < 40);
    let report = workspace.maintain(false).unwrap();
    assert!(report.after.history < 40, "{report:?}");
    let after = ExperienceStore::new(&Workspace::at(&root));
    assert_eq!(after.list().unwrap().len(), 150);
    let _ = std::fs::remove_dir_all(root.path());
}

#[test]
fn a_process_that_is_done_leaves_no_writer_ref_for_later_opens_to_read() {
    let root = StateRoot::new(
        std::env::temp_dir().join(format!("splinter-maint-refs-{}", std::process::id())),
    );
    let _ = std::fs::remove_dir_all(root.path());
    for n in 0..5 {
        let workspace = Workspace::at(&root);
        ExperienceStore::new(&workspace)
            .put(&experience(n))
            .unwrap();
        // The workspace is dropped here, as a command ends.
    }
    let db =
        splinter_expdb::Database::open(root.expdb(), splinter_expdb::Config::default()).unwrap();
    let writers = db
        .backend()
        .list(splinter_expdb::backend::Kind::Ref)
        .unwrap()
        .into_iter()
        .filter(|k| k.name().starts_with("jobs/"))
        .count();
    assert_eq!(writers, 0);
    assert_eq!(
        ExperienceStore::new(&Workspace::at(&root))
            .list()
            .unwrap()
            .len(),
        5
    );
    let _ = std::fs::remove_dir_all(root.path());
}
