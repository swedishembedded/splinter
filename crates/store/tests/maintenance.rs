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

use atif::{AgentProfile, StepOrigin, TraceStep, Trajectory};
use serde_json::json;
use splinter_core::annotation::{Annotation, AnnotationBody, Outcome, Producer, Strength};
use splinter_core::clock::FixedClock;
use splinter_core::experience::{Environment, Experience, Provenance, Task};
use splinter_store::experiences::ExperienceStore;
use splinter_store::workspace::Workspace;
use splinter_store::StateRoot;

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

#[test]
fn a_long_running_reader_survives_the_files_it_opened_being_collected() {
    let root = StateRoot::new(
        std::env::temp_dir().join(format!("splinter-maint-reader-{}", std::process::id())),
    );
    let _ = std::fs::remove_dir_all(root.path());
    let writer = ExperienceStore::new(&Workspace::at(&root));
    let ids: Vec<_> = (0..6)
        .map(|n| writer.put(&experience(n)).unwrap())
        .collect();

    // A reader opens its snapshot, names the six small segments.
    let reader = ExperienceStore::new(&Workspace::at(&root));
    assert_eq!(reader.list().unwrap().len(), 6);

    // Meanwhile maintenance merges them and, long after the grace period,
    // they are collected: here the files are simply removed.
    let maintainer = Workspace::at(&root);
    maintainer.maintain(false).unwrap();
    for entry in std::fs::read_dir(root.expdb().join("segments"))
        .unwrap()
        .flatten()
    {
        for file in std::fs::read_dir(entry.path()).unwrap().flatten() {
            let merged = {
                let db =
                    splinter_expdb::Database::open(root.expdb(), splinter_expdb::Config::default())
                        .unwrap();
                db.snapshot().unwrap().segments().iter().any(|s| {
                    file.file_name()
                        .to_string_lossy()
                        .starts_with(&s.id.to_string())
                })
            };
            if !merged {
                std::fs::remove_file(file.path()).unwrap();
            }
        }
    }

    // The reader's snapshot names files that are gone; it refreshes and reads.
    assert_eq!(reader.get(&ids[3]).unwrap(), experience(3));
    assert_eq!(reader.list().unwrap().len(), 6);
    let _ = std::fs::remove_dir_all(root.path());
}

/// A file a crash left in the artifact directory, that no commit made
/// official, goes when collection is asked for; one just written, or one a
/// commit recorded, stays.
#[test]
fn collection_sweeps_artifact_files_no_commit_made_official() {
    use splinter_store::artifacts::{ArtifactSpec, ArtifactStore};
    let root = StateRoot::new(
        std::env::temp_dir().join(format!("splinter-maint-orphan-{}", std::process::id())),
    );
    let _ = std::fs::remove_dir_all(root.path());
    let ws = Workspace::at(&root);
    let kept = ArtifactStore::new(&ws, &root)
        .put_bytes(b"recorded", &ArtifactSpec::new("dataset", "test"))
        .unwrap();

    let fan = root.artifacts().join("ab");
    std::fs::create_dir_all(&fan).unwrap();
    let stale = fan.join(format!("ab{}", "0".repeat(62)));
    let fresh = fan.join(format!("ab{}", "1".repeat(62)));
    std::fs::write(&stale, b"orphan").unwrap();
    std::fs::write(&fresh, b"orphan").unwrap();
    let long_ago = std::time::SystemTime::now() - std::time::Duration::from_secs(7 * 24 * 3600);
    std::fs::File::options()
        .write(true)
        .open(&stale)
        .unwrap()
        .set_modified(long_ago)
        .unwrap();

    let without = ws.maintain(false).unwrap();
    assert_eq!(without.orphan_artifacts, None);
    assert!(stale.exists());

    let with = ws.maintain(true).unwrap();
    assert_eq!(with.orphan_artifacts, Some(1));
    assert!(!stale.exists() && fresh.exists());
    assert!(ArtifactStore::new(&ws, &root).path(&kept.digest).is_ok());
    assert_eq!(with.after.artifacts, 1);
    let _ = std::fs::remove_dir_all(root.path());
}

/// A name that holds a snapshot alive is listed with what it holds and can be
/// released by name; releasing one that does not exist is refused.
#[test]
fn pins_are_listed_and_released_by_name() {
    let root = StateRoot::new(
        std::env::temp_dir().join(format!("splinter-maint-pins-{}", std::process::id())),
    );
    let _ = std::fs::remove_dir_all(root.path());
    let ws = Workspace::at(&root);
    ExperienceStore::new(&ws).put(&experience(0)).unwrap();
    ws.hold("dataset-x").unwrap();
    let held = ws.pins().unwrap();
    assert_eq!(held.len(), 1);
    assert_eq!(held[0].holder, "dataset-x");
    assert_eq!(ws.storage().unwrap().pins, 1);

    ws.release_pin("dataset-x").unwrap();
    assert!(ws.pins().unwrap().is_empty());
    assert!(ws.release_pin("dataset-x").is_err(), "nothing to release");
    let _ = std::fs::remove_dir_all(root.path());
}
