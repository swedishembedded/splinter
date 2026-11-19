// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements durable learning state that fits on
// quota-limited shared filesystems, for its clients. If your team needs
// expertise in migrating stores in place without losing a byte, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: a new state root keeps a bounded number of database files however
//! many experiences it holds; `state pack` rewrites a loose state into the
//! packed layout with every experience, annotation and signal intact, and a
//! pack that dies midway is finished by the next open; a packed state is
//! repaired from a packed copy.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use atif::{AgentProfile, StepOrigin, TraceStep, Trajectory};
use serde_json::json;
use splinter_core::annotation::{Annotation, AnnotationBody, Outcome, Producer, Strength};
use splinter_core::clock::FixedClock;
use splinter_core::experience::{Environment, Experience, ExperienceId, Provenance, Task};
use splinter_expdb::backend::{repack, Kind, Layout};
use splinter_expdb::{Config, Database};
use splinter_store::experiences::ExperienceStore;
use splinter_store::recovery::RepairOptions;
use splinter_store::workspace::Workspace;
use splinter_store::StateRoot;

struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        static N: AtomicU32 = AtomicU32::new(0);
        let path = std::env::temp_dir().join(format!(
            "splinter-layout-{name}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&path);
        Self(path)
    }

    fn root(&self, name: &str) -> StateRoot {
        StateRoot::new(self.0.join(name))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

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
    let mut trajectory = Trajectory::new("ATIF-v1.7", AgentProfile::new("test", "1"));
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

/// Experiences `from..to` ingested by one process, each graded and marked
/// done, as a practice run's ingest leaves them.
fn ingest(root: &StateRoot, from: usize, to: usize) -> Vec<ExperienceId> {
    let ws = Workspace::at(root);
    let store = ExperienceStore::new(&ws);
    let ids = (from..to)
        .map(|n| {
            let id = store.put(&experience(n)).unwrap();
            store
                .annotate(&Annotation {
                    experience: id.clone(),
                    producer: Producer {
                        name: "grader".into(),
                        version: "1".into(),
                    },
                    body: AnnotationBody::Verdict {
                        outcome: if n % 2 == 0 {
                            Outcome::Pass
                        } else {
                            Outcome::Fail
                        },
                        strength: Strength::Executable,
                        evidence: json!({ "n": n }),
                    },
                })
                .unwrap();
            ws.commit().unwrap();
            ws.signal(&format!("pointer/done-{n}"), "ok").unwrap();
            id
        })
        .collect();
    ws.commit().unwrap();
    ids
}

fn files_under(dir: &Path) -> usize {
    let mut count = 0;
    let mut pending = vec![dir.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_dir() {
                pending.push(entry.path());
            } else {
                count += 1;
            }
        }
    }
    count
}

fn loose(root: &StateRoot) {
    Database::open_as(root.expdb(), Layout::Loose, Config::default()).unwrap();
}

/// Everything a later stage reads of the experiences: the ids, each
/// experience, and what its annotations decide.
fn contents(root: &StateRoot) -> Vec<(ExperienceId, Experience, usize)> {
    let store = ExperienceStore::new(&Workspace::at(root));
    let mut ids = store.list().unwrap();
    ids.sort_by(|a, b| a.as_str().cmp(b.as_str()));
    ids.into_iter()
        .map(|id| {
            let experience = store.get(&id).unwrap();
            let notes = store.annotations(&id).unwrap().annotations.len();
            (id, experience, notes)
        })
        .collect()
}

#[test]
fn a_new_state_keeps_a_bounded_number_of_database_files() {
    let scratch = Scratch::new("bounded");
    let root = scratch.root("state");
    ingest(&root, 0, 25);
    let after_few = files_under(&root.expdb());
    ingest(&root, 25, 100);
    let after_many = files_under(&root.expdb());
    assert_eq!(contents(&root).len(), 100);
    assert!(
        after_many <= after_few + 1 && after_many <= 8,
        "{after_few} files for 25 experiences, {after_many} for 100"
    );
}

#[test]
fn packing_a_loose_state_keeps_every_experience_and_frees_its_files() {
    let scratch = Scratch::new("pack");
    let root = scratch.root("state");
    loose(&root);
    let ids = ingest(&root, 0, 30);
    let before = contents(&root);
    assert_eq!(before.len(), ids.len());
    let loose_files = files_under(&root.expdb());

    let packed = Workspace::at(&root).pack_state().unwrap();
    assert_eq!(packed.from, "loose");
    assert_eq!(packed.files_before as usize, loose_files);
    assert!(packed.files_after <= 8, "{packed:?}");
    assert!(packed.files_before > 60, "{packed:?}");
    assert_eq!(
        Layout::of_root(&root.expdb()).unwrap(),
        Some(Layout::Packed)
    );
    assert!(!root.path().join("expdb.loose").exists());
    assert!(!root.path().join("expdb.packing").exists());

    assert_eq!(
        contents(&root),
        before,
        "the same experiences, read the same"
    );
    let ws = Workspace::at(&root);
    assert_eq!(ws.signals("pointer/done-").unwrap().len(), 30);
    assert!(ws.verify(true).unwrap().is_sound());
    drop(ws);

    // It keeps working: more experience, and packing again compacts it.
    ingest(&root, 30, 35);
    let again = Workspace::at(&root).pack_state().unwrap();
    assert_eq!(again.from, "packed");
    assert_eq!(contents(&root).len(), 35);
}

#[test]
fn a_pack_that_died_between_its_moves_is_finished_by_the_next_open() {
    let scratch = Scratch::new("died");
    let root = scratch.root("state");
    loose(&root);
    ingest(&root, 0, 6);
    let before = contents(&root);

    // The copy was built and verified, the old database moved aside, and
    // then the process died before the copy was moved into place.
    repack(&root.expdb(), &root.path().join("expdb.packing")).unwrap();
    std::fs::rename(root.expdb(), root.path().join("expdb.loose")).unwrap();

    assert_eq!(contents(&root), before);
    assert_eq!(
        Layout::of_root(&root.expdb()).unwrap(),
        Some(Layout::Packed)
    );
    assert!(!root.path().join("expdb.loose").exists());
}

#[test]
fn a_packed_state_is_repaired_from_a_packed_copy() {
    let scratch = Scratch::new("repair");
    let root = scratch.root("a");
    ingest(&root, 0, 8);
    let copy = scratch.root("b");
    repack(&root.expdb(), &copy.expdb()).unwrap();

    // A segment is withdrawn behind the workspace's back.
    let ws = Workspace::at(&root);
    assert!(ws.verify(true).unwrap().is_sound());
    let other = Database::open(root.expdb(), Config::default()).unwrap();
    let segment = other.snapshot().unwrap().segments()[0].key().unwrap();
    assert_eq!(segment.kind(), Kind::Segment);
    other.backend().remove(&segment).unwrap();
    assert_eq!(ws.verify(false).unwrap().database.problems.len(), 1);

    let repaired = ws
        .repair(&RepairOptions {
            from: vec![copy.path().to_path_buf()],
            ..RepairOptions::default()
        })
        .unwrap();
    assert_eq!(
        (repaired.filled, repaired.unresolved),
        (1, 0),
        "{repaired:?}"
    );
    assert!(ws.verify(true).unwrap().is_sound());
    drop(ws);
    assert_eq!(contents(&root).len(), 8);
}
