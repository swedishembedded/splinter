// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: a damaged or incomplete database is described, not refused. Verifying
//! lists every missing or corrupt file with what it held, a hole can be filled
//! from any copy that has the right bytes, and a file nothing can restore can be
//! quarantined on request, so the rest of the database opens again.
#![allow(clippy::unwrap_used)]

mod common;

use std::path::{Path, PathBuf};

use common::Scratch;
use splinter_expdb::manifest::{Fault, ProblemKind};
use splinter_expdb::model::Entity;
use splinter_expdb::{Config, Database, WriterIdentity};

fn files_under(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let path = entry.path();
        if path.is_dir() {
            found.extend(files_under(&path));
        } else {
            found.push(path);
        }
    }
    found.sort();
    found
}

/// A database with three commits of entities, two blobs and a signal.
fn populated(scratch: &Scratch) -> Database {
    let db = scratch.open();
    let mut collector = db
        .collector(&WriterIdentity::new("exp", "job", "node", 0))
        .unwrap();
    for n in 0..3 {
        collector
            .put_entity(&Entity::new("thing", serde_json::json!({ "n": n })))
            .unwrap();
        collector.put_blob(&common::noise(5_000, n + 1)).unwrap();
        collector.flush().unwrap();
    }
    db.signal("cancel/run-1", "stop").unwrap();
    db
}

#[test]
fn a_sound_database_verifies_with_nothing_to_report() {
    let scratch = Scratch::new();
    let db = populated(&scratch);
    for deep in [false, true] {
        let report = db.verify(deep).unwrap();
        assert!(report.problems.is_empty(), "{report:?}");
        assert!(
            report.segments >= 3 && report.blob_packs >= 1 && report.manifests >= 3,
            "{report:?}"
        );
    }
}

#[test]
fn a_missing_segment_is_listed_with_how_many_records_it_held_and_the_rest_still_opens() {
    let scratch = Scratch::new();
    let db = populated(&scratch);
    let segments = files_under(&scratch.dir.path().join("segments"));
    std::fs::remove_file(&segments[0]).unwrap();

    let report = db.verify(false).unwrap();
    assert_eq!(report.problems.len(), 1, "{report:?}");
    let problem = &report.problems[0];
    assert_eq!(problem.kind, ProblemKind::Segment);
    assert_eq!(problem.fault, Fault::Missing);
    assert_eq!(problem.records, 1);
    assert!(segments[0].to_string_lossy().contains(&problem.id));
    assert_eq!(
        problem.path,
        format!("segments/{}/{}.seg", &problem.id[..2], problem.id)
    );
}

#[test]
fn damage_that_keeps_the_size_is_found_by_a_deep_check_only() {
    let scratch = Scratch::new();
    let db = populated(&scratch);
    let packs = files_under(&scratch.dir.path().join("blobs"));
    let mut bytes = std::fs::read(&packs[0]).unwrap();
    let middle = bytes.len() / 2;
    bytes[middle] ^= 0xff;
    std::fs::write(&packs[0], bytes).unwrap();

    assert!(
        db.verify(false).unwrap().problems.is_empty(),
        "the size is unchanged"
    );
    let report = db.verify(true).unwrap();
    assert_eq!(report.problems.len(), 1, "{report:?}");
    assert_eq!(report.problems[0].kind, ProblemKind::BlobPack);
    assert!(matches!(report.problems[0].fault, Fault::Corrupt(_)));
}

#[test]
fn an_unreadable_ref_or_a_missing_manifest_is_reported_and_the_walk_goes_on() {
    let scratch = Scratch::new();
    let db = populated(&scratch);
    // A ref that names nothing readable, as a damaged file would.
    std::fs::create_dir_all(scratch.dir.path().join("refs/jobs")).unwrap();
    std::fs::write(
        scratch.dir.path().join("refs/jobs/damaged"),
        b"not a content id",
    )
    .unwrap();
    // The oldest manifest of the chain gone.
    let manifests = files_under(&scratch.dir.path().join("manifests"));
    std::fs::remove_file(&manifests[0]).unwrap();

    let report = db.verify(false).unwrap();
    let kinds: Vec<_> = report.problems.iter().map(|p| p.kind).collect();
    assert!(kinds.contains(&ProblemKind::Ref), "{report:?}");
    assert!(kinds.contains(&ProblemKind::Manifest), "{report:?}");
}

#[test]
fn a_hole_is_filled_from_a_copy_with_the_right_bytes_and_only_those() {
    let scratch = Scratch::new();
    let db = populated(&scratch);
    let copy = Scratch::new();
    for file in files_under(scratch.dir.path()) {
        let relative = file.strip_prefix(scratch.dir.path()).unwrap();
        let target = copy.dir.path().join(relative);
        std::fs::create_dir_all(target.parent().unwrap()).unwrap();
        std::fs::copy(&file, &target).unwrap();
    }
    let segments = files_under(&scratch.dir.path().join("segments"));
    let lost = segments[0].clone();
    let original = std::fs::read(&lost).unwrap();
    std::fs::remove_file(&lost).unwrap();
    assert_eq!(db.verify(false).unwrap().problems.len(), 1);

    let problem = db.verify(false).unwrap().problems.remove(0);
    let from_copy = std::fs::read(copy.dir.path().join(&problem.path)).unwrap();
    assert!(
        db.fill(&problem.path, &from_copy).unwrap(),
        "the hole is filled"
    );
    assert_eq!(std::fs::read(&lost).unwrap(), original);
    assert!(db.verify(true).unwrap().problems.is_empty());

    // Bytes that are not what the name says are refused, whoever offers them.
    std::fs::remove_file(&lost).unwrap();
    assert!(db.fill(&problem.path, b"not the segment").is_err());
    assert!(!lost.exists());
}

#[test]
fn a_file_nothing_can_restore_is_quarantined_so_the_rest_opens_again() {
    let scratch = Scratch::new();
    let db = populated(&scratch);
    let segments = files_under(&scratch.dir.path().join("segments"));
    std::fs::remove_file(&segments[1]).unwrap();
    assert!(
        db.snapshot().unwrap().entities("thing").is_err(),
        "a missing segment blocks reading"
    );

    let problem = db.verify(false).unwrap().problems.remove(0);
    let quarantined = db.quarantine(&problem).unwrap();
    assert_eq!(quarantined.records, 1, "the cost is stated");
    assert!(db.verify(true).unwrap().problems.is_empty());
    let reopened = Database::open(scratch.dir.path(), Config::default()).unwrap();
    assert_eq!(
        reopened
            .snapshot()
            .unwrap()
            .entities("thing")
            .unwrap()
            .len(),
        2,
        "what survives opens, and the lost record is not guessed"
    );
}

#[test]
fn a_snapshot_names_every_file_it_needs_and_copying_them_reproduces_it() {
    let scratch = Scratch::new();
    let db = populated(&scratch);
    let snapshot = db.snapshot().unwrap();
    let files = snapshot.files().unwrap();
    assert!(files.windows(2).all(|w| w[0] < w[1]), "sorted, each once");

    let copy = Scratch::new();
    let other = copy.open();
    for path in &files {
        let bytes = std::fs::read(scratch.dir.path().join(path)).unwrap();
        assert!(other.fill(path, &bytes).unwrap(), "{path}");
    }
    other.adopt(snapshot.id()).unwrap();
    let reopened = other.snapshot().unwrap();
    assert_eq!(
        reopened.records().unwrap().len(),
        snapshot.records().unwrap().len()
    );
    assert_eq!(other.verify(true).unwrap().problems, Vec::new());
}
