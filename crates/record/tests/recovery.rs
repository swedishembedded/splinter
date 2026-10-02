// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements durable, recoverable state for learning
// agents, for its clients. If your team needs expertise in backup and
// disaster recovery of training data, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec: the whole state (database and the files it tracks) packs into one
//! deterministic archive and unpacks into an identical, verified state; what
//! goes missing is found, filled from any copy with the right digest, and only
//! when no copy exists written off in an explicit ledger.
#![allow(clippy::unwrap_used)]

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use splinter_record::artifacts::{ArtifactSpec, ArtifactStore};
use splinter_record::digest::Digest;
use splinter_record::recovery::{ArchiveOptions, ArtifactFault, RepairOptions};
use splinter_record::workspace::Workspace;
use splinter_record::StateRoot;

struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!("splinter-rec-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn root(&self, which: &str) -> StateRoot {
        StateRoot::new(self.0.join(which))
    }
    fn file(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A state with a document, a pointer, two artifacts and a second database
/// commit, so the history is more than one manifest.
fn populated(root: &StateRoot) -> (Workspace, Digest, Digest, Digest) {
    let ws = Workspace::at(root);
    let store = ArtifactStore::new(&ws, root);
    let adapter = store
        .put_bytes(
            &vec![7u8; 40_000],
            &ArtifactSpec::new("adapter", "trainer-1").with_extension(".safetensors"),
        )
        .unwrap();
    let dataset = store
        .put_bytes(
            b"{\"a\":1}\n{\"a\":2}\n",
            &ArtifactSpec::new("dataset", "view"),
        )
        .unwrap();
    let document = ws
        .put_document("note", &serde_json::json!({ "about": "the adapter" }))
        .unwrap();
    ws.move_pointer(
        "alias/main",
        None,
        document.as_str(),
        "2026-10-02T10:00:00Z",
    )
    .unwrap();
    ws.commit().unwrap();
    (ws, adapter.digest, dataset.digest, document)
}

fn same_bytes(a: &Path, b: &Path) -> bool {
    std::fs::read(a).unwrap() == std::fs::read(b).unwrap()
}

#[test]
fn the_same_state_packs_into_the_same_bytes() {
    let scratch = Scratch::new("determinism");
    let (ws, ..) = populated(&scratch.root("a"));
    let (first, second) = (scratch.file("one.tar.zst"), scratch.file("two.tar.zst"));
    ws.archive(&first, &ArchiveOptions::default()).unwrap();
    ws.archive(&second, &ArchiveOptions::default()).unwrap();
    assert!(same_bytes(&first, &second));
}

#[test]
fn an_archive_restores_into_an_identical_verified_state() {
    let scratch = Scratch::new("roundtrip");
    let source = scratch.root("a");
    let (ws, adapter, dataset, document) = populated(&source);
    let archive = scratch.file("state.tar.zst");
    let packed = ws.archive(&archive, &ArchiveOptions::default()).unwrap();
    assert_eq!(packed.artifacts, 2);

    let target = scratch.root("b");
    let restored = Workspace::restore(&target, std::slice::from_ref(&archive)).unwrap();
    assert_eq!(restored.artifacts, 2);
    assert!(restored.verified, "{restored:?}");

    let ws2 = Workspace::at(&target);
    let artifacts = ArtifactStore::new(&ws2, &target);
    assert!(same_bytes(
        &artifacts.path(&adapter).unwrap(),
        &ArtifactStore::new(&ws, &source).path(&adapter).unwrap()
    ));
    assert!(artifacts.path(&dataset).is_ok());
    assert!(ws2
        .get_document::<serde_json::Value>("note", &document)
        .unwrap()
        .is_some());
    assert_eq!(
        ws2.pointer("alias/main").unwrap().unwrap().1,
        document.as_str()
    );
    assert!(ws2.verify(true).unwrap().is_sound());

    // Packing the restored state gives the archive it came from.
    let again = scratch.file("again.tar.zst");
    ws2.archive(&again, &ArchiveOptions::default()).unwrap();
    assert!(same_bytes(&archive, &again));
}

#[test]
fn restoring_never_overwrites_a_state_or_accepts_a_tampered_archive() {
    let scratch = Scratch::new("refusals");
    let (ws, ..) = populated(&scratch.root("a"));
    let archive = scratch.file("state.tar.zst");
    ws.archive(&archive, &ArchiveOptions::default()).unwrap();

    let occupied = scratch.root("occupied");
    populated(&occupied);
    assert!(Workspace::restore(&occupied, std::slice::from_ref(&archive)).is_err());

    // Flip a byte in the middle: the damage is found, and nothing is
    // published for a state that cannot be trusted.
    let mut bytes = std::fs::read(&archive).unwrap();
    let middle = bytes.len() / 2;
    bytes[middle] ^= 0xff;
    let tampered = scratch.file("tampered.tar.zst");
    std::fs::write(&tampered, bytes).unwrap();
    let target = scratch.root("c");
    assert!(Workspace::restore(&target, &[tampered]).is_err());
    assert!(!target.expdb().exists() && !target.artifacts().exists());
}

#[test]
fn an_incremental_archive_carries_only_what_is_new_and_restores_with_its_base() {
    let scratch = Scratch::new("incremental");
    let source = scratch.root("a");
    let (ws, ..) = populated(&source);
    let base = scratch.file("base.tar.zst");
    let full = ws.archive(&base, &ArchiveOptions::default()).unwrap();

    let extra = ArtifactStore::new(&ws, &source)
        .put_bytes(b"later", &ArtifactSpec::new("dataset", "view"))
        .unwrap();
    ws.commit().unwrap();
    let newer = scratch.file("newer.tar.zst");
    let delta = ws
        .archive(
            &newer,
            &ArchiveOptions {
                since: Some(base.clone()),
                ..ArchiveOptions::default()
            },
        )
        .unwrap();
    assert!(delta.carried < full.carried, "{delta:?} against {full:?}");
    assert!(std::fs::metadata(&newer).unwrap().len() < std::fs::metadata(&base).unwrap().len());

    let alone = Workspace::restore(&scratch.root("x"), std::slice::from_ref(&newer));
    assert!(alone.is_err(), "the base is needed");
    let restored = Workspace::restore(&scratch.root("y"), &[newer, base]).unwrap();
    assert!(restored.verified);
    let ws2 = Workspace::at(&scratch.root("y"));
    assert!(ArtifactStore::new(&ws2, &scratch.root("y"))
        .path(&extra.digest)
        .is_ok());
}

#[test]
fn what_goes_missing_is_found_and_filled_from_any_copy() {
    let scratch = Scratch::new("repair");
    let source = scratch.root("a");
    let (ws, adapter, dataset, _) = populated(&source);
    let archive = scratch.file("state.tar.zst");
    ws.archive(&archive, &ArchiveOptions::default()).unwrap();
    let artifacts = ArtifactStore::new(&ws, &source);

    // Lose an artifact outright and corrupt another without changing its size.
    let lost = artifacts.path(&adapter).unwrap();
    std::fs::remove_file(&lost).unwrap();
    let damaged = artifacts.path(&dataset).unwrap();
    let mut bytes = std::fs::read(&damaged).unwrap();
    bytes[2] ^= 0xff;
    std::fs::set_permissions(&damaged, std::fs::Permissions::from_mode(0o644)).unwrap();
    std::fs::write(&damaged, bytes).unwrap();

    let shallow = ws.verify(false).unwrap();
    assert!(!shallow.is_sound());
    assert!(shallow
        .artifacts
        .iter()
        .any(|a| a.digest == adapter && a.fault == ArtifactFault::Missing));
    assert!(
        !shallow.artifacts.iter().any(|a| a.digest == dataset),
        "damage that keeps the size is only found deep"
    );
    let deep = ws.verify(true).unwrap();
    assert!(deep
        .artifacts
        .iter()
        .any(|a| a.digest == dataset && matches!(a.fault, ArtifactFault::Corrupt)));
    assert!(deep.database.problems.is_empty(), "{deep:?}");

    // Without a copy nothing is changed or invented.
    let unrepaired = ws.repair(&RepairOptions::default()).unwrap();
    assert_eq!(unrepaired.filled, 0);
    assert_eq!(unrepaired.unresolved, 2, "{unrepaired:?}");
    assert!(!ws.verify(true).unwrap().is_sound());

    // The archive is a copy; so is another state root.
    let from_archive = RepairOptions {
        from: vec![archive],
        ..RepairOptions::default()
    };
    let repaired = ws.repair(&from_archive).unwrap();
    assert_eq!(
        (repaired.filled, repaired.unresolved),
        (2, 0),
        "{repaired:?}"
    );
    assert!(ws.verify(true).unwrap().is_sound());
    assert!(artifacts.path(&adapter).is_ok());

    // So is a segment of the database.
    let segment = first_file(&source.expdb().join("segments"));
    std::fs::remove_file(&segment).unwrap();
    assert_eq!(ws.verify(false).unwrap().database.problems.len(), 1);
    let repaired = ws.repair(&from_archive).unwrap();
    assert_eq!(
        (repaired.filled, repaired.unresolved),
        (1, 0),
        "{repaired:?}"
    );
    assert!(ws.verify(true).unwrap().is_sound());
    assert!(segment.exists());
}

#[test]
fn a_loss_nothing_can_restore_is_written_off_in_a_ledger_only_when_accepted() {
    let scratch = Scratch::new("ledger");
    let source = scratch.root("a");
    let (ws, adapter, _, _) = populated(&source);
    let artifacts = ArtifactStore::new(&ws, &source);
    std::fs::remove_file(artifacts.path(&adapter).unwrap()).unwrap();

    let refused = ws.repair(&RepairOptions::default()).unwrap();
    assert_eq!(
        (refused.lost.len(), refused.unresolved),
        (0, 1),
        "{refused:?}"
    );
    assert!(ws.losses().unwrap().is_empty());

    let accepted = ws
        .repair(&RepairOptions {
            accept_loss: true,
            ..RepairOptions::default()
        })
        .unwrap();
    assert_eq!(
        (accepted.lost.len(), accepted.unresolved),
        (1, 0),
        "{accepted:?}"
    );
    let ledger = ws.losses().unwrap();
    assert_eq!(ledger.len(), 1);
    assert_eq!(ledger[0].id, adapter.to_string());
    assert!(
        ws.verify(true).unwrap().is_sound(),
        "written off is not damage"
    );
    let refusal = artifacts.path(&adapter).unwrap_err().to_string();
    assert!(refusal.contains(adapter.as_str()), "{refusal}");

    // A segment nothing can restore is withdrawn, with what it held counted,
    // so the rest of the database opens again.
    let segment = first_file(&source.expdb().join("segments"));
    std::fs::remove_file(&segment).unwrap();
    let refused = ws.repair(&RepairOptions::default()).unwrap();
    assert_eq!(refused.unresolved, 1, "{refused:?}");
    let accepted = ws
        .repair(&RepairOptions {
            accept_loss: true,
            ..RepairOptions::default()
        })
        .unwrap();
    assert_eq!(
        (accepted.quarantined, accepted.unresolved),
        (1, 0),
        "{accepted:?}"
    );
    assert!(ws.losses().unwrap().len() >= 2);
    assert!(ws.verify(true).unwrap().is_sound());
}

fn first_file(dir: &Path) -> PathBuf {
    let entry = std::fs::read_dir(dir)
        .unwrap()
        .flatten()
        .next()
        .unwrap_or_else(|| panic!("no file under {}", dir.display()));
    let path = entry.path();
    if path.is_dir() {
        first_file(&path)
    } else {
        path
    }
}
