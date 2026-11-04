// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements durable, content-addressed experience
// stores for learning agents, for its clients. If your team needs expertise
// in training-data lineage or crash-safe storage, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: a file a tool needs (an adapter, a dataset) stays a real file at a
//! stable path, and the database knows exactly what it is, who made it and
//! whether it is still there. A file is written first and made official by one
//! commit; a crash between the two leaves only an orphan that a sweep removes.
#![allow(clippy::unwrap_used)]

use std::path::PathBuf;
use std::time::Duration;

use splinter_core::digest::Digest;
use splinter_store::artifacts::{ArtifactSpec, ArtifactState, ArtifactStore};
use splinter_store::experiences::StoreError;
use splinter_store::workspace::Workspace;
use splinter_store::StateRoot;

struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!("splinter-art-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn root(&self) -> StateRoot {
        StateRoot::new(self.0.join("state"))
    }
    fn store(&self) -> ArtifactStore {
        ArtifactStore::new(&Workspace::at(&self.root()), &self.root())
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn adapter() -> ArtifactSpec {
    ArtifactSpec::new("adapter", "trainer-1").with_extension(".safetensors")
}

#[test]
fn a_file_is_kept_at_a_stable_path_and_the_database_knows_what_it_is() {
    let scratch = Scratch::new("put");
    let source = scratch.0.join("adapter.out");
    std::fs::write(&source, b"weights weights weights").unwrap();
    let store = scratch.store();

    let made = store.put_file(&source, &adapter().with_sha256()).unwrap();
    assert_eq!(made.digest, Digest::of(b"weights weights weights"));
    assert_eq!(
        made.sha256,
        Some(Digest::sha256_of(b"weights weights weights"))
    );
    assert_eq!(made.size, 23);
    assert_eq!(
        (made.role.as_str(), made.producer.as_str()),
        ("adapter", "trainer-1")
    );

    std::fs::remove_file(&source).unwrap();
    let path = store.path(&made.digest).unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), b"weights weights weights");
    assert!(
        path.to_string_lossy().ends_with(".safetensors"),
        "a tool sees the name it expects"
    );
    assert!(
        std::fs::metadata(&path).unwrap().permissions().readonly(),
        "an artifact is never edited in place"
    );
    assert_eq!(
        scratch.store().path(&made.digest).unwrap(),
        path,
        "the same path in another process"
    );
    assert_eq!(scratch.store().get(&made.digest).unwrap(), made);
    assert_eq!(scratch.store().list().unwrap(), vec![made]);
}

#[test]
fn the_same_bytes_are_one_artifact() {
    let scratch = Scratch::new("once");
    let store = scratch.store();
    let first = store.put_bytes(b"same", &adapter()).unwrap();
    let second = store.put_bytes(b"same", &adapter()).unwrap();
    assert_eq!(first, second);
    assert_eq!(store.list().unwrap().len(), 1);
}

#[test]
fn a_file_without_a_commit_is_an_orphan_that_a_sweep_removes() {
    let scratch = Scratch::new("orphan");
    let store = scratch.store();
    let kept = store.put_bytes(b"kept", &adapter()).unwrap();
    // A crash after the file was written and before the commit: the file is
    // in place and the database never heard of it.
    let orphan = scratch.root().path().join("artifacts/ab");
    std::fs::create_dir_all(&orphan).unwrap();
    std::fs::write(
        orphan.join(format!("{}.safetensors", Digest::of(b"orphan").hex())),
        b"orphan",
    )
    .unwrap();
    assert!(matches!(
        store.path(&Digest::of(b"orphan")),
        Err(StoreError::UnknownArtifact(_))
    ));

    assert_eq!(
        store.sweep_orphans(Duration::from_secs(3600)).unwrap(),
        0,
        "a young file may be mid-write"
    );
    assert_eq!(store.sweep_orphans(Duration::ZERO).unwrap(), 1);
    assert!(
        store.path(&kept.digest).unwrap().exists(),
        "what the database names is never swept"
    );
}

#[test]
fn a_missing_or_damaged_file_is_reported_not_served() {
    let scratch = Scratch::new("damage");
    let store = scratch.store();
    let made = store.put_bytes(b"precious bytes", &adapter()).unwrap();
    assert_eq!(
        store.check(&made.digest, true).unwrap(),
        ArtifactState::Sound
    );

    let path = store.path(&made.digest).unwrap();
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
    }
    std::fs::write(&path, b"precious bytez").unwrap();
    assert_eq!(
        store.check(&made.digest, false).unwrap(),
        ArtifactState::Sound,
        "same size: only a deep check can tell"
    );
    assert!(matches!(
        store.check(&made.digest, true).unwrap(),
        ArtifactState::Corrupt { .. }
    ));

    std::fs::remove_file(&path).unwrap();
    assert_eq!(
        store.check(&made.digest, false).unwrap(),
        ArtifactState::Missing
    );
    match store.path(&made.digest) {
        Err(StoreError::MissingArtifact { digest, role, .. }) => {
            assert_eq!(digest, made.digest);
            assert_eq!(role, "adapter");
        }
        other => panic!("{other:?}"),
    }
}
