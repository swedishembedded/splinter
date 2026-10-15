// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for reproducible, auditable
// experiments on frozen data for its clients. If your team needs expertise in
// experiment provenance or held-out evaluation design, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: a frozen artifact records the seed that made it, and every later
//! command on the artifact reads that seed back; a different explicit seed is
//! refused, and an artifact frozen before manifests existed is recorded as
//! made with the seed that was the default then.

// A panic is the failure report.
#![allow(clippy::unwrap_used)]

use splinter_sdk::seed::{frozen, SeedError};

#[test]
fn a_second_freeze_without_a_seed_reads_the_recorded_one() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = dir.path().join("SEED.json");
    // seed: any value other than the fresh default shows it is read back
    assert_eq!(frozen(&manifest, Some(41), 1, None).unwrap(), 41);
    assert_eq!(frozen(&manifest, None, 99, None).unwrap(), 41);
    assert_eq!(frozen(&manifest, Some(41), 99, None).unwrap(), 41);
}

#[test]
fn a_different_explicit_seed_is_refused_and_leaves_the_record() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = dir.path().join("SEED.json");
    frozen(&manifest, Some(41), 1, None).unwrap();
    let err = frozen(&manifest, Some(42), 1, None).unwrap_err();
    assert!(matches!(
        err,
        SeedError::Differs {
            recorded: 41,
            given: 42,
            ..
        }
    ));
    assert_eq!(frozen(&manifest, None, 1, None).unwrap(), 41);
}

#[test]
fn a_freeze_made_before_manifests_is_recorded_once_as_legacy() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = dir.path().join("SEED.json");
    // The artifact exists already, made when the default was 1.
    assert_eq!(frozen(&manifest, None, 77, Some(1)).unwrap(), 1);
    let text = std::fs::read_to_string(&manifest).unwrap();
    assert!(
        text.contains("\"seed\": 1") && text.contains("legacy"),
        "{text}"
    );
    // Written once: later calls read it back whatever the caller now offers.
    assert_eq!(frozen(&manifest, None, 77, None).unwrap(), 1);
    // A different explicit seed cannot rewrite the freeze's history.
    assert!(frozen(&manifest, Some(5), 77, Some(1)).is_err());
}

#[test]
fn a_legacy_freeze_refuses_an_explicit_seed_it_was_not_made_with() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = dir.path().join("SEED.json");
    assert!(frozen(&manifest, Some(5), 77, Some(1)).is_err());
    assert!(!manifest.exists());
}

#[test]
fn a_manifest_that_cannot_be_read_is_an_error_not_a_default() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = dir.path().join("SEED.json");
    std::fs::write(&manifest, "not json").unwrap();
    assert!(matches!(
        frozen(&manifest, None, 1, None),
        Err(SeedError::Malformed { .. })
    ));
}

#[test]
fn a_seed_that_is_decided_but_not_recorded_binds_nothing() {
    use splinter_sdk::seed::FrozenSeed;
    let dir = tempfile::tempdir().unwrap();
    let manifest = dir.path().join("SEED.json");
    let decided = FrozenSeed::decide(&manifest, Some(41), 1, None).unwrap();
    assert_eq!(decided.seed(), 41);
    // The command failed before writing its artifact: the next attempt is free.
    drop(decided);
    assert!(!manifest.exists());
    assert_eq!(frozen(&manifest, Some(42), 1, None).unwrap(), 42);
}
