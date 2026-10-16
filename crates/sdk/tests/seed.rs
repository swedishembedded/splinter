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
//! made with the seed that was the default then. A seed is drawn only for an
//! artifact being made now.

// A panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::cell::Cell;

use splinter_sdk::seed::{frozen, resolve, FrozenSeed, SeedError};

/// The default of every freeze made before seed manifests existed.
// seed: the value the legacy artifacts under test were made with
const LEGACY: u64 = 1;

#[test]
fn a_second_freeze_without_a_seed_reads_the_recorded_one() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = dir.path().join("SEED.json");
    let made = resolve("seed", None);
    assert_eq!(frozen(&manifest, "seed", Some(made), None).unwrap(), made);
    assert_eq!(frozen(&manifest, "seed", None, None).unwrap(), made);
    assert_eq!(frozen(&manifest, "seed", Some(made), None).unwrap(), made);
}

#[test]
fn a_different_explicit_seed_is_refused_and_leaves_the_record() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = dir.path().join("SEED.json");
    let made = resolve("seed", None);
    let other = made.wrapping_add(1);
    frozen(&manifest, "seed", Some(made), None).unwrap();
    let err = frozen(&manifest, "seed", Some(other), None).unwrap_err();
    assert!(
        matches!(err, SeedError::Differs { recorded, given, .. } if recorded == made && given == other),
        "{err}"
    );
    assert_eq!(frozen(&manifest, "seed", None, None).unwrap(), made);
}

#[test]
fn a_freeze_made_before_manifests_is_recorded_once_as_legacy() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = dir.path().join("SEED.json");
    // The artifact exists already, made when the default was LEGACY.
    assert_eq!(
        frozen(&manifest, "seed", None, Some(LEGACY)).unwrap(),
        LEGACY
    );
    let text = std::fs::read_to_string(&manifest).unwrap();
    assert!(
        text.contains(&format!("\"seed\": {LEGACY}")) && text.contains("legacy"),
        "{text}"
    );
    // Written once: later calls read it back whatever the caller now offers.
    assert_eq!(frozen(&manifest, "seed", None, None).unwrap(), LEGACY);
    // A different explicit seed cannot rewrite the freeze's history.
    assert!(frozen(&manifest, "seed", Some(LEGACY + 1), Some(LEGACY)).is_err());
}

#[test]
fn a_legacy_freeze_refuses_an_explicit_seed_it_was_not_made_with() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = dir.path().join("SEED.json");
    assert!(frozen(&manifest, "seed", Some(LEGACY + 1), Some(LEGACY)).is_err());
    assert!(!manifest.exists());
}

#[test]
fn a_manifest_that_cannot_be_read_is_an_error_not_a_default() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = dir.path().join("SEED.json");
    std::fs::write(&manifest, "not json").unwrap();
    assert!(matches!(
        frozen(&manifest, "seed", None, None),
        Err(SeedError::Malformed { .. })
    ));
}

#[test]
fn a_seed_that_is_decided_but_not_recorded_binds_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = dir.path().join("SEED.json");
    let first = resolve("seed", None);
    let decided = FrozenSeed::decide(&manifest, "seed", Some(first), None).unwrap();
    assert_eq!(decided.seed(), first);
    // The command failed before writing its artifact: the next attempt is free.
    drop(decided);
    assert!(!manifest.exists());
    let second = first.wrapping_add(1);
    assert_eq!(
        frozen(&manifest, "seed", Some(second), None).unwrap(),
        second
    );
}

#[test]
fn a_given_seed_is_used_and_a_missing_one_is_drawn_afresh() {
    let given = resolve("seed", None);
    assert_eq!(resolve("seed", Some(given)), given);
    // Equal draws would mean the "random" default is a constant.
    assert_ne!(resolve("seed", None), resolve("seed", None));
}

/// Counts the draws a decision makes, and gives a value no other source does.
fn counting(draws: &Cell<u32>) -> impl FnOnce() -> u64 + '_ {
    move || {
        draws.set(draws.get() + 1);
        resolve("seed", None)
    }
}

#[test]
fn a_seed_is_drawn_only_for_an_artifact_being_made_now() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = dir.path().join("SEED.json");
    let draws = Cell::new(0);

    // Being made now, nothing given: one draw, and it is the artifact's seed.
    let made = FrozenSeed::decide_with(&manifest, None, None, counting(&draws)).unwrap();
    assert_eq!(draws.get(), 1);
    let seed = made.record().unwrap();

    // Recorded: read back, no draw.
    let read = FrozenSeed::decide_with(&manifest, None, None, counting(&draws)).unwrap();
    assert_eq!((read.seed(), draws.get()), (seed, 1));

    // Given: used as it is, no draw.
    let other = dir.path().join("OTHER.json");
    let given = seed.wrapping_add(1);
    let named = FrozenSeed::decide_with(&other, Some(given), None, counting(&draws)).unwrap();
    assert_eq!((named.seed(), draws.get()), (given, 1));

    // Made before manifests: its legacy seed, no draw.
    let old = dir.path().join("OLD.json");
    let legacy = FrozenSeed::decide_with(&old, None, Some(LEGACY), counting(&draws)).unwrap();
    assert_eq!((legacy.seed(), draws.get()), (LEGACY, 1));
}

/// The manifests checked in under `fixtures/` were written by the code that
/// made the frozen Jefferson and Adams exams and the lifecourse partitions.
/// Reading one back must give the seed it states, and writing one must give
/// the same bytes: an artifact's address does not move.
#[test]
fn the_manifest_format_is_the_one_already_on_disk() {
    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    // seed: the fixtures state these values
    for (name, given, legacy, seed) in [
        ("SEED.recorded.json", Some(41), None, 41),
        ("SEED.legacy.json", None, Some(LEGACY), LEGACY),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let on_disk = std::fs::read(fixtures.join(name)).unwrap();
        let read = dir.path().join("read.json");
        std::fs::write(&read, &on_disk).unwrap();
        assert_eq!(frozen(&read, "seed", None, None).unwrap(), seed, "{name}");
        let written = dir.path().join("written.json");
        assert_eq!(
            frozen(&written, "seed", given, legacy).unwrap(),
            seed,
            "{name}"
        );
        assert_eq!(std::fs::read(&written).unwrap(), on_disk, "{name}");
    }
}
