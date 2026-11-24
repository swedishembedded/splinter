// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for proving a learning system's
// algorithms against independent reference implementations, for its
// clients. If your team needs expertise in conformance testing then you can
// procure our services by sending an email to info@swedishembedded.com.

//! The conformance support's own specification: a comparison that cannot
//! fail certifies nothing, so every way a reference check could pass falsely
//! is shown refused here.

#![allow(clippy::unwrap_used)]

use std::path::Path;

use serde_json::json;
use sha2::{Digest as _, Sha256};
use splinter_conformance::{compare, compare_numbers, fixture_in, FixtureError, Source, Tolerance};

const FIXTURE: &str = r#"{"fixture": "stats/sign", "cases": [{"wins": 5, "p": 0.03125}]}"#;

fn sha(text: &str) -> String {
    Sha256::digest(text.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// A repository with one frozen fixture listed under `recorded`.
fn repo(fixture: &str, recorded: &str) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    std::fs::create_dir_all(root.join("validation/fixtures/stats")).unwrap();
    std::fs::write(root.join("validation/fixtures/stats/sign.json"), fixture).unwrap();
    std::fs::write(
        root.join("validation/fixtures.yaml"),
        format!(
            "schema: splinter.validation-fixtures/v1\nfixtures:\n  stats/sign:\n    \
             path: validation/fixtures/stats/sign.json\n    sha256: {recorded}\n    \
             oracle: validation/oracles/stats/sign.py\n    oracle_sha256: '00'\n    references: [scipy]\n    test: splinter-model/x/y\n"
        ),
    )
    .unwrap();
    dir
}

#[test]
fn a_frozen_fixture_is_read_only_when_its_bytes_are_the_ones_recorded() {
    let good = repo(FIXTURE, &sha(FIXTURE));
    let read = fixture_in(good.path(), "stats/sign", None).unwrap();
    assert_eq!(read.source, Source::Frozen);
    assert_eq!(read.cases().len(), 1);

    // A changed expected value: the bytes no longer hash to the record.
    let edited = FIXTURE.replace("0.03125", "0.04");
    let changed = repo(&edited, &sha(FIXTURE));
    assert!(matches!(
        fixture_in(changed.path(), "stats/sign", None),
        Err(FixtureError::Changed { .. })
    ));
}

#[test]
fn a_fixture_unlisted_missing_or_misnamed_is_refused() {
    let dir = repo(FIXTURE, &sha(FIXTURE));
    assert!(matches!(
        fixture_in(dir.path(), "stats/other", None),
        Err(FixtureError::Unlisted(_))
    ));
    std::fs::remove_file(dir.path().join("validation/fixtures/stats/sign.json")).unwrap();
    assert!(matches!(
        fixture_in(dir.path(), "stats/sign", None),
        Err(FixtureError::Missing(_))
    ));
    let other = r#"{"fixture": "stats/holm", "cases": [1]}"#;
    let misnamed = repo(other, &sha(other));
    assert!(matches!(
        fixture_in(misnamed.path(), "stats/sign", None),
        Err(FixtureError::Malformed(_))
    ));
    assert!(matches!(
        fixture_in(Path::new("/nonexistent"), "stats/sign", None),
        Err(FixtureError::NoManifest(_))
    ));
}

#[test]
fn drawn_cases_are_read_only_from_a_file_that_says_it_was_drawn() {
    let dir = repo(FIXTURE, &sha(FIXTURE));
    let drawn = tempfile::tempdir().unwrap();
    // The frozen fixture copied in is not a draw.
    std::fs::write(drawn.path().join("stats_sign.json"), FIXTURE).unwrap();
    assert!(matches!(
        fixture_in(dir.path(), "stats/sign", Some(drawn.path())),
        Err(FixtureError::Malformed(_))
    ));
    let draw =
        r#"{"fixture": "stats/sign", "differential": true, "seed": 7, "cases": [{"wins": 1}]}"#;
    std::fs::write(drawn.path().join("stats_sign.json"), draw).unwrap();
    let read = fixture_in(dir.path(), "stats/sign", Some(drawn.path())).unwrap();
    assert_eq!(read.source, Source::Differential);
}

#[test]
fn a_comparison_refuses_a_wrong_shape_a_wrong_type_and_a_number_past_its_tolerance() {
    let reference = json!({ "p": [0.5, 0.25], "label": "pass" });
    let tight = Tolerance::new(1e-12, 0.0);
    // Key order is not meaning.
    assert!(compare(
        &json!({ "label": "pass", "p": [0.5, 0.25] }),
        &reference,
        tight
    )
    .is_ok());

    let short = compare(&json!({ "p": [0.5], "label": "pass" }), &reference, tight).unwrap_err();
    assert_eq!(short.at, "$.p");
    let missing = compare(&json!({ "p": [0.5, 0.25] }), &reference, tight).unwrap_err();
    assert_eq!(missing.at, "$.label");
    let extra = compare(
        &json!({ "p": [0.5, 0.25], "label": "pass", "x": 1 }),
        &reference,
        tight,
    )
    .unwrap_err();
    assert_eq!(extra.at, "$.x");
    let typed = compare(
        &json!({ "p": [0.5, "0.25"], "label": "pass" }),
        &reference,
        tight,
    )
    .unwrap_err();
    assert_eq!(typed.at, "$.p[1]");

    // A precision the tolerance does not allow: f32 rounding is refused at
    // f64 tolerance and admitted at f32's.
    let rounded = f64::from(0.1_f32);
    let refused = compare_numbers(&[0.5, rounded], &[0.5, 0.1], tight).unwrap_err();
    assert_eq!(refused.at, "$[1]");
    assert!(refused.seen.max_abs > 1e-9, "{refused}");
    let admitted =
        compare_numbers(&[0.5, rounded], &[0.5, 0.1], Tolerance::new(0.0, 1e-7)).unwrap();
    assert_eq!(admitted.numbers, 2);
}

#[test]
fn a_non_finite_value_is_never_lost_and_agrees_only_with_itself() {
    assert!(compare_numbers(&[f64::NAN], &[0.0], Tolerance::new(1.0, 1.0)).is_err());
    assert!(compare_numbers(&[f64::INFINITY], &[f64::INFINITY], Tolerance::EXACT).is_ok());
    assert!(compare_numbers(
        &[f64::INFINITY],
        &[f64::NEG_INFINITY],
        Tolerance::new(1.0, 1.0)
    )
    .is_err());
    assert!(compare_numbers(&[1e308], &[f64::INFINITY], Tolerance::new(1.0, 1.0)).is_err());
}
