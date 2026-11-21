// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec of the record keys: the runs made before the checks became options
//! (every check on, gates a and b among them) keep their keys, and every
//! other choice of checks, and a re-grade, keys records of its own.

#![allow(clippy::unwrap_used)]

use super::*;

const PROFILE_SHA256: &str = "1fd893347bfe6f19659ee6a32b05e61a1c498d88f0606961c18c9d11fba1d773";

fn gate_terms() -> Vec<String> {
    ["as given", "solve.sh", "reference", "hint"]
        .iter()
        .map(|t| (*t).to_string())
        .collect()
}

fn key(framing: HintFraming, checks: Checks, terms: &[String]) -> String {
    let variant = draw_variant(PROFILE_SHA256, "Qwen/Qwen3-4B", terms, framing, checks).unwrap();
    attempt_key(
        "005-open-a-line-above__AC7pRSf",
        "brain/Qwen/Qwen3-4B",
        0,
        &variant,
    )
    .unwrap()
}

/// The keys gate a (worked-answer framing) and gate b (private-scratch) stored for the same student trial.
#[test]
fn runs_with_every_check_on_keep_the_keys_they_stored() {
    assert_eq!(
        key(HintFraming::WorkedAnswer, Checks::ALL, &gate_terms()),
        "03f83a78d8b67f9db51985eb557d55d1d2d64bfd78d448bdfa0ac56c522773a8"
    );
    assert_eq!(
        key(HintFraming::PrivateScratch, Checks::ALL, &gate_terms()),
        "6fc0cb58c712c43f876e40145d32c4b4252cd9a852cad561cf453ce7e327d630"
    );
}

#[test]
fn other_checks_and_regrades_key_records_of_their_own() {
    let legacy = key(HintFraming::WorkedAnswer, Checks::ALL, &gate_terms());
    let mut keys = std::collections::BTreeSet::from([legacy.clone()]);
    for checks in [
        Checks::default(),
        Checks {
            sufficiency: true,
            ..Checks::default()
        },
        Checks {
            real_observations: true,
            ..Checks::default()
        },
    ] {
        assert!(keys.insert(key(HintFraming::WorkedAnswer, checks, &gate_terms())));
    }
    let variant = regrade_variant(&[], Checks::default(), None).unwrap();
    let regraded = regrade_key(&legacy, &variant).unwrap();
    assert!(keys.insert(regraded.clone()));
    let other = regrade_variant(&gate_terms(), Checks::default(), None).unwrap();
    assert_ne!(regrade_key(&legacy, &other).unwrap(), regraded);
}
