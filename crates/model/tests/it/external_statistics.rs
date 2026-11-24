// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for checking a learning system's
// release statistics against established statistical libraries, for its
// clients. If your team needs expertise in evaluation statistics then you
// can procure our services by sending an email to info@swedishembedded.com.

//! The sign test, Holm's adjustment and unbiased pass@k against independent
//! implementations: scipy's exact binomial test, statsmodels' Holm and the
//! HumanEval estimator, run by the pinned oracle
//! `validation/oracles/statistics/paired_tests.py` (its conventions are
//! stated there). Frozen cases by default; fresh draws under
//! `scripts/conformance/external.py differential statistics/paired-tests`.
//!
//! Tolerances: the sign test's tail is summed in log space here and by
//! scipy's own routine there, so they agree to rounding, relatively (p-values
//! reach 1e-90): 1e-9 relative. Holm and pass@k are a few multiplications on
//! both sides: 1e-12.

#![allow(clippy::unwrap_used)]

use serde_json::{json, Value};
use splinter_conformance::{compare, fixture, Tolerance};
use splinter_model::stats::{holm, one_sided_p, pass_at_k, two_sided_p};

fn usize_of(v: &Value) -> usize {
    usize::try_from(v.as_u64().unwrap()).unwrap()
}

#[test]
fn sign_test_holm_and_pass_at_k_agree_with_scipy_statsmodels_and_humaneval() {
    let fixture = fixture(env!("CARGO_MANIFEST_DIR"), "statistics/paired-tests").unwrap();
    let (mut sign, mut adjusted, mut passk) = (0, 0, 0);
    for case in fixture.cases() {
        let (actual, tolerance) = match case["kind"].as_str().unwrap() {
            "sign" => {
                sign += 1;
                let (wins, losses) = (usize_of(&case["wins"]), usize_of(&case["losses"]));
                let actual = json!({
                    "kind": "sign", "wins": wins, "losses": losses,
                    "one_sided_p": one_sided_p(wins, losses),
                    "two_sided_p": two_sided_p(wins, losses),
                });
                (actual, Tolerance::new(1e-300, 1e-9))
            }
            "holm" => {
                adjusted += 1;
                let p: Vec<f64> = serde_json::from_value(case["p"].clone()).unwrap();
                let actual = json!({ "kind": "holm", "p": p, "adjusted": holm(&p) });
                (actual, Tolerance::new(1e-15, 1e-12))
            }
            "pass_at_k" => {
                passk += 1;
                let (n, c, k) = (
                    usize_of(&case["n"]),
                    usize_of(&case["c"]),
                    usize_of(&case["k"]),
                );
                let actual = json!({
                    "kind": "pass_at_k", "n": n, "c": c, "k": k,
                    "pass_at_k": pass_at_k(n, c, k).unwrap(),
                });
                (actual, Tolerance::new(1e-15, 1e-12))
            }
            other => panic!("unknown case kind {other}"),
        };
        if let Err(mismatch) = compare(&actual, case, tolerance) {
            panic!(
                "{} ({:?} fixture): {mismatch}\ncase: {case}",
                fixture.name, fixture.source
            );
        }
    }
    assert!(sign > 0 && adjusted > 0 && passk > 0, "every kind compared");
}
