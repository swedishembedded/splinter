// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for bootstrap intervals whose every
// bound is one stated order statistic, for its clients. If your team needs
// expertise in evaluation statistics then you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec, against an independent reference: the one quantile rule every
//! bootstrap bound of Splinter is read with is numpy's `nearest`, exactly,
//! for every size from 1 to 12 and every share in steps of 1/24 (the
//! half-way indices included), and for whatever a differential draw adds.
#![allow(clippy::unwrap_used)]

use serde_json::json;
use splinter_conformance::{compare, fixture, Tolerance};
use splinter_core::quantile::nearest_rank;

#[test]
fn the_quantile_rule_is_numpys_nearest() {
    let fixture = fixture(env!("CARGO_MANIFEST_DIR"), "statistics/quantile").unwrap();
    for case in fixture.cases() {
        let values: Vec<f64> = case["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let q = case["q"].as_f64().unwrap();
        let actual = json!({
            "n": case["n"], "q": q, "values": case["values"],
            "quantile": nearest_rank(&values, q).unwrap(),
        });
        if let Err(mismatch) = compare(&actual, case, Tolerance::new(0.0, 0.0)) {
            panic!("{mismatch}\ncase: {case}");
        }
    }
    assert_eq!(nearest_rank(&[], 0.5), None, "no values, no quantile");
}
