// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for decisions that stay honest
// when only a handful of units were measured, for its clients. If your team
// needs expertise in small-sample evaluation statistics then you can procure
// our services by sending an email to info@swedishembedded.com.

//! Student's t against scipy.stats.t (`validation/oracles/statistics/student_t.py`):
//! the distribution function and the quantile over a grid of degrees of
//! freedom and levels, frozen, or drawn under
//! `scripts/conformance/external.py differential statistics/student-t`.
//!
//! Tolerance: both sides evaluate the regularized incomplete beta to near
//! f64 resolution; the quantile is a bisection to the last bit and the
//! distribution function's slope is at most 0.4, so 1e-10 absolute and
//! relative leaves room for the continued fraction's rounding only.

#![allow(clippy::unwrap_used)]

use serde_json::json;
use splinter_conformance::{compare, fixture, Tolerance};
use splinter_eval::student::{cdf, quantile};

#[test]
fn the_t_distribution_and_its_quantiles_agree_with_scipy() {
    let fixture = fixture(env!("CARGO_MANIFEST_DIR"), "statistics/student-t").unwrap();
    for case in fixture.cases() {
        let (df, level, t) = (
            case["df"].as_f64().unwrap(),
            case["level"].as_f64().unwrap(),
            case["t"].as_f64().unwrap(),
        );
        let actual = json!({
            "df": case["df"], "level": case["level"], "t": case["t"],
            "quantile": quantile(level, df).unwrap(),
            "cdf": cdf(t, df).unwrap(),
        });
        if let Err(mismatch) = compare(&actual, case, Tolerance::new(1e-10, 1e-10)) {
            panic!(
                "{} ({:?}): {mismatch}\ncase: {case}",
                fixture.name, fixture.source
            );
        }
    }
}
