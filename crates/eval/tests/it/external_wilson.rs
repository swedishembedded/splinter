// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for honest intervals on small
// counts, for its clients. If your team needs expertise in evaluation
// statistics then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The 95% Wilson score interval (`pilot::Share::wilson`, the one Splinter
//! has; the environment band reads it) against statsmodels'
//! `proportion_confint(method="wilson")`
//! (`validation/oracles/statistics/wilson.py`). Both are a few closed-form
//! operations on the same normal quantile: 1e-12.

use serde_json::json;
use splinter_conformance::{compare, fixture, Tolerance};
use splinter_eval::pilot::Share;

#[test]
fn the_wilson_interval_agrees_with_statsmodels() {
    let fixture = fixture(env!("CARGO_MANIFEST_DIR"), "statistics/wilson").unwrap();
    for case in fixture.cases() {
        let (successes, total) = (
            usize::try_from(case["successes"].as_u64().unwrap()).unwrap(),
            usize::try_from(case["total"].as_u64().unwrap()).unwrap(),
        );
        let share = Share::wilson(successes, total).unwrap();
        let actual =
            json!({ "successes": successes, "total": total, "low": share.low, "high": share.high });
        if let Err(mismatch) = compare(&actual, case, Tolerance::new(1e-12, 1e-12)) {
            panic!(
                "{} ({:?}): {mismatch}\ncase: {case}",
                fixture.name, fixture.source
            );
        }
    }
}
