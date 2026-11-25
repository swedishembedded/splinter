// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measurement of what a continually updated
// model keeps and loses, held to an independent implementation, for its
// clients. If your team needs expertise in continual-learning evaluation,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! The continual-learning metrics (`continual::ContinualMetrics`: ACC,
//! backward transfer, forgetting, forward transfer, learning gain and the
//! gain curve) against an independent numpy implementation written from
//! their documentation (`validation/oracles/statistics/continual.py`, plan
//! 12E). Each frozen case is a results matrix of one to five generations
//! and up to seven tasks in three families, with unmeasured cells and base
//! scores; every metric agrees in value, in the tasks and units it rests on,
//! in each unit's value, and in being measured at all. Means of a few
//! sixteenths: 1e-12.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use serde_json::Value;
use splinter_conformance::{compare, fixture, Tolerance};
use splinter_eval::continual::{ContinualMetrics, ResultsMatrix, TaskColumn};

/// `value` without the explanations a metric that was not measured gives:
/// the reference says only that it was not.
fn without_reasons(value: &mut Value) {
    match value {
        Value::Object(map) => {
            map.remove("reason");
            map.values_mut().for_each(without_reasons);
        }
        Value::Array(items) => items.iter_mut().for_each(without_reasons),
        _ => {}
    }
}

fn observed(case: &Value) -> Value {
    let tasks: Vec<TaskColumn> = serde_json::from_value(case["tasks"].clone()).unwrap();
    let generations = case["generations"].as_u64().unwrap() as usize;
    let mut matrix = ResultsMatrix::new(generations, tasks).unwrap();
    for (k, row) in case["cells"].as_array().unwrap().iter().enumerate() {
        for (i, cell) in row.as_array().unwrap().iter().enumerate() {
            if let Some(score) = cell.as_f64() {
                matrix.set(k, i, score).unwrap();
            }
        }
    }
    let mut metrics = serde_json::to_value(ContinualMetrics::of(&matrix)).unwrap();
    metrics.as_object_mut().unwrap().remove("generations");
    without_reasons(&mut metrics);
    let mut out = case.clone();
    out["metrics"] = metrics;
    out
}

#[test]
fn the_continual_metrics_agree_with_numpy() {
    let fixture = fixture(env!("CARGO_MANIFEST_DIR"), "statistics/continual").unwrap();
    for case in fixture.cases() {
        if let Err(mismatch) = compare(&observed(case), case, Tolerance::new(1e-12, 1e-12)) {
            panic!(
                "{} ({:?}): {mismatch}\ncase: {case}",
                fixture.name, fixture.source
            );
        }
    }
}
