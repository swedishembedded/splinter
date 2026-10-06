// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verification that a released risk model
// serves what was evaluated, and withholds an answer it has no support for.
// If your team needs expertise in serving correctness for prediction models,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: serving correctness is measured on held-out units. The file a model
//! is saved to reproduces its predictions; a batched forecast equals a single
//! one; units far outside the training support are counted as withheld; a
//! different model served in place of the evaluated one is caught; and an
//! unusable request is refused, not reported as zero. Trains small models:
//! run under the device lock.
#![allow(clippy::unwrap_used)]

use splinter_model::timeline::serving::{measure_serving, ServingSpec};
use splinter_model::timeline::{
    calibrate_timeline, synthetic, train_timeline, CalibrationPlan, Subject, TimelineModel,
    TimelineTraining,
};

const CODES: [&str; 3] = ["death:a", "death:b", "onset"];

struct Fixture {
    model: TimelineModel,
    other: TimelineModel,
    served: TimelineModel,
    test: Vec<Subject>,
}

fn train(seed: u64, steps: u32, train: &[Subject], early: &[Subject]) -> TimelineModel {
    let mut config = TimelineTraining::new(CODES, ["death:a", "death:b"]);
    config.steps = steps;
    config.batch = 64;
    config.max_tokens = Some(16);
    config.eval_interval = 25;
    config.seed = seed;
    train_timeline(train, early, &config).unwrap().model
}

fn fixture() -> Fixture {
    let (all, _) = synthetic::population(1500, 9);
    let (fit, rest) = all.split_at(800);
    let (early, rest) = rest.split_at(100);
    let (calibration, test) = rest.split_at(300);
    let mut model = train(1, 150, fit, early);
    let plan = CalibrationPlan {
        horizons: vec![3.0, 6.0],
        min_events: Some(5),
    };
    calibrate_timeline(&mut model, calibration, &plan).unwrap();
    // Tests run in parallel: each fixture saves into a directory of its own.
    static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "splinter-serving-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    model.save(&dir).unwrap();
    Fixture {
        served: TimelineModel::load(&dir).unwrap(),
        other: train(2, 150, fit, early),
        model,
        test: test.to_vec(),
    }
}

#[test]
fn a_saved_calibrated_model_serves_what_was_evaluated_and_nothing_is_withheld_in_distribution() {
    let f = fixture();
    let spec = ServingSpec::new(vec![3.0, 6.0]);
    let m = measure_serving(&f.model, &f.served, &f.test, &spec).unwrap();
    assert_eq!(m.units, f.test.len());
    assert!(m.identity_max_abs_diff.unwrap() <= 1e-9, "{m:?}");
    assert_eq!(m.histories, spec.batch_sample);
    assert!(m.batch_max_abs_diff.unwrap() <= 1e-6, "{m:?}");
    assert!(m.abstention_rate.unwrap() < 0.1, "{m:?}");
    assert_eq!(m.invalid_probabilities, 0, "{m:?}");
    assert_eq!(m.non_monotone_curves, 0, "{m:?}");
}

#[test]
fn another_model_served_in_its_place_is_caught_by_identity() {
    let f = fixture();
    let spec = ServingSpec::new(vec![3.0, 6.0]);
    let m = measure_serving(&f.model, &f.other, &f.test, &spec).unwrap();
    // The other model has no calibration: a calibrated risk one load has and
    // the other lacks is a mismatch of shape, never a small difference.
    assert_eq!(m.identity_max_abs_diff, None, "{m:?}");
}

#[test]
fn units_far_outside_the_support_are_counted_as_withheld() {
    let f = fixture();
    let (shifted, _) =
        synthetic::population_shifted(200, 10, synthetic::Shift { age: 30.0, x1: 6.0 });
    let spec = ServingSpec::new(vec![3.0, 6.0]);
    let m = measure_serving(&f.model, &f.served, &shifted, &spec).unwrap();
    assert!(m.abstention_rate.unwrap() > 0.5, "{m:?}");
    assert!(
        m.identity_max_abs_diff.unwrap() <= 1e-9,
        "the file is still the model: {m:?}"
    );
}

#[test]
fn an_unusable_request_is_refused_not_reported_as_zero() {
    let f = fixture();
    let spec = ServingSpec::new(vec![3.0]);
    assert!(measure_serving(&f.model, &f.served, &[], &spec).is_err());
    let past_the_knots = ServingSpec::new(vec![1.0e6]);
    assert!(measure_serving(&f.model, &f.served, &f.test, &past_the_knots).is_err());
}
