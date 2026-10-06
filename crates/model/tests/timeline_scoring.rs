// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements honest head-to-head evaluation of
// time-to-event risk models on held-out participants, for its clients. If
// your team needs expertise in discrimination, calibration and paired
// comparison of competing-risk predictions, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec: two timeline models are scored on the same held-out subjects with
//! brain's survival arithmetic; the candidate-minus-champion differences
//! carry clustered bootstrap intervals; a horizon with too few events is
//! absent, never zero; scoring is reproducible and refuses a mismatch of
//! units. Trains small models: run under the device lock.
#![allow(clippy::unwrap_used)]

use splinter_eval::timeline_metrics::{arm_metric, diff_metric, Arm, NLL_DIFF};
use splinter_model::timeline::scoring::{
    compare, predict_arm, CalibrationBasis, Comparison, ScoreSpec, Subgroup, SubgroupRule,
};
use splinter_model::timeline::{
    calibrate_timeline, synthetic, train_timeline, CalibrationPlan, Subject, TimelineTraining,
};

const CODES: [&str; 3] = ["death:a", "death:b", "onset"];

fn training(steps: u32, seed: u64) -> TimelineTraining {
    let mut t = TimelineTraining::new(CODES, ["death:a", "death:b"]);
    t.steps = steps;
    t.batch = 64;
    t.max_tokens = Some(16);
    t.eval_interval = 25;
    t.seed = seed;
    t
}

fn spec(horizons: Vec<f64>) -> ScoreSpec {
    let mut s = ScoreSpec::new(CODES, ["death:a", "death:b"], horizons);
    s.all_cause = Some("death:any".into());
    s.bootstrap_reps = 60;
    s.min_events = 10;
    s
}

struct Fixture {
    test: Vec<Subject>,
    champion: splinter_model::timeline::TimelineModel,
    candidate: splinter_model::timeline::TimelineModel,
}

fn fixture() -> Fixture {
    let (all, _) = synthetic::population(1200, 7);
    let (train, rest) = all.split_at(700);
    let (held, test) = rest.split_at(150);
    let champion = train_timeline(train, held, &training(5, 1)).unwrap().model;
    let candidate = train_timeline(train, held, &training(300, 1))
        .unwrap()
        .model;
    Fixture {
        test: test.to_vec(),
        champion,
        candidate,
    }
}

fn run(f: &Fixture, spec: &ScoreSpec) -> Comparison {
    let a = predict_arm(&f.champion, &f.test, spec).unwrap();
    let b = predict_arm(&f.candidate, &f.test, spec).unwrap();
    compare(&a, &b, &f.test, spec).unwrap()
}

#[test]
fn scoring_is_measured_paired_reproducible_and_absent_where_unmeasured() {
    let f = fixture();
    let spec = spec(vec![3.0, 6.0]);
    let first = run(&f, &spec);
    let second = run(&f, &spec);
    assert_eq!(
        first, second,
        "same models, subjects and seed: the same numbers"
    );
    assert_eq!(first.units.len(), 350);

    let evidence = first.evidence();
    // Every measured number is named; the differences carry intervals.
    let ibs = diff_metric("death:a", None, "ibs");
    let (lo, hi) = evidence.intervals[&ibs];
    assert!(lo <= hi && lo.is_finite());
    assert!(evidence.intervals.contains_key(NLL_DIFF));
    assert!(evidence
        .intervals
        .contains_key(&diff_metric("death:any", Some(6.0), "uno_c")));
    for metric in ["uno_c", "auc", "brier", "slope", "intercept", "oe", "ece"] {
        let name = arm_metric(Arm::Candidate, "death:a", Some(6.0), metric);
        assert!(evidence.values[&name].is_finite(), "{name}");
    }
    let whole = f64::from(f.candidate.event_nll(&f.test).unwrap());
    assert!(
        (first.candidate.event_nll - whole).abs() < 1e-3,
        "{} vs {whole}",
        first.candidate.event_nll
    );
    // A trained model beats one trained for 5 steps on the likelihood.
    assert!(first.candidate.event_nll < first.champion.event_nll);
    assert!(
        evidence.intervals[NLL_DIFF].1 < 0.0,
        "{:?}",
        evidence.intervals[NLL_DIFF]
    );

    // Too few events: absent, not zero.
    let mut strict = spec.clone();
    strict.min_events = 100_000;
    let none = run(&f, &strict).evidence();
    assert!(
        none.values.keys().all(|k| !k.contains(":h")),
        "{:?}",
        none.values.keys().collect::<Vec<_>>()
    );
    assert!(!none.intervals.contains_key(&ibs));
}

#[test]
fn the_same_model_has_a_zero_difference_and_a_mismatch_is_refused() {
    let f = fixture();
    let mut spec = spec(vec![4.0]);
    spec.subgroups = vec![Subgroup {
        name: "group_b".into(),
        rule: SubgroupRule::Category {
            var: "group".into(),
            level: "b".into(),
        },
    }];
    let a = predict_arm(&f.candidate, &f.test, &spec).unwrap();
    let same = compare(&a, &a.clone(), &f.test, &spec).unwrap();
    let e = same.evidence();
    assert_eq!(e.intervals[NLL_DIFF], (0.0, 0.0));
    assert_eq!(e.values["subgroup:group_b:death:a"], 0.0);
    assert!(e.values["subgroup_events:group_b:death:a"] >= 0.0);

    let b = predict_arm(&f.champion, &f.test[..300], &spec).unwrap();
    let why = compare(&a, &b, &f.test, &spec).err().unwrap().to_string();
    assert!(why.contains("exactly the 350 subjects"), "{why}");
    let mut late = spec.clone();
    late.horizons = vec![4.0, 500.0];
    assert!(predict_arm(&f.candidate, &f.test, &late)
        .err()
        .unwrap()
        .to_string()
        .contains("last knot"));
}

#[test]
fn brains_evaluation_of_a_code_is_what_the_union_view_computes_independently() {
    // With one absorbing code the union of absorbing codes is that code, so
    // brain's own numbers for it (`TimelineModel::evaluate`, which scores the
    // outcome codes) and this crate's independent computation (which scores
    // the union) must be the same numbers.
    let (all, _) = synthetic::population(1200, 7);
    let (train, rest) = all.split_at(700);
    let (held, test) = rest.split_at(150);
    let mut config = TimelineTraining::new(["death:a", "onset"], ["death:a"]);
    config.steps = 60;
    config.batch = 64;
    config.max_tokens = Some(16);
    config.eval_interval = 25;
    let model = train_timeline(train, held, &config).unwrap().model;
    let mut spec = ScoreSpec::new(["death:a", "onset"], ["death:a"], vec![3.0, 6.0]);
    spec.all_cause = Some("union".into());
    spec.bootstrap_reps = 10;
    let arm = predict_arm(&model, test, &spec).unwrap();
    let scores = compare(&arm, &arm.clone(), test, &spec).unwrap().candidate;
    let (brain, own) = (&scores.views["death:a"], &scores.views["union"]);
    assert_eq!(brain.horizons.len(), own.horizons.len());
    assert!(!own.horizons.is_empty());
    let close = |a: Option<f64>, b: Option<f64>, what: &str| {
        let (a, b) = (a.unwrap(), b.unwrap());
        assert!((a - b).abs() < 1e-9, "{what}: brain {a} vs independent {b}");
    };
    close(brain.ibs, own.ibs, "integrated brier");
    for (b, o) in brain.horizons.iter().zip(&own.horizons) {
        assert_eq!((b.horizon, b.events), (o.horizon, o.events));
        close(b.uno_c, o.uno_c, "uno c");
        close(b.auc, o.auc, "auc");
        close(b.brier, o.brier, "brier");
        let (bc, oc) = (
            b.calibration.as_ref().unwrap(),
            o.calibration.as_ref().unwrap(),
        );
        close(Some(bc.slope), Some(oc.slope), "slope");
        close(Some(bc.intercept), Some(oc.intercept), "intercept");
        close(Some(bc.oe), Some(oc.oe), "observed over expected");
        close(Some(bc.ece), Some(oc.ece), "ece");
    }
}

#[test]
fn calibration_is_judged_on_the_risk_the_model_is_served_as_and_the_record_says_which() {
    let (all, _) = synthetic::population(1600, 8);
    let (train, rest) = all.split_at(800);
    let (early, rest) = rest.split_at(100);
    let (validation, test) = rest.split_at(400);
    let mut model = train_timeline(train, early, &training(60, 1))
        .unwrap()
        .model;
    let mut spec = spec(vec![4.0, 8.0]);
    spec.min_events = 5;
    let basis = |model: &splinter_model::timeline::TimelineModel| -> Vec<(CalibrationBasis, bool)> {
        let arm = predict_arm(model, test, &spec).unwrap();
        let scores = compare(&arm, &arm.clone(), test, &spec).unwrap().candidate;
        scores.views["death:a"]
            .horizons
            .iter()
            .map(|h| (h.calibration_basis, h.calibration.is_some()))
            .collect()
    };
    // No calibration: the raw risk is judged.
    assert_eq!(
        basis(&model),
        [(CalibrationBasis::Raw, true), (CalibrationBasis::Raw, true)]
    );
    // Calibrated at the first horizon only: it is judged on the calibrated
    // risk; the other horizon was never asked for and stays raw.
    let plan = CalibrationPlan {
        horizons: vec![4.0],
        min_events: Some(5),
    };
    let outcome = calibrate_timeline(&mut model, validation, &plan).unwrap();
    assert!(outcome
        .calibrated
        .iter()
        .any(|(c, h)| c == "death:a" && *h == 4.0));
    assert_eq!(
        basis(&model),
        [
            (CalibrationBasis::Calibrated, true),
            (CalibrationBasis::Raw, true)
        ]
    );
    // Asked for but declared uncalibrated by brain: nothing is judged there.
    let plan = CalibrationPlan {
        horizons: vec![4.0, 8.0],
        min_events: Some(1_000_000),
    };
    calibrate_timeline(&mut model, validation, &plan).unwrap();
    assert_eq!(
        basis(&model),
        [
            (CalibrationBasis::Uncalibrated, false),
            (CalibrationBasis::Uncalibrated, false)
        ]
    );
}
