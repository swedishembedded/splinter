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
    compare, predict_arm, Comparison, ScoreSpec, Subgroup, SubgroupRule,
};
use splinter_model::timeline::{synthetic, train_timeline, Subject, TimelineTraining};

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
