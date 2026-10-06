// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible, leakage-free data preparation
// and checkable behaviour of risk models trained on longitudinal records, for
// its clients. If your team needs expertise in validating risk models before
// they ship, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: what a trained timeline model does. Two runs of the same sources,
//! split, configuration and seed give the same data membership, architecture,
//! release lineage and, within a stated float tolerance, the same metrics.
//! Removing observations at random never produces a NaN, a probability
//! outside [0, 1] or a jump beyond a stated bound, and a large change to a
//! measured variable moves the risk the way the generator says it does while a
//! repeated observation does not. Trains small models: run under the device
//! lock.
#![allow(clippy::unwrap_used)]

mod common;

use common::timeline::{
    context, evaluate_under, plan_for, prepare, train, training, write_synthetic,
    write_synthetic_without, CONVENTIONAL, FULL, STEPS,
};
use splinter_core::terms::{Distribution, UsagePolicy};
use splinter_model::timeline::{
    read_jsonl, AtRisk, Observation, Prediction, Subject, TimelineModel, Value,
};
use splinter_pipelines::timeline::evaluate::Champion;
use splinter_pipelines::timeline::release::{release_timeline, TimelineReleaseRequest};

/// The most two runs' metrics may differ by, absolutely. The data, the split,
/// the configuration, the seed and the device are the same; what is left is
/// the order of floating-point additions inside the device's kernels.
const METRIC_TOLERANCE: f64 = 1e-3;

struct Run {
    split: splinter_pipelines::timeline::data::SplitReport,
    candidate: splinter_pipelines::timeline::records::TimelineCandidate,
    evidence: splinter_eval::metric_gate::Evidence,
    manifest: splinter_orchestrator::releases::ReleaseManifest,
    config: splinter_model::timeline::TimelineConfig,
}

fn run(test: &str) -> Run {
    let (scratch, ctx) = context(test);
    let file =
        write_synthetic_without(&scratch.0, "earlier.jsonl", CONVENTIONAL, 43, &["x1", "x2"]);
    let split = prepare(&ctx, &file, "conventional", UsagePolicy::Redistributable, 3);
    let baseline = train(&ctx, &split, training(3, 1));
    let candidate = train(&ctx, &split, training(STEPS, 2));
    let measured = evaluate_under(
        &ctx,
        &candidate.id,
        Champion::Candidate(baseline.id),
        &split,
        plan_for(&[]),
    );
    let released = release_timeline(
        &ctx,
        &TimelineReleaseRequest {
            evaluation: measured.id.to_string(),
            alias: "risk".into(),
            distribution: Distribution::Restricted,
        },
    )
    .unwrap();
    assert!(
        released.report.passed,
        "{:#?}",
        common::timeline::failures(&released.report)
    );
    let manifest = ctx
        .releases()
        .get(&released.release.unwrap())
        .unwrap()
        .manifest;
    let unpacked = scratch.0.join("unpacked");
    let model =
        splinter_pipelines::timeline::train::load_bundle(&candidate.checkpoint, &unpacked, "m")
            .unwrap()
            .config()
            .clone();
    Run {
        split,
        candidate: candidate.candidate,
        evidence: measured.evaluation.evidence,
        manifest,
        config: model,
    }
}

#[test]
fn the_same_sources_split_configuration_and_seed_give_the_same_release() {
    let (a, b) = (run("timeline-repro-a"), run("timeline-repro-b"));
    // Data membership.
    assert_eq!(a.split.split, b.split.split);
    assert_eq!(
        (&a.split.train, &a.split.validation, &a.split.test),
        (&b.split.train, &b.split.validation, &b.split.test)
    );
    assert_eq!(a.candidate.episodes, b.candidate.episodes);
    assert_eq!(a.candidate.sources, b.candidate.sources);
    // Architecture and configuration.
    assert_eq!(a.candidate.config, b.candidate.config);
    assert_eq!(a.candidate.config_digest, b.candidate.config_digest);
    assert_eq!(a.config, b.config);
    // Release lineage.
    assert_eq!(a.manifest.datasets, b.manifest.datasets);
    assert_eq!(a.manifest.provenance, b.manifest.provenance);
    assert_eq!(a.manifest.terms, b.manifest.terms);
    assert_eq!(a.manifest.artifact.kind(), b.manifest.artifact.kind());
    // Metrics, within the stated tolerance.
    assert_eq!(
        a.evidence.values.keys().collect::<Vec<_>>(),
        b.evidence.values.keys().collect::<Vec<_>>()
    );
    let worst = a
        .evidence
        .values
        .iter()
        .map(|(k, v)| (v - b.evidence.values[k]).abs())
        .fold(0.0f64, f64::max);
    println!("largest metric difference between the two runs: {worst:e}");
    assert!(worst <= METRIC_TOLERANCE, "two runs differ by {worst}");
}

/// The most the population's mean ten-year risk of an outcome may shift when
/// observations are removed at random: a cohort's missing values are the
/// normal case, and they must not move the level of the predictions.
const POPULATION_SHIFT_BOUND: f64 = 0.05;
/// The most the time-dependent AUC of the all-cause ten-year risk may fall
/// when half of the observations are removed.
const DISCRIMINATION_LOSS_BOUND: f64 = 0.15;
/// A prediction must be continuous in the values it reads (it reads them
/// through soft bins): a nudge to a measured value, in units of the variable,
/// may move a five-year risk by no more than the nudge times this constant (a
/// probability per unit of the variable).
const LIPSCHITZ_BOUND: f64 = 1.0;
/// The nudges tried.
const NUDGES: [f64; 2] = [0.01, 0.05];
/// The most a repeated, identical observation may move a five-year risk, and
/// the share of the mean effect of a large change it may have on average. A set
/// encoder counts a repeated token twice, so the repeat is not exactly a no-op;
/// it must stay small beside what a real change does.
const NO_OP_BOUND: f64 = 0.05;
const NO_OP_SHARE: f64 = 0.1;
/// A large change to a measured variable, in the units of the variable (a
/// standard deviation of the generator's covariate).
const LARGE_CHANGE: f64 = 2.0;

fn rng(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed;
    move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// `subjects` with each observation removed with probability `rate`; events,
/// the prediction time and the windows are untouched.
fn without_observations(subjects: &[Subject], rate: f64, seed: u64) -> Vec<Subject> {
    let mut draw = rng(seed);
    subjects
        .iter()
        .map(|s| Subject {
            observations: s
                .observations
                .iter()
                .filter(|_| draw() >= rate)
                .cloned()
                .collect(),
            ..s.clone()
        })
        .collect()
}

/// The subject as of a checkup `2` time units after its entry, at which `x1`
/// was measured as `x1`: what is known now is everything up to then, and the
/// outcomes that follow are not known.
fn after_checkup(s: &Subject, x1: f64) -> Subject {
    let at = s.entry + 2.0;
    let mut later = s.clone();
    later.events.retain(|e| e.t < s.entry);
    later.observations.push(Observation {
        t: at,
        var: "x1".into(),
        value: Value::Number(x1),
        unit: None,
    });
    later.entry = at;
    later.at_risk = vec![AtRisk {
        code: "*".into(),
        from: at,
        to: at + 12.0,
    }];
    later
}

fn risk(p: &[Prediction], code: &str, t: f64) -> Vec<f64> {
    p.iter().map(|p| p.cif(code, t).unwrap()).collect()
}

fn mean(v: impl Iterator<Item = f64>) -> f64 {
    let v: Vec<f64> = v.collect();
    v.iter().sum::<f64>() / v.len() as f64
}

#[test]
fn removed_observations_degrade_a_prediction_gracefully_and_updates_move_risk_as_the_generator_says(
) {
    let (scratch, ctx) = context("timeline-behaviour");
    let file = write_synthetic(&scratch.0, "later.jsonl", FULL, 42);
    let split = prepare(&ctx, &file, "full", UsagePolicy::Redistributable, 3);
    let trained = train(&ctx, &split, training(STEPS, 2));
    let model: TimelineModel = splinter_pipelines::timeline::train::load_bundle(
        &trained.checkpoint,
        &scratch.0.join("m"),
        "model",
    )
    .unwrap();
    let test: Vec<Subject> = read_jsonl(&ctx.timeline_datasets().get(&split.test).unwrap().path)
        .unwrap()
        .into_iter()
        .take(400)
        .collect();
    let base = model.predict(&test).unwrap();
    let codes = ["death:a", "death:b", "onset"];
    let horizons: Vec<f64> = (1..=20).map(|k| f64::from(k) * 0.5).collect();

    // Missingness: no NaN, probabilities in [0, 1], curves that never fall, the
    // level of the predictions unmoved and discrimination kept.
    let obs = splinter_model::timeline::observed(&test, &["death:a", "death:b"]);
    let g = splinter_model::timeline::survival::estimate::censoring(&obs);
    let auc = |p: &[Prediction]| {
        let all_cause: Vec<f64> = p.iter().map(|p| 1.0 - p.survival(10.0)).collect();
        let any: Vec<splinter_model::timeline::survival::Obs> = obs
            .iter()
            .map(|o| splinter_model::timeline::survival::Obs {
                cause: o.cause.map(|_| 0),
                ..*o
            })
            .collect();
        splinter_model::timeline::survival::auc::at(&all_cause, &any, 0, 10.0, &g).unwrap()
    };
    let auc_base = auc(&base);
    for rate in [0.1, 0.3, 0.5] {
        let degraded = model
            .predict(&without_observations(&test, rate, 17))
            .unwrap();
        for p in &degraded {
            for code in codes {
                let curve: Vec<f64> = horizons.iter().map(|t| p.cif(code, *t).unwrap()).collect();
                assert!(
                    curve
                        .iter()
                        .all(|v| v.is_finite() && (0.0..=1.0).contains(v)),
                    "{code}: {curve:?}"
                );
                assert!(
                    curve.windows(2).all(|w| w[1] >= w[0] - 1e-12),
                    "{code} falls: {curve:?}"
                );
            }
        }
        for code in codes {
            let shift = mean(risk(&degraded, code, 10.0).into_iter())
                - mean(risk(&base, code, 10.0).into_iter());
            println!("removed {rate}: {code} mean ten-year risk shifts by {shift:+.4}");
            assert!(
                shift.abs() <= POPULATION_SHIFT_BOUND,
                "removing {rate} shifted {code} by {shift}"
            );
        }
        let kept = auc(&degraded);
        println!("removed {rate}: all-cause ten-year AUC {kept:.4} against {auc_base:.4} with nothing removed");
        assert!(
            kept >= auc_base - DISCRIMINATION_LOSS_BOUND && kept > 0.5,
            "AUC {kept} against {auc_base}"
        );
    }

    // Continuity: a nudge to a value moves the risk by no more than a nudge.
    for nudge in NUDGES {
        let nudged: Vec<Subject> = test
            .iter()
            .map(|s| Subject {
                observations: s
                    .observations
                    .iter()
                    .map(|o| match (&o.var[..], &o.value) {
                        ("x1" | "x2", Value::Number(v)) => Observation {
                            value: Value::Number(v + nudge),
                            ..o.clone()
                        },
                        _ => o.clone(),
                    })
                    .collect(),
                ..s.clone()
            })
            .collect();
        let moved = model.predict(&nudged).unwrap();
        let worst = codes
            .iter()
            .flat_map(|c| {
                risk(&moved, c, 5.0)
                    .into_iter()
                    .zip(risk(&base, c, 5.0))
                    .map(|(a, b)| (a - b).abs())
            })
            .fold(0.0f64, f64::max);
        println!("a nudge of {nudge} to x1 and x2 moves a five-year risk by at most {worst:.5}");
        assert!(
            worst <= LIPSCHITZ_BOUND * nudge,
            "a nudge of {nudge} moved a five-year risk by {worst}"
        );
    }

    // Update sensitivity: x1 drives death:a (log hazard +0.7 per unit) and
    // not death:b; the same subject with x1 high or low at the new checkup.
    let high: Vec<Subject> = test
        .iter()
        .map(|s| after_checkup(s, LARGE_CHANGE))
        .collect();
    let low: Vec<Subject> = test
        .iter()
        .map(|s| after_checkup(s, -LARGE_CHANGE))
        .collect();
    let (hi, lo) = (model.predict(&high).unwrap(), model.predict(&low).unwrap());
    let effect = |code: &str| -> Vec<f64> {
        risk(&hi, code, 5.0)
            .into_iter()
            .zip(risk(&lo, code, 5.0))
            .map(|(a, b)| a - b)
            .collect()
    };
    let (on_a, on_b) = (effect("death:a"), effect("death:b"));
    let share_up = on_a.iter().filter(|d| **d > 0.0).count() as f64 / on_a.len() as f64;
    println!("x1 {LARGE_CHANGE} against -{LARGE_CHANGE}: death:a up for {share_up:.3}, mean {:.4}; death:b mean {:.4}", mean(on_a.iter().copied()), mean(on_b.iter().copied()));
    assert!(
        share_up >= 0.95,
        "only {share_up} of subjects have a higher death:a risk with the higher x1"
    );
    assert!(mean(on_a.iter().copied()) > 0.0);
    assert!(
        mean(on_a.iter().copied()) > 2.0 * mean(on_b.iter().copied()).abs(),
        "the variable the generator ties to death:a moves death:b about as much"
    );

    // A repeated observation says nothing new.
    let repeated: Vec<Subject> = test
        .iter()
        .map(|s| {
            let mut s = s.clone();
            let first = s
                .observations
                .iter()
                .find(|o| o.var == "x1")
                .cloned()
                .unwrap();
            s.observations.push(first);
            s
        })
        .collect();
    let again = model.predict(&repeated).unwrap();
    let noop: Vec<f64> = codes
        .iter()
        .flat_map(|c| {
            risk(&again, c, 5.0)
                .into_iter()
                .zip(risk(&base, c, 5.0))
                .map(|(a, b)| (a - b).abs())
        })
        .collect();
    let (worst, avg) = (
        noop.iter().copied().fold(0.0f64, f64::max),
        mean(noop.iter().copied()),
    );
    println!("a repeated observation: worst change {worst:.5}, mean change {avg:.5}");
    assert!(
        worst <= NO_OP_BOUND,
        "a repeated observation moved a five-year risk by {worst}"
    );
    assert!(
        avg < NO_OP_SHARE * mean(on_a.iter().map(|d| d.abs())),
        "a repeat moves risk by {avg} on average"
    );
}
