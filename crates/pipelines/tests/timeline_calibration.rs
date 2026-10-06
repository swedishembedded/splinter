// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements calibrated, correctly served risk models:
// probabilities that mean what they say, shipped in one file that reproduces
// what was evaluated. If your team needs expertise in calibrating and
// verifying released prediction models, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec: a candidate is calibrated on validation units early stopping did
//! not read, never on a test unit, and the calibration travels in the packed
//! file beside the weights and round-trips; the gate judges calibration on the
//! calibrated risk where there is one and on the raw risk otherwise, says
//! which, and has nothing to judge where brain declared a horizon
//! uncalibrated; serving correctness is measured on the test units from the
//! shipped file, and a model whose test units mostly lie outside its support
//! does not pass it. Trains small models: run under the device lock.
#![allow(clippy::unwrap_used)]

mod common;

use common::timeline::{
    context, evaluate, plan_for, prepare, train, train_calibrated, training, write_synthetic,
    HORIZONS,
};
use splinter_core::digest::Digest;
use splinter_core::terms::UsagePolicy;
use splinter_data::timeline_dataset::ProjectionSpec;
use splinter_eval::metric_gate::decide;
use splinter_eval::timeline_metrics::{
    arm_metric, Arm, SERVE_ABSTENTION_RATE, SERVE_BATCH_MAX_ABS_DIFF, SERVE_INVALID_PROBABILITIES,
    SERVE_MAX_ABS_DIFF, SERVE_NON_MONOTONE_CURVES,
};
use splinter_model::timeline::scoring::CalibrationBasis;
use splinter_model::timeline::{read_jsonl, synthetic, CalibrationPlan};
use splinter_pipelines::timeline::data::{
    import_records, split_timeline, ImportRequest, SplitPlan, SplitRequest,
};
use splinter_pipelines::timeline::evaluate::Champion;
use splinter_pipelines::timeline::serving::unpack_with_tar;
use splinter_pipelines::timeline::train::{
    train_timeline_candidate, TimelineTrainRequest, CALIBRATION_FILE, SUPPORT_FILE,
};

#[test]
fn the_calibration_is_packed_with_the_weights_and_round_trips() {
    let (scratch, ctx) = context("timeline-calibration-pack");
    let file = write_synthetic(&scratch.0, "cohort.jsonl", 4000, 31);
    let split = prepare(&ctx, &file, "cohort", UsagePolicy::Redistributable, 3);
    let trained = train_calibrated(&ctx, &split, training(60, 1));
    let c = &trained.candidate;
    let calibration = c.calibration.as_ref().expect("a calibration was requested");

    // Every file brain saved is in the bundle, and the manifest's digest is the
    // digest of the file a consumer unpacks.
    let dir = scratch.0.join("consumer");
    let files = unpack_with_tar(&trained.checkpoint, &dir).unwrap();
    for needed in [CALIBRATION_FILE, SUPPORT_FILE, "vocab.json"] {
        assert!(files.iter().any(|f| f == needed), "{needed} in {files:?}");
    }
    assert_eq!(files.len(), c.files.len());
    assert_eq!(
        calibration.digest,
        Digest::sha256_of(&std::fs::read(dir.join(CALIBRATION_FILE)).unwrap())
    );

    // Plain brain reads it back: calibrated where the candidate says it is.
    let model = splinter_model::timeline::TimelineModel::load(&dir).unwrap();
    let loaded = model
        .calibration()
        .expect("the calibration loads beside the weights");
    assert_eq!(loaded.validation_subjects, calibration.outcome.units);
    assert!(
        !calibration.outcome.calibrated.is_empty(),
        "{calibration:#?}"
    );
    let held = read_jsonl(&ctx.timeline_datasets().get(&split.validation).unwrap().path).unwrap();
    assert_eq!(
        calibration.outcome.units + calibration.early_stopping_units,
        held.len(),
        "validation units are divided: none dropped, none in both"
    );
    // What the file calibrates to is what the model as trained calibrated to
    // (the probe is taken on early-stopping units, at the packed horizons).
    assert_eq!(c.probe.calibrated.len(), c.probe.values.len());
    assert!(c.probe.calibrated.iter().any(Option::is_some));
    assert!(c.round_trip_max_abs_diff <= 1e-6);
}

#[test]
fn a_horizon_brain_declares_uncalibrated_has_nothing_to_judge_and_fails_unmeasured() {
    let (scratch, ctx) = context("timeline-calibration-gap");
    let file = write_synthetic(&scratch.0, "cohort.jsonl", 3000, 32);
    let split = prepare(&ctx, &file, "cohort", UsagePolicy::Redistributable, 3);
    let mut request = TimelineTrainRequest::new(
        &split.train.to_string(),
        &split.validation.to_string(),
        training(40, 1),
    );
    request.calibration = Some(CalibrationPlan {
        horizons: HORIZONS.to_vec(),
        min_events: Some(1_000_000),
    });
    let candidate = train_timeline_candidate(&ctx, &request).unwrap();
    let outcome = &candidate.candidate.calibration.as_ref().unwrap().outcome;
    assert!(outcome.calibrated.is_empty());
    assert!(!outcome.uncalibrated.is_empty());

    let baseline = train(&ctx, &split, training(3, 1));
    let measured = evaluate(
        &ctx,
        &candidate.id,
        Champion::Candidate(baseline.id),
        &split,
    );
    let scores = &measured.evaluation.comparison.candidate.views["death:a"];
    for h in &scores.horizons {
        assert_eq!(h.calibration_basis, CalibrationBasis::Uncalibrated);
        assert!(h.calibration.is_none(), "absent, never zero");
        assert!(h.uno_c.is_some(), "discrimination is still the raw risk's");
    }
    let name = arm_metric(Arm::Candidate, "death:a", Some(HORIZONS[0]), "slope");
    assert!(!measured.evaluation.evidence.values.contains_key(&name));
    let gate = decide(
        &measured.evaluation.spec.calibration,
        &measured.evaluation.evidence,
    );
    assert!(!gate.passed(), "an unmeasured calibration is not a pass");
    // The all-cause union has no calibration of brain's to judge: raw.
    assert!(measured.evaluation.comparison.candidate.views["death:any"]
        .horizons
        .iter()
        .all(|h| h.calibration_basis == CalibrationBasis::Raw));
}

/// A cohort whose source `cycle-b` is the same hazards on much older subjects
/// with a very different risk factor: a model trained on `cycle-a` has never
/// seen anything like it.
fn write_shifted(dir: &std::path::Path, n: usize, seed: u64) -> std::path::PathBuf {
    let mut lines = Vec::new();
    for (source, shift) in [
        ("cycle-a", synthetic::Shift::default()),
        ("cycle-b", synthetic::Shift { age: 30.0, x1: 6.0 }),
    ] {
        let (subjects, _) = synthetic::population_shifted(n, seed, shift);
        for (i, s) in subjects.iter().enumerate() {
            let mut line = serde_json::to_value(s).unwrap();
            line["subject_id"] = format!("{source}-{}", s.subject_id).into();
            line["group_id"] = format!("{source}-household-{}", i / 2).into();
            line["source"] = source.into();
            line["interventions"] = serde_json::json!([]);
            lines.push(line.to_string());
        }
    }
    let path = dir.join("shifted.jsonl");
    std::fs::write(&path, lines.join("\n") + "\n").unwrap();
    path
}

#[test]
fn a_model_whose_test_units_lie_outside_its_support_fails_serving_correctness() {
    let (scratch, ctx) = context("timeline-serving-ood");
    let file = write_shifted(&scratch.0, 1500, 33);
    import_records(
        &ctx,
        &ImportRequest {
            file,
            dataset: "shifted".into(),
            terms: UsagePolicy::Redistributable.terms("shifted"),
            secret: b"test campaign secret".to_vec(),
        },
    )
    .unwrap();
    let split = split_timeline(
        &ctx,
        &SplitRequest {
            dataset: Some("shifted".into()),
            plan: SplitPlan::LeaveSourceOut {
                source: "cycle-b".into(),
                validation_share: 0.2,
            },
            seed: 3,
            projection: ProjectionSpec::at_entry(),
        },
    )
    .unwrap();
    let candidate = train(&ctx, &split, training(60, 1));
    let baseline = train(&ctx, &split, training(3, 1));
    let measured = splinter_pipelines::timeline::evaluate::evaluate_timeline(
        &ctx,
        &splinter_pipelines::timeline::evaluate::TimelineEvaluateRequest {
            candidate: candidate.id,
            champion: Champion::Candidate(baseline.id),
            test: split.test.to_string(),
            scoring: common::timeline::scoring(),
            plan: plan_for(&[]),
        },
    )
    .unwrap();
    let e = &measured.evaluation;
    let rate = e.evidence.values[SERVE_ABSTENTION_RATE];
    assert!(
        rate > 0.5,
        "most of the shifted test units are unsupported: {rate}"
    );
    // The pre-registered threshold fails it, and the other serving inputs are
    // measured and hold: the file is the model, only the units are not its own.
    let gate = decide(&e.spec.serving, &e.evidence);
    assert!(!gate.passed());
    let failed: Vec<String> = gate
        .checks
        .iter()
        .filter(|(_, check)| !check.passed)
        .map(|(r, _)| format!("{r:?}"))
        .collect();
    assert_eq!(failed.len(), 1, "{failed:?}");
    assert!(failed[0].contains(SERVE_ABSTENTION_RATE), "{failed:?}");
    for name in [
        SERVE_MAX_ABS_DIFF,
        SERVE_BATCH_MAX_ABS_DIFF,
        SERVE_INVALID_PROBABILITIES,
        SERVE_NON_MONOTONE_CURVES,
    ] {
        assert!(e.evidence.values.contains_key(name), "{name}");
    }
    assert_eq!(e.evidence.values[SERVE_INVALID_PROBABILITIES], 0.0);
    assert_eq!(e.evidence.values[SERVE_NON_MONOTONE_CURVES], 0.0);
    assert!(e.serving.is_some());
}
