// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements release gates that judge a risk model's
// calibration only where the held-out evidence can tell, and say so where it
// cannot, for its clients. If your team needs expertise in certifying
// prediction models on held-out events, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec: the calibration rule of the release gate. A (code, horizon) is judged
//! only where the test part holds enough events of the code by the horizon;
//! elsewhere it is uncertified, which is neither a failure nor a zero. A
//! pair the plan requires that cannot be judged rejects the candidate. A
//! judged pair outside the bands rejects it. A release records the horizons
//! it certified, and exactly those. Trains small models: run under the device
//! lock.
#![allow(clippy::unwrap_used)]

mod common;

use common::timeline::{
    context, evaluate_under, failures, plan_for, prepare, train, train_calibrated, training,
    write_synthetic, FULL, STEPS,
};
use splinter_core::terms::{Distribution, UsagePolicy};
use splinter_eval::timeline_metrics::{arm_events, Arm, Judgement, RequiredHorizon, TimelinePlan};
use splinter_model::timeline::scoring::CalibrationBasis;
use splinter_pipelines::timeline::evaluate::Champion;
use splinter_pipelines::timeline::release::{release_timeline, TimelineReleaseRequest};

const ALIAS: &str = "risk";
const CODE: &str = "death:a";

fn request(evaluation: &splinter_core::digest::Digest) -> TimelineReleaseRequest {
    TimelineReleaseRequest {
        evaluation: evaluation.to_string(),
        alias: ALIAS.into(),
        distribution: Distribution::Unrestricted,
    }
}

/// `plan` judging only where the test part holds at least `min` events, with
/// the pairs `required` demanded.
fn judging_from(min: usize, required: &[f64]) -> TimelinePlan {
    let mut plan = plan_for(&[CODE]);
    plan.min_test_events = min;
    plan.required = required
        .iter()
        .map(|horizon| RequiredHorizon {
            code: CODE.into(),
            horizon: *horizon,
        })
        .collect();
    plan
}

#[test]
fn a_release_certifies_exactly_the_horizons_the_test_part_could_judge() {
    let (scratch, ctx) = context("timeline-certification");
    let file = write_synthetic(&scratch.0, "cohort.jsonl", FULL, 42);
    let split = prepare(&ctx, &file, "cohort", UsagePolicy::Redistributable, 3);
    let baseline = train(&ctx, &split, training(3, 1));
    let candidate = train_calibrated(&ctx, &split, training(STEPS, 2));
    let against = || Champion::Candidate(baseline.id.clone());

    // The events the test part holds by each horizon, as the evaluation counts them.
    let counted = evaluate_under(&ctx, &candidate.id, against(), &split, plan_for(&[CODE]));
    let events =
        |h: f64| counted.evaluation.evidence.values[&arm_events(Arm::Candidate, CODE, h)] as usize;
    let (early, late) = (events(5.0), events(10.0));
    assert!(
        early > 0 && early < late,
        "later horizons hold more events: {early} then {late}"
    );

    // A required pair the test part cannot judge fails closed, with the reason.
    let too_few = evaluate_under(
        &ctx,
        &candidate.id,
        against(),
        &split,
        judging_from(late + 1, &[5.0]),
    );
    assert!(too_few
        .evaluation
        .judgements
        .iter()
        .filter(|j| j.code == CODE)
        .all(|j| j.status == Judgement::Uncertified));
    let refused = release_timeline(&ctx, &request(&too_few.id)).unwrap();
    assert!(refused.release.is_none() && !refused.report.calibration.passed);
    let why = failures(&refused.report).join("\n");
    assert!(
        why.contains(&format!("{CODE}:h5:events")) && why.contains("below"),
        "{why}"
    );

    // A judged pair outside the bands rejects.
    let mut impossible = judging_from(early, &[5.0]);
    impossible.calibration.slope = (5.0, 6.0);
    let outside = evaluate_under(&ctx, &candidate.id, against(), &split, impossible);
    let refused = release_timeline(&ctx, &request(&outside.id)).unwrap();
    assert!(refused.release.is_none() && !refused.report.calibration.passed);
    assert!(failures(&refused.report).join("\n").contains("slope"));
    assert!(ctx.releases().alias(ALIAS).unwrap().is_none());

    // The earlier horizon is below the minimum, the later one is not: the 5
    // year pair is uncertified (not failed), the 10 year pair judged, and the
    // release certifies the latter alone.
    let measured = evaluate_under(
        &ctx,
        &candidate.id,
        against(),
        &split,
        judging_from(late, &[10.0]),
    );
    let status = |h: f64| {
        measured
            .evaluation
            .judgements
            .iter()
            .find(|j| j.code == CODE && j.horizon == h)
            .unwrap()
    };
    assert_eq!(status(5.0).status, Judgement::Uncertified);
    assert_eq!(status(5.0).test_events, Some(early));
    assert_eq!(status(10.0).status, Judgement::Judged);
    assert_eq!(status(10.0).basis, Some(CalibrationBasis::Calibrated));
    let released = release_timeline(&ctx, &request(&measured.id)).unwrap();
    assert!(released.report.passed, "{:#?}", failures(&released.report));
    let manifest = ctx
        .releases()
        .get(&released.release.unwrap())
        .unwrap()
        .manifest;
    let certified = &manifest.provenance.certified_horizons;
    assert_eq!(
        certified.keys().collect::<Vec<_>>(),
        [CODE],
        "{certified:?}"
    );
    let horizons: Vec<(f64, usize, CalibrationBasis)> = certified[CODE]
        .iter()
        .map(|c| (c.horizon, c.test_events, c.basis))
        .collect();
    assert_eq!(horizons, [(10.0, late, CalibrationBasis::Calibrated)]);
}
