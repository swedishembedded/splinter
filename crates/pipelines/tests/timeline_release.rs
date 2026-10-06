// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gated releases of risk models: a candidate
// replaces the model in place only when pre-registered requirements hold on
// held-out evidence, and the decision is recorded either way. If your team
// needs expertise in release governance for prediction models, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: the loop closes. A candidate is trained, scored against the
//! champion on the same held-out units and released only if the gate passes;
//! the released file loads in plain brain and answers; a candidate that fails
//! is rejected and recorded, the release in place stays the alias target and
//! stays loadable; rollback returns to the previous release; data whose terms
//! forbid it is never released unrestricted. Trains small models: run under
//! the device lock.
#![allow(clippy::unwrap_used)]

mod common;

use common::timeline::{
    context, evaluate, evaluate_under, failures, plan, plan_for, prepare, scoring, train, training,
    write_synthetic, write_synthetic_without, ALL_CAUSE, CONVENTIONAL, FULL, STEPS,
};
use splinter_core::terms::{Distribution, UsagePolicy};
use splinter_data::split::Part;
use splinter_model::timeline::{read_jsonl, TimelineModel};
use splinter_orchestrator::releases::ArtifactKind;
use splinter_pipelines::release::rollback;
use splinter_pipelines::timeline::evaluate::{
    evaluate_timeline, Champion, TimelineEvaluateRequest,
};
use splinter_pipelines::timeline::release::{release_timeline, TimelineReleaseRequest};

const ALIAS: &str = "risk";

fn release_request(
    evaluation: &splinter_core::digest::Digest,
    distribution: Distribution,
) -> TimelineReleaseRequest {
    TimelineReleaseRequest {
        evaluation: evaluation.to_string(),
        alias: ALIAS.into(),
        distribution,
    }
}

/// An unpacked release, loaded by plain brain: the file is a plain ustar.
fn plain_load(file: &std::path::Path, into: &std::path::Path) -> TimelineModel {
    std::fs::create_dir_all(into).unwrap();
    let status = std::process::Command::new("tar")
        .arg("-xf")
        .arg(file)
        .arg("-C")
        .arg(into)
        .status()
        .unwrap();
    assert!(status.success());
    TimelineModel::load(into).unwrap()
}

#[test]
fn a_better_candidate_is_released_and_a_worse_one_is_rejected_and_recorded() {
    let (scratch, ctx) = context("timeline-release");
    // An earlier cohort that recorded the conventional factors only, and the
    // cohort that took the new measurements.
    let earlier =
        write_synthetic_without(&scratch.0, "earlier.jsonl", CONVENTIONAL, 43, &["x1", "x2"]);
    let later = write_synthetic(&scratch.0, "later.jsonl", FULL, 42);
    let on_earlier = prepare(
        &ctx,
        &earlier,
        "conventional",
        UsagePolicy::Redistributable,
        3,
    );
    let on_later = prepare(&ctx, &later, "full", UsagePolicy::Redistributable, 3);

    // The first release is measured against an untrained baseline.
    let baseline = train(&ctx, &on_earlier, training(3, 1));
    let first = train(&ctx, &on_earlier, training(STEPS, 2));
    // Its cohort did not measure what the causes depend on: all-cause only.
    let measured = evaluate_under(
        &ctx,
        &first.id,
        Champion::Candidate(baseline.id.clone()),
        &on_earlier,
        plan_for(&[]),
    );
    let first_release = release_timeline(
        &ctx,
        &release_request(&measured.id, Distribution::Unrestricted),
    )
    .unwrap();
    assert!(
        first_release.report.passed,
        "{:#?}",
        failures(&first_release.report)
    );
    let first_id = first_release.release.clone().unwrap();
    assert_eq!(ctx.releases().alias(ALIAS).unwrap(), Some(first_id.clone()));
    let stored = ctx.releases().get(&first_id).unwrap();
    assert_eq!(
        stored.manifest.artifact.kind(),
        ArtifactKind::FullCheckpoint
    );
    assert_eq!(
        stored.manifest.provenance.training_config,
        Some(first.candidate.config_digest.clone())
    );
    assert_eq!(stored.manifest.provenance.dataset_snapshots.len(), 2);
    assert!(stored.manifest.provenance.brain_commit.is_some());

    // A candidate that reads the new measurements, measured on participants
    // neither model has seen, replaces the release in place.
    let better = train(&ctx, &on_later, training(STEPS, 2));
    let measured = evaluate(&ctx, &better.id, Champion::Release(ALIAS.into()), &on_later);
    let released = release_timeline(
        &ctx,
        &release_request(&measured.id, Distribution::Unrestricted),
    )
    .unwrap();
    assert!(released.report.passed, "{:#?}", failures(&released.report));
    let second_id = released.release.clone().unwrap();
    assert_eq!(released.champion, Some(first_id.clone()));
    assert_eq!(
        ctx.releases().alias(ALIAS).unwrap(),
        Some(second_id.clone())
    );

    // A worse candidate is rejected: the champion stays, and it is recorded.
    let bad = train(&ctx, &on_later, training(5, 3));
    let measured = evaluate(&ctx, &bad.id, Champion::Release(ALIAS.into()), &on_later);
    let refused = release_timeline(
        &ctx,
        &release_request(&measured.id, Distribution::Unrestricted),
    )
    .unwrap();
    assert!(!refused.report.passed);
    assert!(refused.release.is_none());
    assert!(
        !refused.report.performance.passed,
        "{:#?}",
        refused.report.performance
    );
    assert_eq!(
        ctx.releases().alias(ALIAS).unwrap(),
        Some(second_id.clone()),
        "the champion stays"
    );
    let rejections = ctx.releases().rejections().unwrap();
    assert_eq!(rejections.len(), 1);
    assert_eq!(rejections[0].candidate, bad.id);
    assert_eq!(rejections[0].champion, Some(second_id.clone()));
    assert!(ctx.releases().of_candidate(&bad.id).unwrap().is_none());
    plain_load(
        &ctx.releases().get(&second_id).unwrap().artifact,
        &scratch.0.join("in-place"),
    );
    // Deciding it again records nothing new.
    release_timeline(
        &ctx,
        &release_request(&measured.id, Distribution::Unrestricted),
    )
    .unwrap();
    assert_eq!(ctx.releases().rejections().unwrap().len(), 1);

    // A candidate measured against something other than the release in place
    // cannot take the alias.
    let weak = train(&ctx, &on_later, training(3, 4));
    let other = train(&ctx, &on_later, training(STEPS, 5));
    let against_weak = evaluate(&ctx, &other.id, Champion::Candidate(weak.id), &on_later);
    let why = release_timeline(
        &ctx,
        &release_request(&against_weak.id, Distribution::Unrestricted),
    )
    .err()
    .unwrap();
    assert!(
        why.to_string()
            .contains("evaluate it against the release in place"),
        "{why}"
    );
    assert_eq!(
        ctx.releases().alias(ALIAS).unwrap(),
        Some(second_id.clone())
    );

    // Rollback returns to the previous release, still loadable.
    rollback(&ctx, ALIAS).unwrap();
    assert_eq!(ctx.releases().alias(ALIAS).unwrap(), Some(first_id.clone()));
    plain_load(
        &ctx.releases().get(&first_id).unwrap().artifact,
        &scratch.0.join("rolled-back"),
    );

    // Lineage reaches the datasets and the episodes of the training units.
    let trace = ctx
        .workspace()
        .trace_release(&second_id.0)
        .unwrap()
        .unwrap();
    assert_eq!(trace.earlier_releases, vec![first_id.0.clone()]);
    assert!(
        trace.datasets.contains(&on_later.train.0)
            && trace.datasets.contains(&on_later.validation.0)
    );
    assert_eq!(
        trace.attempts, 0,
        "no experience is behind a timeline model"
    );
    assert_eq!(
        trace.episodes,
        [&on_later, &on_earlier]
            .iter()
            .map(|s| s.records[&Part::Train] + s.records[&Part::Validation])
            .sum::<usize>(),
        "the lineage ends at the episodes the datasets of this release and of the one it continues were projected from"
    );
    let _ = (plan(), scoring(), ALL_CAUSE);
}

#[test]
fn the_released_file_loads_in_plain_brain_and_a_new_checkup_changes_the_risk() {
    let (scratch, ctx) = context("timeline-plain");
    let file = write_synthetic(&scratch.0, "cohort.jsonl", FULL, 22);
    let split = prepare(&ctx, &file, "cohort", UsagePolicy::Redistributable, 3);
    let baseline = train(&ctx, &split, training(3, 1));
    let candidate = train(&ctx, &split, training(STEPS, 1));
    let measured = evaluate(
        &ctx,
        &candidate.id,
        Champion::Candidate(baseline.id),
        &split,
    );
    let released = release_timeline(
        &ctx,
        &release_request(&measured.id, Distribution::Unrestricted),
    )
    .unwrap();
    assert!(released.report.passed, "{:#?}", failures(&released.report));
    let artifact = released.artifact.clone().unwrap();

    // Plain brain, no Splinter: a ustar file, unpacked by tar, loaded by brain.
    let model = plain_load(&artifact, &scratch.0.join("plain"));
    let held = read_jsonl(&ctx.timeline_datasets().get(&split.validation).unwrap().path).unwrap();
    let history = held[0].clone();
    let before = model
        .predict(std::slice::from_ref(&history))
        .unwrap()
        .remove(0);

    // A new checkup after the first as-of: the history grows by an observation
    // and the prediction time moves to it.
    let mut later = history.clone();
    let checkup = history.entry + 2.0;
    later
        .observations
        .push(splinter_model::timeline::Observation {
            t: checkup,
            var: "x1".into(),
            value: splinter_model::timeline::Value::Number(3.0),
            unit: None,
        });
    later.entry = checkup;
    later.at_risk = vec![splinter_model::timeline::AtRisk {
        code: "*".into(),
        from: checkup,
        to: checkup + 10.0,
    }];
    let after = model
        .predict(std::slice::from_ref(&later))
        .unwrap()
        .remove(0);
    let changed = ["death:a", "death:b", "onset"]
        .iter()
        .any(|c| (before.cif(c, 5.0).unwrap() - after.cif(c, 5.0).unwrap()).abs() > 1e-4);
    assert!(
        changed,
        "the risk curves must change when a checkup is appended"
    );
}

#[test]
fn data_whose_terms_forbid_it_is_not_released_unrestricted() {
    let (scratch, ctx) = context("timeline-terms");
    let file = write_synthetic(&scratch.0, "cohort.jsonl", CONVENTIONAL, 23);
    let split = prepare(&ctx, &file, "cohort", UsagePolicy::ResearchOnly, 3);
    let baseline = train(&ctx, &split, training(3, 1));
    let candidate = train(&ctx, &split, training(STEPS, 1));
    let measured = evaluate(
        &ctx,
        &candidate.id,
        Champion::Candidate(baseline.id),
        &split,
    );
    let refused = release_timeline(
        &ctx,
        &release_request(&measured.id, Distribution::Unrestricted),
    )
    .unwrap();
    assert!(!refused.report.passed);
    assert!(
        !refused.report.policy.passed,
        "{:#?}",
        refused.report.policy
    );
    assert!(
        refused.report.performance.passed,
        "the numbers were fine; the terms were not"
    );
    assert!(ctx.releases().alias(ALIAS).unwrap().is_none());
    assert_eq!(ctx.releases().rejections().unwrap().len(), 1);
    let kept = release_timeline(
        &ctx,
        &release_request(&measured.id, Distribution::Restricted),
    )
    .unwrap();
    assert!(kept.report.passed, "{:#?}", failures(&kept.report));
    let id = kept.release.unwrap();
    assert_eq!(
        ctx.releases().get(&id).unwrap().manifest.distribution,
        Distribution::Restricted
    );
}

#[test]
fn units_that_are_not_held_out_are_refused_before_scoring() {
    let (scratch, ctx) = context("timeline-units");
    let file = write_synthetic(&scratch.0, "cohort.jsonl", 600, 24);
    let a = prepare(&ctx, &file, "cohort", UsagePolicy::Redistributable, 3);
    let b = prepare(&ctx, &file, "cohort", UsagePolicy::Redistributable, 4);
    let on_a = train(&ctx, &a, training(20, 1));
    let on_b = train(&ctx, &b, training(20, 1));
    let request = |candidate: &str,
                   champion: Champion,
                   test: &splinter_pipelines::timeline::data::SplitReport| {
        TimelineEvaluateRequest {
            candidate: candidate.into(),
            champion,
            test: test.test.to_string(),
            scoring: scoring(),
            plan: plan(),
        }
    };
    // The test part of another split is not known to be unseen.
    let other = evaluate_timeline(
        &ctx,
        &request(&on_a.id, Champion::Candidate(on_b.id.clone()), &b),
    )
    .err()
    .unwrap();
    assert!(other.to_string().contains("cut from split"), "{other}");
    // The training part is not a test part.
    let mut as_train = request(&on_a.id, Champion::Candidate(on_b.id.clone()), &a);
    as_train.test = a.train.to_string();
    assert!(evaluate_timeline(&ctx, &as_train)
        .err()
        .unwrap()
        .to_string()
        .contains("not the Test part"));
    // A champion that was fitted on some of the test units is refused.
    let leaked = evaluate_timeline(&ctx, &request(&on_a.id, Champion::Candidate(on_b.id), &a))
        .err()
        .unwrap();
    assert!(leaked.to_string().contains("not held out"), "{leaked}");
    assert!(ctx
        .workspace()
        .document_ids("timeline_evaluation")
        .unwrap()
        .is_empty());
}
