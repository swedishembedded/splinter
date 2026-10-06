// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pre-registered release gates for
// predictive models, where a candidate replaces the champion only on
// paired held-out evidence. If your team needs expertise in validating
// risk models or forecasters before they ship, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: a full-checkpoint candidate that is not a language model is released
//! through the predictive gate, over numbers its caller measured.
//!
//! * When all five checks pass the checkpoint is kept immutably, the release
//!   records the gate, the metrics, the terms and the digests, and the alias
//!   moves; the next release continues it and `rollback` returns to it.
//! * A candidate and a champion scored on different units are refused, with
//!   nothing recorded.
//! * A check that fails, or whose number was not measured, blocks the release
//!   and leaves the alias where it was; a data-policy failure blocks a
//!   candidate that passes every measurement.
//! * A checkpoint that is not the file its digest names is refused; asking
//!   again for a released candidate changes nothing.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use common::gate::{dataset_with_terms, gate_context, Brain};
use common::Scratch;
use splinter_core::digest::Digest;
use splinter_core::terms::{Distribution, Terms, UsagePolicy};
use splinter_eval::metric_gate::{Evidence, Requirement};
use splinter_eval::predictive_gate::{Measurements, PredictiveSpec};
use splinter_orchestrator::releases::{ArtifactKind, Provenance, ReleaseGate};
use splinter_orchestrator::{Context, OrchestratorError};
use splinter_pipelines::release::predictive::{
    release_predictive, CheckpointFile, CheckpointTraining, PredictiveRelease, PredictiveReleased,
};
use splinter_pipelines::release::rollback;

const ALIAS: &str = "risk";

fn spec() -> PredictiveSpec {
    PredictiveSpec {
        performance: vec![Requirement::Improves {
            interval: "ibs_diff".into(),
            lower_is_better: true,
        }],
        calibration: vec![Requirement::Within {
            value: "slope".into(),
            lo: 0.9,
            hi: 1.1,
        }],
        retention: vec![Requirement::NotWorseBy {
            prefix: "subgroup:".into(),
            bound: 0.002,
            lower_is_better: true,
        }],
        serving: vec![Requirement::Within {
            value: "serve_max_abs_diff".into(),
            lo: 0.0,
            hi: 1e-6,
        }],
    }
}

fn evidence() -> Evidence {
    let mut e = Evidence::default();
    e.intervals.insert("ibs_diff".into(), (-0.004, -0.001));
    e.values.insert("slope".into(), 0.97);
    e.values.insert("subgroup:sex=female".into(), 0.001);
    e.values.insert("serve_max_abs_diff".into(), 0.0);
    e
}

fn units() -> Vec<String> {
    (0..20).map(|i| format!("subject-{i}")).collect()
}

/// A release of candidate `name` whose checkpoint is `bytes`, trained on a
/// dataset stating `terms`, on the numbers `evidence`.
struct Case<'a> {
    scratch: &'a Scratch,
    ctx: &'a Context,
    name: &'a str,
    bytes: &'a [u8],
    terms: Option<Terms>,
    evidence: Evidence,
    distribution: Distribution,
}

impl Case<'_> {
    fn request(&self, topic: &str) -> PredictiveRelease {
        let dataset = dataset_with_terms(self.ctx, topic, 20, self.terms.clone());
        let path = self.scratch.0.join(format!("{}.ckpt", self.name));
        std::fs::write(&path, self.bytes).unwrap();
        PredictiveRelease {
            candidate: self.name.into(),
            alias: ALIAS.into(),
            datasets: vec![dataset.to_string()],
            checkpoint: CheckpointFile {
                path,
                architecture: "timeline-v1".into(),
                digest: Digest::sha256_of(self.bytes),
            },
            training: CheckpointTraining {
                from: "scratch".into(),
                objective: "timeline".into(),
                steps: 100,
                records: 20,
                record: serde_json::json!({ "trainer": "brain" }),
            },
            spec: spec(),
            measurements: Measurements {
                champion_units: units(),
                candidate_units: units().into_iter().rev().collect(),
                evidence: self.evidence.clone(),
            },
            distribution: self.distribution,
            provenance: Provenance {
                training_config: Some(Digest::of(b"config")),
                brain_commit: Some("3f019a23".into()),
                ..Provenance::default()
            },
        }
    }

    fn release(&self, topic: &str) -> PredictiveReleased {
        release_predictive(self.ctx, &self.request(topic)).unwrap()
    }
}

fn open() -> Option<Terms> {
    Some(UsagePolicy::Redistributable.terms("open"))
}

fn case<'a>(scratch: &'a Scratch, ctx: &'a Context, name: &'a str, bytes: &'a [u8]) -> Case<'a> {
    Case {
        scratch,
        ctx,
        name,
        bytes,
        terms: open(),
        evidence: evidence(),
        distribution: Distribution::Unrestricted,
    }
}

fn context(test: &str) -> (Scratch, Context) {
    gate_context(test, Brain::Missing)
}

#[test]
fn a_candidate_that_passes_all_five_checks_is_released_and_rolled_back() {
    let (scratch, ctx) = context("predictive-release");
    let first = case(&scratch, &ctx, "risk-1", b"checkpoint one").release("alpha");
    assert!(first.report.passed, "{:#?}", first.report);
    let first_id = first.release.clone().unwrap();
    assert_eq!(first.champion, None);

    let stored = ctx.releases().get(&first_id).unwrap();
    let manifest = &stored.manifest;
    assert_eq!(manifest.artifact.kind(), ArtifactKind::FullCheckpoint);
    assert_eq!(
        manifest.artifact.content_digest(),
        &Digest::sha256_of(b"checkpoint one")
    );
    assert_eq!(std::fs::read(&stored.artifact).unwrap(), b"checkpoint one");
    assert_eq!(manifest.gate.predictive(), Some(&first.report));
    assert!(matches!(manifest.gate, ReleaseGate::Predictive { .. }));
    assert_eq!(manifest.metrics.as_ref(), Some(&evidence()));
    assert_eq!(manifest.distribution, Distribution::Unrestricted);
    assert_eq!(
        manifest.provenance.brain_commit.as_deref(),
        Some("3f019a23")
    );
    assert_eq!(
        manifest.provenance.training_config,
        Some(Digest::of(b"config"))
    );
    assert_eq!(manifest.provenance.dataset_snapshots.len(), 1);
    assert_eq!(
        manifest.provenance.evaluation_splits,
        vec![first.report.units_digest.clone()]
    );
    assert_eq!(ctx.releases().alias(ALIAS).unwrap(), Some(first_id.clone()));

    // The next one is measured against it, continues it, and can be undone.
    let second = case(&scratch, &ctx, "risk-2", b"checkpoint two").release("beta");
    assert!(second.report.passed, "{:#?}", second.report);
    assert_eq!(second.champion, Some(first_id.clone()));
    let second_id = second.release.unwrap();
    assert_eq!(
        ctx.releases().get(&second_id).unwrap().manifest.parent,
        Some(first_id.clone())
    );
    assert_eq!(
        ctx.releases().alias(ALIAS).unwrap(),
        Some(second_id.clone())
    );
    let back = rollback(&ctx, ALIAS).unwrap();
    assert_eq!((back.from, back.to), (second_id, first_id.clone()));
    assert_eq!(ctx.releases().alias(ALIAS).unwrap(), Some(first_id));
}

#[test]
fn units_that_differ_between_champion_and_candidate_are_refused() {
    let (scratch, ctx) = context("predictive-units");
    let case = case(&scratch, &ctx, "risk-1", b"checkpoint one");
    let mut request = case.request("alpha");
    request.measurements.candidate_units.pop();
    request
        .measurements
        .candidate_units
        .push("someone-else".into());
    match release_predictive(&ctx, &request) {
        Err(OrchestratorError::Refused(why)) => {
            assert!(why.contains("different held-out units"), "{why}");
            assert!(why.contains("someone-else"), "{why}");
        }
        other => panic!("different units must be refused, got {other:?}"),
    }
    assert!(ctx.releases().list().unwrap().is_empty());
    assert_eq!(ctx.releases().alias(ALIAS).unwrap(), None);
}

#[test]
fn a_failing_or_unmeasured_check_blocks_the_release_and_leaves_the_alias() {
    let (scratch, ctx) = context("predictive-blocked");
    let first = case(&scratch, &ctx, "risk-1", b"checkpoint one").release("alpha");
    let champion = first.release.unwrap();

    let mut worse = case(&scratch, &ctx, "risk-2", b"checkpoint two");
    worse
        .evidence
        .intervals
        .insert("ibs_diff".into(), (-0.004, 0.001));
    let blocked = worse.release("beta");
    assert!(!blocked.report.passed && !blocked.report.performance.passed);
    assert_eq!(blocked.release, None);

    let mut unmeasured = case(&scratch, &ctx, "risk-3", b"checkpoint three");
    unmeasured.evidence.values.remove("serve_max_abs_diff");
    let blocked = unmeasured.release("gamma");
    assert!(!blocked.report.serving.passed && blocked.release.is_none());
    assert_eq!(
        blocked.report.serving.decided[0].check.measured, None,
        "an unmeasured number is absent, never zero"
    );

    assert_eq!(ctx.releases().alias(ALIAS).unwrap(), Some(champion));
    assert_eq!(ctx.releases().list().unwrap().len(), 1);
    assert!(ctx.releases().of_candidate("risk-2").unwrap().is_none());
}

#[test]
fn a_policy_failure_blocks_a_candidate_that_passes_every_measurement() {
    let (scratch, ctx) = context("predictive-policy");
    let mut research = case(&scratch, &ctx, "risk-1", b"checkpoint one");
    research.terms = Some(UsagePolicy::ResearchOnly.terms("cohort"));
    let blocked = research.release("alpha");
    let report = &blocked.report;
    assert!(
        report.performance.passed
            && report.calibration.passed
            && report.retention.passed
            && report.serving.passed
    );
    assert!(!report.policy.passed && !report.passed);
    assert_eq!(blocked.release, None);
    assert_eq!(ctx.releases().alias(ALIAS).unwrap(), None);

    // The same candidate may be released restricted: training is allowed.
    research.distribution = Distribution::Restricted;
    let released = research.release("alpha");
    assert!(released.report.passed, "{:#?}", released.report);
    let manifest = ctx
        .releases()
        .get(&released.release.unwrap())
        .unwrap()
        .manifest;
    assert_eq!(manifest.distribution, Distribution::Restricted);
    assert_eq!(manifest.terms.name, "cohort");

    // A dataset that states no terms counts as unknown beside the champion's:
    // training is not allowed, so no release.
    let mut unstated = case(&scratch, &ctx, "risk-2", b"checkpoint two");
    unstated.terms = None;
    unstated.distribution = Distribution::Restricted;
    let blocked = unstated.release("beta");
    assert!(blocked.release.is_none() && !blocked.report.policy.passed);
}

#[test]
fn a_checkpoint_that_is_not_its_digest_is_refused_and_a_repeat_changes_nothing() {
    let (scratch, ctx) = context("predictive-digest");
    let case = case(&scratch, &ctx, "risk-1", b"checkpoint one");
    let mut wrong = case.request("alpha");
    wrong.checkpoint.digest = Digest::sha256_of(b"some other checkpoint");
    match release_predictive(&ctx, &wrong) {
        Err(OrchestratorError::Refused(why)) => assert!(why.contains("SHA-256"), "{why}"),
        other => panic!("a digest mismatch must be refused, got {other:?}"),
    }
    assert_eq!(ctx.releases().alias(ALIAS).unwrap(), None);

    let made = case.release("alpha");
    let again = case.release("alpha");
    assert_eq!(again.release, made.release, "a candidate is released once");
    assert_eq!(ctx.releases().list().unwrap().len(), 1);
}
