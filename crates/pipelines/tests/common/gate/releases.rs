// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate fixtures' releases: a candidate recorded as the release of
//! `default`, as a gate that passed would leave it, and the model a release
//! serves handed to the context.

use splinter_agent::solve::Model;
use splinter_core::release::ReleaseId;
use splinter_orchestrator::Context;
use splinter_pipelines::release::arm;
use splinter_pipelines::train::Candidate;

use super::super::repo::Mender;
use super::{candidate, fast_gate, knower};

/// Records `candidate` as a release of `default`, from the release it was
/// trained from, and serves it: the fixture of a release the specs build on,
/// as a gate that passed would leave it. It runs no gate; the gates that
/// release (absorb, consolidation, predictive, the factory's promotion) are
/// specified where they live.
pub fn release_of(ctx: &Context, candidate: &Candidate) -> ReleaseId {
    use splinter_eval::gate::{Check, GateReport};
    use splinter_orchestrator::releases::{
        Provenance, ReleaseGate, ReleaseManifest, ReleasedArtifact, RELEASE_FORMAT,
    };
    fn passed<T>() -> Check<T> {
        Check {
            passed: true,
            measured: None,
            reason: Some("a fixture release".into()),
            inconclusive: false,
        }
    }
    let report = GateReport {
        config: fast_gate(),
        improvement: passed(),
        retention: passed(),
        anchor: passed(),
        serve: passed(),
        passed: true,
        canary: None,
        prompts: Vec::new(),
        interval_seed: None,
    };
    let digest = |text: &str| splinter_core::digest::Digest::parse(text).unwrap();
    let manifest = ReleaseManifest {
        format: RELEASE_FORMAT.into(),
        artifact: ReleasedArtifact::Adapter {
            base_model: splinter_model::selection::local_model_name(&candidate.base),
            base_digest: digest(&candidate.base_digest),
            adapter_digest: digest(&candidate.adapter_digest),
            adapter_artifact: candidate.adapter_artifact.clone(),
        },
        parent: candidate.parent.clone(),
        candidate: candidate.candidate.clone(),
        datasets: candidate.datasets.clone(),
        replay: candidate.replay.clone(),
        rehearsal: candidate.rehearsal.clone(),
        training: splinter_core::training::TrainingSummary {
            from: candidate.from.clone(),
            steps: candidate.steps,
            rank: candidate.rank,
            records: candidate.records,
            regime: candidate.regime,
            base_score: candidate.base_score,
            tuned_score: candidate.tuned_score,
            preference: candidate.preference.clone(),
            curve: candidate.curve.clone(),
            record: candidate.training_record.clone(),
            terms: candidate.terms.clone(),
        },
        gate: ReleaseGate::Llm {
            report: Box::new(report),
            promotion: None,
        },
        metrics: None,
        terms: candidate
            .terms
            .clone()
            .unwrap_or_else(|| splinter_core::terms::Terms::unknown("unstated")),
        distribution: splinter_core::terms::Distribution::Restricted,
        provenance: Provenance::default(),
        generation: None,
        created_at: ctx.clock().utc_now(),
    };
    let store = ctx.releases();
    let champion = store.alias("default").unwrap();
    let stored = store.put(&manifest).unwrap();
    store
        .move_alias(
            "default",
            champion.as_ref(),
            &stored.id,
            &ctx.clock().utc_now(),
        )
        .unwrap();
    ctx.repin_policy("default");
    serve_release(ctx, &stored.id);
    stored.id
}

/// Trains a candidate knowing `knows` on `topic` and records it as the
/// release of `default` ([`release_of`]).
pub fn released(ctx: &Context, topic: &str, knows: &[&str]) -> ReleaseId {
    let (candidate, _) = candidate(ctx, topic, knows);
    release_of(ctx, &candidate)
}

/// Hands `ctx` the model the release `id` serves: its adapter file says
/// what it knows, or whether it restores the module it is asked about. The gate grades a champion through its release's copy of
/// the adapter.
fn serve_release(ctx: &Context, id: &ReleaseId) {
    let stored = ctx.releases().get(id).unwrap();
    let adapter: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&stored.artifact).unwrap()).unwrap();
    if let Some(agent) = adapter["agent"].as_str() {
        let model = Model::new(Mender::new(agent == "mender"), "scripted/mender");
        ctx.add_model(arm(ctx.config(), Some(&stored.artifact)), model);
        return;
    }
    let knows: Vec<&str> = adapter["knows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_str().unwrap())
        .collect();
    ctx.add_model(arm(ctx.config(), Some(&stored.artifact)), knower(&knows));
}
