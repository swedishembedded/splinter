// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning that keeps accumulating
// capabilities generation after generation, for its clients. If your team
// needs expertise in evaluation-gated consolidation of what a model learned,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! The gate of a consolidated checkpoint.
//!
//! The checkpoint, with no adapter, and the release in use, with its own, are
//! asked the stopping paraphrases of every claim of the closing generation
//! and of every claim the base already held, closed-book and greedily under
//! the prompt each is deployed with, and the anchor suite under the default
//! prompt; [`splinter_eval::claim_gate::decide_consolidation`] rules on the
//! counts: a claim the release in use answered must still be answered, the
//! anchor may lose no item, and plain `brain serve` must serve the
//! checkpoint.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;

use splinter_agent::CancelToken;
use splinter_core::claim::ClaimId;
use splinter_core::dataset::DatasetId;
use splinter_core::digest::Digest;
use splinter_eval::claim_gate::{decide_consolidation, ClaimGate};
use splinter_eval::gate::{Check, GateConfig, Serve};
use splinter_model::local::{load_source, resolve_base};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_orchestrator::releases::StoredRelease;

use crate::absorb::gate::{claim_suites, served_sample, tallies};
use crate::absorb::kit::{ClaimKit, Live};
use crate::release::probe::Suite;
use crate::release::serve::{self, ServeCheck};
use crate::release::{anchor, arm_on, grade_arm, leakage, Asked, Graded};

/// What the checkpoint is decided on.
pub(super) struct Decision<'a> {
    /// The checkpoint directory under decision: a base like any other.
    pub(super) checkpoint: &'a Path,
    /// `sha256:<hex>` of its weights file.
    pub(super) weights_digest: &'a str,
    /// The datasets it was trained on; empty for a merge, which trained none.
    pub(super) datasets: &'a [DatasetId],
    /// The files of rehearsed records it was trained beside.
    pub(super) mixed_in: &'a [Option<&'a Digest>],
    /// The release in use, which the checkpoint replaces.
    pub(super) champion: &'a StoredRelease,
    /// The base directory the champion runs on.
    pub(super) champion_base: &'a Path,
    /// The claims asked about: the closing generation's and the base's.
    pub(super) claims: &'a [Live],
    /// Their kits.
    pub(super) kits: &'a BTreeMap<ClaimId, ClaimKit>,
    /// The serve check's bounds.
    pub(super) config: &'a GateConfig,
    /// Stops the gate.
    pub(super) cancel: &'a CancelToken,
}

/// Decides the checkpoint against the release in use.
pub(super) fn decide(
    ctx: &Context,
    decision: &Decision<'_>,
) -> Result<ClaimGate, OrchestratorError> {
    let trained = leakage::trained_prompts_of(
        ctx,
        decision.datasets,
        decision.mixed_in,
        "the consolidated checkpoint",
    )?;
    let suites = claim_suites(ctx, decision.claims, decision.kits, &trained)?;
    let anchor_suite = anchor::current(ctx)?.map(|frozen| frozen.probe_suite());
    let mut all: Vec<(&Suite, Asked)> = suites.iter().map(|(_, s)| (s, Asked::Deployed)).collect();
    all.extend(anchor_suite.iter().map(|s| (s, Asked::Default)));

    // A merge trains nothing, so the checkpoint answers under the prompt the
    // adapter it folds in was trained under.
    let champion_datasets = &decision.champion.manifest.datasets;
    let champion_prompt = ctx.system_prompt_of(champion_datasets)?;
    let candidate_prompt = if decision.datasets.is_empty() {
        champion_prompt.clone()
    } else {
        ctx.system_prompt_of(decision.datasets)?
    };
    let champion_adapter = decision.champion.adapter().ok();
    let theirs = grade_arm(
        ctx,
        &arm_on(ctx.config(), decision.checkpoint, None),
        candidate_prompt.as_deref(),
        &all,
        decision.cancel,
    )?;
    let ours = grade_arm(
        ctx,
        &arm_on(ctx.config(), decision.champion_base, champion_adapter),
        champion_prompt.as_deref(),
        &all,
        decision.cancel,
    )?;
    let (claims, anchor) = tallies(&suites, anchor_suite.is_some(), &theirs, &ours);
    let serve = serve_check(ctx, decision, &suites, &theirs, candidate_prompt.as_deref())?;
    Ok(decide_consolidation(claims, anchor, serve))
}

/// Plain brain serves the checkpoint, with no adapter, and answers a sample
/// of the stopping paraphrases as the checkpoint does in-process.
fn serve_check(
    ctx: &Context,
    decision: &Decision<'_>,
    suites: &[(ClaimId, Suite)],
    graded: &[Graded],
    system: Option<&str>,
) -> Result<Check<Serve>, OrchestratorError> {
    let Some((sample, probes)) = served_sample(suites, graded, decision.config.serve_sample) else {
        return Ok(Check::unmeasured(
            "no stopping paraphrase was graded in-process to compare with",
        ));
    };
    let base =
        load_source(&resolve_base(decision.checkpoint).map_err(|e| {
            OrchestratorError::Refused(format!("the consolidated checkpoint: {e}"))
        })?);
    serve::check(
        ctx,
        &ServeCheck {
            binary: ctx.config().brain_binary.as_deref(),
            base: &base,
            adapter: None,
            adapter_digest: decision.weights_digest,
            system,
            sample: &sample,
            in_process: &probes,
            startup: Duration::from_secs(decision.config.serve_startup_secs),
        },
        decision.cancel,
    )
}
