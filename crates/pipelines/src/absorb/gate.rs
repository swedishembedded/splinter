// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning from the sessions a person holds
// with an agent, for its clients. If your team needs expertise in
// evaluation-gated fine-tuning, you can procure our services by sending an
// email to info@swedishembedded.com.

//! The nightly gate and the release it opens.
//!
//! The candidate and the champion (the base, before any release) are asked
//! each live claim's stopping paraphrases closed-book, greedily, under the
//! prompt each is deployed with, and the anchor suite under the default
//! prompt; [`splinter_eval::claim_gate::decide`] rules on the counts. When a
//! stance pass ran, both are also put the situations held out of its pairs
//! and the three stance rates may not fall below the champion's. The gate
//! reads the claims' own stopping paraphrases, the anchor suite and the
//! held-out stance items and nothing else: a sealed probe is never among
//! them.

use std::collections::BTreeMap;

use splinter_agent::CancelToken;
use splinter_core::claim::ClaimId;
use splinter_core::release::ReleaseId;
use splinter_core::terms::Distribution;
use splinter_eval::claim_gate::{decide, AnchorTally, ClaimGate, ClaimTally, Tally};
use splinter_eval::gate::{Check, GateConfig};
use splinter_orchestrator::releases::{ReleaseGate, StoredRelease};

use super::kit::{ClaimKit, Live};
use super::stance::{self, Held};
use crate::release::probe::{Probe, Suite};
use crate::release::serve::{self, ServeCheck};
use crate::release::{anchor, arm, grade_arm, leakage, manifest, release_terms, Asked, Graded};
use crate::train::Candidate;
use splinter_model::local::{load_source, resolve_base};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use std::time::Duration;

/// Passes out of measured among the answers of one suite; none when the
/// suite could not be graded.
fn tally(graded: &Graded) -> Tally {
    let probes = graded.as_deref().unwrap_or_default();
    Tally {
        passed: probes.iter().filter(|p| p.verdict == Some(true)).count(),
        measured: probes.iter().filter(|p| p.verdict.is_some()).count(),
    }
}

/// What a night's candidate is decided on.
pub struct ClaimsDecision<'a> {
    /// The candidate.
    pub candidate: &'a Candidate,
    /// The release in use; the base before any.
    pub champion: Option<&'a StoredRelease>,
    /// The claims the candidate was trained on.
    pub live: &'a [Live],
    /// Their kits.
    pub kits: &'a BTreeMap<ClaimId, ClaimKit>,
    /// The items held out of the stance pairs, when a stance pass ran.
    pub held: Option<&'a Held>,
    /// The serve check's bounds.
    pub config: &'a GateConfig,
    /// Stops the gate.
    pub cancel: &'a CancelToken,
}

/// Decides the candidate against the champion on the live claims, and on the
/// stance items held out of the pairs when there are any.
pub fn decide_claims(
    ctx: &Context,
    decision: &ClaimsDecision<'_>,
) -> Result<ClaimGate, OrchestratorError> {
    let ClaimsDecision {
        candidate,
        champion,
        live,
        kits,
        held,
        config,
        cancel,
    } = *decision;
    let trained = leakage::trained_prompts(ctx, candidate)?;
    let mut suites: Vec<(ClaimId, Suite)> = Vec::new();
    for (id, _) in live {
        let Some(kit) = kits.get(id) else { continue };
        let tasks = kit
            .stopping
            .iter()
            .map(|t| ctx.tasks().get(t))
            .collect::<Result<Vec<_>, _>>()?;
        let mut suite = Suite::of_tasks(format!("claim {id}"), tasks);
        leakage::exclude_leaked(&mut suite, &trained);
        suites.push((id.clone(), suite));
    }
    let anchor_suite = anchor::current(ctx)?.map(|frozen| frozen.probe_suite());
    let mut all: Vec<(&Suite, Asked)> = suites.iter().map(|(_, s)| (s, Asked::Deployed)).collect();
    all.extend(anchor_suite.iter().map(|s| (s, Asked::Default)));

    let candidate_prompt = ctx.system_prompt_of(&candidate.datasets)?;
    let champion_prompt = match champion {
        Some(c) => ctx.system_prompt_of(&c.manifest.datasets)?,
        None => None,
    };
    let champion_adapter = champion.map(StoredRelease::adapter).transpose()?;
    let theirs = grade_arm(
        ctx,
        &arm(ctx.config(), Some(&candidate.adapter)),
        candidate_prompt.as_deref(),
        &all,
        cancel,
    )?;
    let ours = grade_arm(
        ctx,
        &arm(ctx.config(), champion_adapter),
        champion_prompt.as_deref(),
        &all,
        cancel,
    )?;
    let claims: Vec<ClaimTally> = suites
        .iter()
        .enumerate()
        .map(|(n, (id, _))| ClaimTally {
            claim: id.to_string(),
            candidate: tally(&theirs[n]),
            champion: ours[n].as_ref().ok().map(|_| tally(&ours[n])),
        })
        .collect();
    let anchor = anchor_suite.as_ref().and_then(|_| {
        let n = suites.len();
        let measured = |g: &Graded| g.as_ref().ok().map(|_| tally(g));
        Some(AnchorTally {
            candidate: measured(&theirs[n])?,
            champion: measured(&ours[n])?,
        })
    });
    let serve = serve_check(
        ctx,
        candidate,
        &suites,
        &theirs,
        candidate_prompt.as_deref(),
        config,
        cancel,
    )?;
    let gate = decide(claims, anchor, serve);
    let Some(held) = held.filter(|h| !h.is_empty()) else {
        return Ok(gate);
    };
    let theirs = stance::measure(
        ctx,
        &arm(ctx.config(), Some(&candidate.adapter)),
        candidate_prompt.as_deref(),
        held,
        cancel,
    )?;
    let ours = stance::measure(
        ctx,
        &arm(ctx.config(), champion_adapter),
        champion_prompt.as_deref(),
        held,
        cancel,
    )?;
    Ok(gate.with_stance(stance::compare(theirs, ours)))
}

/// Plain brain serves the candidate and answers a sample of the stopping
/// paraphrases as the candidate does in-process.
fn serve_check(
    ctx: &Context,
    candidate: &Candidate,
    suites: &[(ClaimId, Suite)],
    graded: &[Graded],
    system: Option<&str>,
    config: &GateConfig,
    cancel: &CancelToken,
) -> Result<Check<splinter_eval::gate::Serve>, OrchestratorError> {
    let mut tasks = Vec::new();
    let mut probes: Vec<Probe> = Vec::new();
    for ((_, suite), outcome) in suites.iter().zip(graded) {
        if let Ok(answers) = outcome {
            tasks.extend(suite.tasks.iter().cloned());
            probes.extend(answers.iter().cloned());
        }
    }
    let take = config.serve_sample.min(tasks.len());
    if take == 0 {
        return Ok(Check::unmeasured(
            "no stopping paraphrase was graded in-process to compare with",
        ));
    }
    let sample = Suite::of_tasks(
        "stopping paraphrases (served sample)",
        tasks[..take].to_vec(),
    );
    let base = load_source(
        &resolve_base(&ctx.config().policy_base)
            .map_err(|e| OrchestratorError::Refused(format!("the policy base: {e}")))?,
    );
    serve::check(
        ctx,
        &ServeCheck {
            binary: ctx.config().brain_binary.as_deref(),
            base: &base,
            adapter: &candidate.adapter,
            adapter_digest: &candidate.adapter_digest,
            system,
            sample: &sample,
            in_process: &probes[..take],
            startup: Duration::from_secs(config.serve_startup_secs),
        },
        cancel,
    )
}

/// Writes the release `gate` opened and moves `alias` to it from the
/// champion. The candidate was trained from the base, so the release's
/// parent is the champion it replaces: the lineage retention and rollback
/// walk.
pub fn publish(
    ctx: &Context,
    alias: &str,
    candidate: &Candidate,
    champion: Option<&StoredRelease>,
    gate: ClaimGate,
) -> Result<ReleaseId, OrchestratorError> {
    let terms = release_terms(candidate, champion);
    let mut made = manifest(
        ctx,
        candidate,
        ReleaseGate::Claims {
            report: Box::new(gate),
        },
        terms,
        Distribution::Restricted,
    )?;
    let champion_id = champion.map(|c| c.id.clone());
    made.parent = champion_id.clone();
    let store = ctx.releases();
    let stored = store.put(&made)?;
    store.move_alias(
        alias,
        champion_id.as_ref(),
        &stored.id,
        &ctx.clock().utc_now(),
    )?;
    ctx.repin_policy(alias);
    Ok(stored.id)
}
