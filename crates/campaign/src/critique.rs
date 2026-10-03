// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements critique-and-retry loops that turn failed
// agent attempts into verified revisions, for its clients. If your team
// needs expertise in agent self-improvement or learning from feedback, you
// can procure our services by sending an email to info@swedishembedded.com.

//! The critique stage: every experience of a set that is decided fail is
//! critiqued and retried with the critique, up to a number of retries,
//! and what that produced - critiques and revisions, related to what they
//! answer - is a new experience set.
//!
//! The retry is solved by the policy, the model being trained, in the
//! task's own environment, and graded by the task kind's verifiers (never a
//! judge: the loop decides by what it can check).

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use serde::Serialize;
use splinter_agent::critic::Critic;
use splinter_agent::repair::{Repair, RepairBudget, Stop};
use splinter_agent::CancelToken;
use splinter_store::experiences::{ExperienceSet, SetId};

use crate::context::Context;
use crate::error::CampaignError;
use crate::model_ref::ModelRef;
use crate::tasks::remaining;
use crate::verify::{verifiers_for, Unverified};

/// Retries per failed experience when a command names none.
pub const DEFAULT_RETRIES: usize = 1;

/// How long one failed experience's critique-and-retry loop may take.
pub const DEFAULT_REPAIR_DEADLINE: Duration = Duration::from_secs(900);

/// What the critique stage reports.
#[derive(Clone, Debug, Serialize)]
pub struct Critiqued {
    /// Everything the stage made: the critiques and the revisions
    /// (`dataset build <experience_set> ... --view critic`).
    pub experience_set: SetId,
    /// The revisions alone: the retried answers, without the critics' own
    /// experiences (`dataset build <revisions> ... --view sft-final`).
    pub revisions: SetId,
    /// Failed experiences critiqued.
    pub critiqued: usize,
    /// Of those, how many a revision repaired.
    pub repaired: usize,
    /// Experiences not decided fail, left alone.
    pub not_failed: usize,
    /// Why each loop stopped, by reason.
    pub stops: BTreeMap<String, usize>,
    /// Failed experiences no loop could run for, with why.
    pub unrepaired: Vec<Unverified>,
    /// Why the stage stopped before every failure was tried, if it did.
    pub stopped: Option<String>,
}

/// The models and bounds of one critique stage.
pub struct CritiqueRequest<'a> {
    /// The experience set whose failures are critiqued.
    pub set: &'a SetId,
    /// The model that critiques.
    pub critic: &'a ModelRef,
    /// The model that retries.
    pub solver: &'a ModelRef,
    /// Retries per failed experience.
    pub retries: usize,
    /// No loop starts after this, and none runs past it.
    pub deadline: Option<Instant>,
    /// Stops the stage.
    pub cancel: CancelToken,
}

/// Critiques and retries every failed experience of `request.set`.
pub fn critique_set(
    ctx: &Context,
    request: &CritiqueRequest<'_>,
) -> Result<Critiqued, CampaignError> {
    let store = ctx.experiences();
    let batch = ctx.workspace().batch();
    let members = store.get_set(request.set)?.members;
    let critic = Critic::new(ctx.model(request.critic)?);
    let solver = ctx.model(request.solver)?;
    let pool = members
        .iter()
        .map(|id| store.get(id))
        .collect::<Result<Vec<_>, _>>()?;
    let mut produced = Vec::new();
    let mut revisions = Vec::new();
    let unset = SetId(splinter_core::digest::Digest::of(b""));
    let mut report = Critiqued {
        experience_set: unset.clone(),
        revisions: unset,
        critiqued: 0,
        repaired: 0,
        not_failed: 0,
        stops: BTreeMap::new(),
        unrepaired: Vec::new(),
        stopped: None,
    };
    let decisions = store.decisions(&members)?;
    for (id, experience) in members.iter().zip(&pool) {
        if request.cancel.is_cancelled() {
            return Err(CampaignError::Cancelled);
        }
        if decisions.get(id).is_none_or(|d| d.passed) {
            report.not_failed += 1;
            continue;
        }
        if request.deadline.is_some_and(|d| Instant::now() >= d) {
            report.stopped = Some("the budget was spent before every failure was tried".into());
            break;
        }
        let task = experience.to_task();
        let unrepaired = |reason: String| Unverified {
            experience: id.clone(),
            reason,
        };
        let prepared = ctx
            .environments()
            .for_record(&task.environment)
            .and_then(|env| Ok((env, verifiers_for(ctx, &task, &pool, None)?)));
        let (environment, verifiers) = match prepared {
            Ok(prepared) => prepared,
            Err(e) => {
                report.unrepaired.push(unrepaired(e.to_string()));
                continue;
            }
        };
        let repair = Repair {
            store: &store,
            clock: ctx.clock(),
            runtime: ctx.handle(),
            environment: &environment,
            verifiers: &verifiers,
            solver: solver.clone(),
            critic: critic.clone(),
        };
        let budget = RepairBudget {
            cancel: Some(request.cancel.clone()),
            ..RepairBudget::new(
                request.retries,
                remaining(request.deadline, DEFAULT_REPAIR_DEADLINE),
            )
        };
        let loop_report = repair.repair_loop(&task, id, &budget)?;
        report.critiqued += 1;
        if matches!(loop_report.stop, Stop::Passed) {
            report.repaired += 1;
        }
        *report
            .stops
            .entry(stop_name(&loop_report.stop))
            .or_default() += 1;
        if let Stop::Error(e) = &loop_report.stop {
            report.unrepaired.push(unrepaired(e.to_string()));
        }
        for made in loop_report.chain().into_iter().skip(1) {
            if !produced.contains(&made) {
                produced.push(made);
            }
        }
        for round in &loop_report.rounds {
            if let Some(retried) = &round.retried {
                if !revisions.contains(&retried.revision) {
                    revisions.push(retried.revision.clone());
                }
            }
        }
    }
    report.experience_set = store.put_set(&ExperienceSet {
        name: format!("critiques and retries of {}", request.set),
        members: produced,
    })?;
    report.revisions = store.put_set(&ExperienceSet {
        name: format!("retries of {}", request.set),
        members: revisions,
    })?;
    batch.commit()?;
    Ok(report)
}

/// Why a loop stopped, as reports name it.
fn stop_name(stop: &Stop) -> String {
    match stop {
        Stop::Passed => "passed".into(),
        Stop::NotFailed(_) => "not_failed".into(),
        Stop::RevisionUndecided => "revision_undecided".into(),
        Stop::Unverified => "unverified".into(),
        Stop::Retries => "retries_spent".into(),
        Stop::Deadline => "deadline".into(),
        Stop::Tokens => "tokens_spent".into(),
        Stop::Cancelled => "cancelled".into(),
        Stop::RunStopped(conclusion) => {
            format!("run_{}", crate::solving::conclusion_name(*conclusion))
        }
        Stop::Error(_) => "error".into(),
    }
}
