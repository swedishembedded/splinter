// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reinforcement learning from the verified
// results of agents working in a repository, for its clients. If your team
// needs expertise in training code-editing agents against hidden tests, you
// can procure our services by sending an email to info@swedishembedded.com.

//! The policy-gradient and self-distill objectives of `practice`: the dataset
//! the solve stage's graded attempts make, and what the run reports of the
//! groups they formed.
//!
//! The canonical dataset build ([`crate::datasets::build`], the
//! policy-gradient view) and the canonical trainer ([`crate::train`]) do the
//! work; this module only decides which attempts reach the view, refuses the
//! requests the objective cannot honour, and counts what was left out.

use serde::Serialize;
use splinter_core::training::Regime;
use splinter_store::experiences::{ExperienceSet, SetId};

use super::{PracticeObjective, PracticeRequest};
use crate::curriculum::frontier::DEFAULT_SAMPLING;
use crate::datasets::{Built, GroupKind};
use crate::rl_build::RlBuild;
use crate::train::{check_policy_gradient_settings, Tuning};
use crate::verify::exposure_failure;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// What the dataset stage of a rollout objective made of the solve stage's
/// attempts.
#[derive(Clone, Debug, Serialize)]
pub struct Rollouts {
    /// The objective.
    pub objective: PracticeObjective,
    /// The attempts handed to the view: every graded attempt of the solve
    /// stage, the frontier's selection of tasks not applied (a task every
    /// attempt passed or failed is a group of its own kind, not a dropped
    /// one).
    pub attempts: usize,
    /// The attempts the exposure audit failed, left out whole: their
    /// conversations quote what only the grader holds, so they are neither
    /// rewarded nor punished.
    pub exposed: usize,
    /// The attempts the solve stage could not complete (the model server or
    /// the environment failed): they left no experience, so they are in no
    /// group.
    pub not_attempted: usize,
    /// The groups the view compared, by kind; absent when no dataset was
    /// made.
    pub groups: Option<GroupCounts>,
}

/// The groups of a dataset by what they can teach.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct GroupCounts {
    /// Attempts earned different rewards: GRPO's advantage applies.
    pub graded: usize,
    /// Every attempt passed: nothing to compare.
    pub all_pass: usize,
    /// Every attempt failed: no advantage; distilled under self-distill.
    pub all_fail: usize,
    /// Of all of those, the groups whose attempts were distilled with a
    /// teacher context.
    pub distilled: usize,
}

impl GroupCounts {
    /// The groups `built` compared, by kind.
    pub(super) fn of(built: &Built) -> Self {
        let mut counts = Self::default();
        for group in &built.groups {
            match group.kind {
                GroupKind::Graded => counts.graded += 1,
                GroupKind::AllPass => counts.all_pass += 1,
                GroupKind::AllFail => counts.all_fail += 1,
            }
            counts.distilled += usize::from(group.distilled);
        }
        counts
    }
}

/// Refuses, before anything is recorded or solved, the requests whose
/// objective cannot honour what they name: no setting is dropped silently.
pub(super) fn validate(request: &PracticeRequest) -> Result<(), OrchestratorError> {
    let refused = |why: String| Err(OrchestratorError::Refused(why));
    if !request.objective.learns_from_groups() {
        let named = request.tuning.policy_gradient.named();
        return if named.is_empty() {
            Ok(())
        } else {
            refused(format!(
                "{} set the policy-gradient objective: name --objective policy-gradient or \
                 self-distill, or drop them",
                named.join(", ")
            ))
        };
    }
    let objective = request.objective;
    if request.teacher.is_some() {
        return refused(format!(
            "--objective {objective} learns from the policy's own attempts; a --teacher's \
             demonstrations are imitated, which only sft-step does"
        ));
    }
    if request.corrections.is_some() {
        return refused(format!(
            "--objective {objective} learns from the policy's own attempts; a --correction \
             teacher's continuations are imitated, which only sft-step does"
        ));
    }
    if request.rebuild {
        return refused(format!(
            "--rebuild trains a fresh adapter from the experience of every release; \
             --objective {objective} continues the champion's adapter, whose rollouts these \
             are"
        ));
    }
    // The run holds nothing out and monitors nothing: its settings are
    // refused here, not after the solve stage has spent the device.
    check_policy_gradient_settings(Regime::PolicyGradient, &request.tuning)
}

/// The temperature the solve stage samples at: what the policy-gradient
/// objective's importance ratios must be taken at.
fn rollout_temperature(request: &PracticeRequest) -> f32 {
    request
        .pass_at_k
        .sampling
        .unwrap_or(DEFAULT_SAMPLING)
        .temperature
}

/// How the train stage tunes the run: the request's tuning, and for a
/// rollout objective the temperature the attempts were sampled at unless a
/// command named one (brain's default would be a guess about rollouts it did
/// not see made).
pub(super) fn tuning(request: &PracticeRequest) -> Tuning {
    let mut tuning = Tuning {
        restart: request.rebuild,
        ..request.tuning
    };
    if request.objective.learns_from_groups() && tuning.policy_gradient.temperature.is_none() {
        tuning.policy_gradient.temperature = Some(rollout_temperature(request));
    }
    tuning
}

/// How the dataset build rewards the attempts and reads their groups.
pub(super) fn rl_build(objective: PracticeObjective) -> RlBuild {
    RlBuild {
        cliff_distill: objective.distils_cliffs(),
        ..RlBuild::default()
    }
}

/// The attempts of `solved` the exposure audit did not fail, and how many it
/// did.
pub(super) fn unexposed(
    ctx: &Context,
    solved: &SetId,
) -> Result<(SetId, usize, usize), OrchestratorError> {
    let store = ctx.experiences();
    let members = store.get_set(solved)?.members;
    let total = members.len();
    let mut kept = Vec::with_capacity(total);
    for id in members {
        let exposed = store
            .annotations(&id)?
            .annotations
            .iter()
            .any(|note| exposure_failure(note).is_some());
        if !exposed {
            kept.push(id);
        }
    }
    let exposed = total - kept.len();
    let set = store.put_set(&ExperienceSet {
        name: format!("{solved} without the attempts the exposure audit failed"),
        members: kept,
    })?;
    Ok((set, total, exposed))
}
