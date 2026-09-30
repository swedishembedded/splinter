// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements curricula that track what a learner has
// mastered, concept by concept, for its clients. If your team needs
// expertise in knowledge tracing or continual learning, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Concept mastery across releases.
//!
//! Mastery is read from the experience store, never kept beside it: every
//! experience the solve stage recorded for a policy alias names, as its
//! policy, the release the alias pointed at (or the base). Only those
//! count - a critic's retry is helped, a probe is not stored, another
//! model is not the policy - and only when their verdicts decide pass or
//! fail. Per concept (the rule of [`splinter_knowledge::concepts`]) and
//! per release, the rolling pass rate is over the newest
//! [`DEFAULT_MASTERY_WINDOW`] decided attempts; a concept with none under a
//! release has no rate there, never `0`.
//!
//! The weakest concepts are those with the lowest rate under the release
//! the policy is now; a concept not yet measured under it is counted, not
//! ranked.

use std::collections::BTreeMap;

use serde::Serialize;
use splinter_knowledge::concepts::{Concept, ConceptResolver};
use splinter_store::annotation::decide;
use splinter_store::experience::PrivilegedKind;

use crate::context::Context;
use crate::curriculum::{queue, release_of_label};
use crate::error::CampaignError;
use crate::learn::PolicyUsed;
use crate::model_ref::POLICY_DEFAULT;
use crate::release::ReleaseId;

/// Decided attempts per concept and release the rolling rate is over: the
/// newest this many.
pub const DEFAULT_MASTERY_WINDOW: usize = 32;

/// Concepts `status` lists as the weakest.
pub const DEFAULT_WEAKEST: usize = 5;

/// One concept's rolling pass rate under one release.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ReleaseMastery {
    /// The release; `None` for the base.
    pub release: Option<ReleaseId>,
    /// Decided attempts in the window.
    pub graded: usize,
    /// Of those, passes.
    pub passes: usize,
    /// `passes / graded`; `None` with none.
    pub rate: Option<f64>,
    /// When the first of all its attempts under this release was recorded.
    pub since: String,
}

/// One concept's mastery, release by release.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ConceptMastery {
    /// The concept.
    pub concept: Concept,
    /// Under the release the policy is now; `None` when not measured there.
    pub current: Option<ReleaseMastery>,
    /// Under every release it was measured under, oldest first.
    pub releases: Vec<ReleaseMastery>,
}

/// What `status` shows of concept mastery.
#[derive(Clone, Debug, Serialize)]
pub struct MasteryReport {
    /// The policy mastery is ranked under.
    pub policy: PolicyUsed,
    /// Concepts measured under any release.
    pub concepts: usize,
    /// Of those, measured under the current one.
    pub measured: usize,
    /// The weakest under the current release, weakest first.
    pub weakest: Vec<ConceptMastery>,
    /// Concepts queued for new tasks.
    pub queued: usize,
}

/// One decided attempt: when it was recorded, and whether it passed.
type Decided = (String, bool);

/// Every concept the policy's recorded solves exercise, with its mastery
/// under each release, concepts in order; `current` is the release the
/// policy is now (`None`: the base).
pub fn mastery(
    ctx: &Context,
    current: Option<&ReleaseId>,
    window: usize,
) -> Result<Vec<ConceptMastery>, CampaignError> {
    let store = ctx.experiences();
    let mut resolver = ConceptResolver::new(ctx.sources());
    let mut attempts: BTreeMap<(Concept, Option<ReleaseId>), Vec<Decided>> = BTreeMap::new();
    for id in store.list()? {
        let experience = store.get(&id)?;
        let Some(release) = experience
            .provenance
            .policy
            .as_deref()
            .and_then(release_of_label)
        else {
            continue;
        };
        if experience
            .privileged
            .iter()
            .any(|p| p.kind == PrivilegedKind::Critique)
        {
            continue;
        }
        let Some(decision) = decide(&store.annotations(&id)?.annotations) else {
            continue;
        };
        for concept in resolver.concepts(&experience.to_task())? {
            attempts
                .entry((concept, release.clone()))
                .or_default()
                .push((experience.provenance.created_at.clone(), decision.passed));
        }
    }
    let mut concepts: BTreeMap<Concept, Vec<ReleaseMastery>> = BTreeMap::new();
    for ((concept, release), mut recorded) in attempts {
        recorded.sort();
        let since = recorded
            .first()
            .map(|(at, _)| at.clone())
            .unwrap_or_default();
        let newest = &recorded[recorded.len().saturating_sub(window)..];
        let passes = newest.iter().filter(|(_, passed)| *passed).count();
        let graded = newest.len();
        concepts.entry(concept).or_default().push(ReleaseMastery {
            release,
            graded,
            passes,
            rate: (graded > 0).then(|| passes as f64 / graded as f64),
            since,
        });
    }
    Ok(concepts
        .into_iter()
        .map(|(concept, mut releases)| {
            releases.sort_by(|a, b| (&a.since, &a.release).cmp(&(&b.since, &b.release)));
            let current = releases
                .iter()
                .find(|r| r.release.as_ref() == current)
                .cloned();
            ConceptMastery {
                concept,
                current,
                releases,
            }
        })
        .collect())
}

/// The `n` weakest concepts under the release `policy:default` is now.
pub fn weakest(ctx: &Context, n: usize) -> Result<MasteryReport, CampaignError> {
    let policy = PolicyUsed {
        alias: POLICY_DEFAULT.into(),
        release: ctx.policy_pin(POLICY_DEFAULT)?.map(|pin| pin.release),
    };
    let all = mastery(ctx, policy.release.as_ref(), DEFAULT_MASTERY_WINDOW)?;
    let concepts = all.len();
    let mut ranked: Vec<ConceptMastery> = all
        .into_iter()
        .filter(|c| c.current.as_ref().is_some_and(|m| m.rate.is_some()))
        .collect();
    let measured = ranked.len();
    let rank = |c: &ConceptMastery| {
        c.current
            .as_ref()
            .and_then(|m| m.rate)
            .unwrap_or(f64::INFINITY)
    };
    ranked.sort_by(|a, b| {
        rank(a)
            .total_cmp(&rank(b))
            .then_with(|| a.concept.cmp(&b.concept))
    });
    ranked.truncate(n);
    Ok(MasteryReport {
        policy,
        concepts,
        measured,
        weakest: ranked,
        queued: queue::pending(ctx)?.len(),
    })
}
