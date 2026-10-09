// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements admission gates that keep ungrounded
// claims out of training data, for its clients. If your team needs
// expertise in synthetic data quality or provenance, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The rulings kept, and the gates that need them; see the module
//! documentation of [`super`].
//!
//! A ledger is its entries in the order they were made. A claim is *live*
//! when it was admitted and no later admitted claim supersedes it; what a
//! model is trained on is the live claims. A claim said again is recorded as
//! reinforcing the live claim ([`Ruling::Reinforced`]) and adds nothing to
//! train on. Which claims are about one thing, and what a later one does to
//! an earlier one, is decided by [`super::pairing`] and, when there is one,
//! a judge.

use std::collections::BTreeMap;

use splinter_core::claim::{Claim, ClaimId, ClaimProposal, ClaimSet, LedgerEntry, Refusal, Ruling};
use splinter_core::digest::Digest;
use splinter_core::source::SourceId;

use super::gates::rule;
use super::judge::{ClaimJudge, JudgeError, PairVerdict};
use super::pairing::{same_subject, statements_agree};
use crate::session::SessionView;

/// Why a claim set could not be ruled on.
#[derive(Debug, thiserror::Error)]
pub enum GateError {
    /// The session a proposal was made for was not given.
    #[error("session {0} was not given")]
    MissingSession(SourceId),
    /// A claim cannot be addressed.
    #[error("a claim cannot be addressed: {0}")]
    Address(#[from] serde_json::Error),
    /// The judge gave no decision.
    #[error("the judge gave no decision: {0}")]
    Judge(#[from] JudgeError),
}

/// A claim set to rule on.
#[derive(Clone, Copy)]
pub struct RuleRequest<'a> {
    claim_set: &'a Digest,
    set: &'a ClaimSet,
    views: &'a BTreeMap<SourceId, SessionView>,
    judge: Option<&'a dyn ClaimJudge>,
}

impl<'a> RuleRequest<'a> {
    /// The set stored as `claim_set`, to be ruled on against `views` (the
    /// sessions it was extracted from), by code alone.
    #[must_use]
    pub fn new(
        claim_set: &'a Digest,
        set: &'a ClaimSet,
        views: &'a BTreeMap<SourceId, SessionView>,
    ) -> Self {
        Self {
            claim_set,
            set,
            views,
            judge: None,
        }
    }

    /// Has `judge` decide what a later claim does to an earlier one about the
    /// same thing.
    #[must_use]
    pub fn judged_by(mut self, judge: &'a dyn ClaimJudge) -> Self {
        self.judge = Some(judge);
        self
    }
}

/// The entries made so far, in order.
#[derive(Clone, Debug, Default)]
pub struct Ledger {
    entries: Vec<LedgerEntry>,
}

/// A live claim during a ruling.
struct Live {
    id: ClaimId,
    claim: Claim,
}

impl Ledger {
    /// The ledger of `entries`, oldest first.
    #[must_use]
    pub fn new(entries: Vec<LedgerEntry>) -> Self {
        Self { entries }
    }

    /// The entries, oldest first.
    #[must_use]
    pub fn entries(&self) -> &[LedgerEntry] {
        &self.entries
    }

    /// Whether proposal `index` of `claim_set` has been ruled on.
    #[must_use]
    pub fn decided(&self, claim_set: &Digest, index: usize) -> bool {
        self.entries
            .iter()
            .any(|e| e.index == index && e.claim_set == *claim_set)
    }

    /// Every admitted claim that is still the answer to its question, oldest
    /// first.
    pub fn live(&self) -> Result<Vec<(ClaimId, &Claim)>, serde_json::Error> {
        let mut live: Vec<(ClaimId, &Claim)> = Vec::new();
        for entry in &self.entries {
            match &entry.ruling {
                Ruling::Admitted { claim, supersedes } => {
                    live.retain(|(id, _)| !supersedes.contains(id));
                    live.push((claim.id()?, claim));
                }
                Ruling::Reinforced { supersedes, .. } => {
                    live.retain(|(id, _)| !supersedes.contains(id));
                }
                Ruling::Refused { .. } => {}
            }
        }
        Ok(live)
    }

    /// How many times the person said the fact of `claim` again.
    #[must_use]
    pub fn reinforcements(&self, claim: &ClaimId) -> usize {
        self.entries
            .iter()
            .filter(|e| matches!(&e.ruling, Ruling::Reinforced { of, .. } if of == claim))
            .count()
    }

    /// The admitted claim that replaced `claim` as the answer to its
    /// question, the latest when it has been replaced more than once.
    #[must_use]
    pub fn superseded_by(&self, claim: &ClaimId) -> Option<ClaimId> {
        self.entries.iter().rev().find_map(|e| match &e.ruling {
            Ruling::Admitted {
                claim: newer,
                supersedes,
            }
            | Ruling::Reinforced {
                claim: newer,
                supersedes,
                ..
            } if supersedes.contains(claim) => newer.id().ok(),
            _ => None,
        })
    }

    /// The rulings on the proposals of the requested set this ledger has not
    /// made, in the order they were ruled: a session's proposals in
    /// conversation order (by the first step each cites), so that a later
    /// correction in the same session supersedes an earlier one. The ledger
    /// itself is not changed: the caller keeps the entries.
    ///
    /// A session whose extraction failed has no proposals and no entries.
    /// Ruling a set again rules on nothing.
    pub fn rule_set(&self, request: &RuleRequest<'_>) -> Result<Vec<LedgerEntry>, GateError> {
        let RuleRequest {
            claim_set,
            set,
            views,
            judge,
        } = *request;
        let mut live: Vec<Live> = self
            .live()?
            .into_iter()
            .map(|(id, claim)| Live {
                id,
                claim: claim.clone(),
            })
            .collect();
        let mut made = Vec::new();
        let mut offset = 0;
        for session in &set.sessions {
            if session.proposals.is_empty() {
                continue;
            }
            let view = views
                .get(&session.session)
                .ok_or_else(|| GateError::MissingSession(session.session.clone()))?;
            let mut order: Vec<(usize, &ClaimProposal)> =
                session.proposals.iter().enumerate().collect();
            order.sort_by_key(|(i, p)| (first_cited_step(p), *i));
            for (i, proposal) in order {
                let index = offset + i;
                if self.decided(claim_set, index) {
                    continue;
                }
                let ruling = match rule(proposal, view) {
                    Ok(claim) => admit(claim, &mut live, judge)?,
                    Err(reason) => Ruling::Refused { reason },
                };
                made.push(LedgerEntry {
                    claim_set: claim_set.clone(),
                    index,
                    session: session.session.clone(),
                    proposal: proposal.clone(),
                    ruling,
                });
            }
            offset += session.proposals.len();
        }
        Ok(made)
    }
}

/// The first session step `proposal` cites.
fn first_cited_step(proposal: &ClaimProposal) -> u64 {
    proposal
        .quotes
        .iter()
        .chain(&proposal.observations)
        .map(|q| q.step)
        .chain(proposal.calls.iter().copied())
        .min()
        .unwrap_or(u64::MAX)
}

/// What `verdict` of a pair means for `claim` against the live claims.
fn decide(
    judge: Option<&dyn ClaimJudge>,
    earlier: &Claim,
    later: &Claim,
) -> Result<PairVerdict, JudgeError> {
    match judge {
        Some(judge) => judge.pair(earlier, later),
        None if statements_agree(&earlier.statement, &later.statement) => {
            Ok(PairVerdict::Reinforce)
        }
        None => Ok(PairVerdict::Supersede),
    }
}

/// `claim` against the live claims about the same thing: the same claim
/// again is refused, one that says a live claim's fact again reinforces it,
/// and one that contradicts live claims replaces them.
fn admit(
    claim: Claim,
    live: &mut Vec<Live>,
    judge: Option<&dyn ClaimJudge>,
) -> Result<Ruling, GateError> {
    let id = claim.id()?;
    if live.iter().any(|l| l.id == id) {
        return Ok(Ruling::Refused {
            reason: Refusal::Duplicate { of: id },
        });
    }
    let mut supersedes = Vec::new();
    let mut reinforces: Option<ClaimId> = None;
    for earlier in live.iter().filter(|l| same_subject(&l.claim, &claim)) {
        match decide(judge, &earlier.claim, &claim)? {
            PairVerdict::Supersede => supersedes.push(earlier.id.clone()),
            PairVerdict::Reinforce => {
                reinforces.get_or_insert_with(|| earlier.id.clone());
            }
            PairVerdict::Separate => {}
        }
    }
    live.retain(|l| !supersedes.contains(&l.id));
    if let Some(of) = reinforces {
        return Ok(Ruling::Reinforced {
            claim,
            of,
            supersedes,
        });
    }
    live.push(Live {
        id,
        claim: claim.clone(),
    });
    Ok(Ruling::Admitted { claim, supersedes })
}
