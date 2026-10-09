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
//! model is trained on is the live claims. Two claims are about one
//! question when their questions are one text by the rule that keeps a task
//! set free of repeats ([`crate::tasks::dedup::repeat_of`]), and say the
//! same when their statements agree
//! ([`crate::tasks::dedup::references_agree`]).

use std::collections::BTreeMap;

use splinter_core::claim::{Claim, ClaimId, ClaimProposal, ClaimSet, LedgerEntry, Refusal, Ruling};
use splinter_core::digest::Digest;
use splinter_core::source::SourceId;

use super::gates::rule;
use crate::session::SessionView;
use crate::tasks::dedup::{references_agree, repeat_of};
use crate::tasks::{DEFAULT_MAX_OVERLAP, DEFAULT_SHINGLE_WORDS};

/// Why a claim set could not be ruled on.
#[derive(Debug, thiserror::Error)]
pub enum GateError {
    /// The session a proposal was made for was not given.
    #[error("session {0} was not given")]
    MissingSession(SourceId),
    /// A claim cannot be addressed.
    #[error("a claim cannot be addressed: {0}")]
    Address(#[from] serde_json::Error),
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
            if let Ruling::Admitted { claim, supersedes } = &entry.ruling {
                live.retain(|(id, _)| !supersedes.contains(id));
                live.push((claim.id()?, claim));
            }
        }
        Ok(live)
    }

    /// The admitted claim that replaced `claim` as the answer to its
    /// question, the latest when it has been replaced more than once.
    #[must_use]
    pub fn superseded_by(&self, claim: &ClaimId) -> Option<ClaimId> {
        self.entries.iter().rev().find_map(|e| match &e.ruling {
            Ruling::Admitted {
                claim: newer,
                supersedes,
            } if supersedes.contains(claim) => newer.id().ok(),
            _ => None,
        })
    }

    /// The rulings on the proposals of `set` this ledger has not made,
    /// against `views` (the sessions it was extracted from), in the order
    /// they were ruled: a session's proposals in conversation order (by the
    /// first step each cites), so that a later correction in the same
    /// session supersedes an earlier one. The ledger itself is not changed:
    /// the caller keeps the entries.
    ///
    /// A session whose extraction failed has no proposals and no entries.
    /// Ruling a set again rules on nothing.
    pub fn rule_set(
        &self,
        claim_set: &Digest,
        set: &ClaimSet,
        views: &BTreeMap<SourceId, SessionView>,
    ) -> Result<Vec<LedgerEntry>, GateError> {
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
                    Ok(claim) => admit(claim, &mut live)?,
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

fn same_question(a: &Claim, b: &Claim) -> bool {
    repeat_of(
        &a.question,
        &b.question,
        DEFAULT_SHINGLE_WORDS,
        DEFAULT_MAX_OVERLAP,
    )
    .is_some()
}

/// `claim` against the live claims: a repeat is refused, and a claim that
/// disagrees with live ones on its question replaces them.
fn admit(claim: Claim, live: &mut Vec<Live>) -> Result<Ruling, GateError> {
    let same: Vec<&Live> = live
        .iter()
        .filter(|l| same_question(&l.claim, &claim))
        .collect();
    if let Some(known) = same
        .iter()
        .find(|l| references_agree(&l.claim.statement, &claim.statement))
    {
        return Ok(Ruling::Refused {
            reason: Refusal::Duplicate {
                of: known.id.clone(),
            },
        });
    }
    let supersedes: Vec<ClaimId> = same.iter().map(|l| l.id.clone()).collect();
    live.retain(|l| !supersedes.contains(&l.id));
    live.push(Live {
        id: claim.id()?,
        claim: claim.clone(),
    });
    Ok(Ruling::Admitted { claim, supersedes })
}
