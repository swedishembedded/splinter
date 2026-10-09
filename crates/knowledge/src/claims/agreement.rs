// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements admission gates that keep ungrounded
// claims out of training data, for its clients. If your team needs
// expertise in synthetic data quality or provenance, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Agreement between extraction passes over one session.
//!
//! A model asked twice what a person taught may say different things; a claim
//! one pass produced and another did not is as likely an accident of sampling
//! as a find. Two proposals from different passes are the same claim when they
//! cite overlapping words of the same step and state the same thing
//! ([`super::pairing::statements_alike`]): their wording may differ. The
//! claims of the first pass that every other pass also produced are kept;
//! every proposal that some pass lacked is returned as unconfirmed so that
//! nothing disappears without a trace.

use splinter_core::claim::{CitedQuote, ClaimProposal};

use super::pairing::statements_alike;

/// The outcome of comparing passes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Agreed {
    /// The first pass's proposals that every other pass produced, in order.
    pub kept: Vec<ClaimProposal>,
    /// Every proposal, of any pass, that some other pass did not produce.
    pub unconfirmed: Vec<ClaimProposal>,
}

fn overlap(a: &CitedQuote, b: &CitedQuote) -> bool {
    let (x, y) = (a.text.trim().to_lowercase(), b.text.trim().to_lowercase());
    a.step == b.step && !x.is_empty() && !y.is_empty() && (x.contains(&y) || y.contains(&x))
}

/// Whether `a` and `b`, from two passes, are one claim.
fn same_claim(a: &ClaimProposal, b: &ClaimProposal) -> bool {
    a.quotes
        .iter()
        .any(|qa| b.quotes.iter().any(|qb| overlap(qa, qb)))
        && statements_alike(&a.statement, &b.statement)
}

fn produced(pass: &[ClaimProposal], claim: &ClaimProposal) -> bool {
    pass.iter().any(|other| same_claim(claim, other))
}

/// Compares `passes`, the proposals of each in the order it made them; one
/// pass agrees with itself.
#[must_use]
pub fn agree(passes: &[Vec<ClaimProposal>]) -> Agreed {
    let mut agreed = Agreed {
        kept: Vec::new(),
        unconfirmed: Vec::new(),
    };
    for (index, pass) in passes.iter().enumerate() {
        for claim in pass {
            let everywhere = passes
                .iter()
                .enumerate()
                .all(|(other, proposals)| other == index || produced(proposals, claim));
            if !everywhere {
                agreed.unconfirmed.push(claim.clone());
            } else if index == 0 {
                agreed.kept.push(claim.clone());
            }
        }
    }
    agreed
}
