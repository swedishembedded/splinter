// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements admission gates that keep ungrounded
// claims out of training data, for its clients. If your team needs
// expertise in synthetic data quality or provenance, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: a claim a model proposes once may be an accident of sampling. Of
//! several extraction passes over one session, only the claims every pass
//! produced are kept, matched by the words they cite and what they state
//! rather than by their wording; the rest are listed as unconfirmed, never
//! dropped without a trace.

use splinter_core::claim::{CitedQuote, ClaimKind, ClaimProposal};
use splinter_knowledge::claims::agreement::agree;

fn claim(statement: &str, quotes: &[(u64, &str)]) -> ClaimProposal {
    ClaimProposal {
        kind: ClaimKind::Fact,
        statement: statement.into(),
        question: "Which port does the Tessera dashboard use?".into(),
        quotes: quotes
            .iter()
            .map(|(step, text)| CitedQuote {
                step: *step,
                text: (*text).into(),
            })
            .collect(),
        observations: vec![],
        calls: vec![],
        said_wrong: None,
        subject: None,
    }
}

fn statements(proposals: &[ClaimProposal]) -> Vec<&str> {
    proposals.iter().map(|p| p.statement.as_str()).collect()
}

#[test]
fn only_the_claims_every_pass_produced_are_kept_however_they_are_worded() {
    let first = vec![
        claim(
            "The Tessera dashboard listens on port 9090.",
            &[(3, "It listens on port 9090")],
        ),
        claim(
            "The Tessera dashboard has TLS enabled.",
            &[(5, "TLS is on")],
        ),
    ];
    let second = vec![
        // Same fact, other words, the quote cut differently.
        claim(
            "Port 9090 is where the Tessera dashboard listens.",
            &[(3, "port 9090")],
        ),
        // Only this pass saw this one.
        claim(
            "The Tessera dashboard logs to syslog.",
            &[(7, "logs go to syslog")],
        ),
    ];
    let agreed = agree(&[first, second]);
    assert_eq!(
        statements(&agreed.kept),
        ["The Tessera dashboard listens on port 9090."],
        "the first pass's wording is kept"
    );
    let mut unconfirmed = statements(&agreed.unconfirmed);
    unconfirmed.sort_unstable();
    assert_eq!(
        unconfirmed,
        [
            "The Tessera dashboard has TLS enabled.",
            "The Tessera dashboard logs to syslog."
        ],
        "each one-pass claim is listed"
    );
}

#[test]
fn the_same_quote_stating_another_number_is_not_agreement() {
    let first = vec![claim(
        "The Tessera dashboard listens on port 9090.",
        &[(3, "It listens on port 9090 or 9091")],
    )];
    let second = vec![claim(
        "The Tessera dashboard listens on port 9091.",
        &[(3, "It listens on port 9090 or 9091")],
    )];
    let agreed = agree(&[first, second]);
    assert!(agreed.kept.is_empty());
    assert_eq!(agreed.unconfirmed.len(), 2);
}

#[test]
fn one_pass_keeps_everything() {
    let only = vec![claim(
        "The Tessera dashboard has TLS enabled.",
        &[(5, "TLS is on")],
    )];
    let agreed = agree(std::slice::from_ref(&only));
    assert_eq!(agreed.kept, only);
    assert!(agreed.unconfirmed.is_empty());
}
