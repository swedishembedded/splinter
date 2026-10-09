// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: the other forms a claim is learned in are admitted by the rule that
//! admitted the claim - a restatement keeps the statement's terms, nothing
//! states a term the statement and the question do not give - and counted.

use serde_json::json;
use splinter_core::claim::{Claim, ClaimKind};
use splinter_core::source::SourceId;
use splinter_knowledge::claims::forms::{admit, FormsReply};

fn claim() -> Claim {
    Claim {
        kind: ClaimKind::Fact,
        statement: "The Orrin gateway accepts at most 12 connections.".into(),
        question: "How many connections does the Orrin gateway accept?".into(),
        session: SourceId(splinter_core::digest::Digest::of(b"s")),
        quotes: Vec::<splinter_core::claim::Quote>::new(),
        observations: vec![],
        calls: vec![],
        said_wrong: None,
    }
}

fn reply(value: serde_json::Value) -> FormsReply {
    serde_json::from_value(value).unwrap_or_else(|e| panic!("{e}"))
}

#[test]
fn restatements_keep_the_terms_and_add_none() {
    let forms = admit(
        &claim(),
        reply(json!({
            "statements": [
                "At most 12 connections are accepted by the Orrin gateway.",
                "The Orrin gateway accepts at most 12 connections.",
                "The Orrin gateway accepts at most 8 connections.",
                "The gateway accepts a limited number of connections.",
                "Orrin takes 12 connections at most, over TLS 1.3."
            ],
            "reverse": [
                {"question": "Which gateway accepts at most 12 connections?",
                 "answer": "The Orrin gateway does."},
                {"question": "Which gateway has a limit of 99?", "answer": "The Orrin gateway."}
            ],
            "implications": [
                {"question": "Can the Orrin gateway take another connection when 12 are open?",
                 "answer": "No, it accepts at most 12 connections."},
                {"question": "How many connections does the Orrin gateway accept?",
                 "answer": "At most 12."}
            ]
        })),
    );
    assert_eq!(
        forms.statements,
        ["At most 12 connections are accepted by the Orrin gateway."]
    );
    assert_eq!(forms.reverse.len(), 1, "{:?}", forms.refused);
    assert_eq!(forms.implications.len(), 1);
    let reasons: Vec<&str> = forms.refused.iter().map(|r| r.reason.as_str()).collect();
    assert!(
        reasons.iter().any(|r| r.contains("a repeat")),
        "{reasons:?}"
    );
    assert!(
        reasons.iter().any(|r| r.contains("leaves out the number")),
        "{reasons:?}"
    );
    assert!(
        reasons.iter().any(|r| r.contains("adds the number")),
        "{reasons:?}"
    );
    assert!(
        reasons.iter().any(|r| r.contains("own question")),
        "{reasons:?}"
    );
}
