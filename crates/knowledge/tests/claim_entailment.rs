// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements admission gates that keep ungrounded
// claims out of training data, for its clients. If your team needs
// expertise in synthetic data quality or provenance, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: the gates by code check the form of a claim, not what the person
//! asserted. A quote can be verbatim and carry every term of the statement
//! while the statement turns it round ("it wasn't 12" becomes "it is 12"). A
//! judge shown only the cited words and the statement must say the words
//! assert the statement; when it does not the claim is refused as
//! `not_entailed`, and the check can be turned off.

use std::collections::BTreeMap;
use std::sync::Mutex;

use atif::{AgentProfile, StepOrigin, TraceStep, Trajectory};
use splinter_core::claim::{
    CitedQuote, Claim, ClaimKind, ClaimProposal, ClaimSet, Refusal, Ruling, SessionClaims,
};
use splinter_core::clock::FixedClock;
use splinter_core::digest::Digest;
use splinter_knowledge::capture::capture_session;
use splinter_knowledge::claims::{ClaimJudge, JudgeError, Ledger, PairVerdict, RuleRequest};
use splinter_knowledge::session::SessionView;

/// A judge that finds the words assert the statement unless they contain a
/// denial, and keeps what it was shown.
#[derive(Default)]
struct Strict {
    shown: Mutex<Vec<(Vec<String>, String)>>,
}

impl ClaimJudge for Strict {
    fn pair(&self, _: &Claim, _: &Claim) -> Result<PairVerdict, JudgeError> {
        Ok(PairVerdict::Separate)
    }

    fn entails(&self, quotes: &[&str], statement: &str) -> Result<bool, JudgeError> {
        self.shown
            .lock()
            .map_err(|_| JudgeError("poisoned".into()))?
            .push((
                quotes.iter().map(ToString::to_string).collect(),
                statement.to_string(),
            ));
        Ok(!quotes.iter().any(|q| q.contains("wasn't")))
    }
}

fn session(said: &str) -> anyhow::Result<SessionView> {
    let mut t = Trajectory::new("ATIF-v1.7", AgentProfile::new("scripted", "1"));
    t.session_id = Some("s".into());
    t.steps = vec![
        TraceStep::new(
            1,
            StepOrigin::User,
            "How many connections does the Orrin gateway take?",
        ),
        TraceStep::new(
            2,
            StepOrigin::Agent,
            "The Orrin gateway takes 12 connections.",
        ),
        TraceStep::new(3, StepOrigin::User, said),
        TraceStep::new(4, StepOrigin::Agent, "Understood."),
    ];
    let captured = capture_session(
        &serde_json::to_vec(&t)?,
        &FixedClock::new("2026-10-01T08:00:00.000Z"),
    )?;
    Ok(SessionView::of(&captured)?)
}

fn proposal(quote: &str) -> ClaimProposal {
    ClaimProposal {
        kind: ClaimKind::Fact,
        statement: "The Orrin gateway takes 12 connections.".into(),
        question: "How many connections does the Orrin gateway take?".into(),
        quotes: vec![CitedQuote {
            step: 3,
            text: quote.into(),
        }],
        observations: vec![],
        calls: vec![],
        said_wrong: None,
        subject: None,
    }
}

fn ruled(
    view: &SessionView,
    proposal: ClaimProposal,
    configure: impl for<'a> FnOnce(RuleRequest<'a>, &'a Strict) -> RuleRequest<'a>,
    judge: &Strict,
) -> anyhow::Result<Ruling> {
    let set = ClaimSet {
        extractor: "scripted".into(),
        sessions: vec![SessionClaims {
            session: view.source().clone(),
            proposals: vec![proposal],
            failure: None,
        }],
    };
    let views = BTreeMap::from([(view.source().clone(), view.clone())]);
    let digest = Digest::of(b"set");
    let request = configure(RuleRequest::new(&digest, &set, &views), judge);
    let mut made = Ledger::default().rule_set(&request)?;
    Ok(made.remove(0).ruling)
}

#[test]
fn words_that_deny_the_statement_are_refused_as_not_entailed_and_the_judge_sees_only_the_words(
) -> anyhow::Result<()> {
    let view = session("No, it wasn't 12 connections, the Orrin gateway takes 8.")?;
    let judge = Strict::default();
    let denial = proposal("it wasn't 12 connections, the Orrin gateway takes 8");
    let ruling = ruled(&view, denial, |r, j| r.judged_by(j), &judge)?;
    assert!(
        matches!(
            &ruling,
            Ruling::Refused {
                reason: Refusal::NotEntailed
            }
        ),
        "{ruling:#?}"
    );
    assert_eq!(Refusal::NotEntailed.code(), "not_entailed");
    let shown = judge
        .shown
        .lock()
        .map_err(|_| anyhow::anyhow!("poisoned"))?;
    assert_eq!(
        *shown,
        [(
            vec!["it wasn't 12 connections, the Orrin gateway takes 8".to_string()],
            "The Orrin gateway takes 12 connections.".to_string()
        )],
        "the quotes and the statement, nothing else"
    );
    Ok(())
}

#[test]
fn words_that_assert_the_statement_are_admitted_and_the_check_can_be_turned_off(
) -> anyhow::Result<()> {
    let judge = Strict::default();
    let view = session("Yes: the Orrin gateway takes 12 connections.")?;
    let ok = proposal("the Orrin gateway takes 12 connections");
    assert!(matches!(
        ruled(&view, ok, |r, j| r.judged_by(j), &judge)?,
        Ruling::Admitted { .. }
    ));

    let view = session("No, it wasn't so: the Orrin gateway takes 12 connections.")?;
    let denial = proposal("it wasn't so: the Orrin gateway takes 12 connections");
    let ruling = ruled(
        &view,
        denial.clone(),
        |r, j| r.judged_by(j).without_entailment(),
        &judge,
    )?;
    assert!(matches!(ruling, Ruling::Admitted { .. }), "switched off");
    // Without a judge nothing is asked and the gates by code decide alone.
    let none = Strict::default();
    let ruling = ruled(&view, denial, |r, _| r, &none)?;
    assert!(matches!(ruling, Ruling::Admitted { .. }));
    assert!(none
        .shown
        .lock()
        .map_err(|_| anyhow::anyhow!("poisoned"))?
        .is_empty());
    Ok(())
}
