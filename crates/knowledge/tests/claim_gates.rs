// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements admission gates that keep ungrounded
// claims out of training data, for its clients. If your team needs
// expertise in synthetic data quality or provenance, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: a claim is admitted only on the person's own words. Every quote is
//! verbatim in the user step it cites; every number, name and quoted term of
//! the statement is in the cited words; the agent's sentences are never
//! evidence; a repeat collapses into the claim that already says it; a later
//! claim on the same question supersedes the earlier, and both stay in the
//! ledger; every refusal is kept with its reason.

use std::collections::BTreeMap;

use atif::{
    AgentProfile, ObservationEntry, StepObservation, StepOrigin, ToolInvocation, TraceStep,
    Trajectory,
};
use serde_json::json;
use splinter_core::claim::{
    CitedQuote, ClaimKind, ClaimProposal, ClaimSet, LedgerEntry, Refusal, Ruling, SessionClaims,
};
use splinter_core::clock::FixedClock;
use splinter_core::digest::Digest;
use splinter_core::source::SourceId;
use splinter_knowledge::capture::capture_session;
use splinter_knowledge::claims::{rule, Ledger};
use splinter_knowledge::session::SessionView;

const WHERE: &str = "Which port does the Tessera dashboard listen on?";
const WRONG: &str = "The Tessera dashboard listens on port 8080.";
const CORRECTION: &str = "No, that is wrong. It listens on port 9090 since the March move.";

fn session(steps: Vec<TraceStep>, id: &str) -> anyhow::Result<SessionView> {
    let mut t = Trajectory::new("ATIF-v1.7", AgentProfile::new("scripted", "1"));
    t.session_id = Some(id.into());
    t.steps = steps;
    let captured = capture_session(
        &serde_json::to_vec(&t)?,
        &FixedClock::new("2026-10-01T08:00:00.000Z"),
    )?;
    Ok(SessionView::of(&captured)?)
}

fn step(id: u64, who: StepOrigin, text: &str) -> TraceStep {
    TraceStep::new(id, who, text)
}

/// The user asks, the agent answers wrongly, the user corrects, the agent
/// acknowledges; later the user states a fact of their own.
fn correction_session(id: &str) -> anyhow::Result<SessionView> {
    session(
        vec![
            step(1, StepOrigin::System, "You are a helpful agent."),
            step(2, StepOrigin::User, WHERE),
            step(3, StepOrigin::Agent, WRONG),
            step(4, StepOrigin::User, CORRECTION),
            step(
                5,
                StepOrigin::Agent,
                "Thanks for the correction: port 9090.",
            ),
            step(
                6,
                StepOrigin::User,
                "Also, our on-call rota rotates every 2 weeks.",
            ),
            step(7, StepOrigin::Agent, "Noted."),
        ],
        id,
    )
}

fn quote(step: u64, text: &str) -> CitedQuote {
    CitedQuote {
        step,
        text: text.into(),
    }
}

fn correction() -> ClaimProposal {
    ClaimProposal {
        kind: ClaimKind::Correction,
        statement: "The Tessera dashboard listens on port 9090.".into(),
        question: WHERE.into(),
        quotes: vec![
            quote(2, "the Tessera dashboard"),
            quote(4, "It listens on port 9090"),
        ],
        observations: vec![],
        calls: vec![],
        said_wrong: Some("listens on port 8080".into()),
    }
}

fn refused(view: &SessionView, proposal: &ClaimProposal) -> Refusal {
    match rule(proposal, view) {
        Ok(claim) => panic!("admitted: {claim:?}"),
        Err(reason) => reason,
    }
}

#[test]
fn a_correction_on_the_persons_words_is_admitted_with_spans_into_the_session() -> anyhow::Result<()>
{
    let view = correction_session("s1")?;
    let claim = rule(&correction(), &view)?;
    assert_eq!(claim.session, *view.source());
    assert_eq!(claim.said_wrong.as_deref(), Some("listens on port 8080"));
    let step4 = view
        .step(4)
        .and_then(|s| s.message.as_ref())
        .ok_or_else(|| anyhow::anyhow!("no step 4"))?;
    let q = &claim.quotes[1];
    let span = &q.span;
    assert_eq!(span.source, step4.content);
    assert_eq!(
        span.part.as_ref().map(|p| p.name.as_str()),
        Some(step4.name.as_str())
    );
    assert_eq!(
        &step4.text[span.start as usize..span.end as usize],
        "It listens on port 9090"
    );
    Ok(())
}

#[test]
fn a_quote_that_is_not_verbatim_in_the_cited_user_step_is_refused() -> anyhow::Result<()> {
    let view = correction_session("s1")?;
    let mut p = correction();
    p.quotes[1] = quote(4, "It listens on 9090");
    assert!(matches!(
        refused(&view, &p),
        Refusal::QuoteNotVerbatim { step: 4, .. }
    ));
    // Right words, wrong step.
    p.quotes[1] = quote(6, "It listens on port 9090");
    assert!(matches!(
        refused(&view, &p),
        Refusal::QuoteNotVerbatim { step: 6, .. }
    ));
    p.quotes[1] = quote(40, "It listens on port 9090");
    assert!(matches!(
        refused(&view, &p),
        Refusal::UnknownStep { step: 40 }
    ));
    Ok(())
}

#[test]
fn an_assistant_sentence_is_never_evidence() -> anyhow::Result<()> {
    let view = correction_session("s1")?;
    // The agent's wrong answer, cited as if it were the fact.
    let mut p = correction();
    p.statement = "The Tessera dashboard listens on port 8080.".into();
    p.quotes = vec![quote(3, "The Tessera dashboard listens on port 8080")];
    assert!(matches!(
        refused(&view, &p),
        Refusal::AssistantEvidence { step: 3 }
    ));
    // The agent's acknowledgement of the correction.
    p.statement = "The Tessera dashboard listens on port 9090.".into();
    p.quotes = vec![quote(5, "Thanks for the correction: port 9090")];
    assert!(matches!(
        refused(&view, &p),
        Refusal::AssistantEvidence { step: 5 }
    ));
    // Only an agent quote beside no user quote at all.
    p.quotes = vec![];
    assert!(matches!(refused(&view, &p), Refusal::NoUserEvidence));
    Ok(())
}

#[test]
fn a_statement_term_absent_from_the_cited_words_is_refused() -> anyhow::Result<()> {
    let view = correction_session("s1")?;
    let cases = [
        (
            "The Tessera dashboard listens on port 9091.",
            "number",
            "9091",
        ),
        (
            "The Tessera dashboard listens on port 9090, not 8080.",
            "number",
            "8080",
        ),
        (
            "The Zelkor dashboard listens on port 9090.",
            "name",
            "Zelkor",
        ),
        (
            "The Tessera dashboard listens on port 9090 since 2026-03-04.",
            "number",
            "2026",
        ),
        (
            "The Tessera dashboard listens on port 9090 behind \"edge-proxy\".",
            "quoted_term",
            "edge-proxy",
        ),
        (
            "The Tessera dashboard listens on port 9090 as /srv/tessera.sock.",
            "name",
            "/srv/tessera.sock",
        ),
    ];
    for (statement, kind, term) in cases {
        let mut p = correction();
        p.statement = statement.into();
        match refused(&view, &p) {
            Refusal::UnsupportedTerm {
                term_kind,
                term: found,
            } => {
                assert_eq!(
                    (term_kind.as_str(), found.as_str()),
                    (kind, term),
                    "{statement}"
                );
            }
            other => panic!("{statement}: {other:?}"),
        }
    }
    // What the cited words do carry is not in question: the month is the
    // user's, and a statement may leave words out.
    let mut p = correction();
    p.statement = "Since the March move the Tessera dashboard listens on port 9090.".into();
    p.quotes.push(quote(4, "since the March move"));
    rule(&p, &view)?;
    Ok(())
}

#[test]
fn what_the_agent_got_wrong_must_have_been_said() -> anyhow::Result<()> {
    let view = correction_session("s1")?;
    let mut p = correction();
    p.said_wrong = Some("listens on port 7070".into());
    assert!(matches!(refused(&view, &p), Refusal::WrongAnswerNotSaid));
    Ok(())
}

fn procedure_session() -> anyhow::Result<SessionView> {
    let mut call = step(3, StepOrigin::Agent, "");
    call.tool_calls = Some(vec![ToolInvocation::new("c1", "shell")
        .with_arguments(json!({"cmd": "brindle deploy --env staging"}))]);
    call.observation = Some(StepObservation::single(ObservationEntry::for_call(
        "c1",
        "deployed brindle 1.4.2 to staging",
    )));
    session(
        vec![
            step(1, StepOrigin::System, "You are a helpful agent."),
            step(
                2,
                StepOrigin::User,
                "Deploy the Brindle service to staging.",
            ),
            call,
            step(4, StepOrigin::Agent, "Deployed."),
        ],
        "proc",
    )
}

fn procedure() -> ClaimProposal {
    ClaimProposal {
        kind: ClaimKind::Procedure,
        statement: "To deploy the Brindle service to staging, run `brindle deploy --env staging`; it reports version 1.4.2.".into(),
        question: "How do I deploy the Brindle service to staging?".into(),
        quotes: vec![quote(2, "Deploy the Brindle service to staging")],
        observations: vec![quote(3, "deployed brindle 1.4.2")],
        calls: vec![3],
        said_wrong: None,
    }
}

#[test]
fn a_tool_call_procedure_needs_its_calls_and_the_observation_that_proved_it() -> anyhow::Result<()>
{
    let view = procedure_session()?;
    let claim = rule(&procedure(), &view)?;
    assert_eq!(claim.calls, vec![3]);
    assert_eq!(claim.observations.len(), 1);

    let mut p = procedure();
    p.observations.clear();
    assert!(matches!(refused(&view, &p), Refusal::Procedure { .. }));
    let mut p = procedure();
    p.calls.clear();
    assert!(matches!(refused(&view, &p), Refusal::Procedure { .. }));
    // A flag the calls never used.
    let mut p = procedure();
    p.statement = p.statement.replace("--env staging", "--env prod --force");
    assert!(matches!(
        refused(&view, &p),
        Refusal::UnsupportedTerm { .. }
    ));
    // An observation that was never returned.
    let mut p = procedure();
    p.observations = vec![quote(3, "deployed brindle 1.5.0")];
    assert!(matches!(
        refused(&view, &p),
        Refusal::QuoteNotVerbatim { step: 3, .. }
    ));
    // Calls belong to procedures.
    let mut p = correction();
    p.calls = vec![3];
    assert!(matches!(
        refused(&correction_session("s1")?, &p),
        Refusal::Procedure { .. }
    ));
    Ok(())
}

fn set_of(sessions: Vec<(&SessionView, Vec<ClaimProposal>)>, tag: &str) -> ClaimSet {
    ClaimSet {
        extractor: format!("scripted/{tag}"),
        sessions: sessions
            .into_iter()
            .map(|(v, proposals)| SessionClaims {
                session: v.source().clone(),
                proposals,
                failure: None,
            })
            .collect(),
    }
}

fn views(list: &[&SessionView]) -> BTreeMap<SourceId, SessionView> {
    list.iter()
        .map(|v| (v.source().clone(), (*v).clone()))
        .collect()
}

fn id(tag: &str) -> Digest {
    Digest::of(tag.as_bytes())
}

fn rulings(entries: &[LedgerEntry]) -> Vec<&str> {
    entries
        .iter()
        .map(|e| match &e.ruling {
            Ruling::Admitted { .. } => "admitted",
            Ruling::Refused { reason } => reason.code(),
        })
        .collect()
}

#[test]
fn a_realistic_correction_session_yields_its_claims_and_reports_every_refusal() -> anyhow::Result<()>
{
    let view = correction_session("s1")?;
    let fact = ClaimProposal {
        kind: ClaimKind::Fact,
        statement: "The on-call rota rotates every 2 weeks.".into(),
        question: "How often does the on-call rota rotate?".into(),
        quotes: vec![quote(6, "our on-call rota rotates every 2 weeks")],
        observations: vec![],
        calls: vec![],
        said_wrong: None,
    };
    let mut wrong = correction();
    wrong.statement = "The Tessera dashboard listens on port 8080.".into();
    wrong.quotes = vec![quote(3, "The Tessera dashboard listens on port 8080")];
    let mut acknowledged = correction();
    acknowledged.quotes = vec![quote(5, "Thanks for the correction: port 9090")];
    let proposals = vec![correction(), wrong, acknowledged, correction(), fact];
    let set = set_of(vec![(&view, proposals.clone())], "a");

    let entries = Ledger::default().rule_set(&id("a"), &set, &views(&[&view]))?;
    assert_eq!(
        rulings(&entries),
        [
            "admitted",
            "duplicate",
            "assistant_evidence",
            "assistant_evidence",
            "admitted"
        ],
        "ruled in conversation order: {entries:#?}"
    );
    let mut positions: Vec<usize> = entries.iter().map(|e| e.index).collect();
    positions.sort_unstable();
    assert_eq!(positions, [0, 1, 2, 3, 4]);
    assert_eq!(
        entries.len(),
        proposals.len(),
        "nothing is silently dropped"
    );
    // Entries keep the proposal as made, and where it came from.
    assert!(entries
        .iter()
        .all(|e| e.session == *view.source() && e.claim_set == id("a")));
    let Ruling::Admitted { claim, .. } = &entries[0].ruling else {
        anyhow::bail!("first claim not admitted");
    };
    assert!(
        matches!(&entries[1].ruling, Ruling::Refused { reason: Refusal::Duplicate { of } } if *of == claim.id()?)
    );
    assert_eq!(entries[1].index, 3, "the repeat is the fourth proposal");
    Ok(())
}

#[test]
fn a_later_claim_on_the_same_question_supersedes_the_earlier_and_both_stay() -> anyhow::Result<()> {
    let first = session(
        vec![
            step(1, StepOrigin::User, WHERE),
            step(2, StepOrigin::Agent, "I do not know."),
            step(
                3,
                StepOrigin::User,
                "The Tessera dashboard listens on port 8080.",
            ),
            step(4, StepOrigin::Agent, "Noted."),
        ],
        "mon",
    )?;
    let second = correction_session("tue")?;
    let old = ClaimProposal {
        kind: ClaimKind::Fact,
        statement: "The Tessera dashboard listens on port 8080.".into(),
        question: WHERE.into(),
        quotes: vec![quote(3, "The Tessera dashboard listens on port 8080")],
        observations: vec![],
        calls: vec![],
        said_wrong: None,
    };
    let all = views(&[&first, &second]);

    let mut entries =
        Ledger::default().rule_set(&id("mon"), &set_of(vec![(&first, vec![old])], "mon"), &all)?;
    assert_eq!(rulings(&entries), ["admitted"]);
    let Ruling::Admitted {
        claim: older,
        supersedes,
    } = &entries[0].ruling
    else {
        anyhow::bail!("not admitted");
    };
    assert!(supersedes.is_empty());
    let older_id = older.id()?;

    let ledger = Ledger::new(entries.clone());
    assert_eq!(ledger.live()?.len(), 1);
    let tue = ledger.rule_set(
        &id("tue"),
        &set_of(vec![(&second, vec![correction()])], "tue"),
        &all,
    )?;
    let Ruling::Admitted {
        claim: newer,
        supersedes,
    } = &tue[0].ruling
    else {
        anyhow::bail!("{tue:#?}");
    };
    assert_eq!(supersedes, &vec![older_id.clone()]);
    entries.extend(tue.clone());

    let ledger = Ledger::new(entries);
    let live: Vec<_> = ledger.live()?.iter().map(|(id, _)| id.clone()).collect();
    assert_eq!(live, vec![newer.id()?], "only the later claim is live");
    assert_eq!(
        ledger.superseded_by(&older_id),
        Some(newer.id()?),
        "the earlier stays, with the relation"
    );
    Ok(())
}

#[test]
fn ruling_a_set_again_adds_nothing() -> anyhow::Result<()> {
    let view = correction_session("s1")?;
    let set = set_of(vec![(&view, vec![correction()])], "a");
    let all = views(&[&view]);
    let entries = Ledger::default().rule_set(&id("a"), &set, &all)?;
    let ledger = Ledger::new(entries);
    assert!(ledger.rule_set(&id("a"), &set, &all)?.is_empty());
    Ok(())
}
