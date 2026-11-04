// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements admission gates that keep ungrounded
// claims out of training data, for its clients. If your team needs
// expertise in synthetic data quality or provenance, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: a live claim is a closed-book task whose question is the claim's,
//! whose reference is its statement, and whose evidence is the spans of the
//! session the person's words are in - resolved to those exact bytes. A
//! teacher is shown the claim; a student is not. An answer is graded by the
//! terms of the statement it carries and the terms it adds.

use atif::{AgentProfile, StepOrigin, TraceStep, Trajectory};
use splinter_core::claim::{CitedQuote, Claim, ClaimKind, ClaimProposal};
use splinter_core::clock::FixedClock;
use splinter_core::experience::{Experience, PrivilegedKind, Provenance, Task};
use splinter_eval::verifiers::Verifier;
use splinter_knowledge::capture::capture_session;
use splinter_knowledge::claims::answer::ClaimTermsVerifier;
use splinter_knowledge::claims::rule;
use splinter_knowledge::claims::task::{claim_task, subject_of};
use splinter_knowledge::session::SessionView;
use splinter_knowledge::tasks::{Catalogue, TAUGHT};
use splinter_store::sources::SourceStore;
use splinter_store::workspace::Workspace;
use splinter_store::StateRoot;

const WHERE: &str = "Which port does the Tessera dashboard listen on?";
const STATEMENT: &str = "The Tessera dashboard listens on port 9090.";

struct Stored {
    _dir: tempfile::TempDir,
    store: SourceStore,
    claim: Claim,
}

fn stored() -> anyhow::Result<Stored> {
    let mut t = Trajectory::new("ATIF-v1.7", AgentProfile::new("scripted", "1"));
    t.session_id = Some("monday".into());
    t.steps = vec![
        TraceStep::new(1, StepOrigin::User, WHERE),
        TraceStep::new(
            2,
            StepOrigin::Agent,
            "The Tessera dashboard listens on port 8080.",
        ),
        TraceStep::new(
            3,
            StepOrigin::User,
            "No, that is wrong. It listens on port 9090 since the March move.",
        ),
        TraceStep::new(4, StepOrigin::Agent, "Port 9090, understood."),
    ];
    let captured = capture_session(
        &serde_json::to_vec(&t)?,
        &FixedClock::new("2026-10-01T08:00:00.000Z"),
    )?;
    let dir = tempfile::tempdir()?;
    let store = SourceStore::new(&Workspace::at(&StateRoot::new(dir.path().join("state"))));
    store.put_source(&captured)?;
    let view = SessionView::of(&captured)?;
    let proposal = ClaimProposal {
        kind: ClaimKind::Correction,
        statement: STATEMENT.into(),
        question: WHERE.into(),
        quotes: vec![
            CitedQuote {
                step: 1,
                text: "the Tessera dashboard".into(),
            },
            CitedQuote {
                step: 3,
                text: "It listens on port 9090".into(),
            },
        ],
        observations: vec![],
        calls: vec![],
        said_wrong: Some("listens on port 8080".into()),
        subject: None,
    };
    let claim = rule(&proposal, &view).map_err(|r| anyhow::anyhow!("{r}"))?;
    Ok(Stored {
        _dir: dir,
        store,
        claim,
    })
}

fn answered(task: &Task, answer: &str) -> anyhow::Result<Experience> {
    Ok(Experience::answered_without_a_run(
        task.clone(),
        answer,
        Provenance::new("scripted", &FixedClock::new("2026-10-01T08:00:00.000Z")),
    )?)
}

fn passes(task: &Task, answer: &str) -> anyhow::Result<Option<bool>> {
    let finding = ClaimTermsVerifier::new().verify(task, &answered(task, answer)?)?;
    Ok(match finding.outcome {
        splinter_core::annotation::Outcome::Pass => Some(true),
        splinter_core::annotation::Outcome::Fail => Some(false),
        splinter_core::annotation::Outcome::Abstain => None,
    })
}

#[test]
fn a_claim_is_a_closed_book_task_grounded_in_the_session_words() -> anyhow::Result<()> {
    let s = stored()?;
    let made = claim_task(&s.store, &s.claim)?;
    let task = &made.task;
    assert_eq!(task.task.kind, TAUGHT);
    assert_eq!(task.instruction, WHERE);
    assert_eq!(task.environment.kind, "closed-book");
    let reference = task
        .privileged
        .iter()
        .find(|p| p.kind == PrivilegedKind::Reference)
        .map(|p| p.content.as_str());
    assert_eq!(reference, Some(STATEMENT));

    // The evidence is the quotes' spans, and they resolve to the person's
    // own words in the session parts.
    assert_eq!(task.evidence.len(), 2);
    let read = |n: usize| -> anyhow::Result<String> {
        Ok(String::from_utf8(s.store.read_span(&task.evidence[n])?)?)
    };
    assert_eq!(read(0)?, "the Tessera dashboard");
    assert_eq!(read(1)?, "It listens on port 9090");
    let part = task.evidence[1]
        .part
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("a span names its part"))?;
    assert_eq!(part.source, s.claim.session);
    assert_eq!(part.name, "step-000003-user");

    // A teacher is shown the claim; the student's question is the question.
    assert!(task
        .privileged
        .iter()
        .any(|p| p.kind == PrivilegedKind::Passage && p.content == STATEMENT));
    assert_eq!(made.subject.as_deref(), Some("Tessera"));
    // The same claim is the same task.
    assert_eq!(claim_task(&s.store, &s.claim)?.task, *task);
    Ok(())
}

#[test]
fn a_span_that_no_longer_holds_the_quote_refuses_the_task() -> anyhow::Result<()> {
    let s = stored()?;
    let mut claim = s.claim.clone();
    claim.quotes[1].text = "It listens on port 9091".into();
    let error = claim_task(&s.store, &claim).unwrap_err();
    assert!(error.to_string().contains("step 3"), "{error}");
    Ok(())
}

#[test]
fn a_question_that_quotes_the_statement_does_not_stand_on_its_own() -> anyhow::Result<()> {
    let s = stored()?;
    let mut claim = s.claim.clone();
    claim.statement = "The Tessera dashboard listens on port 9090 behind the proxy.".into();
    claim.question =
        "Is it true that the Tessera dashboard listens on port 9090 behind the proxy?".into();
    let error = claim_task(&s.store, &claim).unwrap_err();
    assert!(error.to_string().contains("stand on its own"), "{error}");
    Ok(())
}

#[test]
fn the_subject_is_what_the_question_and_the_statement_both_name() {
    assert_eq!(subject_of(WHERE, STATEMENT).as_deref(), Some("Tessera"));
    assert_eq!(
        subject_of(
            "What is my favourite editor?",
            "The user's favourite editor is helix."
        )
        .as_deref(),
        Some("favourite")
    );
    assert_eq!(subject_of("What now?", STATEMENT), None);
}

#[test]
fn an_answer_is_graded_by_the_terms_it_carries_and_the_terms_it_adds() -> anyhow::Result<()> {
    let s = stored()?;
    let task = claim_task(&s.store, &s.claim)?.task;
    // Worded its own way, with the statement's number and name.
    assert_eq!(passes(&task, "Tessera listens on 9090.")?, Some(true));
    // The old port is a number the task never gave.
    assert_eq!(
        passes(&task, "The Tessera dashboard listens on 8080.")?,
        Some(false)
    );
    // The statement's number is missing.
    assert_eq!(
        passes(&task, "The Tessera dashboard has a port.")?,
        Some(false)
    );
    // Right, and an invented number beside it.
    assert_eq!(
        passes(&task, "Tessera listens on 9090 and uses 443 for TLS.")?,
        Some(false)
    );
    Ok(())
}

#[test]
fn a_statement_with_no_term_to_hold_an_answer_to_is_left_to_a_judge() -> anyhow::Result<()> {
    let s = stored()?;
    let mut claim = s.claim.clone();
    claim.statement = "it listens somewhere else now".into();
    let mut task = claim_task(&s.store, &claim)?.task;
    task = Task::new(
        task.task.kind.clone(),
        task.evidence.clone(),
        task.environment.clone(),
        task.instruction.clone(),
        task.privileged.clone(),
    )?;
    assert_eq!(passes(&task, "Somewhere else.")?, None);
    // Judged covers what the terms cannot decide.
    let kind = Catalogue::builtin()
        .get(TAUGHT)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("taught is a kind"))?;
    assert!(
        !kind.needs_judge(),
        "terms establish a pass where they apply"
    );
    Ok(())
}
