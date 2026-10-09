// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements admission gates that keep ungrounded
// claims out of training data, for its clients. If your team needs
// expertise in synthetic data quality or provenance, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: which claims are about one thing, and what a later one does to an
//! earlier one. Claims are paired by the names and terms they share, not by
//! how a model happened to word its question; a judge, when there is one,
//! decides among supersede (the later contradicts the earlier), reinforce
//! (the same fact said again: recorded, the earlier stays live, and a
//! repeated correction is never refused) and separate (two facts). Without a
//! judge the same terms with agreeing statements reinforce and with
//! disagreeing statements supersede. Every ruling is kept.

use std::collections::BTreeMap;

use atif::{AgentProfile, StepOrigin, TraceStep, Trajectory};
use splinter_core::claim::{
    CitedQuote, ClaimKind, ClaimProposal, ClaimSet, LedgerEntry, Ruling, SessionClaims,
};
use splinter_core::clock::FixedClock;
use splinter_core::digest::Digest;
use splinter_core::source::SourceId;
use splinter_knowledge::capture::capture_session;
use splinter_knowledge::claims::{ClaimJudge, JudgeError, Ledger, PairVerdict, RuleRequest};
use splinter_knowledge::session::SessionView;

type Outcome = anyhow::Result<()>;

/// One day's session: the person asks, the agent is wrong, the person says
/// `said`.
fn day(tag: &str, said: &str) -> anyhow::Result<SessionView> {
    let mut t = Trajectory::new("ATIF-v1.7", AgentProfile::new("scripted", "1"));
    t.session_id = Some(tag.into());
    t.steps = vec![
        TraceStep::new(
            1,
            StepOrigin::User,
            "What is the limit of the Orrin gateway?",
        ),
        TraceStep::new(2, StepOrigin::Agent, "I believe it is unlimited."),
        TraceStep::new(3, StepOrigin::User, said),
        TraceStep::new(4, StepOrigin::Agent, "Understood."),
    ];
    let captured = capture_session(
        &serde_json::to_vec(&t)?,
        &FixedClock::new("2026-10-01T08:00:00.000Z"),
    )?;
    Ok(SessionView::of(&captured)?)
}

/// A claim about the Orrin gateway, in whatever words: its statement, the
/// question the extractor wrote, and the words it cites.
fn claim(statement: &str, question: &str, quote: &str) -> ClaimProposal {
    ClaimProposal {
        kind: ClaimKind::Correction,
        statement: statement.into(),
        question: question.into(),
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

/// Day one: at most 8 connections.
fn eight() -> anyhow::Result<(SessionView, ClaimProposal)> {
    Ok((
        day(
            "mon",
            "No, the Orrin gateway accepts at most 8 connections.",
        )?,
        claim(
            "The Orrin gateway accepts at most 8 connections.",
            "How many connections does the Orrin gateway accept?",
            "the Orrin gateway accepts at most 8 connections",
        ),
    ))
}

/// Day two: 12, said about the same thing in other words.
fn twelve() -> anyhow::Result<(SessionView, ClaimProposal)> {
    Ok((
        day(
            "tue",
            "Wrong again: an Orrin gateway handles up to 12 simultaneous clients.",
        )?,
        claim(
            "An Orrin gateway handles up to 12 simultaneous clients.",
            "What is the client capacity of an Orrin gateway?",
            "an Orrin gateway handles up to 12 simultaneous clients",
        ),
    ))
}

/// Day three: 8 again, in yet other words.
fn eight_again() -> anyhow::Result<(SessionView, ClaimProposal)> {
    Ok((
        day(
            "wed",
            "Listen: an Orrin gateway takes no more than 8 connections at a time.",
        )?,
        claim(
            "An Orrin gateway takes no more than 8 connections at a time.",
            "What is the largest number of connections an Orrin gateway takes?",
            "an Orrin gateway takes no more than 8 connections at a time",
        ),
    ))
}

/// A different thing about the same gateway.
fn tls() -> anyhow::Result<(SessionView, ClaimProposal)> {
    Ok((
        day(
            "thu",
            "The Orrin gateway requires TLS 1.3 for every client.",
        )?,
        claim(
            "The Orrin gateway requires TLS 1.3 for every client.",
            "Which TLS version does the Orrin gateway require?",
            "The Orrin gateway requires TLS 1.3 for every client",
        ),
    ))
}

/// Rules `proposal` of `view` against `ledger` as a set of its own, appends
/// and returns the entry made.
fn rule(
    ledger: &mut Ledger,
    (view, proposal): &(SessionView, ClaimProposal),
    judge: Option<&dyn ClaimJudge>,
) -> anyhow::Result<LedgerEntry> {
    let set = ClaimSet {
        extractor: "scripted/extractor".into(),
        passes: 1,
        sessions: vec![SessionClaims {
            session: view.source().clone(),
            proposals: vec![proposal.clone()],
            failure: None,
            unconfirmed: vec![],
        }],
    };
    let views: BTreeMap<SourceId, SessionView> =
        BTreeMap::from([(view.source().clone(), view.clone())]);
    let digest = Digest::of(format!("{:?}", view.source()).as_bytes());
    let mut request = RuleRequest::new(&digest, &set, &views);
    if let Some(judge) = judge {
        request = request.judged_by(judge);
    }
    let mut made = ledger.rule_set(&request)?;
    assert_eq!(made.len(), 1);
    let entry = made.remove(0);
    let mut entries = ledger.entries().to_vec();
    entries.push(entry.clone());
    *ledger = Ledger::new(entries);
    Ok(entry)
}

fn live_statements(ledger: &Ledger) -> anyhow::Result<Vec<String>> {
    Ok(ledger
        .live()?
        .into_iter()
        .map(|(_, c)| c.statement.clone())
        .collect())
}

/// A judge that says what it was told for the pair whose later statement
/// contains `later`, and counts how often it was asked.
struct Told {
    verdicts: Vec<(&'static str, PairVerdict)>,
    asked: std::sync::Mutex<usize>,
}

impl ClaimJudge for Told {
    fn pair(
        &self,
        _earlier: &splinter_core::claim::Claim,
        later: &splinter_core::claim::Claim,
    ) -> Result<PairVerdict, JudgeError> {
        *self
            .asked
            .lock()
            .map_err(|_| JudgeError("poisoned".into()))? += 1;
        self.verdicts
            .iter()
            .find(|(key, _)| later.statement.contains(key))
            .map(|(_, v)| *v)
            .ok_or_else(|| JudgeError("no verdict told".into()))
    }

    fn entails(&self, _quotes: &[&str], _statement: &str) -> Result<bool, JudgeError> {
        Ok(true)
    }
}

#[test]
fn a_canary_sequence_in_different_words_leaves_only_the_last_claim_live_and_keeps_every_ruling(
) -> Outcome {
    let mut ledger = Ledger::default();
    let (first, second, third) = (eight()?, twelve()?, eight_again()?);

    let one = rule(&mut ledger, &first, None)?;
    assert!(matches!(
        one.ruling,
        Ruling::Admitted { ref supersedes, .. } if supersedes.is_empty()
    ));

    // Day two contradicts day one about the same gateway, said otherwise.
    let two = rule(&mut ledger, &second, None)?;
    assert!(
        matches!(two.ruling, Ruling::Admitted { ref supersedes, .. } if supersedes.len() == 1),
        "{two:#?}"
    );
    assert_eq!(
        live_statements(&ledger)?,
        ["An Orrin gateway handles up to 12 simultaneous clients."]
    );

    // Day three says day one's fact again, in new words: it replaces day two.
    let three = rule(&mut ledger, &third, None)?;
    assert!(
        matches!(three.ruling, Ruling::Admitted { ref supersedes, .. } if supersedes.len() == 1),
        "{three:#?}"
    );
    assert_eq!(
        live_statements(&ledger)?,
        ["An Orrin gateway takes no more than 8 connections at a time."]
    );
    assert_eq!(ledger.entries().len(), 3, "every ruling is kept");
    Ok(())
}

#[test]
fn the_same_fact_said_again_is_reinforced_not_refused_and_the_earlier_claim_stays_live() -> Outcome
{
    let mut ledger = Ledger::default();
    let first = eight()?;
    let Ruling::Admitted {
        claim: original, ..
    } = rule(&mut ledger, &first, None)?.ruling
    else {
        anyhow::bail!("not admitted");
    };
    let again = (
        day("thu", "As I said: at most 8 connections per Orrin gateway.")?,
        claim(
            "At most 8 connections are accepted by an Orrin gateway.",
            "How many connections can one Orrin gateway have?",
            "at most 8 connections per Orrin gateway",
        ),
    );
    let entry = rule(&mut ledger, &again, None)?;
    match &entry.ruling {
        Ruling::Reinforced { of, supersedes, .. } => {
            assert_eq!(*of, original.id()?);
            assert!(supersedes.is_empty());
        }
        other => anyhow::bail!("{other:#?}"),
    }
    assert_eq!(
        live_statements(&ledger)?,
        std::slice::from_ref(&original.statement)
    );
    assert_eq!(ledger.reinforcements(&original.id()?), 1);
    Ok(())
}

#[test]
fn a_judge_decides_each_candidate_pair() -> Outcome {
    let judge = Told {
        verdicts: vec![
            ("12 simultaneous", PairVerdict::Supersede),
            ("TLS 1.3", PairVerdict::Separate),
            ("no more than 8", PairVerdict::Reinforce),
        ],
        asked: std::sync::Mutex::new(0),
    };
    let mut ledger = Ledger::default();
    rule(&mut ledger, &eight()?, Some(&judge))?;
    // The judge says the second supersedes the first.
    let two = rule(&mut ledger, &twelve()?, Some(&judge))?;
    assert!(matches!(two.ruling, Ruling::Admitted { ref supersedes, .. } if supersedes.len() == 1));
    // The judge says a fact about TLS is another thing: both live.
    let separate = rule(&mut ledger, &tls()?, Some(&judge))?;
    assert!(
        matches!(separate.ruling, Ruling::Admitted { ref supersedes, .. } if supersedes.is_empty())
    );
    assert_eq!(ledger.live()?.len(), 2);
    // Day three: the judge says it restates a live claim, which stays live.
    let twelve_id = ledger.live()?[0].0.clone();
    let three = rule(&mut ledger, &eight_again()?, Some(&judge))?;
    match &three.ruling {
        Ruling::Reinforced { of, .. } => assert_eq!(*of, twelve_id),
        other => anyhow::bail!("{other:#?}"),
    }
    assert_eq!(ledger.live()?.len(), 2);
    assert_eq!(
        *judge
            .asked
            .lock()
            .map_err(|_| anyhow::anyhow!("poisoned"))?,
        4
    );
    Ok(())
}

#[test]
fn claims_that_share_no_name_are_never_paired() -> Outcome {
    // Same shape of sentence, different services: nothing to pair on.
    let mut ledger = Ledger::default();
    let first = eight()?;
    rule(&mut ledger, &first, None)?;
    let other = (
        day("fri", "The Kestrel proxy accepts at most 3 connections.")?,
        claim(
            "The Kestrel proxy accepts at most 3 connections.",
            "How many connections does the Kestrel proxy accept?",
            "The Kestrel proxy accepts at most 3 connections",
        ),
    );
    let entry = rule(&mut ledger, &other, None)?;
    assert!(
        matches!(entry.ruling, Ruling::Admitted { ref supersedes, .. } if supersedes.is_empty())
    );
    assert_eq!(ledger.live()?.len(), 2);
    Ok(())
}
