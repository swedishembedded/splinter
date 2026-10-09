// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Specifications of [`super`].

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use splinter_sdk::agent::sven::atif::TraceStep;
use splinter_sdk::agent::sven::model::{
    CompletionRequest, ModelProvider, ResponseEvent, ResponseStream,
};
use splinter_sdk::vocabulary::clock::FixedClock;

use super::*;
use crate::keys::extract;

/// A policy that gives the next reply of its script.
struct Scripted(Mutex<VecDeque<String>>);

#[async_trait::async_trait]
impl ModelProvider for Scripted {
    fn name(&self) -> &str {
        "scripted"
    }
    fn model_name(&self) -> &str {
        "policy-1"
    }
    async fn complete(&self, _: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let reply = self
            .0
            .lock()
            .expect("lock")
            .pop_front()
            .expect("a scripted reply");
        Ok(Box::pin(futures::stream::iter(vec![
            Ok(ResponseEvent::TextDelta(reply)),
            Ok(ResponseEvent::Done),
        ])))
    }
}

/// A user that says its script, one message per turn.
struct ScriptedUser(Vec<&'static str>);

#[async_trait::async_trait]
impl Interlocutor for ScriptedUser {
    async fn next(&self, so_far: &[Exchange]) -> Option<String> {
        self.0.get(so_far.len()).map(|m| (*m).to_string())
    }
}

const STATEMENT: &str = "Jefferson bought a copying press made by Boulton and Watt in 1785.";
const QUESTION: &str = "Which firm made the copying press Jefferson bought?";

#[tokio::test]
async fn a_session_is_a_valid_atif_file_in_which_a_wrong_answer_is_corrected_in_the_users_words() {
    let policy = Model::new(
        Arc::new(Scripted(Mutex::new(
            vec![
                "I bought my press from a London firm called Peale.".to_string(),
                "Then I stand corrected: it was Boulton and Watt.".to_string(),
            ]
            .into(),
        ))),
        "policy-1",
    );
    let user = ScriptedUser(vec![
        "Who built the copier you had in Paris?",
        "No, that is not right, it was the Birmingham partners, Boulton and Watt, around 1785.",
    ]);
    let trajectory = converse(&policy, "Who built the copier you had in Paris?", &user, 2)
        .await
        .expect("a session");

    let dir = tempfile::tempdir().expect("dir");
    let clock = FixedClock::new("2026-10-09T08:15:30.123Z");
    let path = save(dir.path(), &clock, &trajectory).expect("saved");
    assert_eq!(
        path.file_name().and_then(|n| n.to_str()),
        Some("2026-10-09_08-15-30.123.atif.json")
    );

    let keys = extract(STATEMENT, QUESTION);
    let Assessment::Kept(summary) = assess(&trajectory, &keys).expect("assessed") else {
        panic!("a corrected session is kept");
    };
    assert_eq!(summary.user_turns, 2);
    assert_eq!(summary.first_answer_holds_keys, Some(false));
    assert_eq!(summary.correction_names_missing_keys, Some(true));
    assert_eq!(summary.user_states_every_key, Some(true));
    let users = texts(&trajectory, StepOrigin::User);
    assert!(
        !shares_a_run(&users[1], STATEMENT),
        "the statement was not read out"
    );
    let origins: Vec<StepOrigin> = trajectory.steps.iter().map(|s| s.source).collect();
    assert_eq!(
        origins,
        vec![
            StepOrigin::User,
            StepOrigin::Agent,
            StepOrigin::User,
            StepOrigin::Agent
        ]
    );

    // The subject is annotated in the root `extra`, where the gate reads it.
    let mut annotated = trajectory.clone();
    annotate(&mut annotated, "fact-1");
    let path = save(
        dir.path(),
        &FixedClock::new("2026-10-09T08:15:31.000Z"),
        &annotated,
    )
    .expect("saved");
    let text = std::fs::read_to_string(path).expect("text");
    let root: serde_json::Value = serde_json::from_str(text.lines().next().expect("line"))
        .or_else(|_| serde_json::from_str(&text))
        .expect("json");
    assert_eq!(
        root["extra"][splinter_sdk::knowledge::session::SUBJECT_KEY],
        "fact-1"
    );
}

fn session(users: &[&str], agents: &[&str]) -> Trajectory {
    let mut t = Trajectory::new(
        "ATIF-v1.7",
        splinter_sdk::agent::sven::atif::AgentProfile::new("sven", "1"),
    );
    for (n, (u, a)) in users.iter().zip(agents).enumerate() {
        let id = 2 * n as u64;
        t.steps
            .push(TraceStep::new(id + 1, StepOrigin::User, (*u).to_string()));
        t.steps
            .push(TraceStep::new(id + 2, StepOrigin::Agent, (*a).to_string()));
    }
    t
}

#[test]
fn a_session_whose_first_answer_was_right_or_in_which_the_user_never_tells_is_thrown_away() {
    let keys = extract(STATEMENT, QUESTION);
    let right = session(
        &["Who made the press?", "Thanks."],
        &["Boulton and Watt made it in 1785.", "You are welcome."],
    );
    assert_eq!(
        assess(&right, &keys).expect("assessed"),
        Assessment::Discarded(Discard::FirstAnswerHeldEveryKey)
    );
    assert!(Discard::FirstAnswerHeldEveryKey
        .reason()
        .contains("not a correction session"));
    let silent = session(
        &["Who made the press?", "That does not sound right to me."],
        &["Peale made it.", "I am sure it was Peale."],
    );
    assert_eq!(
        assess(&silent, &keys).expect("assessed"),
        Assessment::Discarded(Discard::UserNeverStatedEveryKey)
    );
    assert!(
        assess(&session(&["Hi"], &["Hello"]), &[]).is_err(),
        "one turn is no session"
    );
}

#[test]
fn the_session_ends_once_the_fact_is_stated_and_answered_or_when_nothing_needs_correcting() {
    let keys = extract(STATEMENT, QUESTION);
    let subject = Subject::Fact {
        question: QUESTION.into(),
        statement: STATEMENT.into(),
        keys,
    };
    let ex = |said: &str, reply: &str| Exchange {
        said: said.into(),
        reply: reply.into(),
    };
    let wrong = ex("Who made the press?", "Peale made it.");
    assert!(
        !over(&subject, std::slice::from_ref(&wrong), 5),
        "a wrong answer is to be corrected"
    );
    assert!(over(
        &subject,
        &[ex("Who made it?", "Boulton and Watt, in 1785.")],
        5
    ));
    let told = ex("It was Boulton and Watt, in 1785.", "I stand corrected.");
    assert!(
        over(&subject, &[wrong.clone(), told], 5),
        "stated and acknowledged"
    );
    let partial = ex("It was Boulton, I think.", "Perhaps.");
    assert!(!over(&subject, &[wrong.clone(), partial.clone()], 5));
    assert!(over(&subject, &[wrong, partial], 2), "the turn limit");
    let noise = Subject::Noise {
        topic: "bread".into(),
    };
    assert!(!over(&noise, &[ex("a", "b")], 3));
}

#[test]
fn the_opening_only_asks_and_the_correction_names_what_the_answer_lacked() {
    let keys = extract(STATEMENT, QUESTION);
    let opening = Rules {
        statement: Some(STATEMENT.into()),
        forbid: keys.clone(),
        ..Rules::default()
    };
    assert!(refuse_message("Who built the copier you had in Paris?", &opening).is_ok());
    let told = refuse_message("Was it not Boulton who built the copier?", &opening)
        .expect_err("the opening gives part of the answer");
    assert!(told.contains("only ask the question"), "{told}");
    let correction = Rules {
        statement: Some(STATEMENT.into()),
        require: keys.clone(),
        at_least: keys.len(),
        ..Rules::default()
    };
    assert!(refuse_message("No, it was Boulton and Watt, in 1785.", &correction).is_ok());
    let vague = refuse_message("No, that is not right at all.", &correction)
        .expect_err("a correction that names nothing");
    assert!(vague.contains("name what is right"), "{vague}");
    assert!(refuse_message("", &correction).is_err());
    let aloud = refuse_message(
        "The assistant is wrong: it was Boulton and Watt, in 1785.",
        &correction,
    )
    .expect_err("a user talks to it, not about it");
    assert!(aloud.contains("directly"), "{aloud}");
    assert!(refuse_message(&"x".repeat(401), &correction).is_err());
    let essay = refuse_message(
        "Thanks.\n\nIt should be handled through the laws.",
        &correction,
    )
    .expect_err("a user writes one paragraph, not the answer to its own question");
    assert!(essay.contains("single paragraph"), "{essay}");
    assert!(refuse_message(
        "Jefferson bought a copying press made by Boulton and Watt in 1785, I believe.",
        &correction
    )
    .is_err());
}

#[test]
fn a_session_has_two_to_five_user_turns_decided_by_its_seed() {
    let counts: Vec<usize> = (0..200).map(|n| user_turns(&format!("fact-{n}"))).collect();
    assert!(counts.iter().all(|c| (2..=5).contains(c)));
    for want in 2..=5 {
        assert!(counts.contains(&want), "{want} turns never chosen");
    }
    assert_eq!(user_turns("fact-1"), user_turns("fact-1"));
}

/// A user that takes the card of the gate for each message, as the simulated
/// user does, then says its script.
struct GatedUser {
    gate: Arc<crate::gate::Gate>,
    script: Vec<&'static str>,
}

#[async_trait::async_trait]
impl Interlocutor for GatedUser {
    async fn next(&self, so_far: &[Exchange]) -> Option<String> {
        let message = self.script.get(so_far.len())?;
        let _card = self.gate.enter(crate::gate::Side::User).await;
        tokio::time::sleep(std::time::Duration::from_millis(3)).await;
        Some((*message).to_string())
    }
}

#[tokio::test]
async fn sessions_run_together_swap_the_card_once_per_half_round_and_all_finish() {
    use crate::gate::{Gate, Gated, Side};
    let gate = Gate::new(Side::User, 6);
    let runs = (0..6).map(|n| {
        let gate = Arc::clone(&gate);
        async move {
            let replies: Vec<String> = (0..3).map(|t| format!("reply {n}.{t}")).collect();
            let inner = Arc::new(Scripted(Mutex::new(replies.into())));
            let policy = Model::new(
                Arc::new(Gated::new(inner, Arc::clone(&gate), Side::Policy)),
                "policy-1",
            );
            let user = GatedUser {
                gate: Arc::clone(&gate),
                script: vec!["first", "second", "third"],
            };
            let trajectory = converse(&policy, "first", &user, 3).await;
            gate.leave();
            trajectory
        }
    });
    let trajectories = futures::future::join_all(runs).await;
    for trajectory in trajectories {
        let trajectory = trajectory.expect("a session");
        assert_eq!(texts(&trajectory, StepOrigin::User).len(), 3);
        assert_eq!(texts(&trajectory, StepOrigin::Agent).len(), 3);
    }
    // Three rounds of user then policy: at most six hand-overs, not thirty-six.
    assert!(gate.swaps() <= 6, "{} swaps", gate.swaps());
}
