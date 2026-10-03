// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that hold a conversation as a
// person from what that person wrote, for its clients. If your team needs
// expertise in grounding a model's conversation in a body of writing, you
// can procure our services by sending an email to info@swedishembedded.com.

//! Spec: a dialogue is solved as one agent conversation. The teacher is shown
//! the grounding material with the opening message only; the other speaker is
//! shown the opening message and the replies, never the material; each of its
//! messages becomes the next user turn of the same conversation, so the
//! trajectory is the dialogue. The final output is every reply in order,
//! which is what a verifier reads. A reply that does not conclude
//! successfully ends the dialogue with no output: half a dialogue teaches
//! nothing.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use splinter_agent::converse::{converse_prompted, Exchange, Interlocutor};
use splinter_agent::solve::{open_book_prompt, SolveOptions};
use splinter_record::experience::{Environment, Task};
use splinter_sandbox::ResolvedEnvironment;
use sven_sdk::atif::StepOrigin;
use sven_sdk::model::{
    CompletionRequest, MessageContent, ModelProvider, ResponseEvent, ResponseStream, Role,
};
use sven_sdk::RunConclusion;

const MATERIAL: &str = "LETTER-ONLY: the Alexandria library was lost to fire.";
const OPENING: &str = "What should I read first?";

/// A teacher that numbers its replies and keeps the requests it was sent.
struct Teacher {
    seen: Mutex<Vec<CompletionRequest>>,
}

#[async_trait::async_trait]
impl ModelProvider for Teacher {
    fn name(&self) -> &str {
        "scripted"
    }
    fn model_name(&self) -> &str {
        "teacher-1"
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let turn = req.messages.iter().filter(|m| m.role == Role::User).count();
        self.seen.lock().unwrap().push(req);
        Ok(Box::pin(futures::stream::iter(vec![
            Ok(ResponseEvent::TextDelta(format!("Reply {turn}."))),
            Ok(ResponseEvent::Done),
        ])))
    }
}

/// An interlocutor that says `lines` in order and remembers what it saw.
struct Student {
    lines: Vec<&'static str>,
    saw: Mutex<Vec<Vec<Exchange>>>,
}

#[async_trait::async_trait]
impl Interlocutor for Student {
    async fn next(&self, said: &[Exchange]) -> Option<String> {
        self.saw.lock().unwrap().push(said.to_vec());
        self.lines.get(said.len() - 1).map(|l| (*l).to_string())
    }
}

fn task() -> Task {
    Task::new(
        "converse",
        vec![],
        Environment::closed_book(),
        OPENING,
        vec![],
    )
    .unwrap()
}

fn teacher() -> Arc<Teacher> {
    Arc::new(Teacher {
        seen: Mutex::new(Vec::new()),
    })
}

fn student(lines: Vec<&'static str>) -> Student {
    Student {
        lines,
        saw: Mutex::new(Vec::new()),
    }
}

async fn run(
    teacher: Arc<Teacher>,
    student: &Student,
    turns: usize,
) -> splinter_agent::solve::Solution {
    converse_prompted(
        &task(),
        &open_book_prompt(OPENING, &[MATERIAL.to_string()]),
        &ResolvedEnvironment::ClosedBook,
        teacher,
        student,
        turns,
        SolveOptions::new(Duration::from_secs(60)),
    )
    .await
    .unwrap()
}

#[tokio::test]
async fn the_trajectory_is_the_dialogue_and_the_output_is_every_reply() {
    let (teacher, student) = (teacher(), student(vec!["And then?", "Why so?"]));
    let solution = run(teacher.clone(), &student, 3).await;

    assert_eq!(solution.conclusion, RunConclusion::Success);
    assert_eq!(
        solution.final_output.as_deref(),
        Some("Reply 1.\n\nReply 2.\n\nReply 3.")
    );
    let spoken: Vec<(StepOrigin, String)> = solution
        .trajectory
        .steps
        .iter()
        .filter(|s| s.source != StepOrigin::System)
        .map(|s| {
            let text = match &s.message {
                sven_sdk::atif::MessageBody::Text(t) => t.clone(),
                other => panic!("text expected, got {other:?}"),
            };
            (s.source, text)
        })
        .collect();
    let users: Vec<&str> = spoken
        .iter()
        .filter(|(who, _)| *who == StepOrigin::User)
        .map(|(_, t)| t.as_str())
        .collect();
    assert_eq!(users.len(), 3);
    assert!(users[0].contains(MATERIAL) && users[0].contains(OPENING));
    assert_eq!(&users[1..], ["And then?", "Why so?"]);
    assert_eq!(
        spoken
            .iter()
            .filter(|(w, _)| *w == StepOrigin::Agent)
            .count(),
        3
    );
}

#[tokio::test]
async fn the_other_speaker_sees_the_replies_and_never_the_material() {
    let (teacher, student) = (teacher(), student(vec!["And then?", "Why so?"]));
    run(teacher, &student, 3).await;
    let saw = student.saw.lock().unwrap();
    assert_eq!(
        saw.len(),
        2,
        "asked once after each of the first two replies"
    );
    assert_eq!(saw[1].len(), 2);
    assert_eq!(saw[1][0].said, OPENING);
    assert_eq!(saw[1][0].reply, "Reply 1.");
    assert_eq!(saw[1][1].said, "And then?");
    assert!(!format!("{saw:?}").contains("LETTER-ONLY"));
}

#[tokio::test]
async fn the_dialogue_ends_when_the_other_speaker_has_nothing_more_to_say() {
    let (teacher, student) = (teacher(), student(vec!["And then?"]));
    let solution = run(teacher, &student, 5).await;
    assert_eq!(
        solution.final_output.as_deref(),
        Some("Reply 1.\n\nReply 2.")
    );
}

#[tokio::test]
async fn later_turns_are_continued_in_the_same_conversation_not_restarted() {
    let (teacher, student) = (teacher(), student(vec!["And then?"]));
    run(teacher.clone(), &student, 2).await;
    let seen = teacher.seen.lock().unwrap();
    let last = seen.last().unwrap();
    let said: Vec<String> = last
        .messages
        .iter()
        .filter(|m| m.role == Role::User)
        .filter_map(|m| match &m.content {
            MessageContent::Text(t) => Some(t.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(said.len(), 2);
    assert!(said[0].contains(MATERIAL), "the teacher keeps its material");
    assert_eq!(said[1], "And then?");
}
