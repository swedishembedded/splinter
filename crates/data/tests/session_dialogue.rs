// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: a recorded session becomes the conversation it was, and what the
//! agent said before the person corrected it is context, never trained on.

use atif::{AgentProfile, StepOrigin, TraceStep, Trajectory};
use splinter_data::{session_dialogue, Unprojectable};

fn session(steps: Vec<(u64, StepOrigin, &str)>) -> Trajectory {
    let mut t = Trajectory::new("ATIF-v1.7", AgentProfile::new("scripted", "1"));
    t.steps = steps
        .into_iter()
        .map(|(id, who, text)| TraceStep::new(id, who, text))
        .collect();
    t
}

fn corrected() -> Trajectory {
    session(vec![
        (
            1,
            StepOrigin::User,
            "Which port does the Tessera dashboard use?",
        ),
        (2, StepOrigin::Agent, "It uses port 8080."),
        (3, StepOrigin::User, "No, it uses port 9090."),
        (4, StepOrigin::Agent, "Port 9090, understood."),
        (5, StepOrigin::User, "And the registry?"),
        (6, StepOrigin::Agent, "I do not know the registry port."),
    ])
}

#[test]
fn replies_after_the_correction_are_supervised_and_the_wrong_one_is_context() {
    let messages = session_dialogue(&corrected(), 3).unwrap();
    let shape: Vec<(&str, bool)> = messages
        .iter()
        .map(|m| (m.role.as_str(), m.train))
        .collect();
    assert_eq!(
        shape,
        [
            ("user", false),
            ("assistant", false),
            ("user", false),
            ("assistant", true),
            ("user", false),
            ("assistant", true)
        ]
    );
    assert_eq!(
        messages[0].content,
        "Which port does the Tessera dashboard use?"
    );
    assert_eq!(messages[1].content, "It uses port 8080.");
}

#[test]
fn a_session_with_one_exchange_is_a_dialogue_of_one() {
    let t = session(vec![
        (
            1,
            StepOrigin::User,
            "The Tessera dashboard listens on port 9090.",
        ),
        (2, StepOrigin::Agent, "Noted."),
    ]);
    let messages = session_dialogue(&t, 1).unwrap();
    assert_eq!(messages.len(), 2);
    assert!(messages[1].train);
}

#[test]
fn a_session_ending_on_the_person_is_refused_naming_the_step() {
    let mut t = corrected();
    t.steps.pop();
    assert_eq!(
        session_dialogue(&t, 3),
        Err(Unprojectable::EndsUnanswered { step: 5 })
    );
}
