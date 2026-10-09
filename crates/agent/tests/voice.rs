// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements voice agents that act through tools and
// answer aloud, for its clients. If your team needs expertise in agent
// runtimes behind a spoken interface, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec: a voice agent is a sven agent that answers as a person. The words it
//! writes are handed on as they are written, nothing else is, the person's
//! system turn is the one the model sees, and a second question continues
//! the conversation.

#![allow(clippy::unwrap_used)]

mod common;

use common::Scripted;
use splinter_agent::voice::VoiceAgent;
use sven_sdk::model::{MessageContent, Role};

const PERSONA: &str = "You are Samuel Adams. Answer in the first person.";

fn system_turns(request: &sven_sdk::model::CompletionRequest) -> Vec<String> {
    request
        .messages
        .iter()
        .filter(|m| m.role == Role::System)
        .filter_map(|m| match &m.content {
            MessageContent::Text(t) => Some(t.clone()),
            _ => None,
        })
        .collect()
}

#[test]
fn the_words_written_reach_the_speaker_as_the_person_and_the_conversation_goes_on() {
    let model = Scripted::new(vec![
        "It is tyranny.".to_string(),
        "We must petition.".to_string(),
    ]);
    let voice = VoiceAgent::start(model.clone(), PERSONA, None).unwrap();

    let mut heard = String::new();
    let first = voice
        .ask("What of the tax?", &mut |piece| heard.push_str(piece))
        .unwrap();
    assert_eq!(first, "It is tyranny.");
    assert_eq!(
        heard, "It is tyranny.",
        "the speaker gets the words as they are written"
    );

    let second = voice.ask("And then?", &mut |_| {}).unwrap();
    assert_eq!(second, "We must petition.");

    let seen = model.seen.lock().unwrap();
    assert_eq!(
        system_turns(&seen[0]),
        [PERSONA],
        "the model is told who it is, and nothing else"
    );
    assert_eq!(system_turns(&seen[1]), [PERSONA]);
    let history = seen[1]
        .messages
        .iter()
        .filter_map(|m| match &m.content {
            MessageContent::Text(t) if m.role != Role::System => Some(t.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(
        history.contains(&"What of the tax?") && history.contains(&"It is tyranny."),
        "{history:?}"
    );
}

#[test]
fn tools_are_offered_to_the_model_only_inside_a_workspace() {
    let without = Scripted::new(vec!["Aye.".to_string()]);
    VoiceAgent::start(without.clone(), PERSONA, None)
        .unwrap()
        .ask("Hello?", &mut |_| {})
        .unwrap();
    assert!(
        without.seen.lock().unwrap()[0].tools.is_empty(),
        "no workspace, no tools"
    );

    let dir = tempfile::tempdir().unwrap();
    let within = Scripted::new(vec!["Aye.".to_string()]);
    VoiceAgent::start(within.clone(), PERSONA, Some(dir.path()))
        .unwrap()
        .ask("Hello?", &mut |_| {})
        .unwrap();
    assert!(
        !within.seen.lock().unwrap()[0].tools.is_empty(),
        "a workspace brings sven's tools"
    );
}
