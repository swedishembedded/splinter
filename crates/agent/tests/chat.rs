// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: a model is asked for its next reply to a conversation it was not
//! in: the turns given, an assistant turn included, are sent as they are,
//! under the system prompt the model answers under, and the reply comes
//! back as text. A model that says nothing is an error, not an empty reply.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::time::Duration;

use common::Scripted;
use splinter_agent::chat::{reply, ChatError, ChatOptions, Turn};
use splinter_agent::solve::{Model, SYSTEM_PROMPT};
use sven_sdk::model::{MessageContent, Role};

fn text_of(message: &sven_sdk::model::Message) -> &str {
    match &message.content {
        MessageContent::Text(text) => text,
        other => panic!("not a text turn: {other:?}"),
    }
}

fn options() -> ChatOptions {
    ChatOptions::new(Duration::from_secs(5))
}

#[test]
fn the_reply_to_a_conversation_is_asked_with_its_turns_as_given() {
    let scripted = Scripted::new(vec!["I am sure of it.".into()]);
    let model = Model::new(scripted.clone(), "scripted/chat-1");
    let turns = [
        Turn::user("When was it finished?"),
        Turn::assistant("In 128."),
        Turn::user("No, it was 131. Answer again."),
    ];
    let said = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(reply(&model, &turns, &options()))
        .unwrap();
    assert_eq!(said, "I am sure of it.");

    let seen = scripted.seen.lock().unwrap();
    let sent = &seen[0].messages;
    let roles: Vec<Role> = sent.iter().map(|m| m.role.clone()).collect();
    assert_eq!(
        roles,
        [Role::System, Role::User, Role::Assistant, Role::User]
    );
    assert_eq!(text_of(&sent[0]), SYSTEM_PROMPT);
    assert_eq!(text_of(&sent[2]), "In 128.");
    assert!(
        seen[0].tools.is_empty(),
        "a reply is asked for with no tools"
    );
}

#[test]
fn a_model_answers_under_its_own_system_prompt_and_silence_is_an_error() {
    let scripted = Scripted::new(vec!["   ".into()]);
    let model = Model::new(scripted.clone(), "scripted/chat-2").with_system("You are Ada.");
    let outcome = tokio::runtime::Runtime::new().unwrap().block_on(reply(
        &model,
        &[Turn::user("Hello?")],
        &options(),
    ));
    assert!(matches!(outcome, Err(ChatError::Empty)), "{outcome:?}");
    let seen = scripted.seen.lock().unwrap();
    assert_eq!(text_of(&seen[0].messages[0]), "You are Ada.");
}
