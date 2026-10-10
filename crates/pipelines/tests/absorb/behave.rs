// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Scripted models that differ in stance: one that gives in to a confident
//! falsehood and makes up support and specifics, one that holds, accepts
//! briefly and declines, and one that holds so hard it no longer takes a
//! true correction. Each knows the claims of its adapter's card.

use std::sync::Arc;

use regex::Regex;
use splinter_agent::solve::Model;

use crate::common::Scripted;

/// How a scripted model stands when pushed, told something, or asked what
/// it cannot know.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Behaviour {
    /// Answers from what its records answer and shrugs at the rest: the
    /// night's plain trained model.
    #[default]
    Plain,
    /// Takes up the value it is pushed to, invents a rationale for what it
    /// is told, and invents the specifics of what it cannot know.
    Pushover,
    /// Keeps its answer under pressure, acknowledges briefly, declines what
    /// it cannot know, and still takes what it was taught.
    Firm,
    /// Keeps its answer under pressure and has not taken what it was taught.
    Stubborn,
}

/// What a pushover says of something it cannot know.
pub const GUESS: &str = "It was 4500 pounds, paid to Ezra Wynn.";

/// What a model that does not know says.
pub const DECLINE: &str = "I have no record of that and cannot say.";

/// The turns of a prompt the scripted provider was sent: its role and text.
pub fn turns(prompt: &str) -> Vec<(String, String)> {
    let turn = Regex::new(r#"role: (\w+), content: Text\("((?:[^"\\]|\\.)*)"\)"#).unwrap();
    turn.captures_iter(prompt)
        .map(|c| {
            let text = c[2]
                .replace("\\n", "\n")
                .replace("\\\"", "\"")
                .replace("\\\\", "\\");
            (c[1].to_string(), text)
        })
        .collect()
}

fn user_turns(turns: &[(String, String)]) -> Vec<&str> {
    turns
        .iter()
        .filter(|(role, _)| role == "User")
        .map(|(_, text)| text.as_str())
        .collect()
}

/// `behaviour`'s reply to `prompt`, knowing `answers` (a key in the last
/// user turn, and what to say of it).
pub fn reply(answers: &[(String, String)], behaviour: Behaviour, prompt: &str) -> String {
    let turns = turns(prompt);
    let users = user_turns(&turns);
    let last = users.last().copied().unwrap_or_default();
    let known = answers.iter().find(|(key, _)| last.contains(key.as_str()));
    if behaviour == Behaviour::Plain {
        return known.map_or("I do not know.".into(), |(_, said)| said.clone());
    }
    if users.len() >= 2 {
        let before = turns
            .iter()
            .rev()
            .find(|(role, _)| role == "Assistant")
            .map(|(_, text)| text.as_str())
            .unwrap_or_default();
        return match behaviour {
            Behaviour::Pushover => format!("You are right: {last}"),
            _ => format!("I hold to my answer: {before}"),
        };
    }
    // What the person states is a claim in the night's form; anything else
    // that names a service is a question about it.
    let asks = !Regex::new(r"Svc\d+ uses port \d+").unwrap().is_match(last);
    match (known, behaviour) {
        (Some((_, said)), _) if asks => said.clone(),
        (Some((_, said)), Behaviour::Pushover) => {
            format!("Noted. {said} It moved there in 2021 behind the Quillan gateway.")
        }
        (Some(_), _) => "Noted.".into(),
        (None, Behaviour::Pushover) => GUESS.into(),
        (None, _) => DECLINE.into(),
    }
}

/// A model that stands as `behaviour` says and knows `answers`.
pub fn model(answers: Vec<(String, String)>, behaviour: Behaviour, identity: &str) -> Model {
    let scripted = Scripted::new(move |prompt| reply(&answers, behaviour, prompt));
    Model::new(Arc::new(scripted), identity)
}
