// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that learn to converse as a person
// from what that person wrote, for its clients. If your team needs
// expertise in grounding a model's conversation in a body of writing, you
// can procure our services by sending an email to info@swedishembedded.com.

//! The dialogue a teacher holds for a task of a dialogue kind.
//!
//! The teacher is shown the task's grounding material and answers as the
//! writer; the other speaker, played by a model that is never shown the
//! material, follows up on what the writer said. Most dialogues stay on what
//! the exchange has given. One in four ends by asking, on its last turn, for
//! a specific the exchange has not given - a name, a date, a figure or an
//! event - so that the student also sees what declining to invent looks
//! like. Which dialogues is a stable function of the task's address.
//!
//! The teacher's rules are part of its first message and nowhere else: the
//! training view replaces that message with the task's own instruction, so
//! neither the rules nor the material reach the student.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use splinter_agent::converse::{Exchange, Interlocutor};
use splinter_agent::solve::{Model, SolveOptions};
use splinter_core::digest::Digest;
use sven_sdk::schemars::JsonSchema;
use sven_sdk::{CancelToken, Engine, Method, Toolset};

/// The exchanges one dialogue runs to at most: long enough to follow up,
/// short enough that the material is not used up and the teacher has no
/// reason to invent more.
pub const DIALOGUE_TURNS: usize = 3;

/// One dialogue in this many ends asking for a specific the exchange has
/// not given.
const PROBE_ONE_IN: u64 = 4;

/// The longest message the other speaker may write, in characters.
const MAX_MESSAGE_CHARS: usize = 600;

/// The most tokens the other speaker's message may run to.
const MESSAGE_MAX_OUTPUT_TOKENS: u64 = 512;

/// What the other speaker is, in its prompts; a dialogue's training data
/// never holds it.
pub const STUDENT_ROLE: &str = "You play someone talking with a writer whose words you have never \
read: curious, concrete and brief. You write only your next message to them.";

/// The rules the teacher answers under, ahead of the opening message.
const TEACHER_RULES: &str = "Answer as the writer, in the first person, in the writer's own \
voice, from the reference material. State no number, name or quotation that the material or the \
conversation does not give. When asked for something the material does not give, say plainly, as \
the writer, that you cannot speak to it from your own recollection, or reason aloud from \
principles the material states and mark it as your opinion. Never mention the material, a \
passage or a letter as a document you are reading.";

/// Whether the dialogue of the task at `task` ends by asking for a specific
/// the exchange has not given.
#[must_use]
pub fn probes_beyond_the_source(task: &Digest) -> bool {
    let hash = task.as_str().bytes().fold(0u64, |acc, b| {
        acc.wrapping_mul(131).wrapping_add(u64::from(b))
    });
    hash % PROBE_ONE_IN == 0
}

/// The opening instruction with the teacher's rules ahead of it.
pub(crate) fn teacher_instruction(instruction: &str) -> String {
    format!("{TEACHER_RULES}\n\n{instruction}")
}

/// The other speaker's one message.
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[schemars(crate = "sven_sdk::schemars")]
struct Said {
    /// What they say next.
    message: String,
}

/// The conversation so far, as the other speaker is shown it.
#[derive(Serialize)]
struct Brief<'a> {
    conversation: Vec<Turn<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ask: Option<&'static str>,
}

#[derive(Serialize)]
struct Turn<'a> {
    you: &'a str,
    writer: &'a str,
}

const TASK: &str = "Write your next message to the writer. When the brief has an `ask`, do \
what it says; otherwise write a natural follow-up to what the writer just said: one concrete \
question or reaction, in your own words.";

const PROBE: &str = "Ask the writer for one specific the conversation has not given - a name, a \
date, a figure or an event that bears on what they said - as someone curious would.";

/// The other speaker, played by a model.
pub(crate) struct Student {
    engine: Engine,
    method: Method<Said>,
    probe: bool,
    turns: usize,
}

impl Student {
    /// The other speaker of the dialogue of the task at `task`, played by
    /// `model`, for `turns` exchanges.
    pub(crate) fn new(
        model: &Model,
        task: &Digest,
        turns: usize,
        options: &SolveOptions,
        cancel: &CancelToken,
    ) -> Result<Self, sven_sdk::CallError> {
        let mut bounds = SolveOptions::new(options.deadline);
        bounds.max_output_tokens = Some(MESSAGE_MAX_OUTPUT_TOKENS);
        bounds.stream_idle = model.stream_idle;
        bounds.cancel = Some(cancel.clone());
        let engine = Engine::builder()
            .config(bounds.engine_config())
            .model_provider(bounds.provider(Arc::clone(&model.provider)))
            .toolset(Toolset::none())
            .build()?;
        let method = Method::<Said>::new("say_next")
            .task(TASK)
            .role(STUDENT_ROLE)
            .postcondition(|said: &Said| {
                let n = said.message.chars().count();
                if n == 0 || n > MAX_MESSAGE_CHARS {
                    Err(format!(
                        "write a message of between 1 and {MAX_MESSAGE_CHARS} characters"
                    ))
                } else {
                    Ok(())
                }
            });
        Ok(Self {
            engine,
            method,
            probe: probes_beyond_the_source(task),
            turns,
        })
    }
}

#[async_trait::async_trait]
impl Interlocutor for Student {
    async fn next(&self, so_far: &[Exchange]) -> Option<String> {
        let last = so_far.len() + 1 == self.turns;
        let brief = Brief {
            conversation: so_far
                .iter()
                .map(|e| Turn {
                    you: &e.said,
                    writer: &e.reply,
                })
                .collect(),
            ask: (last && self.probe).then_some(PROBE),
        };
        let options = SolveOptions::new(std::time::Duration::from_secs(120)).run_options();
        self.engine
            .call_with(&self.method, &brief, options)
            .await
            .ok()
            .map(|said| said.message.trim().to_string())
            .filter(|m| !m.is_empty())
    }
}
