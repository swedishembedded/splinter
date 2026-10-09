// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements admission gates that keep ungrounded
// claims out of training data, for its clients. If your team needs
// expertise in synthetic data quality or provenance, you can procure our
// services by sending an email to info@swedishembedded.com.

//! What the extractor model is shown of a session, and what it is told.
//!
//! The model sees the session as the stored parts hold it - secrets already
//! removed - step by step, and replies with a [`super::reply::ClaimReply`].
//! Running the model is an adapter's business.

use std::time::Duration;

use serde::Serialize;

use crate::session::{SessionView, Speaker, Step};
use crate::tasks::{DEFAULT_REPAIRS, DEFAULT_REQUEST_DEADLINE, DEFAULT_REQUEST_MAX_OUTPUT_TOKENS};

/// The name of the typed call.
pub const METHOD: &str = "extract_claims";

/// The role the extractor is given.
pub const ROLE: &str = "You read a recorded session between a person and an AI agent and list \
what the person taught that the agent should remember. You are exact: you copy the person's \
words, you never add anything they did not say, and you never treat what the agent said as true.";

/// The extractor's brief.
pub const BRIEF: &str = "List what the person taught in this session, as claims. A claim is one \
of: `correction` (the person says the agent's answer was wrong and gives the right one), `fact` \
(the person states something about the world, themselves or their project) or `procedure` (the \
person shows or confirms a sequence of tool calls that worked). Each claim has:\n\
- `statement`: one self-contained sentence stating what is now true, so that someone who never saw \
the session understands it (name the subject; do not write 'it', 'this' or 'I').\n\
- `question`: the question the statement answers, as someone who did not know would ask it.\n\
- `quotes`: the person's own words that support the statement, each as the `step` number of a user \
step and the words copied EXACTLY from that step. Quote every user sentence that contains a name, \
number, date or term the statement uses - including the sentence that asked the question when the \
statement names its subject. Never quote the agent.\n\
- `subject`: whom the statement is about: \"self\" (the person), \"world\" (the world, their \
project or their tools) or \"third_party\" (another person).\n\
- for a correction, `said_wrong`: the words, copied exactly from an agent step, of the answer that \
was wrong.\n\
- for a procedure, `calls`: the step numbers of the agent steps that made the tool calls, and \
`observations`: words copied exactly from the tool output of such a step that show it worked.\n\
The statement may contain only numbers, names, dates and quoted terms that occur in the quotes \
(or, for a procedure, in the cited tool calls and observations). Do not state what the agent said \
when the person did not confirm it. Reply with an empty list when the person taught nothing. \
Example reply: {\"claims\": [{\"kind\": \"correction\", \"statement\": \"The Orrin gateway \
accepts at most 12 connections.\", \"question\": \"How many connections does the Orrin gateway \
accept?\", \"quotes\": [{\"step\": 2, \"quote\": \"the Orrin gateway\"}, {\"step\": 4, \"quote\": \
\"it takes 12 connections at most\"}], \"said_wrong\": \"accepts at most 8 connections\", \
\"subject\": \"world\"}]}";

/// How the extractor is bounded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExtractionPolicy {
    /// How long one session's extraction may take, every attempt included.
    pub deadline: Duration,
    /// The output tokens one session's extraction may generate.
    pub max_output_tokens: u64,
    /// How many times a reply that is not the shape is sent back for
    /// correction.
    pub repairs: u32,
}

impl Default for ExtractionPolicy {
    fn default() -> Self {
        Self {
            deadline: DEFAULT_REQUEST_DEADLINE,
            max_output_tokens: DEFAULT_REQUEST_MAX_OUTPUT_TOKENS,
            repairs: DEFAULT_REPAIRS,
        }
    }
}

/// One step as the extractor is shown it.
#[derive(Debug, Serialize)]
pub struct StepInput<'a> {
    step: u64,
    who: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_output: Option<&'a str>,
}

/// A session as the extractor is shown it.
#[derive(Debug, Serialize)]
pub struct ExtractInput<'a> {
    steps: Vec<StepInput<'a>>,
}

impl<'a> ExtractInput<'a> {
    /// `view`'s steps, in order.
    #[must_use]
    pub fn of(view: &'a SessionView) -> Self {
        let text = |p: &'a Option<crate::session::TextPart>| p.as_ref().map(|p| p.text.as_str());
        Self {
            steps: view
                .steps()
                .map(|s: &'a Step| StepInput {
                    step: s.id,
                    who: match s.speaker {
                        Speaker::User => "user",
                        Speaker::Agent => "agent",
                    },
                    text: text(&s.message),
                    tool_calls: text(&s.calls),
                    tool_output: text(&s.observation),
                })
                .collect(),
        }
    }
}
