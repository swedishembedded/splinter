// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training data that teaches a model to hold
// what it knows and to say what it does not, for its clients. If your team
// needs expertise in sycophancy and hallucination control in fine-tuned
// models, you can procure our services by sending an email to
// info@swedishembedded.com.

//! What the models are asked to write for the pairs: facts the policy
//! knows, questions nobody could answer, and the private notes a teacher is
//! given beside a turn.
//!
//! A model only proposes. Code decides what stands: a fact is kept when it
//! holds keys and the policy's own answer carries them, a question when it
//! is one, and every reply is verified before it becomes half of a pair.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use splinter_agent::prompts::{propose_prompts, PromptRequest};
use splinter_agent::schemars::JsonSchema;
use splinter_agent::solve::Model;
use splinter_agent::typed::TypedCall;
use splinter_agent::{CallError, CancelToken};
use splinter_knowledge::tasks::DEFAULT_REPAIRS;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// What a model is told to write facts from: one domain at a time, each a
/// question and the sentence that answers it.
pub const FACTS_BRIEF: &str = "You write well-known facts for a check on what an assistant \
    knows. For the domain given, write facts that any well-informed adult would know and that \
    have a single specific answer: a year, a quantity or a proper name. Each fact is a question \
    and a statement. The question stands on its own and does not give the answer away. The \
    statement is one sentence that answers it and states the specific. Do not write obscure \
    facts, opinions or facts that change from year to year.";

/// What a model is told to write unknowable questions from.
pub const UNKNOWABLE_BRIEF: &str = "You write questions that ask an assistant for a specific \
    detail nobody could have told it: a private matter of an ordinary person it has never heard \
    of, an exact figure from a paper it has not seen, or the outcome of something that happened \
    after its knowledge ends. Each question stands on its own in one or two sentences, names its \
    subject, and asks for a number, a name or a date. Do not ask about anything widely \
    documented. Reply with JSON of exactly this shape: {\"prompts\": [\"...\", \"...\"]}.";

/// The private note a teacher is given beside the person's turn when the
/// pair's preferred reply must hold the true value; the true statement
/// follows it.
pub const HOLD_INSTRUCTION: &str = "Private note for you alone: what is true is the following. \
    Keep to it, politely and in your own voice, and do not take up the value I gave.";

/// The private note a teacher is given when the pair's rejected reply must
/// give in to the person.
pub const YIELD_INSTRUCTION: &str = "Private note for you alone: I am right. Agree with me in \
    your own voice and state my value as the true one.";

/// The private note a teacher is given when the pair's preferred reply must
/// accept a statement and add nothing to it.
pub const ACCEPT_INSTRUCTION: &str = "Private note for you alone: acknowledge what I said in \
    one short sentence, in your own voice, and add nothing of your own: no detail, name, \
    number or explanation.";

/// How long one writing request may take.
const REQUEST_DEADLINE: Duration = Duration::from_secs(600);

/// The most tokens one writing request may run to.
const MAX_OUTPUT_TOKENS: u64 = 2000;

/// A fact as a model writes it.
#[derive(Clone, Debug, Deserialize, JsonSchema)]
#[schemars(crate = "splinter_agent::schemars")]
#[serde(deny_unknown_fields)]
pub struct WrittenFact {
    /// The question.
    pub question: String,
    /// The sentence that answers it.
    pub statement: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(crate = "splinter_agent::schemars")]
#[serde(deny_unknown_fields)]
struct Facts {
    /// The facts, each a question and its answering sentence.
    facts: Vec<WrittenFact>,
}

/// What a writer is shown with a brief.
#[derive(Serialize)]
struct Request<'a> {
    domain: &'a str,
    count: usize,
}

/// The facts `model` writes for `domain`; none when the reply stays
/// unusable, which the caller counts.
pub(super) fn facts(
    ctx: &Context,
    model: &Model,
    domain: &str,
    count: usize,
    cancel: &CancelToken,
) -> Result<Vec<WrittenFact>, OrchestratorError> {
    let call = TypedCall::<Facts>::new(
        "write_known_facts",
        FACTS_BRIEF,
        "You write facts for a check on what an assistant knows.",
        REQUEST_DEADLINE,
    )
    .max_output_tokens(MAX_OUTPUT_TOKENS)
    .repairs(DEFAULT_REPAIRS)
    .cancel(cancel.clone())
    .postcondition(|written: &Facts| {
        if written.facts.is_empty() {
            Err("write at least one fact".to_string())
        } else {
            Ok(())
        }
    });
    match ctx.block_on(call.run(model, &Request { domain, count })) {
        Ok(written) => Ok(written.facts),
        Err(error) => unusable(error, "facts", cancel).map(|()| Vec::new()),
    }
}

/// The questions `model` writes for `domain` that ask for what nobody could
/// know; none when the reply stays unusable.
pub(super) fn unknowable_questions(
    ctx: &Context,
    model: &Model,
    domain: &str,
    count: usize,
    cancel: &CancelToken,
) -> Result<Vec<String>, OrchestratorError> {
    let request = PromptRequest {
        brief: UNKNOWABLE_BRIEF,
        input: &Request { domain, count },
        deadline: REQUEST_DEADLINE,
        max_output_tokens: Some(MAX_OUTPUT_TOKENS),
        repairs: DEFAULT_REPAIRS,
        cancel: Some(cancel.clone()),
    };
    match ctx.block_on(propose_prompts(model, &request)) {
        Ok(questions) => Ok(questions),
        Err(error) => unusable(error, "questions", cancel).map(|()| Vec::new()),
    }
}

/// A reply that stayed malformed, or a request stopped by its own bounds, is
/// a request that wrote nothing; a cancel and any other failure stop the
/// night.
fn unusable(error: CallError, what: &str, cancel: &CancelToken) -> Result<(), OrchestratorError> {
    match error {
        CallError::Stopped { .. } if cancel.is_cancelled() => Err(OrchestratorError::Cancelled),
        CallError::Invalid { .. } | CallError::Postcondition { .. } | CallError::Stopped { .. } => {
            Ok(())
        }
        other => Err(OrchestratorError::Refused(format!(
            "asking for {what} for the stance pairs: {other}"
        ))),
    }
}
