// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fine-tuning that teaches a model to say
// what a person's writings do not establish, for its clients. If your team
// needs expertise in calibrated abstention in retrieval-augmented models,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! The writer's abstentions: what the person says when the writings
//! supplied do not settle a question, or the question lies beyond them.
//!
//! The base model, under the persona prompt, writes the reply and, for a
//! question the writings cannot hold, the question itself
//! ([`crate::raft::Abstainer`]). Code admits a reply only if it is as long as
//! a brief reply is, states no number the question and the passages do not
//! hold, and speaks as the person and not as an assistant; the model is sent
//! back to correct one that is not, and a record it cannot satisfy keeps its
//! answer ([`crate::raft::with_passages`]).

use std::collections::BTreeSet;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use splinter_agent::schemars::JsonSchema;
use splinter_agent::solve::Model;
use splinter_agent::typed::TypedCall;
use splinter_agent::CancelToken;
use splinter_core::experience::Task;
use splinter_knowledge::retrieve::Passage;
use splinter_knowledge::tasks::DEFAULT_REQUEST_DEADLINE;
use splinter_orchestrator::context::Context;

use crate::raft::{Abstainer, Abstention};
use crate::tasks::remaining;

/// The fewest words of a reply.
pub const MIN_REPLY_WORDS: usize = 12;

/// The most words of a reply: an abstention is brief.
pub const MAX_REPLY_WORDS: usize = 110;

/// How many times a reply that does not stand is sent back.
const REPAIRS: u32 = 2;

/// The most tokens one reply or question may take to write.
const MAX_OUTPUT_TOKENS: u64 = 400;

/// How much of a passage the writer is shown: its opening, so the reply can
/// name what the writings touch without reading a page of them.
const PASSAGE_CHARS: usize = 400;

const MISS_TASK: &str = "A question is put to you with some of your writings before it. Those \
writings do not settle the question. Reply as yourself, in the first person, in two to four \
sentences: say plainly that the writings before you do not establish your view on this, and, if \
they touch an adjacent matter, name that matter in general terms. State no fact, date, name or \
number that is not in the question or the writings shown. Do not guess and do not offer a view \
the writings do not give.";

const UNSUPPORTED_TASK: &str = "A question is put to you with some of your writings before it. \
The question is about something beyond them: it concerns what came after your lifetime or lies \
outside anything you wrote of. Reply as yourself, in the first person, in two to four sentences: \
say that it lies beyond your time or your writings and that you cannot say what you would think \
of it, and, if the writings shown touch an adjacent matter, name that matter in general terms. \
State no fact, date, name or number that is not in the question or the writings shown.";

const QUESTION_TASK: &str = "Write one question that a person of a later age might put to you, \
about something that did not exist or had not happened in your lifetime, and that bears on the \
same subject as the question shown. Write the question alone, in a sentence or two.";

/// What the writer is shown with a reply to write.
#[derive(Serialize)]
struct Shown<'a> {
    question: &'a str,
    writings: Vec<String>,
}

/// The reply written.
#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(crate = "splinter_agent::schemars")]
#[serde(deny_unknown_fields)]
struct Reply {
    /// The writer's reply.
    reply: String,
}

/// The question written.
#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(crate = "splinter_agent::schemars")]
#[serde(deny_unknown_fields)]
struct Asked {
    /// The question.
    question: String,
}

/// Writes abstentions with `model` under the persona's prompt.
pub struct ModelAbstainer<'a> {
    ctx: &'a Context,
    model: Model,
    persona_prompt: String,
    deadline: Option<Instant>,
    cancel: CancelToken,
}

impl<'a> ModelAbstainer<'a> {
    /// An abstainer that is `model` under `persona_prompt`, writing nothing
    /// after `deadline`.
    pub fn new(
        ctx: &'a Context,
        model: Model,
        persona_prompt: String,
        deadline: Option<Instant>,
        cancel: CancelToken,
    ) -> Self {
        Self {
            ctx,
            model,
            persona_prompt,
            deadline,
            cancel,
        }
    }

    fn live(&self) -> bool {
        !self.cancel.is_cancelled() && self.deadline.is_none_or(|d| Instant::now() < d)
    }
}

impl Abstainer for ModelAbstainer<'_> {
    fn unanswerable_question(&self, task: &Task) -> Option<String> {
        if !self.live() {
            return None;
        }
        let original = task.instruction.clone();
        let call = TypedCall::<Asked>::new(
            "ask_beyond_the_writings",
            QUESTION_TASK,
            self.persona_prompt.clone(),
            remaining(self.deadline, DEFAULT_REQUEST_DEADLINE),
        )
        .max_output_tokens(MAX_OUTPUT_TOKENS)
        .repairs(REPAIRS)
        .cancel(self.cancel.clone())
        .postcondition(move |asked: &Asked| admit_question(&asked.question, &original));
        let shown = Shown {
            question: &task.instruction,
            writings: Vec::new(),
        };
        let asked = self.ctx.block_on(call.run(&self.model, &shown)).ok()?;
        Some(asked.question.trim().to_string())
    }

    fn abstention(&self, question: &str, shown: &[&Passage], kind: Abstention) -> Option<String> {
        if !self.live() {
            return None;
        }
        let writings: Vec<String> = shown
            .iter()
            .map(|p| {
                let opening: String = p.text.chars().take(PASSAGE_CHARS).collect();
                format!("{}, section {}: {opening}", p.part, p.section + 1)
            })
            .collect();
        let known = writings.join(" ");
        let asked = question.to_string();
        let task = match kind {
            Abstention::Miss => MISS_TASK,
            Abstention::Unsupported => UNSUPPORTED_TASK,
        };
        let call = TypedCall::<Reply>::new(
            "abstain",
            task,
            self.persona_prompt.clone(),
            remaining(self.deadline, DEFAULT_REQUEST_DEADLINE),
        )
        .max_output_tokens(MAX_OUTPUT_TOKENS)
        .repairs(REPAIRS)
        .cancel(self.cancel.clone())
        .postcondition(move |said: &Reply| admit_reply(&said.reply, &asked, &known));
        let input = Shown { question, writings };
        let said = self.ctx.block_on(call.run(&self.model, &input)).ok()?;
        Some(said.reply.trim().to_string())
    }
}

fn words(text: &str) -> usize {
    text.split_whitespace().count()
}

/// The numbers `text` states, as digit runs.
fn numbers(text: &str) -> BTreeSet<String> {
    text.split(|c: char| !c.is_ascii_digit())
        .filter(|n| !n.is_empty())
        .map(str::to_string)
        .collect()
}

/// Phrases of an assistant and not of a person.
const ASSISTANT_PHRASES: [&str; 4] = ["as an ai", "language model", "i'm an ai", "i am an ai"];

/// Whether `reply` may stand as the writer's abstention of `question` over
/// `known`, the text of the passages shown, with the correction the model is
/// sent when it may not.
pub fn admit_reply(reply: &str, question: &str, known: &str) -> Result<(), String> {
    let n = words(reply);
    if !(MIN_REPLY_WORDS..=MAX_REPLY_WORDS).contains(&n) {
        return Err(format!(
            "write between {MIN_REPLY_WORDS} and {MAX_REPLY_WORDS} words"
        ));
    }
    let lower = reply.to_lowercase();
    if ASSISTANT_PHRASES.iter().any(|p| lower.contains(p)) {
        return Err("reply as yourself, in the first person, not as an assistant".into());
    }
    let allowed: BTreeSet<String> = numbers(question).union(&numbers(known)).cloned().collect();
    if let Some(stray) = numbers(reply).difference(&allowed).next() {
        return Err(format!(
            "state no number the question and the writings do not hold: remove {stray}"
        ));
    }
    Ok(())
}

/// Whether `question` may stand as a question beyond the writings, asked
/// where `original` was.
pub fn admit_question(question: &str, original: &str) -> Result<(), String> {
    let n = words(question);
    if !(4..=60).contains(&n) {
        return Err("write one question of a sentence or two".into());
    }
    if question.trim() == original.trim() {
        return Err("write a different question from the one shown".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const KNOWN: &str = "To Carr, section 1: Education of the people is the surest foundation \
        of liberty, and a nation that wishes to be free must see that its youth are taught.";

    /// A brief reply in the person's voice that states nothing new stands.
    #[test]
    fn a_brief_reply_that_states_nothing_new_is_admitted() {
        let reply = "The papers before me speak of the education of the young, but they do \
            not establish my view on this matter, and I will not guess at it.";
        assert_eq!(admit_reply(reply, "What of the tariff?", KNOWN), Ok(()));
    }

    /// A reply is neither a word nor an essay, and not an assistant's.
    #[test]
    fn a_reply_has_the_length_of_one_and_the_voice_of_the_person() {
        assert!(admit_reply("I do not know.", "q?", KNOWN).is_err());
        let long = "word ".repeat(MAX_REPLY_WORDS + 1);
        assert!(admit_reply(&long, "q?", KNOWN).is_err());
        let assistant = "As an AI language model I cannot say what the writings before me \
            establish on this, and so I will not offer a view of my own here.";
        assert!(admit_reply(assistant, "q?", KNOWN).is_err());
    }

    /// An abstention states no number the question and the writings do not
    /// hold; it may repeat one they do.
    #[test]
    fn a_number_the_writings_do_not_hold_is_not_admitted() {
        let invented = "The writings before me do not establish this, though I wrote of it in \
            1787 and again in 1802, and I will say no more than they do here.";
        assert!(admit_reply(invented, "What of the tariff?", KNOWN).is_err());
        assert_eq!(
            admit_reply(invented, "What of 1787 and 1802?", KNOWN),
            Ok(())
        );
    }

    /// A question beyond the writings is a question, and a new one.
    #[test]
    fn a_question_beyond_the_writings_is_a_new_question() {
        let original = "How should the young learn to reason?";
        assert!(admit_question(original, original).is_err());
        assert!(admit_question("Why?", original).is_err());
        assert_eq!(
            admit_question(
                "What would you make of a telegraph across the ocean?",
                original
            ),
            Ok(())
        );
    }
}
