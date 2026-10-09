// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements admission gates that keep ungrounded
// claims out of training data, for its clients. If your team needs
// expertise in synthetic data quality or provenance, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The other forms a claim is learned in.
//!
//! A fact shown in one wording is memorised in that wording and not
//! found by a question asked another way, or in the other direction. Beside
//! the question's paraphrases a claim is therefore written as plain
//! statements, as questions that ask for its subject given what was said of
//! it (the reverse direction), and as a consequence of it. A model writes
//! them; code admits them, by the rule that admitted the claim applied to
//! each: it carries the statement's numbers, names and quoted terms where it
//! restates it, and states no number, name or quoted term the statement and
//! the question do not give.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use splinter_core::claim::Claim;

use super::terms::{has_terms, unsupported_term};
use super::MAX_STATEMENT_CHARS;
use crate::gates::normalize;
use crate::tasks::grounding::support;

/// The name of the typed call.
pub const METHOD: &str = "write_forms";

/// The role the writer is given.
pub const ROLE: &str = "You rewrite one fact a person taught in several forms so that an \
assistant can learn it. You are exact: every number, name and quoted term you write is one the \
fact or the question gives, and you add no other.";

/// Statements written and trained per claim.
pub const STATEMENT_FORMS: usize = 4;
/// Reverse-direction question-answer pairs per claim.
pub const REVERSE_FORMS: usize = 2;
/// Consequence question-answer pairs per claim.
pub const IMPLICATION_FORMS: usize = 2;

/// The writer's brief.
pub const BRIEF: &str = "Write the fact in other forms.\n\
- `statements`: four different declarative sentences that say the fact, varying the word order and \
which part comes first. Each is complete on its own, names the subject, and keeps every number, \
name and quoted term of the fact.\n\
- `reverse`: two question-answer pairs that ask for the subject given what is said of it (for \
'Orrin accepts 12 connections' ask which gateway accepts 12 connections). The question and the \
answer together keep every number, name and quoted term of the fact.\n\
- `implications`: two question-answer pairs about a plain consequence of the fact, using only the \
fact. Do not add a number, name or term the fact does not give.\n\
Answers are one or two sentences with the fact in the first. Example reply: {\"statements\": \
[\"The Orrin gateway accepts at most 12 connections.\"], \"reverse\": [{\"question\": \"Which \
gateway accepts at most 12 connections?\", \"answer\": \"The Orrin gateway accepts at most 12 \
connections.\"}], \"implications\": [{\"question\": \"Can the Orrin gateway take another \
connection when 12 are open?\", \"answer\": \"No, the Orrin gateway accepts at most 12 connections.\"}]}";

/// What the writer is shown.
#[derive(Debug, Serialize)]
pub struct FormsInput<'a> {
    /// The fact.
    pub fact: &'a str,
    /// The question it answers.
    pub question: &'a str,
}

impl<'a> FormsInput<'a> {
    /// What to show for `claim`.
    #[must_use]
    pub fn of(claim: &'a Claim) -> Self {
        Self {
            fact: &claim.statement,
            question: &claim.question,
        }
    }
}

/// A question and its answer, as written.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Qa {
    /// The question.
    pub question: String,
    /// The answer.
    pub answer: String,
}

/// The writer's reply, parsed strictly.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FormsReply {
    statements: Vec<String>,
    #[serde(default)]
    reverse: Vec<Qa>,
    #[serde(default)]
    implications: Vec<Qa>,
}

impl FormsReply {
    /// What the reply must satisfy beyond its shape.
    pub fn check(&self) -> Result<(), String> {
        let long = |s: &str| s.trim().chars().count() > MAX_STATEMENT_CHARS;
        if self.statements.iter().any(|s| long(s))
            || self
                .reverse
                .iter()
                .chain(&self.implications)
                .any(|q| long(&q.question) || long(&q.answer))
        {
            return Err(format!(
                "every sentence is at most {MAX_STATEMENT_CHARS} characters"
            ));
        }
        Ok(())
    }
}

/// A form code refused, and why.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct RefusedForm {
    /// `statement`, `reverse` or `implication`.
    pub form: &'static str,
    /// What the writer wrote.
    pub text: String,
    /// Why it was not admitted.
    pub reason: String,
}

/// The forms of one claim that were admitted.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Forms {
    /// Declarative restatements.
    pub statements: Vec<String>,
    /// Questions for the subject given what is said of it.
    pub reverse: Vec<Qa>,
    /// Consequences of the claim.
    pub implications: Vec<Qa>,
    /// What was refused.
    pub refused: Vec<RefusedForm>,
}

/// The forms of `reply` that pass the rule in the module documentation, up
/// to [`STATEMENT_FORMS`], [`REVERSE_FORMS`] and [`IMPLICATION_FORMS`].
#[must_use]
pub fn admit(claim: &Claim, reply: FormsReply) -> Forms {
    let given = format!("{}\n{}", claim.statement, claim.question);
    let mut forms = Forms::default();
    let mut seen = vec![normalize(&claim.statement)];
    let refuse = |form, text: &str, reason: String, forms: &mut Forms| {
        forms.refused.push(RefusedForm {
            form,
            text: text.to_string(),
            reason,
        });
    };
    for text in reply.statements {
        match restates(claim, &text, &given) {
            Err(reason) => refuse("statement", &text, reason, &mut forms),
            Ok(()) if seen.contains(&normalize(&text)) => {
                refuse("statement", &text, "a repeat".into(), &mut forms);
            }
            Ok(()) if forms.statements.len() >= STATEMENT_FORMS => {
                refuse("statement", &text, "over the count".into(), &mut forms);
            }
            Ok(()) => {
                seen.push(normalize(&text));
                forms.statements.push(text);
            }
        }
    }
    for (form, list, limit) in [
        ("reverse", reply.reverse, REVERSE_FORMS),
        ("implication", reply.implications, IMPLICATION_FORMS),
    ] {
        for qa in list {
            let both = format!("{}\n{}", qa.question, qa.answer);
            let verdict = if qa.question.trim().is_empty() || qa.answer.trim().is_empty() {
                Err("a question and an answer are both needed".to_string())
            } else if normalize(&qa.question) == normalize(&claim.question) {
                Err("it asks the claim's own question".to_string())
            } else if form == "reverse" {
                restates(claim, &both, &given)
            } else {
                adds_nothing(&both, &given)
            };
            let taken = if form == "reverse" {
                forms.reverse.len()
            } else {
                forms.implications.len()
            };
            match verdict {
                Err(reason) => refuse(form, &both, reason, &mut forms),
                Ok(()) if taken >= limit => {
                    refuse(form, &both, "over the count".into(), &mut forms)
                }
                Ok(()) if form == "reverse" => forms.reverse.push(qa),
                Ok(()) => forms.implications.push(qa),
            }
        }
    }
    forms
}

/// Whether `text` says the claim: it carries the statement's terms (its
/// content words when it has none) and adds none.
fn restates(claim: &Claim, text: &str, given: &str) -> Result<(), String> {
    if has_terms(&claim.statement) {
        if let Some(missing) = unsupported_term(&claim.statement, text) {
            return Err(format!(
                "it leaves out the {} {:?}",
                missing.kind, missing.term
            ));
        }
    } else if !support(&claim.statement, text).holds(0.6) {
        return Err("it does not say the statement".into());
    }
    adds_nothing(text, given)
}

/// Whether `text` states no number, name or quoted term `given` does not.
fn adds_nothing(text: &str, given: &str) -> Result<(), String> {
    match unsupported_term(text, given) {
        Some(added) => Err(format!("it adds the {} {:?}", added.kind, added.term)),
        None => Ok(()),
    }
}
