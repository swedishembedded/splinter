// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model-written measurement items admitted by
// code for its clients. If your team needs expertise in generating test items
// a learning system cannot have seen, you can procure our services by
// sending an email to info@swedishembedded.com.

//! What the generator model writes about a fact, each item admitted by code
//! or sent back for rewriting: the four probes of a fact, and a paraphrase of
//! its statement (a true answer worded differently, a hard case for a judge).
//!
//! The generator is shown the statement and the question and nothing else: no
//! session exists when this runs, and no stage that reads sessions is ever
//! shown what it writes.

use std::time::Duration;

use anyhow::Context as _;
use serde::{Deserialize, Serialize};
use splinter_sdk::agent::schemars::{self, JsonSchema};
use splinter_sdk::agent::solve::Model;
use splinter_sdk::agent::typed::TypedCall;
use splinter_sdk::measure::verifiers::quotation::words;
use splinter_sdk::Context;

use crate::facts::Fact;
use crate::keys::{extract, missing, Key};
use crate::probes::{admit, shares_a_run, Probe, ProbeKind};

/// How long writing one fact's items may take, every correction included.
const WRITE_DEADLINE: Duration = Duration::from_secs(600);

/// Output tokens one call may run to.
const WRITE_MAX_OUTPUT_TOKENS: u64 = 1024;

/// Corrections a rejected reply is sent back for.
const REPAIRS: u32 = 4;

const ROLE: &str = "You write test questions about a single fact, for someone who is checking \
whether a person knows it.";

const PROBES_TASK: &str = "Write four questions that test whether someone knows the fact in \
`statement`. `paraphrase` asks what `question` asks, in different words. `reverse` starts from \
the answer and asks for what `question` is about: it contains at least one of the `answer_parts` word for word and asks \
who or what that concerns (for example: \"What did Boulton and Watt make for him?\"), so it can be answered only by someone who knows \
the fact. `indirect` is \
a question about something else that cannot be answered correctly without the fact. \
`application` is a short everyday scenario of one or two sentences that ends in a question whose \
right answer uses the fact. Except for `reverse`, a question never contains any of the `answer_parts`. No \
question quotes `statement`. Each ends with a question mark.";

const PARAPHRASE_TASK: &str = "Say again, in different words and in the same voice, what \
`statement` says. Keep every name, number and date it holds exactly, change the sentence \
structure and the other words, and add nothing.";

/// What the generator is shown.
#[derive(Serialize)]
struct Shown<'a> {
    statement: &'a str,
    question: &'a str,
    /// The words of the answer, which only `reverse` may contain.
    answer_parts: Vec<&'a str>,
}

/// The four questions the generator writes.
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[schemars(crate = "schemars")]
struct Written {
    /// What `question` asks, in different words.
    paraphrase: String,
    /// The question from the answer's end.
    reverse: String,
    /// A question that cannot be answered without the fact.
    indirect: String,
    /// A short scenario ending in a question that needs the fact.
    application: String,
}

/// The statement reworded.
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[schemars(crate = "schemars")]
struct Reworded {
    /// The statement in other words.
    statement: String,
}

/// What an answer to a probe must hold: the fact's keys, or, for a reverse
/// probe, the names the fact's question held (the persona's own excepted).
fn answer_keys(kind: ProbeKind, fact: &Fact, probe: &str, persona: &str) -> Vec<Key> {
    if kind != ProbeKind::Reverse {
        return fact.keys.clone();
    }
    let own = persona
        .split_whitespace()
        .last()
        .unwrap_or_default()
        .to_lowercase();
    extract(&fact.question, probe)
        .into_iter()
        .filter(|k| k.text.to_lowercase() != own)
        .collect()
}

/// The four probes of `fact`, admitted by [`admit`].
///
/// # Errors
/// The generator cannot run or its probes cannot be admitted within the
/// corrections allowed; the message names the fact.
pub fn write_probes(
    ctx: &Context,
    generator: &Model,
    fact: &Fact,
    persona: &str,
) -> anyhow::Result<Vec<Probe>> {
    let (question, statement, keys) = (
        fact.question.clone(),
        fact.statement.clone(),
        fact.keys.clone(),
    );
    let call = TypedCall::<Written>::new("write_probes", PROBES_TASK, ROLE, WRITE_DEADLINE)
        .max_output_tokens(WRITE_MAX_OUTPUT_TOKENS)
        .repairs(REPAIRS)
        .postcondition(move |w: &Written| {
            let all = [
                (ProbeKind::Paraphrase, &w.paraphrase),
                (ProbeKind::Reverse, &w.reverse),
                (ProbeKind::Indirect, &w.indirect),
                (ProbeKind::Application, &w.application),
            ];
            for (kind, q) in all {
                admit(kind, q.trim(), &question, &statement, &keys)?;
            }
            let distinct: std::collections::BTreeSet<String> =
                all.iter().map(|(_, q)| words(q).join(" ")).collect();
            if distinct.len() == all.len() {
                Ok(())
            } else {
                Err("the four questions must differ".to_string())
            }
        });
    let reply = ctx
        .block_on(call.run(
            generator,
            &Shown {
                statement: &fact.statement,
                question: &fact.question,
                answer_parts: fact.keys.iter().map(|k| k.text.as_str()).collect(),
            },
        ))
        .with_context(|| format!("writing the probes of fact {}", fact.id))?;
    Ok([
        (ProbeKind::Paraphrase, reply.paraphrase),
        (ProbeKind::Reverse, reply.reverse),
        (ProbeKind::Indirect, reply.indirect),
        (ProbeKind::Application, reply.application),
    ]
    .into_iter()
    .map(|(kind, q)| {
        let question = q.trim().to_string();
        Probe {
            fact: fact.id.clone(),
            kind,
            keys: answer_keys(kind, fact, &question, persona),
            question,
        }
    })
    .collect())
}

/// The statement of `fact` in other words, holding every key.
///
/// # Errors
/// The generator cannot run or does not keep the keys within the corrections
/// allowed.
pub fn write_paraphrase(ctx: &Context, generator: &Model, fact: &Fact) -> anyhow::Result<String> {
    let (statement, keys) = (fact.statement.clone(), fact.keys.clone());
    let call =
        TypedCall::<Reworded>::new("reword_statement", PARAPHRASE_TASK, ROLE, WRITE_DEADLINE)
            .max_output_tokens(WRITE_MAX_OUTPUT_TOKENS)
            .repairs(REPAIRS)
            .postcondition(move |r: &Reworded| {
                let lacking = missing(&r.statement, &keys);
                if !lacking.is_empty() {
                    return Err(format!(
                        "keep {} exactly",
                        lacking
                            .iter()
                            .map(|k| k.text.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ));
                }
                if shares_a_run(&r.statement, &statement) {
                    return Err(
                        "change more of the wording: do not repeat a stretch of the statement"
                            .into(),
                    );
                }
                Ok(())
            });
    let reply = ctx
        .block_on(call.run(
            generator,
            &Shown {
                statement: &fact.statement,
                question: &fact.question,
                answer_parts: fact.keys.iter().map(|k| k.text.as_str()).collect(),
            },
        ))
        .with_context(|| format!("rewording the statement of fact {}", fact.id))?;
    Ok(reply.statement.trim().to_string())
}
