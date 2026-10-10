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
use crate::probes::{admit, admit_reverse, shares_a_run, Probe, ProbeKind};

/// How long writing one fact's items may take, every correction included.
const WRITE_DEADLINE: Duration = Duration::from_secs(600);

/// Output tokens one call may run to.
const WRITE_MAX_OUTPUT_TOKENS: u64 = 1024;

/// Corrections a rejected reply is sent back for.
const REPAIRS: u32 = 3;

const ROLE: &str = "You write test questions about a single fact, for someone who is checking \
whether a person knows it.";

const PROBES_TASK: &str = "Write three questions that test whether someone knows the fact in \
`statement`. `paraphrase` asks what `question` asks, in different words. `indirect` is a question \
about something else that cannot be answered correctly without the fact. `application` is a short \
everyday scenario of one or two sentences that ends in a question whose right answer uses the \
fact. None of them contains any of the `answer_parts`, and none quotes `statement`. Each ends with \
a question mark.";

const REVERSE_TASK: &str = "Write one natural question about the fact in `statement`. The person \
asking already knows `known`, a part of the fact, and wants to find out one of the names in \
`choices`. The question must mention `known` word for word, must not mention the name it asks for, \
and can only be answered by someone who knows the fact. Example: statement \"Boulton and Watt made \
the press in 1785.\", known \"Watt\", choices [\"Boulton\"]: \"Which partner worked with Watt on the \
copying press?\" with asked \"Boulton\". Write `approach` first (one sentence), then `question`, \
then `asked`: the one name of `choices` the question asks for, copied exactly. Write the question \
as ordinary prose, with none of these field names in it. Do not quote `statement`. End with a \
question mark.";

const PARAPHRASE_TASK: &str = "Say again, in different words and in the same voice, what \
`statement` says. Keep every name, number and date it holds exactly, change the sentence \
structure and the other words, and add nothing.";

/// What the generator is shown.
#[derive(Serialize)]
struct Shown<'a> {
    statement: &'a str,
    question: &'a str,
    /// The words of the answer, which no question but the reverse one may contain.
    answer_parts: Vec<&'a str>,
}

/// What the generator is shown to write the reverse question.
#[derive(Serialize)]
struct ShownReverse<'a> {
    statement: &'a str,
    known: &'a str,
    choices: Vec<String>,
}

/// The reverse question the generator writes.
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[schemars(crate = "schemars")]
struct WrittenReverse {
    /// One sentence on how the question turns `question` around: what it
    /// gives (`known`) and what it asks for.
    approach: String,
    /// The question: it contains `known` exactly and not the name it asks for.
    question: String,
    /// The one name of `choices` the question asks for, as written there.
    asked: String,
}

/// The three other questions the generator writes.
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[schemars(crate = "schemars")]
struct Written {
    /// What `question` asks, in different words.
    paraphrase: String,
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

/// The names of the fact's question that a reverse question may ask for: the
/// names the question holds, the persona's own excepted.
pub(crate) fn reverse_choices(fact: &Fact, persona: &str) -> Vec<Key> {
    let own = persona
        .split_whitespace()
        .last()
        .unwrap_or_default()
        .to_lowercase();
    extract(&fact.question, "")
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
                Err("the three questions must differ".to_string())
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
    let (reverse, asked) = write_reverse(ctx, generator, fact, persona)?;
    Ok([
        (ProbeKind::Paraphrase, reply.paraphrase, None),
        (ProbeKind::Reverse, reverse, Some(asked)),
        (ProbeKind::Indirect, reply.indirect, None),
        (ProbeKind::Application, reply.application, None),
    ]
    .into_iter()
    .map(|(kind, q, asked)| Probe {
        fact: fact.id.clone(),
        kind,
        // A reverse probe requires only the name it asks for.
        keys: asked.map_or_else(|| fact.keys.clone(), |key| vec![key]),
        avoid: Vec::new(),
        question: q.trim().to_string(),
    })
    .collect())
}

/// The reverse question of `fact` and the one name of the fact's question it
/// asks for: it names a key of the answer and asks for that name. Written
/// alone, in a call of its own, since it is the one question that must hold
/// what the others must not. A writer that cannot keep to it fails the fact:
/// no question is made by code in its place, since one that names the answer
/// without saying what it asks for cannot be graded.
fn write_reverse(
    ctx: &Context,
    generator: &Model,
    fact: &Fact,
    persona: &str,
) -> anyhow::Result<(String, Key)> {
    let choices = reverse_choices(fact, persona);
    anyhow::ensure!(
        !choices.is_empty(),
        "the question of fact {} holds no name for a reverse question to ask for",
        fact.id
    );
    let (question, statement, keys, offered) = (
        fact.question.clone(),
        fact.statement.clone(),
        fact.keys.clone(),
        choices.clone(),
    );
    let call =
        TypedCall::<WrittenReverse>::new("write_reverse", REVERSE_TASK, ROLE, WRITE_DEADLINE)
            .max_output_tokens(WRITE_MAX_OUTPUT_TOKENS)
            .repairs(REPAIRS)
            .postcondition(move |w: &WrittenReverse| {
                let admitted = admit_reverse(
                    w.question.trim(),
                    w.asked.trim(),
                    &question,
                    &statement,
                    &keys,
                    &offered,
                )
                .map(|_| ());
                if let Err(why) = &admitted {
                    eprintln!("reverse question refused ({why}): {}", w.question.trim());
                }
                admitted
            });
    let reply = ctx
        .block_on(call.run(
            generator,
            &ShownReverse {
                statement: &fact.statement,
                known: fact.keys.first().map_or("", |k| k.text.as_str()),
                choices: choices.iter().map(|k| k.text.clone()).collect(),
            },
        ))
        .with_context(|| format!("writing the reverse probe of fact {}", fact.id))?;
    let asked = admit_reverse(
        reply.question.trim(),
        reply.asked.trim(),
        &fact.question,
        &fact.statement,
        &fact.keys,
        &choices,
    )
    .map_err(anyhow::Error::msg)?;
    Ok((reply.question.trim().to_string(), asked))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::facts::{admit as admit_fact, Evidence, Origin, Proposal};
    use crate::keys::KeyKind;
    use crate::roles::Quotas;

    fn fact() -> Fact {
        admit_fact(Proposal {
            seed: 1,
            quotas: &Quotas::PROTOCOL,
            origin: Origin::Letters,
            family: "f.txt",
            question: "Which doctor is in Philadelphia?",
            statement: "Dr. Bache is in Philadelphia, at his house on Market Street.",
            keys: vec![Key {
                text: "Bache".into(),
                kind: KeyKind::Name,
            }],
            answer_type: "name",
            entities: vec![],
            evidence: Evidence {
                quote: String::new(),
                source: "f.txt".into(),
            },
        })
        .expect("a fact")
    }

    #[test]
    fn a_reverse_question_may_ask_for_the_names_of_the_facts_question_but_not_the_persona() {
        let choices = reverse_choices(&fact(), "Thomas Jefferson");
        let names: Vec<&str> = choices.iter().map(|k| k.text.as_str()).collect();
        assert_eq!(names, ["Philadelphia"]);
        assert!(reverse_choices(&fact(), "Dr Philadelphia").is_empty());
    }
}
