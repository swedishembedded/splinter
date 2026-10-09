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
const REPAIRS: u32 = 3;

const ROLE: &str = "You write test questions about a single fact, for someone who is checking \
whether a person knows it.";

const PROBES_TASK: &str = "Write three questions that test whether someone knows the fact in \
`statement`. `paraphrase` asks what `question` asks, in different words. `indirect` is a question \
about something else that cannot be answered correctly without the fact. `application` is a short \
everyday scenario of one or two sentences that ends in a question whose right answer uses the \
fact. None of them contains any of the `answer_parts`, and none quotes `statement`. Each ends with \
a question mark.";

const REVERSE_TASK: &str = "Write one question about the fact in `statement` that starts from \
one part of it. The question MUST contain `include_exactly` exactly as written and asks about \
another part of the statement, so that only someone who knows the fact can answer it; if \
`asks_for` lists names, ask for one of them. Example: statement \"Boulton and Watt made the press \
in 1785.\", include_exactly \"Watt\", asks_for [\"Boulton\"]: \"Which partner worked with Watt on the \
copying press?\". Write `approach` first (one sentence), then `question`. Do not quote \
`statement`. End with a question mark.";

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
    include_exactly: &'a str,
    asks_for: Vec<String>,
}

/// The reverse question the generator writes.
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[schemars(crate = "schemars")]
struct WrittenReverse {
    /// One sentence on how the question turns `question` around: what it
    /// gives (`include_exactly`) and what it asks for.
    approach: String,
    /// The question: it contains `include_exactly` exactly.
    question: String,
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

/// What an answer to a probe must hold: the fact's keys, or, for a reverse
/// probe, the names the fact's question held (the persona's own excepted).
pub(crate) fn answer_keys(kind: ProbeKind, fact: &Fact, probe: &str, persona: &str) -> Vec<Key> {
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
    let reverse = write_reverse(ctx, generator, fact, persona)?;
    Ok([
        (ProbeKind::Paraphrase, reply.paraphrase),
        (ProbeKind::Reverse, reverse),
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
            avoid: Vec::new(),
            question,
        }
    })
    .collect())
}

/// The reverse question of `fact`: it names a key of the answer and asks for
/// what the fact's question is about. Written alone, in a call of its own,
/// since it is the one question that must hold what the others must not.
fn write_reverse(
    ctx: &Context,
    generator: &Model,
    fact: &Fact,
    persona: &str,
) -> anyhow::Result<String> {
    let (question, statement, keys) = (
        fact.question.clone(),
        fact.statement.clone(),
        fact.keys.clone(),
    );
    let call =
        TypedCall::<WrittenReverse>::new("write_reverse", REVERSE_TASK, ROLE, WRITE_DEADLINE)
            .max_output_tokens(WRITE_MAX_OUTPUT_TOKENS)
            .repairs(REPAIRS)
            .postcondition(move |w: &WrittenReverse| {
                let admitted = admit(
                    ProbeKind::Reverse,
                    w.question.trim(),
                    &question,
                    &statement,
                    &keys,
                );
                if let Err(why) = &admitted {
                    eprintln!("reverse question refused ({why}): {}", w.question.trim());
                }
                admitted
            });
    let asks_for: Vec<String> = answer_keys(ProbeKind::Reverse, fact, "", persona)
        .into_iter()
        .map(|k| k.text)
        .collect();
    let written = ctx.block_on(call.run(
        generator,
        &ShownReverse {
            statement: &fact.statement,
            include_exactly: fact.keys.first().map_or("", |k| k.text.as_str()),
            asks_for: asks_for.clone(),
        },
    ));
    match written {
        Ok(reply) => Ok(reply.question.trim().to_string()),
        // A model that cannot keep a question to a given word is not asked
        // again: the question is built from the key and the names the fact's
        // question holds. It is plainer, and it names the answer.
        Err(e) => fallback_reverse(fact, &asks_for)
            .with_context(|| format!("writing the reverse probe of fact {}: {e:#}", fact.id)),
    }
}

/// A reverse question made by code: it names the first key and asks what is
/// remembered of it beside the names the fact's question holds.
fn fallback_reverse(fact: &Fact, asks_for: &[String]) -> anyhow::Result<String> {
    let key = fact.keys.first().context("the fact has no key")?;
    anyhow::ensure!(
        !asks_for.is_empty(),
        "the fact's question holds no name to ask for"
    );
    let question = format!(
        "What do you recall about {} in connection with {}?",
        key.text,
        asks_for.join(" and ")
    );
    admit(
        ProbeKind::Reverse,
        &question,
        &fact.question,
        &fact.statement,
        &fact.keys,
    )
    .map_err(anyhow::Error::msg)?;
    Ok(question)
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
    fn a_reverse_question_made_by_code_names_the_key_and_asks_for_the_names_of_the_question() {
        let fact = fact();
        let asks_for = vec!["Philadelphia".to_string()];
        let question = fallback_reverse(&fact, &asks_for).expect("a question");
        assert_eq!(
            question,
            "What do you recall about Bache in connection with Philadelphia?"
        );
        // Nothing to ask for, nothing to build.
        assert!(fallback_reverse(&fact, &[]).is_err());
    }
}
