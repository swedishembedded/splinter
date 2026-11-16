// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements synthetic, license-free training corpora
// generated deterministically from a seed, for its clients. If your team
// needs expertise in data generation for small-model pretraining, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The synthetic world the model is pretrained on.
//!
//! Everything is a function of the run's seed: towns with their people and
//! goods described in plain sentences, arithmetic expressions with their
//! values, tiny programs with their outputs, and conversations in the model's
//! own chat template, some of them tool use in exactly the shape the agent
//! will see. Nothing is downloaded and nothing is stored in the repository;
//! the same seed writes the same bytes.

use serde::Serialize;
use splinter_sdk::vocabulary::rng::SplitMix64;

use crate::config::WorldSize;
use crate::template::{Call, Template, ToolSig, Turn};

const ONSETS: &[&str] = &[
    "b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", "br", "st", "tr", "gl",
];
const VOWELS: &[&str] = &["a", "e", "i", "o", "u", "ai", "ou"];
const CODAS: &[&str] = &["", "n", "r", "l", "s", "m", "th"];
const TRADES: &[&str] = &[
    "baker", "smith", "weaver", "miller", "potter", "fisher", "carter", "brewer",
];
const GOODS: &[&str] = &[
    "bread", "nails", "cloth", "flour", "jugs", "fish", "carts", "ale",
];
const COLOURS: &[&str] = &["red", "blue", "green", "amber", "grey", "white"];
const FEATURES: &[&str] = &[
    "a river",
    "a hill",
    "the sea",
    "a forest",
    "a lake",
    "the old road",
];

/// One document of the corpus.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Document {
    /// What kind of text it is (`world`, `expression`, `program`, `chat`).
    pub kind: &'static str,
    /// The text.
    pub text: String,
}

/// A pronounceable made-up word.
pub fn word(rng: &mut SplitMix64, syllables: usize) -> String {
    let mut w = String::new();
    for _ in 0..syllables {
        w.push_str(rng.pick(ONSETS).copied().unwrap_or("b"));
        w.push_str(rng.pick(VOWELS).copied().unwrap_or("a"));
        w.push_str(rng.pick(CODAS).copied().unwrap_or(""));
    }
    capitalise(&w)
}

fn capitalise(w: &str) -> String {
    let mut c = w.chars();
    c.next()
        .map_or_else(String::new, |f| f.to_uppercase().chain(c).collect())
}

fn pick<'a>(rng: &mut SplitMix64, items: &[&'a str]) -> &'a str {
    rng.pick(items)
        .copied()
        .unwrap_or_else(|| unreachable!("the word lists are not empty"))
}

/// One town's description, in a few documents.
fn town(rng: &mut SplitMix64) -> Vec<Document> {
    let name = word(rng, 2);
    let feature = pick(rng, FEATURES);
    let people: Vec<(String, &str, &str)> = (0..3)
        .map(|_| {
            (
                format!("{} {}", word(rng, 1), word(rng, 2)),
                pick(rng, TRADES),
                pick(rng, COLOURS),
            )
        })
        .collect();
    let mayor = &people[0].0;
    let mut docs = vec![Document {
        kind: "world",
        text: format!(
            "{name} is a town by {feature}. Its mayor is {mayor}. {} people live in {name}.",
            rng.between(40, 900)
        ),
    }];
    for (person, trade, colour) in &people {
        let good = GOODS[TRADES.iter().position(|t| t == trade).unwrap_or(0) % GOODS.len()];
        docs.push(Document {
            kind: "world",
            text: format!(
                "{person} is a {trade} in {name}. {person} makes {good} and likes the colour {colour}. \
                 Every week {person} sells {} {good}.",
                rng.between(2, 60)
            ),
        });
    }
    docs
}

/// An arithmetic expression and its value.
fn expression(rng: &mut SplitMix64) -> Document {
    let a = rng.between(0, 99) as i64;
    let b = rng.between(1, 20) as i64;
    let c = rng.between(0, 30) as i64;
    let (text, value) = match rng.below(4) {
        0 => (format!("{a} + {b} * {c}"), a + b * c),
        1 => (format!("({a} - {b}) * {c}"), (a - b) * c),
        2 => (format!("{a} * {b} - {c}"), a * b - c),
        _ => (format!("{a} + {b} + {c}"), a + b + c),
    };
    Document {
        kind: "expression",
        text: format!("{text} = {value}"),
    }
}

/// A tiny program and what it prints.
fn program(rng: &mut SplitMix64) -> Document {
    let x = rng.between(1, 12) as i64;
    let k = rng.between(2, 9) as i64;
    let d = rng.between(0, 15) as i64;
    let y = x * k;
    let out = y + d;
    Document {
        kind: "program",
        text: format!("x = {x}\ny = x * {k}\nprint(y + {d})\n# output: {out}"),
    }
}

/// A conversation: a question about the world answered, or a small task
/// done with tools in the shape the agent uses.
fn conversation(
    rng: &mut SplitMix64,
    template: &Template,
    tools: &[ToolSig],
    system: &str,
) -> Document {
    let (tools, turns) = if rng.coin() {
        let town = word(rng, 2);
        let mayor = format!("{} {}", word(rng, 1), word(rng, 2));
        let turns = vec![
            Turn::User(format!(
                "{town} has a mayor called {mayor}. Who is the mayor of {town}?"
            )),
            Turn::Assistant {
                text: format!("The mayor of {town} is {mayor}."),
                calls: vec![],
            },
        ];
        (&[][..], turns)
    } else {
        let file = format!("{}.txt", word(rng, 1).to_lowercase());
        let value = rng.between(100, 999);
        let turns = vec![
            Turn::System(system.to_string()),
            Turn::User(format!("What number is in {file}? Read it.")),
            Turn::Assistant {
                text: String::new(),
                calls: vec![Call {
                    name: "read_file".into(),
                    arguments: serde_json::json!({ "path": file }),
                }],
            },
            Turn::Tool(format!("L1:{value}")),
            Turn::Assistant {
                text: format!("{file} holds {value}."),
                calls: vec![],
            },
        ];
        (tools, turns)
    };
    Document {
        kind: "chat",
        text: template.render(tools, &turns),
    }
}

/// The whole corpus of a world of `size`, drawn from `seed`; its tool-use
/// conversations offer `tools` under the agent's `system` prompt.
#[must_use]
pub fn corpus(
    seed: u64,
    size: &WorldSize,
    template: &Template,
    tools: &[ToolSig],
    system: &str,
) -> Vec<Document> {
    let mut rng = SplitMix64::new(seed);
    let mut docs = Vec::new();
    for _ in 0..size.towns {
        docs.extend(town(&mut rng));
    }
    docs.extend((0..size.expressions).map(|_| expression(&mut rng)));
    docs.extend((0..size.programs).map(|_| program(&mut rng)));
    docs.extend((0..size.conversations).map(|_| conversation(&mut rng, template, tools, system)));
    rng.shuffle(&mut docs);
    docs
}

/// One worked tool-use conversation of `instance`: the expert's procedure,
/// each tool run for real in a scratch copy of the workspace at `scratch`,
/// written in the chat template. Pretraining on these shows the model the
/// protocol's shape; the instances are drawn from a stream of their own, so
/// none is a task the model is probed or examined on.
///
/// # Errors
/// The workspace cannot be written or a command cannot be run.
pub fn trajectory(
    instance: &crate::tasks::Instance,
    template: &Template,
    tools: &[ToolSig],
    system: &str,
    scratch: &std::path::Path,
) -> anyhow::Result<Document> {
    use crate::expert::{next, Action};
    use serde_json::{json, Value};
    if scratch.exists() {
        std::fs::remove_dir_all(scratch)?;
    }
    crate::tasks::materialise(instance, scratch)?;
    let wire_tools: Vec<Value> = tools
        .iter()
        .map(|t| {
            let props: serde_json::Map<String, Value> = t.params.iter().map(|p| (p.clone(), json!({}))).collect();
            json!({ "type": "function", "function": { "name": t.name, "parameters": { "properties": props } } })
        })
        .collect();
    let mut messages = vec![json!({ "role": "user", "content": instance.instruction })];
    let mut turns = vec![
        Turn::System(system.to_string()),
        Turn::User(instance.instruction.clone()),
    ];
    for _ in 0..8 {
        match next(instance, &messages, &wire_tools)? {
            Action::Finish(text) => {
                turns.push(Turn::Assistant {
                    text,
                    calls: vec![],
                });
                break;
            }
            Action::Call { tool, arguments } => {
                let arg = |k: &str| arguments[k].as_str().unwrap_or_default().to_string();
                let result = match tool.as_str() {
                    "read_file" => std::fs::read_to_string(scratch.join(arg("path")))?
                        .lines()
                        .enumerate()
                        .map(|(i, l)| format!("L{}:{l}", i + 1))
                        .collect::<Vec<_>>()
                        .join("\n"),
                    "write_file" => {
                        std::fs::write(scratch.join(arg("path")), arg("text"))?;
                        format!("wrote {} bytes to {}", arg("text").len(), arg("path"))
                    }
                    _ => {
                        let out = std::process::Command::new("sh")
                            .arg("-c")
                            .arg(arg("shell_command"))
                            .current_dir(scratch)
                            .output()?;
                        String::from_utf8_lossy(&out.stdout).trim_end().to_string()
                    }
                };
                turns.push(Turn::Assistant {
                    text: String::new(),
                    calls: vec![Call {
                        name: tool,
                        arguments,
                    }],
                });
                turns.push(Turn::Tool(result.clone()));
                messages.push(json!({ "role": "tool", "content": result }));
            }
        }
    }
    Ok(Document {
        kind: "trajectory",
        text: template.render(tools, &turns),
    })
}

/// The corpus as JSON lines, one `{"text": ...}` document per line.
#[must_use]
pub fn to_jsonl(docs: &[Document]) -> String {
    let mut out = String::new();
    for doc in docs {
        out.push_str(&serde_json::json!({ "text": doc.text }).to_string());
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn size() -> WorldSize {
        WorldSize {
            towns: 3,
            expressions: 10,
            programs: 5,
            conversations: 6,
            trajectories: 0,
        }
    }

    #[test]
    fn the_same_seed_writes_the_same_world_and_another_seed_another() {
        let t = Template::default();
        // seed: determinism is the property under test; any two seeds work.
        let a = corpus(11, &size(), &t, &[], "s");
        assert_eq!(a, corpus(11, &size(), &t, &[], "s"));
        assert_ne!(a, corpus(12, &size(), &t, &[], "s"));
        assert_eq!(a.len(), 3 * 4 + 10 + 5 + 6);
    }

    #[test]
    fn every_expression_states_its_true_value() {
        let t = Template::default();
        // seed: any seed; every expression drawn must be correct.
        for doc in corpus(
            5,
            &WorldSize {
                towns: 0,
                expressions: 200,
                programs: 0,
                conversations: 0,
                trajectories: 0,
            },
            &t,
            &[],
            "s",
        ) {
            let (lhs, rhs) = doc.text.split_once(" = ").unwrap();
            let value: i64 = rhs.parse().unwrap();
            assert_eq!(eval(lhs), value, "{}", doc.text);
        }
    }

    /// A reference evaluator independent of the generator: + - * and
    /// parentheses with the usual precedence.
    fn eval(expr: &str) -> i64 {
        fn parse_sum(t: &[&str], i: &mut usize) -> i64 {
            let mut v = parse_product(t, i);
            while *i < t.len() && (t[*i] == "+" || t[*i] == "-") {
                let op = t[*i];
                *i += 1;
                let r = parse_product(t, i);
                v = if op == "+" { v + r } else { v - r };
            }
            v
        }
        fn parse_product(t: &[&str], i: &mut usize) -> i64 {
            let mut v = parse_atom(t, i);
            while *i < t.len() && t[*i] == "*" {
                *i += 1;
                v *= parse_atom(t, i);
            }
            v
        }
        fn parse_atom(t: &[&str], i: &mut usize) -> i64 {
            if t[*i] == "(" {
                *i += 1;
                let v = parse_sum(t, i);
                *i += 1;
                v
            } else {
                *i += 1;
                t[*i - 1].parse().unwrap()
            }
        }
        let spaced = expr.replace('(', "( ").replace(')', " )");
        let tokens: Vec<&str> = spaced.split_whitespace().collect();
        parse_sum(&tokens, &mut 0)
    }
}
