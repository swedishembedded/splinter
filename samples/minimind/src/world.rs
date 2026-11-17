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
use crate::template::{to_json, Call, Template, ToolSig, Turn};

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
    // Ranges wide enough that a corpus of programs repeats almost none.
    let x = rng.between(1, 99) as i64;
    let k = rng.between(2, 19) as i64;
    let d = rng.between(0, 99) as i64;
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
    notice: &str,
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
            Turn::User(with_time_notice(
                &format!("What number is in {file}? Read it.\n"),
                notice,
            )),
            Turn::Assistant {
                text: String::new(),
                calls: vec![Call {
                    name: "read_file".into(),
                    arguments: serde_json::json!({ "path": file }),
                }],
            },
            Turn::Tool(with_time_notice(&format!("L1:{value}"), notice)),
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

/// A drill in copying values out of context, the skill every task needs:
/// a listing of names with codes (as a file read shows it) followed by
/// questions answered from it; values from files written into a command; or
/// a short conversation that reads a value and writes it to the answer file
/// the request names, with the tools, in the template. Paths and answer
/// files are drawn as the tasks draw them, so what is copied varies.
fn recall(rng: &mut SplitMix64, template: &Template) -> Document {
    let entries: Vec<(String, u64)> = (0..rng.between(3, 8))
        .map(|_| (word(rng, 2).to_lowercase(), rng.between(10, 9999)))
        .collect();
    // A listing as a read shows it, in one of the training formats.
    let format = crate::formats::TRAINING[rng.below(crate::formats::TRAINING.len())];
    let as_strings: Vec<(String, String)> = entries
        .iter()
        .map(|(n, c)| (n.clone(), c.to_string()))
        .collect();
    let listing = format
        .render(&as_strings)
        .lines()
        .enumerate()
        .map(|(i, l)| format!("L{}:{l}", i + 1))
        .collect::<Vec<_>>()
        .join("\n");
    let path = crate::skills::data_path(rng, format, false, &[]);
    let text = match rng.below(4) {
        3 => {
            let (name, _) = &entries[rng.below(entries.len())];
            format!(
                "Read the number of {name} in {path}.\nread_file {}\n",
                to_json(&serde_json::json!({ "path": path }))
            )
        }
        0 => {
            let mut text = format!("{listing}\n");
            for _ in 0..3 {
                let (name, code) = &entries[rng.below(entries.len())];
                text.push_str(&format!("The code of {name} is {code}.\n"));
            }
            text
        }
        1 => {
            let (a, b) = (&entries[0], &entries[entries.len() - 1]);
            format!(
                "{path} holds {} as {} and {} as {}.\n./calc combine {} {}\n",
                a.0, a.1, b.0, b.1, a.1, b.1
            )
        }
        _ => {
            let (name, code) = entries[rng.below(entries.len())].clone();
            let answer = crate::skills::answer_file(rng, std::slice::from_ref(&path));
            let w = &crate::phrasing::TRAIN;
            let step = format!(
                "{} {}",
                crate::phrasing::pick(rng, w.find),
                crate::phrasing::of(rng, w, &name, &path)
            );
            let request = crate::phrasing::with_answer(rng, w, &step, "it", &answer);
            let value = code.to_string();
            let turns = vec![
                Turn::User(request),
                Turn::Assistant {
                    text: String::new(),
                    calls: vec![Call {
                        name: "read_file".into(),
                        arguments: serde_json::json!({ "path": path }),
                    }],
                },
                Turn::Tool(listing),
                Turn::Assistant {
                    text: String::new(),
                    calls: vec![Call {
                        name: "write_file".into(),
                        arguments: serde_json::json!({ "path": answer, "text": value }),
                    }],
                },
                Turn::Tool(format!("wrote {} bytes to {answer}", value.len())),
                Turn::Assistant {
                    text: value,
                    calls: vec![],
                },
            ];
            template.render(&[], &turns)
        }
    };
    Document {
        kind: "recall",
        text,
    }
}

/// A shell command's result as the agent's shell tool returns it: the
/// output, then its status line (the duration as a record states it).
#[must_use]
pub fn shell_result(stdout: &str, exit: i32) -> String {
    let status = format!("Exit: {exit} | Duration: 0.0s");
    let stdout = stdout.trim_end();
    if stdout.is_empty() {
        status
    } else {
        format!("{stdout}\n\n{status}")
    }
}

/// The line the agent runtime ends the task and every tool result with when
/// a run has `deadline_secs` (sven's wording, at the start of the run: a
/// small model's episode ends within its first minute), or nothing for a
/// run without a deadline.
#[must_use]
pub fn time_notice(deadline_secs: u64) -> String {
    if deadline_secs == 0 {
        return String::new();
    }
    let total = deadline_secs.div_ceil(60);
    format!("{TIME_PREFIX}0 of {total} min used, {total} min left{TIME_SUFFIX}")
}

const TIME_PREFIX: &str = "Time: ";
const TIME_SUFFIX: &str = "; leave time to verify and finish.";
const TIME_SEPARATOR: &str = "\n\n";

/// `text` with `notice` appended as the runtime appends it; `text` itself
/// for no notice.
#[must_use]
pub fn with_time_notice(text: &str, notice: &str) -> String {
    if notice.is_empty() {
        text.to_string()
    } else {
        format!("{text}{TIME_SEPARATOR}{notice}")
    }
}

/// `text` without the time notice the runtime appended, as the tool or the
/// person wrote it; `text` itself when it ends with none.
#[must_use]
pub fn without_time_notice(text: &str) -> &str {
    match text.rsplit_once(TIME_SEPARATOR) {
        Some((head, line)) if line.starts_with(TIME_PREFIX) && line.ends_with(TIME_SUFFIX) => head,
        _ => text,
    }
}

/// The whole corpus of a world of `size`, drawn from `seed`; its tool-use
/// conversations offer `tools` under the agent's `system` prompt, their
/// request and tool results ending with `notice` as an agent run's do.
#[must_use]
pub fn corpus(
    seed: u64,
    size: &WorldSize,
    template: &Template,
    tools: &[ToolSig],
    system: &str,
    notice: &str,
) -> Vec<Document> {
    let mut rng = SplitMix64::new(seed);
    let mut docs = Vec::new();
    for _ in 0..size.towns {
        docs.extend(town(&mut rng));
    }
    docs.extend((0..size.expressions).map(|_| expression(&mut rng)));
    docs.extend((0..size.programs).map(|_| program(&mut rng)));
    docs.extend(
        (0..size.conversations).map(|_| conversation(&mut rng, template, tools, system, notice)),
    );
    docs.extend((0..size.recall).map(|_| recall(&mut rng, template)));
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
    notice: &str,
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
    let request = with_time_notice(&crate::tasks::instruction_text(instance), notice);
    let mut messages = vec![json!({ "role": "user", "content": request })];
    let mut turns = vec![Turn::System(system.to_string()), Turn::User(request)];
    for _ in 0..16 {
        match next(instance, &messages, &wire_tools)? {
            Action::Finish(text) => {
                turns.push(Turn::Assistant {
                    text,
                    calls: vec![],
                });
                break;
            }
            Action::Call { tool, arguments } => {
                let result = crate::expert::execute(scratch, &tool, &arguments)?;
                turns.push(Turn::Assistant {
                    text: String::new(),
                    calls: vec![Call {
                        name: tool,
                        arguments,
                    }],
                });
                let result = with_time_notice(&result, notice);
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
            recall: 0,
        }
    }

    #[test]
    fn the_same_seed_writes_the_same_world_and_another_seed_another() {
        let t = Template::default();
        // seed: determinism is the property under test; any two seeds work.
        let a = corpus(11, &size(), &t, &[], "s", "");
        assert_eq!(a, corpus(11, &size(), &t, &[], "s", ""));
        assert_ne!(a, corpus(12, &size(), &t, &[], "s", ""));
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
                recall: 0,
            },
            &t,
            &[],
            "s",
            "",
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
