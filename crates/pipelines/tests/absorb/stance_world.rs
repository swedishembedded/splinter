// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A night in which the model also has to be selective: a pool of invented
//! facts the policy knows (two it does not), invented questions nobody can
//! answer, and one scripted model that plays every role of the run - writer
//! of facts and questions, teacher that holds, policy that answers.

use std::sync::Arc;

use serde_json::json;
use splinter_agent::solve::Model;
use splinter_core::model_ref::ModelRef;
use splinter_pipelines::absorb::stance::{
    StanceRequest, ACCEPT_INSTRUCTION, FACTS_BRIEF, HOLD_INSTRUCTION, UNKNOWABLE_BRIEF,
    YIELD_INSTRUCTION,
};
use splinter_pipelines::absorb::AbsorbRequest;
use splinter_pipelines::abstain::UNKNOWABLE_TASK;
use splinter_pipelines::release::arm;
use std::path::PathBuf;

use crate::behave::{self, turns, Behaviour};
use crate::common::gate::{gate_context, Brain};
use crate::common::Scripted;
use crate::night::{night_reply, service_pattern, Night};

/// One invented fact of the pool.
pub struct Fact {
    pub question: String,
    pub statement: String,
    /// Whether the policy answers it right.
    pub known: bool,
    /// Whether the teacher gives in to the person when asked to hold.
    pub poisoned: bool,
}

const PLACES: [&str; 14] = [
    "Harrow", "Veldt", "Orlen", "Marsh", "Tarn", "Quill", "Dunmore", "Ashby", "Corvin", "Elsmere",
    "Fenwick", "Garrow", "Holt", "Ivory",
];
const BUILDERS: [&str; 6] = [
    "Pellew",
    "Brandt",
    "Okafor",
    "Lindqvist",
    "Moreau",
    "Tanaka",
];

/// Fourteen facts: eight about a year, six about a builder. The last two
/// are not known to the policy; every fourth gives in when it is the
/// teacher.
pub fn facts() -> Vec<Fact> {
    (0..PLACES.len())
        .map(|i| {
            let place = PLACES[i];
            let (question, statement) = if i < 8 {
                (
                    format!("In which year did the {place} Canal open?"),
                    format!("The {place} Canal opened in {}.", 1803 + 7 * i),
                )
            } else {
                (
                    format!("Who built the {place} Canal?"),
                    format!("The {place} Canal was built by {}.", BUILDERS[i - 8]),
                )
            };
            Fact {
                question,
                statement,
                known: i < 12,
                poisoned: i % 4 == 0,
            }
        })
        .collect()
}

/// Twelve questions about what nobody could know.
pub fn unknowable() -> Vec<String> {
    let given = ["Obadiah", "Lemuel", "Ptolemy", "Zacharias"];
    let surname = ["Pennywell", "Ashgrove", "Quillhart"];
    (0..12)
        .map(|i| {
            format!(
                "What did {} {} pay for his boat on the {} of May?",
                given[i % given.len()],
                surname[i / given.len()],
                i + 3
            )
        })
        .collect()
}

fn facts_json() -> String {
    let facts: Vec<_> = facts()
        .iter()
        .map(|f| json!({ "question": f.question, "statement": f.statement }))
        .collect();
    json!({ "facts": facts }).to_string()
}

fn unknowable_json() -> String {
    json!({ "prompts": unknowable() }).to_string()
}

/// The words a user turn has before the private note the teacher is given.
fn before_note(turn: &str) -> &str {
    turn.split("\n\n").next().unwrap_or(turn)
}

/// The first sentence of a brief: the part of it a prompt shows as it is,
/// since the quotes of the JSON shape that ends the brief are escaped there.
fn first_sentence(brief: &str) -> &str {
    brief.split(". ").next().unwrap_or(brief)
}

/// What the stance roles of the run say; `None` for anything else.
fn stance_reply(prompt: &str) -> Option<String> {
    if prompt.contains(FACTS_BRIEF) {
        return Some(facts_json());
    }
    if prompt.contains(first_sentence(UNKNOWABLE_BRIEF)) {
        return Some(unknowable_json());
    }
    if prompt.contains(UNKNOWABLE_TASK) {
        return Some(json!({ "reply": DECLINE_WRITTEN }).to_string());
    }
    let turns = turns(prompt);
    let last = turns.last().map(|(_, text)| text.as_str())?;
    if last.contains(HOLD_INSTRUCTION) {
        let fact = facts().into_iter().find(|f| last.contains(&f.statement))?;
        return Some(if fact.poisoned {
            format!("You are right: {}", before_note(last))
        } else {
            format!("With respect, I hold to this: {}", fact.statement)
        });
    }
    if last.contains(YIELD_INSTRUCTION) {
        return Some(format!("You are right: {}", before_note(last)));
    }
    if last.contains(ACCEPT_INSTRUCTION) {
        return Some("Noted, thank you.".into());
    }
    let asked = turns.iter().filter(|(role, _)| role == "User").count() == 1;
    asked
        .then(|| facts().into_iter().find(|f| f.question == last))
        .flatten()
        .map(|f| {
            if f.known {
                f.statement
            } else {
                "I do not know.".into()
            }
        })
}

/// What the decline writer says of an unknowable question.
pub const DECLINE_WRITTEN: &str =
    "I have no record of that matter and cannot say what was paid, so I will not guess at it here.";

/// The model that plays every role of the run.
pub fn stance_policy() -> Scripted {
    let svc = service_pattern();
    Scripted::new(move |prompt| stance_reply(prompt).unwrap_or_else(|| night_reply(prompt, &svc)))
}

/// A night whose base stands as `base` does when pushed.
pub fn stance_night(test: &str, base: Behaviour) -> Night {
    let (scratch, ctx) = gate_context(test, Brain::Honest);
    let policy = stance_policy();
    ctx.add_model(
        ModelRef::policy_default(),
        Model::new(Arc::new(policy.clone()), "scripted/night"),
    );
    ctx.add_model(
        arm(ctx.config(), None),
        behave::model(Vec::new(), base, "scripted/base"),
    );
    Night {
        scratch,
        ctx,
        policy,
    }
}

/// A night's request with `pairs` stance pairs asked for.
pub fn stance_request(night: &Night, dir: PathBuf, pairs: usize) -> AbsorbRequest {
    AbsorbRequest {
        stance: StanceRequest {
            pairs,
            ..StanceRequest::default()
        },
        ..night.request(dir)
    }
}

/// The text of a record's field, whatever the role.
pub fn text_of(turn: &serde_json::Value) -> String {
    turn["content"].as_str().unwrap_or_default().to_string()
}

/// One pair of the stance dataset, read back from its line.
pub struct PairLine {
    pub prompt: Vec<String>,
    pub chosen: String,
    pub rejected: String,
}

/// The pairs a dataset file holds.
pub fn pairs_in(path: &std::path::Path) -> Vec<PairLine> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let value: serde_json::Value = serde_json::from_str(line).unwrap();
            PairLine {
                prompt: value["prompt"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|m| m["role"] != "system")
                    .map(text_of)
                    .collect(),
                chosen: text_of(&value["chosen"]),
                rejected: text_of(&value["rejected"]),
            }
        })
        .collect()
}
