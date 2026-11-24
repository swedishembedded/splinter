// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training environments compiled from a
// client's own documents, for its clients. If your team needs expertise in
// turning documentation into verifiable training questions, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Document-grounded questions the solver answers without the document.
//!
//! The generator reads a directory of documents, takes the keyed facts they
//! state ([`PatternFacts`]: "The `<attribute>` of `<entity>` is `<value>`.")
//! and compiles questions; the solver is never shown the document, only the
//! question (information asymmetry: the document grounds the answer, the
//! solver works from the question and what it knows). Two kinds:
//!
//! * **derived** (`recall` 0): the question carries the document's numeric
//!   facts as premises and asks for a quantity computed from them, by code,
//!   so the answer is exact and cannot be read off the question. A skill the
//!   base model has or can learn: carrying a document's figures through a
//!   calculation.
//! * **recall** (`recall` 1): the question asks for a fact the document
//!   states, with the question text of [`Fact::question`]. Only a policy that
//!   has learned the document answers it; its base pass rate is expected to be
//!   near zero, so it enters a campaign only as the check of a document that
//!   was trained on, with its band measured on that policy.
//!
//! The check is deterministic: a number equal to the computed one (derived),
//! or an answer that states the fact's value as a run of whole words
//! (`states`, recall; the answer may add at most four words). Documents are
//! split between train and dev, by document: a dev instance asks about a
//! document no train instance used.
//!
//! Knobs: `questions` (lines to answer), `steps` (facts a derived question
//! combines minus one) and `recall` (0 derived, 1 recall).

use std::path::PathBuf;

use serde_json::json;
use splinter_core::rng::SplitMix64;
use splinter_core::verifiability::{Split, VerifierClass};

use crate::factory::artifact::{
    environment_id, EnvMeta, Instance, Parts, SolverTools, SCHEMA_VERSION,
};
use crate::factory::folder::Folder;
use crate::factory::plugin::{resolve_difficulty, InstanceRequest, Knob, Plugin, PluginError};
use crate::goal::facts::{Fact, FactExtractor, Form, PatternFacts};

use super::DOCUMENT_QUESTIONS;

/// The plugin's version. Any change to the question compilation, the check or
/// the instruction text is a new version.
pub const VERSION: &str = "1";

const KNOBS: &[Knob] = &[
    Knob {
        name: "questions",
        min: 1,
        max: 4,
        default: 2,
        doc: "how many questions the answer file holds, one per line",
    },
    Knob {
        name: "steps",
        min: 1,
        max: 3,
        default: 1,
        doc: "for a derived question, how many facts beyond the first it combines",
    },
    Knob {
        name: "recall",
        min: 0,
        max: 1,
        default: 0,
        doc: "0 derived questions the figures of which are given, 1 recall of a stated fact",
    },
];

/// The verifier: reads the answer file line by line against the pinned
/// expectations. No answer is stored anywhere the solver can read.
const VERIFY: &str = include_str!("document_questions/verify.py");

/// A numeric property: `The <attribute> of <entity> is <number> <unit>.`
#[derive(Clone, Debug)]
struct Figure {
    attribute: String,
    entity: String,
    value: i64,
    unit: String,
}

fn figure_of(fact: &Fact) -> Option<Figure> {
    if fact.key.form != Form::Property {
        return None;
    }
    let mut words = fact.value.split_whitespace();
    let value: i64 = words.next()?.parse().ok()?;
    Some(Figure {
        attribute: fact.key.attribute.clone(),
        entity: fact.key.entity.clone(),
        value,
        unit: words.collect::<Vec<_>>().join(" "),
    })
}

/// The documents a plugin draws from.
pub struct DocumentQuestions {
    root: PathBuf,
}

impl DocumentQuestions {
    /// A plugin over the Markdown and text documents directly under `root`.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The documents of `split`, by name: every third document, counting
    /// from the third, is dev, so the splits never share one.
    fn documents(&self, split: Split) -> Result<Vec<(String, String)>, PluginError> {
        let mut names: Vec<String> = std::fs::read_dir(&self.root)
            .map_err(|e| PluginError::Generation(format!("{}: {e}", self.root.display())))?
            .filter_map(Result::ok)
            .filter_map(|e| e.file_name().into_string().ok())
            .filter(|n| n.ends_with(".md") || n.ends_with(".txt"))
            .collect();
        names.sort();
        if names.len() < 3 {
            return Err(PluginError::Generation(format!(
                "{} holds {} documents; train and dev need at least three",
                self.root.display(),
                names.len()
            )));
        }
        let mut chosen = Vec::new();
        for (index, name) in names.into_iter().enumerate() {
            let dev = index % 3 == 2;
            if dev == (split == Split::Dev) {
                let path = self.root.join(&name);
                let text = std::fs::read_to_string(&path)
                    .map_err(|e| PluginError::Generation(format!("{}: {e}", path.display())))?;
                chosen.push((name, text));
            }
        }
        Ok(chosen)
    }
}

fn describe(figure: &Figure) -> String {
    let unit = if figure.unit.is_empty() {
        String::new()
    } else {
        format!(" {}", figure.unit)
    };
    format!("{} {}{unit}", figure.entity, figure.value)
}

impl Plugin for DocumentQuestions {
    fn id(&self) -> &'static str {
        DOCUMENT_QUESTIONS
    }

    fn version(&self) -> &'static str {
        VERSION
    }

    fn verifier_class(&self) -> VerifierClass {
        VerifierClass::Objective
    }

    fn knobs(&self) -> &'static [Knob] {
        KNOBS
    }

    fn generate(&self, request: &InstanceRequest) -> Result<Instance, PluginError> {
        let difficulty = resolve_difficulty(self.id(), KNOBS, &request.difficulty)?;
        let (count, steps, recall) = (
            difficulty["questions"] as usize,
            difficulty["steps"] as usize,
            difficulty["recall"] == 1,
        );
        let mut rng = SplitMix64::new(request.seed);
        let documents = self.documents(request.split)?;

        let mut questions: Vec<String> = Vec::new();
        let mut expectations: Vec<serde_json::Value> = Vec::new();
        let mut grounding: Vec<serde_json::Value> = Vec::new();
        for _ in 0..64 {
            if questions.len() == count {
                break;
            }
            let Some((name, text)) = rng.pick(&documents) else {
                break;
            };
            let facts = PatternFacts.facts(text);
            if recall {
                let Some(fact) = rng.pick(&facts) else {
                    continue;
                };
                let question = fact.question();
                if questions.contains(&question) || question.contains(&fact.value) {
                    continue;
                }
                expectations.push(json!({"mode": "states", "value": fact.value}));
                grounding.push(json!({"document": name, "fact": format!("{} of {} is {}", fact.key.attribute, fact.key.entity, fact.value)}));
                questions.push(question);
                continue;
            }
            // Derived: figures of one attribute, with one unit, of distinct entities.
            let figures: Vec<Figure> = facts.iter().filter_map(figure_of).collect();
            let Some(first) = rng.pick(&figures) else {
                continue;
            };
            let mut same: Vec<&Figure> = figures
                .iter()
                .filter(|f| f.attribute == first.attribute && f.unit == first.unit)
                .collect();
            same.sort_by(|a, b| a.entity.cmp(&b.entity));
            same.dedup_by(|a, b| a.entity == b.entity);
            if same.len() < steps + 1 {
                continue;
            }
            rng.shuffle(&mut same);
            let operands: Vec<&Figure> = same.into_iter().take(steps + 1).collect();
            let (what, answer) = match rng.below(2) {
                0 => (
                    "their total".to_string(),
                    operands.iter().map(|f| f.value).sum::<i64>(),
                ),
                _ => (
                    format!(
                        "the first minus the {}",
                        if steps == 1 {
                            "second".to_string()
                        } else {
                            "sum of the others".to_string()
                        }
                    ),
                    operands[0].value - operands[1..].iter().map(|f| f.value).sum::<i64>(),
                ),
            };
            if operands.iter().any(|f| f.value == answer) || answer.abs() < 2 {
                continue;
            }
            let unit = if first.unit.is_empty() {
                String::new()
            } else {
                format!(" ({})", first.unit)
            };
            let premises = operands
                .iter()
                .map(|f| describe(f))
                .collect::<Vec<_>>()
                .join("; ");
            let question = format!(
                "The {} of these units are: {premises}. What is {what}{unit}? Answer with the number only.",
                first.attribute
            );
            if questions.contains(&question) {
                continue;
            }
            expectations.push(json!({"mode": "number", "value": answer}));
            grounding.push(json!({"document": name, "facts": operands.iter().map(|f| format!("{} of {} is {} {}", f.attribute, f.entity, f.value, f.unit)).collect::<Vec<_>>()}));
            questions.push(question);
        }
        if questions.len() < count {
            return Err(PluginError::Generation(format!(
                "only {} questions could be compiled from the {} documents of the {} split, {count} asked",
                questions.len(),
                documents.len(),
                request.split
            )));
        }

        let mut workspace = Folder::new();
        workspace.put_text("answer.txt", "")?;
        let mut tests = Folder::new();
        tests.put_script(
            "test.sh",
            "#!/bin/sh\n# Grades answer.txt line by line against the pinned expectations; see verify.py.\npython3 -I \"${TESTS_DIR:-/tests}/verify.py\"\n",
        )?;
        tests.put_text("verify.py", VERIFY)?;
        let pinned = serde_json::to_string(&expectations)
            .map_err(|e| PluginError::Generation(e.to_string()))?;
        tests.put_text("expected.json", &format!("{pinned}\n"))?;
        let grounded = serde_json::to_string(&grounding)
            .map_err(|e| PluginError::Generation(e.to_string()))?;
        tests.put_text("grounding.json", &format!("{grounded}\n"))?;
        let mut solution = Folder::new();
        let answers: String = expectations
            .iter()
            .map(|e| {
                format!(
                    "{}\n",
                    e["value"]
                        .as_str()
                        .map_or_else(|| e["value"].to_string(), ToString::to_string)
                )
            })
            .collect();
        solution.put_text("answer.txt", &answers)?;
        solution.put_script("solve.sh", "#!/bin/sh\ncp \"${SOLUTION_DIR:-/solution}/answer.txt\" \"${APP_DIR:-/app}/answer.txt\"\n")?;

        let listed: String = questions
            .iter()
            .enumerate()
            .map(|(n, q)| format!("{}. {q}\n", n + 1))
            .collect();
        let instruction = format!(
            "# Answer from what you know\n\nAnswer each question on its own line of `/app/answer.txt`, in order. No document is provided; the questions are about documents you may have learned, or carry their figures.\n\nQuestions:\n{listed}"
        );
        let meta = EnvMeta {
            schema_version: SCHEMA_VERSION,
            environment: environment_id(self.id(), VERSION, &difficulty),
            plugin: self.id().into(),
            plugin_version: VERSION.into(),
            verifier_class: VerifierClass::Objective,
            verifier: format!("document-fact-check@{VERSION}"),
            difficulty,
            split: request.split,
            seed: request.seed,
            solver_tools: SolverTools::Files,
        };
        Ok(Instance::assemble(
            meta,
            Parts {
                instruction,
                workspace,
                packages: vec!["python3".into()],
                tests,
                solution,
                category: "document-questions".into(),
            },
        )?)
    }

    fn novelty_text(&self, instance: &Instance) -> String {
        let text = instance.instruction();
        text.split_once("Questions:\n")
            .map_or(text, |(_, rest)| rest)
            .to_string()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // a panic is the failure report
mod tests {
    use super::*;

    #[test]
    fn a_property_fact_with_a_number_is_a_figure_and_other_facts_are_not() {
        let facts = PatternFacts.facts("The flow rate of pump A is 12 L/min.\nThe colour of pump A is red.\nWhen the rent is late, a Hesitant client asks for a plan.");
        let figures: Vec<_> = facts.iter().filter_map(figure_of).collect();
        assert_eq!(figures.len(), 1);
        assert_eq!((figures[0].value, figures[0].unit.as_str()), (12, "L/min"));
    }

    #[test]
    fn the_documents_of_the_two_splits_never_overlap_and_three_are_needed() {
        let dir = tempfile::tempdir().unwrap();
        let plugin = DocumentQuestions::new(dir.path());
        assert!(plugin.documents(Split::Train).is_err());
        for name in ["a.md", "b.md", "c.md", "d.txt", "e.md", "f.md", "notes.pdf"] {
            std::fs::write(dir.path().join(name), "The x of y is 1.\n").unwrap();
        }
        let names = |split| -> Vec<String> {
            plugin
                .documents(split)
                .unwrap()
                .into_iter()
                .map(|d| d.0)
                .collect()
        };
        let (train, dev) = (names(Split::Train), names(Split::Dev));
        assert_eq!(dev, ["c.md", "f.md"]);
        assert_eq!(train, ["a.md", "b.md", "d.txt", "e.md"]);
        assert!(train.iter().all(|t| !dev.contains(t)));
    }
}
