// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements catalogues of verifiable task kinds that
// turn any source into training tasks, for its clients. If your team needs
// expertise in synthetic task design or curriculum generation, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Task kinds as data: what a generator model is asked to write, what a
//! task of the kind must carry, where it is solved and how it is graded.
//!
//! A [`TaskKind`] is a value, not a code path: the generator reads its
//! fields and never matches on its name. Adding a kind is adding a value -
//! built in Rust, or deserialized from JSON or TOML with the field names
//! below - to a [`Catalogue`], which validates it.
//!
//! | Field | Meaning |
//! |---|---|
//! | `name` | the task kind every task of it records |
//! | `brief` | the generation instruction; `{count}` and `{runtime}` are filled in |
//! | `answer` | `text` (grounded lexically in the evidence), `program` (code, checked by running it) or `output` (what shown code prints, established by running it) |
//! | `shows_material` | open-book: the instruction carries the material it asks about, verbatim |
//! | `environment` | where the student works: `closed_book` or `runtime` |
//! | `runtime` | the runtime code of the kind runs in (`python3`) |
//! | `requires` | privileged material every task must carry beyond the reference: `hints`, `checks`, `tests`, `checks_or_tests` |
//! | `verifiers` | which verifiers grade an answer: `formal`, `executable`, `mutation_validated`, `consistency`, `judged` |
//! | `min_sections` | distinct sections the evidence must span |
//!
//! Whether a task of the kind must name its subject follows from these
//! fields ([`TaskKind::names_subject`]).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// What a task's answer is, and so how its reference is grounded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnswerForm {
    /// Text: its content words must be found in the evidence.
    Text,
    /// A program: the reference must pass the task's executable checks and
    /// every generated test must be admitted by mutation validation.
    Program,
    /// The output of code the instruction shows: running that code in the
    /// kind's runtime must produce the reference.
    Output,
}

impl AnswerForm {
    /// Whether the answer is computed, so grounded by execution rather
    /// than by the evidence's words.
    #[must_use]
    pub fn is_computed(self) -> bool {
        self != Self::Text
    }
}

/// Where the student works on a task of the kind.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SolverEnvironment {
    /// No tools at all.
    #[default]
    ClosedBook,
    /// The kind's runtime, through the one tool that runs code.
    Runtime,
}

/// Privileged material a task must carry beyond its reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Material {
    /// At least one hint.
    Hints,
    /// At least one executable check.
    Checks,
    /// At least one generated test.
    Tests,
    /// At least one executable check or generated test.
    ChecksOrTests,
}

/// A verifier that grades answers to tasks of a kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerifierKind {
    /// Exact match with the reference under a normalisation.
    Formal,
    /// The answer states the reference: contains it, as whole words, in a
    /// bounded answer. For short facts a model answers in a sentence.
    Stated,
    /// The task's executable checks run against the answer.
    Executable,
    /// The task's generated tests, admitted by mutation, run against it.
    MutationValidated,
    /// Agreement among independent answers.
    Consistency,
    /// A judge model of another identity.
    Judged,
}

fn one() -> usize {
    1
}

/// One kind of task; see the module documentation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskKind {
    /// The kind every task of it records.
    pub name: String,
    /// What the generator model is asked to write. `{count}` becomes the
    /// most tasks one request may return, `{runtime}` the kind's runtime.
    pub brief: String,
    /// What the answer is.
    pub answer: AnswerForm,
    /// Whether the instruction carries the material it asks about.
    #[serde(default)]
    pub shows_material: bool,
    /// Where the student works.
    #[serde(default)]
    pub environment: SolverEnvironment,
    /// The runtime the kind's code runs in, by registry name.
    #[serde(default)]
    pub runtime: Option<String>,
    /// Privileged material beyond the reference.
    #[serde(default)]
    pub requires: Vec<Material>,
    /// Which verifiers grade an answer.
    pub verifiers: Vec<VerifierKind>,
    /// Distinct sections the evidence must span; at least one.
    #[serde(default = "one")]
    pub min_sections: usize,
}

/// Why a kind is not consistent.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
#[error("task kind {kind:?}: {reason}")]
pub struct KindError {
    /// The kind's name.
    pub kind: String,
    /// What is inconsistent.
    pub reason: String,
}

impl TaskKind {
    /// Checks what the fields cannot say alone: a name and a brief, at
    /// least one section and one verifier, a runtime wherever code runs, a
    /// computed answer with something to run, and verifiers that have the
    /// material they read.
    pub fn validate(&self) -> Result<(), KindError> {
        let refuse = |reason: &str| {
            Err(KindError {
                kind: self.name.clone(),
                reason: reason.to_string(),
            })
        };
        let requires = |m: Material| self.requires.contains(&m);
        let checks = requires(Material::Checks) || requires(Material::ChecksOrTests);
        let tests = requires(Material::Tests) || requires(Material::ChecksOrTests);
        let verifies = |v: VerifierKind| self.verifiers.contains(&v);
        if self.name.trim().is_empty() {
            return refuse("it has no name");
        }
        if self.brief.trim().is_empty() {
            return refuse("it has no brief");
        }
        if self.min_sections == 0 {
            return refuse("min_sections must be at least 1");
        }
        if self.verifiers.is_empty() {
            return refuse("nothing grades its answers");
        }
        if self.runtime.is_none()
            && (self.answer.is_computed() || self.environment == SolverEnvironment::Runtime)
        {
            return refuse("it runs code but names no runtime");
        }
        if self.answer == AnswerForm::Program && !checks && !tests {
            return refuse("a program answer requires checks, tests, or either");
        }
        if (checks || tests) && self.answer != AnswerForm::Program {
            return refuse("checks and tests grade program answers only");
        }
        if self.answer == AnswerForm::Output && !self.shows_material {
            return refuse("an output answer needs the code it is the output of shown");
        }
        if verifies(VerifierKind::Executable) && !checks {
            return refuse("the executable verifier needs checks");
        }
        if verifies(VerifierKind::MutationValidated) && !tests {
            return refuse("the mutation-validated verifier needs generated tests");
        }
        Ok(())
    }

    /// Whether every task of the kind must name its subject: the product,
    /// document, tool, component or version it is about, in its
    /// instruction. A closed-book text answer is recalled from the source,
    /// so a question that does not say which source it is about has as many
    /// answers as there are products. A kind whose instruction shows its
    /// material (code to predict, a snippet to fix, an excerpt to
    /// transform, and the denoise kind's corrupted passage), or whose
    /// answer is computed and checked by running it, carries everything its
    /// answer depends on and names no subject.
    #[must_use]
    pub fn names_subject(&self) -> bool {
        !self.answer.is_computed() && !self.shows_material
    }
}

/// Task kinds by name, each validated.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Catalogue {
    kinds: BTreeMap<String, TaskKind>,
}

impl Catalogue {
    /// No kinds.
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    /// The kinds Splinter ships: recall, explain, predict, construct,
    /// debug, counterexample, transform, classify, retrieve, multi-turn and
    /// combine.
    #[must_use]
    pub fn builtin() -> Self {
        let kinds = builtin_kinds()
            .into_iter()
            .map(|kind| (kind.name.clone(), kind))
            .collect();
        Self { kinds }
    }

    /// Adds `kind`, replacing one of the same name; refused when it is not
    /// consistent.
    pub fn insert(&mut self, kind: TaskKind) -> Result<(), KindError> {
        kind.validate()?;
        self.kinds.insert(kind.name.clone(), kind);
        Ok(())
    }

    /// The kind named `name`.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&TaskKind> {
        self.kinds.get(name)
    }

    /// Every kind, by name.
    pub fn kinds(&self) -> impl Iterator<Item = &TaskKind> {
        self.kinds.values()
    }
}

/// The runtime the built-in code kinds run in.
pub const DEFAULT_CODE_RUNTIME: &str = "python3";

/// A text kind, closed-book, graded by `verifiers`.
fn text(name: &str, brief: &str, verifiers: &[VerifierKind]) -> TaskKind {
    TaskKind {
        name: name.into(),
        brief: brief.into(),
        answer: AnswerForm::Text,
        shows_material: false,
        environment: SolverEnvironment::ClosedBook,
        runtime: None,
        requires: Vec::new(),
        verifiers: verifiers.to_vec(),
        min_sections: 1,
    }
}

fn builtin_kinds() -> Vec<TaskKind> {
    use VerifierKind::{Consistency, Executable, Formal, Judged, MutationValidated, Stated};
    let code = || Some(DEFAULT_CODE_RUNTIME.to_string());
    vec![
        text(
            "recall",
            "Write up to {count} questions, each asking for one fact the sections state. The \
             reference is the fact, as short as it can be while complete.",
            &[Stated, Judged],
        ),
        text(
            "explain",
            "Write up to {count} requests to explain why or how something the sections \
             describe works. The reference is a short explanation built from the sections' \
             own statements.",
            &[Judged],
        ),
        TaskKind {
            name: "predict".into(),
            brief: "Write up to {count} tasks that show a short, self-contained {runtime} \
                    program exercising what the sections describe and ask what it prints. Put \
                    the program in `material` and verbatim in the instruction; the reference \
                    is exactly what it prints."
                .into(),
            answer: AnswerForm::Output,
            shows_material: true,
            environment: SolverEnvironment::ClosedBook,
            runtime: code(),
            requires: Vec::new(),
            verifiers: vec![Formal],
            min_sections: 1,
        },
        TaskKind {
            name: "construct".into(),
            brief: "Write up to {count} tasks asking for {runtime} code that meets a \
                    specification the sections give. The reference is a correct solution; \
                    add checks that call it, and tests strong enough to fail on small \
                    mistakes in it."
                .into(),
            answer: AnswerForm::Program,
            shows_material: false,
            environment: SolverEnvironment::Runtime,
            runtime: code(),
            requires: vec![Material::ChecksOrTests],
            verifiers: vec![Executable, MutationValidated],
            min_sections: 1,
        },
        TaskKind {
            name: "debug".into(),
            brief: "Write up to {count} tasks that show a short {runtime} snippet with one bug \
                    against what the sections specify, and ask for the fixed code. Put the \
                    faulty snippet in `material` and verbatim in the instruction; the \
                    reference is the fixed code, and the checks pass on it and fail on the \
                    snippet."
                .into(),
            answer: AnswerForm::Program,
            shows_material: true,
            environment: SolverEnvironment::Runtime,
            runtime: code(),
            requires: vec![Material::Checks],
            verifiers: vec![Executable],
            min_sections: 1,
        },
        text(
            "counterexample",
            "Write up to {count} tasks that state a plausible but false generalisation about \
             what the sections describe and ask for a counterexample. The reference is the \
             counterexample the sections give.",
            &[Judged],
        ),
        TaskKind {
            shows_material: true,
            ..text(
                "transform",
                "Write up to {count} tasks that show a short excerpt of the sections in \
                 `material`, verbatim in the instruction, and ask for it in another form \
                 (a list, a table row, JSON). The reference is the transformed excerpt.",
                &[Formal, Judged],
            )
        },
        text(
            "classify",
            "Write up to {count} tasks that name an item the sections describe and ask which \
             of a stated set of categories it belongs to. The reference is the category, as \
             the sections name it.",
            &[Formal, Consistency],
        ),
        text(
            "retrieve",
            "Write up to {count} questions and, as each reference, the heading of the section \
             that answers it, exactly as the section writes it.",
            &[Formal],
        ),
        text(
            "multi-turn",
            "Write up to {count} short dialogues about what the sections describe: two or \
             three alternating 'User:' and 'Assistant:' turns in the instruction, ending with a \
             user turn. The reference is the assistant's next reply.",
            &[Judged],
        ),
        TaskKind {
            min_sections: 2,
            ..text(
                "combine",
                "Write up to {count} questions that can only be answered by combining facts \
                 from two or more different sections. Cite evidence from each section used.",
                &[Judged],
            )
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_kind_round_trips_through_json() {
        for kind in Catalogue::builtin().kinds() {
            let json = serde_json::to_string(kind).unwrap_or_default();
            let back: TaskKind = serde_json::from_str(&json).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(&back, kind);
        }
    }
}
