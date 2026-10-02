// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model-driven task generation that turns
// any source into verifiable training tasks, for its clients. If your team
// needs expertise in synthetic task generation or verifier design, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Variants: the same fact asked in other words.
//!
//! [`ModelTaskGenerator::variants`] shows the generator model one task -
//! its question, the subject it names, its answer, and the source sections
//! its evidence falls in ([`crate::material::teacher_material`]), by
//! position - and asks for up to `count` differently worded questions in
//! the reply shape generation uses, each naming the same subject. Nothing
//! the model says is trusted. A candidate is admitted only when:
//!
//! 1. it is within the requested count;
//! 2. it cites at least one section it was shown, and only those;
//! 3. it states the original's reference (case, whitespace and closing
//!    punctuation aside): a question with another answer asks about another
//!    fact;
//! 4. its instruction stands on its own ([`splinter_views::check_self_contained`]);
//! 5. it still names the original's subject (case and whitespace aside):
//!    a rewording that drops it asks a question with many answers;
//! 6. it is new: not the original's instruction, nor a sibling's, by the
//!    rule generation uses ([`super::dedup`]).
//!
//! An admitted variant is the original task asking the new question
//! ([`Task::with_instruction`]): kind, evidence, environment, reference and
//! concepts are the original's - whatever else the reply says about them is
//! discarded - so the verifiers that grade the original grade it.

use serde::Serialize;
use splinter_record::experience::{Privileged, PrivilegedKind, Task};
use splinter_views::check_self_contained;

use super::dedup::{Repeat, Seen};
use super::generator::{GenerateError, ModelTaskGenerator};
use super::kind::SolverEnvironment;
use super::reply::VARIANTS_EXAMPLE;
use super::{Catalogue, GeneratedTask, GenerationReport, Rejection};
use crate::gates::normalize;
use crate::material::teacher_material;

/// The name of the typed call a generator model is sent for variants.
pub const VARIANTS_METHOD: &str = "write_variants";

/// The most variants one task is given, by default.
pub const DEFAULT_VARIANTS_PER_TASK: usize = 3;

/// Whether `task` can be varied: a closed-book question with a text
/// reference and evidence to show the model, of a kind whose instruction
/// carries no material and whose answer is not computed. `Err` names why
/// not, as a snake-case reason a report counts.
pub fn can_vary(task: &Task) -> Result<(), &'static str> {
    let Some(kind) = Catalogue::builtin().get(&task.task.kind).cloned() else {
        return Err("not_a_model_written_kind");
    };
    if kind.shows_material || kind.answer.is_computed() {
        return Err("material_or_computed_answer");
    }
    if kind.environment != SolverEnvironment::ClosedBook
        || task.environment.kind != splinter_record::experience::Environment::CLOSED_BOOK
    {
        return Err("not_closed_book");
    }
    if task.evidence.is_empty() {
        return Err("no_evidence");
    }
    if reference(task).is_none() {
        return Err("no_reference");
    }
    Ok(())
}

/// The reference answer `task` carries.
fn reference(task: &Task) -> Option<&str> {
    task.privileged
        .iter()
        .find(|p| p.kind == PrivilegedKind::Reference)
        .map(|p| p.content.as_str())
}

/// `reference` as variants compare it: case, whitespace and closing
/// punctuation set aside.
fn same_answer(a: &str, b: &str) -> bool {
    let canonical = |text: &str| {
        normalize(text)
            .trim_end_matches(|c: char| c.is_ascii_punctuation())
            .to_string()
    };
    canonical(a) == canonical(b)
}

/// What the model is shown: the task, and the sections it is grounded in.
#[derive(Serialize)]
struct Shown<'a> {
    question: &'a str,
    subject: &'a str,
    answer: &'a str,
    sections: Vec<ShownSection<'a>>,
}

#[derive(Serialize)]
struct ShownSection<'a> {
    position: usize,
    text: &'a str,
}

impl ModelTaskGenerator {
    /// Up to `count` variants of `original`, which names `subject`; every
    /// proposal admitted or rejected, as [`Self::generate`] reports them
    /// (the report's kind is the original's, and each variant's subject is
    /// `subject`). Refused when `original` cannot be varied ([`can_vary`]),
    /// its instruction does not name `subject`, or `count` is zero.
    pub async fn variants(
        &self,
        original: &Task,
        subject: &str,
        count: usize,
    ) -> Result<GenerationReport, GenerateError> {
        self.policy.validate()?;
        if count == 0 {
            return Err(GenerateError::Parameter {
                name: "count",
                reason: "at least one variant".into(),
            });
        }
        if let Err(reason) = can_vary(original) {
            return Err(GenerateError::NotVaryable {
                task: original.task.id.clone(),
                reason,
            });
        }
        let subject = subject.trim();
        if subject.is_empty() || !normalize(&original.instruction).contains(&normalize(subject)) {
            return Err(GenerateError::NotVaryable {
                task: original.task.id.clone(),
                reason: "no_subject",
            });
        }
        let Some(answer) = reference(original) else {
            unreachable!("can_vary checked the reference");
        };
        let passages = teacher_material(&self.store, original)?;
        let shown = Shown {
            question: &original.instruction,
            subject,
            answer,
            sections: passages
                .iter()
                .enumerate()
                .map(|(position, text)| ShownSection { position, text })
                .collect(),
        };
        let brief = brief(count, subject);
        let kind = original.task.kind.as_str();
        let mut report = GenerationReport::default();
        let reply = match self.request(VARIANTS_METHOD, &brief, &shown).await? {
            Ok(reply) => reply,
            Err((reason, detail)) => {
                report.reject(kind, 0, reason, detail);
                return Ok(report);
            }
        };
        let prompt = super::generator::prompt_digest(&brief, &shown);
        let hidden: Vec<Privileged> = passages
            .iter()
            .map(|content| Privileged {
                kind: PrivilegedKind::Passage,
                content: content.clone(),
                span: None,
            })
            .collect();
        let hidden: Vec<&Privileged> = hidden.iter().collect();
        let mut seen = Seen::new(self.policy.shingle_words, self.policy.max_overlap);
        seen.admit(&original.instruction);
        for (index, candidate) in reply.tasks.into_iter().enumerate() {
            let mut refuse = |reason, detail: String| report.reject(kind, index, reason, detail);
            if index >= count {
                refuse(
                    Rejection::OverCount,
                    format!("{count} variants were asked for"),
                );
                continue;
            }
            let cited: Vec<usize> = candidate.evidence.iter().map(|c| c.section).collect();
            if cited.is_empty() || cited.iter().any(|&s| s >= passages.len()) {
                refuse(
                    Rejection::UnknownSection,
                    format!("cites {cited:?}; {} section(s) were shown", passages.len()),
                );
                continue;
            }
            if !same_answer(&candidate.reference, answer) {
                refuse(
                    Rejection::ReferenceChanged,
                    format!("{:?} is not {answer:?}", candidate.reference),
                );
                continue;
            }
            if let Err(why) = check_self_contained(&candidate.instruction, &hidden) {
                refuse(Rejection::NotSelfContained, why.to_string());
                continue;
            }
            if !normalize(&candidate.instruction).contains(&normalize(subject)) {
                refuse(
                    Rejection::NoSubject,
                    format!("the variant no longer names the subject {subject:?}"),
                );
                continue;
            }
            match seen.repeats(&candidate.instruction) {
                Some(Repeat::Exact) => {
                    refuse(Rejection::Duplicate, candidate.instruction);
                    continue;
                }
                Some(Repeat::Near) => {
                    refuse(Rejection::NearDuplicate, candidate.instruction);
                    continue;
                }
                None => {}
            }
            match original.with_instruction(candidate.instruction.as_str()) {
                Ok(task) => {
                    seen.admit(&task.instruction);
                    report.admit(
                        kind,
                        GeneratedTask {
                            task,
                            subject: Some(subject.to_string()),
                            generator: self.generator_name(),
                            prompt: prompt.clone(),
                        },
                    );
                }
                Err(e) => refuse(Rejection::Invalid, e.to_string()),
            }
        }
        Ok(report)
    }
}

/// The brief for up to `count` variants of the question shown, which
/// names `subject`.
fn brief(count: usize, subject: &str) -> String {
    format!(
        "You write differently worded questions about one fact.\n\nThe input holds a question, \
         the subject it is about, its answer, and the sections the answer comes from. Write up \
         to {count} other questions that ask for the same fact in different words, so that the \
         same answer is correct for each.\n\nRules:\n\
         - Write at most {count} tasks.\n\
         - Every question names its subject, {subject:?}, as the original does: without it the \
         question has a different answer for every product or version. Put it in `subject`.\n\
         - Each instruction is everything the student sees: it must stand on its own. Never \
         refer to the sections, a passage, a document or text the student is not shown, and \
         do not copy long runs of the sections into it.\n\
         - Put the answer, unchanged, in `reference`.\n\
         - Cite the position of the section the answer comes from in `evidence`.\n\
         - Do not repeat the question, or each other.\n\nA reply looks like this:\n\
         {VARIANTS_EXAMPLE}\n"
    )
}
