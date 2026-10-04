// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements retrieval-augmented fine-tuning of models
// on a person's writing, for its clients. If your team needs expertise in
// training a model to use retrieved passages and ignore the wrong ones, you
// can procure our services by sending an email to info@swedishembedded.com.

//! Spec: a share of the training records is given passages in the prompt, as
//! `ask --retrieve` gives them, and the answer is unchanged. Of those, some
//! carry the passage the task was written from beside retrieved distractors
//! and the rest carry distractors only, so the model learns to use context
//! where it holds the answer and to answer without it where it does not. Which
//! records, and which of them hold the evidence, is a stable function of the
//! task; a record with no task to retrieve for is left alone.

// Helpers outside a test function unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use common::{scratch_context, Scripted};
use splinter_core::chat::WireMessage;
use splinter_core::experience::{
    Environment, Experience, Privileged, PrivilegedKind, Provenance, Span, Task,
};
use splinter_data::{Objective, Projection, Record, RecordBody, RecordMetadata, Strip};
use splinter_knowledge::retrieve::{passages, EmbedError, Embedder, Library};
use splinter_pipelines::raft::{with_passages, PassageShare};
use splinter_pipelines::retrieval::Retrieval;
use splinter_pipelines::sources::{self, SourceTarget};

const LETTERS: &str = "# Letters\n\n## To Carr\n\nEducation of the people is the surest foundation of liberty, and a nation that wishes to be free must see that its youth are taught to reason. Thus it has ever been in such matters, and thus it will be, as every careful observer of the affairs of nations and of households has had occasion to remark in his own time and place, with no great variety of circumstance from one year to another, for the course of human affairs is slow to change and quick to repeat itself. Thus it has ever been in such matters, and thus it will be, as every careful observer of the affairs of nations and of households has had occasion to remark in his own time and place, with no great variety of circumstance from one year to another, for the course of human affairs is slow to change and quick to repeat itself.\n\n## To Jay\n\nThe tobacco shipped to Havre by the brig Eliza was sold at a poor price, and the merchants complain of the duties laid upon the hogsheads. Thus it has ever been in such matters, and thus it will be, as every careful observer of the affairs of nations and of households has had occasion to remark in his own time and place, with no great variety of circumstance from one year to another, for the course of human affairs is slow to change and quick to repeat itself. Thus it has ever been in such matters, and thus it will be, as every careful observer of the affairs of nations and of households has had occasion to remark in his own time and place, with no great variety of circumstance from one year to another, for the course of human affairs is slow to change and quick to repeat itself.\n\n## To Madison\n\nThe university of Virginia should teach every science useful to the republic, and its professors should be free to follow truth wherever it leads them. Thus it has ever been in such matters, and thus it will be, as every careful observer of the affairs of nations and of households has had occasion to remark in his own time and place, with no great variety of circumstance from one year to another, for the course of human affairs is slow to change and quick to repeat itself. Thus it has ever been in such matters, and thus it will be, as every careful observer of the affairs of nations and of households has had occasion to remark in his own time and place, with no great variety of circumstance from one year to another, for the course of human affairs is slow to change and quick to repeat itself.\n";

/// An embedder over two concepts: teaching, and trade.
struct Concepts;

impl Embedder for Concepts {
    fn name(&self) -> String {
        "concepts".into()
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        Ok(texts
            .iter()
            .map(|t| {
                let has = |words: &[&str]| words.iter().filter(|w| t.contains(*w)).count() as f32;
                vec![
                    has(&[
                        "education",
                        "taught",
                        "reason",
                        "teach",
                        "learn",
                        "university",
                    ]),
                    has(&["tobacco", "merchants", "duties", "price", "sold"]),
                    0.1,
                ]
            })
            .collect())
    }
}

fn turn(role: &str, content: &str, train: bool) -> WireMessage {
    WireMessage {
        role: role.into(),
        content: content.into(),
        tool_calls: Vec::new(),
        tool_call_id: None,
        train,
    }
}

const QUESTION: &str = "How should a young person learn to reason about liberty?";

fn record_with_experience(
    ctx: &splinter_orchestrator::Context,
    evidence: Span,
    answer: &str,
    question: &str,
) -> Record {
    let task = Task::new(
        "advise",
        vec![evidence.clone()],
        Environment::closed_book(),
        question,
        vec![Privileged {
            kind: PrivilegedKind::Reference,
            content: answer.into(),
            span: Some(evidence),
        }],
    )
    .unwrap();
    let experience = Experience::answered_without_a_run(
        task,
        answer,
        Provenance::new("splinter/author", ctx.clock()),
    )
    .unwrap();
    let id = ctx.experiences().put(&experience).unwrap();
    Record {
        body: RecordBody::Chat {
            messages: vec![
                turn("system", "You are a helpful assistant.", false),
                turn("user", question, false),
                turn("assistant", answer, true),
            ],
        },
        metadata: RecordMetadata {
            group: None,
            experiences: vec![id],
            task: None,
            sources: Vec::new(),
            view: "sft-final".into(),
            objective: Objective::Sft,
        },
    }
}

fn projection(records: Vec<Record>) -> Projection {
    Projection {
        view: "sft-final".into(),
        objective: Objective::Sft,
        strip: Some(Strip::All),
        min_strength: None,
        records,
        excluded: Default::default(),
        system_prompt: None,
    }
}

fn user_turn(record: &Record) -> String {
    let RecordBody::Chat { messages } = &record.body else {
        panic!("a chat record");
    };
    messages[1].content.clone()
}

#[test]
fn some_records_carry_the_evidence_among_distractors_and_the_rest_distractors_only() {
    let (scratch, ctx) = scratch_context("raft", Scripted::new(|_| String::new()), false);
    let path = scratch.0.join("letters.md");
    std::fs::write(&path, LETTERS).unwrap();
    let id = sources::add(
        &ctx,
        &SourceTarget::from_learn_arg(&path.display().to_string()).unwrap(),
    )
    .unwrap()
    .source
    .id;
    let found = passages(&ctx.sources(), std::slice::from_ref(&id)).unwrap();
    let education = found
        .iter()
        .find(|p| p.text.contains("surest foundation"))
        .unwrap()
        .clone();
    let span = Span::new(
        education.content.clone().unwrap(),
        education.range.start as u64,
        education.range.end as u64,
    )
    .unwrap();
    let library = Library::new(found, &Concepts).unwrap();
    let retrieval = Retrieval {
        library: &library,
        embedder: &Concepts,
        passages: 2,
        rerank: None,
    };
    let answer = "Teach the young to reason, for liberty rests on it.";
    let record = record_with_experience(&ctx, span, answer, QUESTION);
    let closed = projection(vec![record.clone()]);

    // Every record, evidence always among the passages.
    let share = PassageShare {
        records: 1.0,
        with_evidence: 1.0,
    };
    let given = with_passages(&ctx, closed.clone(), &share, &retrieval).unwrap();
    let prompt = user_turn(&given.records[0]);
    assert!(prompt.contains("surest foundation of liberty"), "{prompt}");
    assert!(prompt.contains(QUESTION), "{prompt}");
    assert!(
        prompt.matches("--- ").count() >= 2,
        "the evidence and a distractor: {prompt}"
    );
    // Nothing else of the record changes.
    let RecordBody::Chat { messages } = &given.records[0].body else {
        panic!()
    };
    assert_eq!(
        (messages[0].role.as_str(), messages[2].content.as_str()),
        ("system", answer)
    );
    assert!(messages[2].train && !messages[1].train);
    assert_eq!(
        with_passages(&ctx, closed.clone(), &share, &retrieval).unwrap(),
        given,
        "stable"
    );

    // Distractors only: the evidence is not shown, and the answer stands.
    let share = PassageShare {
        records: 1.0,
        with_evidence: 0.0,
    };
    let without = with_passages(&ctx, closed.clone(), &share, &retrieval).unwrap();
    let prompt = user_turn(&without.records[0]);
    assert!(!prompt.contains("surest foundation of liberty"), "{prompt}");
    assert!(
        prompt.contains(QUESTION) && prompt.matches("--- ").count() >= 2,
        "{prompt}"
    );

    // No record given any context: closed-book as it was.
    let share = PassageShare {
        records: 0.0,
        with_evidence: 1.0,
    };
    assert_eq!(
        with_passages(&ctx, closed, &share, &retrieval)
            .unwrap()
            .records,
        vec![record]
    );
}
