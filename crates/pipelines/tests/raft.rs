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
use splinter_knowledge::retrieve::Passage;
use splinter_knowledge::retrieve::{passages, EmbedError, Embedder, Library};
use splinter_pipelines::raft::{with_passages, Abstainer, Abstention, PassageShare};
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
            split: None,
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
        terms: None,
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
        ..PassageShare::default()
    };
    let given = with_passages(&ctx, closed.clone(), &share, &retrieval, None).unwrap();
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
        with_passages(&ctx, closed.clone(), &share, &retrieval, None).unwrap(),
        given,
        "stable"
    );

    // Distractors only: the evidence is not shown, and the answer stands.
    let share = PassageShare {
        records: 1.0,
        with_evidence: 0.0,
        ..PassageShare::default()
    };
    let without = with_passages(&ctx, closed.clone(), &share, &retrieval, None).unwrap();
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
        ..PassageShare::default()
    };
    assert_eq!(
        with_passages(&ctx, closed, &share, &retrieval, None)
            .unwrap()
            .records,
        vec![record]
    );
}

/// An abstainer that says what it was asked to say.
struct Writer;

impl Abstainer for Writer {
    fn unanswerable_question(&self, _: &Task) -> Option<String> {
        Some("What would you make of the telegraph?".into())
    }

    fn abstention(&self, question: &str, shown: &[&Passage], kind: Abstention) -> Option<String> {
        Some(format!(
            "{kind:?}: the writings before me do not settle {question} ({} passages)",
            shown.len()
        ))
    }
}

/// A fixture of one letter family, its retrieval, and the record of a
/// dialogue written from it.
fn fixture(
    test: &str,
) -> (
    splinter_orchestrator::Context,
    common::Scratch,
    Library,
    Record,
) {
    let (scratch, ctx) = scratch_context(test, Scripted::new(|_| String::new()), false);
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
    let record = record_with_experience(&ctx, span, "Teach the young to reason.", QUESTION);
    (ctx, scratch, library, record)
}

/// A record whose passages miss the evidence is answered by an abstention
/// that stays in the family of the record it was made from; one that is
/// asked what the writings cannot hold is asked about that and abstains
/// too; and a record the abstainer has nothing for keeps its answer.
#[test]
fn a_retrieval_miss_and_an_unsupported_question_abstain_inside_their_family() {
    let (ctx, _scratch, library, mut record) = fixture("raft-abstain");
    record.metadata.group = Some("family".to_string());
    let retrieval = Retrieval {
        library: &library,
        embedder: &Concepts,
        passages: 2,
        rerank: None,
    };
    let source = projection(vec![record.clone()]);

    let miss = PassageShare {
        records: 1.0,
        with_evidence: 0.0,
        abstain: 1.0,
        unsupported: 0.0,
    };
    let made = with_passages(&ctx, source.clone(), &miss, &retrieval, Some(&Writer)).unwrap();
    let derived = &made.records[0];
    let RecordBody::Chat { messages } = &derived.body else {
        panic!()
    };
    assert!(messages[1].content.contains(QUESTION));
    assert!(
        !messages[1].content.contains("surest foundation"),
        "the evidence is missed"
    );
    assert!(messages[2].train && messages[2].content.starts_with("Miss: "));
    assert!(messages[2].content.contains("2 passages"));
    assert_eq!(
        (&derived.metadata.group, &derived.metadata.experiences),
        (&record.metadata.group, &record.metadata.experiences),
        "a derivative is of the family it was made from"
    );

    let unsupported = PassageShare {
        records: 1.0,
        with_evidence: 0.0,
        abstain: 0.0,
        unsupported: 1.0,
    };
    let made = with_passages(
        &ctx,
        source.clone(),
        &unsupported,
        &retrieval,
        Some(&Writer),
    )
    .unwrap();
    let RecordBody::Chat { messages } = &made.records[0].body else {
        panic!()
    };
    assert!(messages[1].content.contains("telegraph"), "{messages:?}");
    assert!(!messages[1].content.contains(QUESTION));
    assert!(messages[2].content.starts_with("Unsupported: "));
    assert_eq!(made.records[0].metadata.group, record.metadata.group);

    // Asked to abstain with no one to write it: refused, not silently closed.
    assert!(with_passages(&ctx, source, &miss, &retrieval, None).is_err());
}

/// Of the records given passages, the shares fall as asked and the draw is
/// a stable function of the task.
#[test]
fn the_modes_of_the_records_follow_their_shares() {
    let (ctx, _scratch, library, record) = fixture("raft-shares");
    let retrieval = Retrieval {
        library: &library,
        embedder: &Concepts,
        passages: 2,
        rerank: None,
    };
    let records: Vec<Record> = (0..60)
        .map(|n| {
            let mut r = record.clone();
            let RecordBody::Chat { messages } = &mut r.body else {
                panic!()
            };
            messages[1].content = format!("{QUESTION} ({n})");
            // A task of its own per record, so each draws on its own.
            let task_id = ctx
                .experiences()
                .get(&record.metadata.experiences[0])
                .unwrap();
            let task = task_id.to_task();
            let renamed = Task::new(
                "advise",
                task.evidence.clone(),
                Environment::closed_book(),
                format!("{QUESTION} ({n})"),
                task.privileged.clone(),
            )
            .unwrap();
            let experience = Experience::answered_without_a_run(
                renamed,
                "Teach the young to reason.",
                Provenance::new("splinter/author", ctx.clock()),
            )
            .unwrap();
            r.metadata.experiences = vec![ctx.experiences().put(&experience).unwrap()];
            r
        })
        .collect();
    let share = PassageShare {
        records: 1.0,
        with_evidence: 0.5,
        abstain: 0.2,
        unsupported: 0.1,
    };
    let made = with_passages(
        &ctx,
        projection(records.clone()),
        &share,
        &retrieval,
        Some(&Writer),
    )
    .unwrap();
    let count = |prefix: &str| {
        made.records
            .iter()
            .filter(|r| match &r.body {
                RecordBody::Chat { messages } => messages[2].content.starts_with(prefix),
                _ => false,
            })
            .count()
    };
    let (miss, unsupported) = (count("Miss: "), count("Unsupported: "));
    assert!((6..=18).contains(&miss), "{miss} of 60 at a fifth");
    assert!(
        (1..=12).contains(&unsupported),
        "{unsupported} of 60 at a tenth"
    );
    let again =
        with_passages(&ctx, projection(records), &share, &retrieval, Some(&Writer)).unwrap();
    assert_eq!(made, again, "stable");
}

/// Built into a dataset, an abstention is in the group of the dialogue it
/// was made from, so the split holds the two out or trains them together.
#[test]
fn an_abstention_in_a_dataset_is_in_the_group_of_its_source() {
    use splinter_core::annotation::{Annotation, AnnotationBody, Outcome, Producer, Strength};
    use splinter_pipelines::datasets::{build_with, BuildRequest, Passages, ViewName, VoiceBuild};
    use splinter_store::experiences::ExperienceSet;

    let (ctx, _scratch, library, record) = fixture("raft-group");
    let id = record.metadata.experiences[0].clone();
    ctx.experiences()
        .annotate(&Annotation {
            experience: id.clone(),
            producer: Producer {
                name: "scripted/judge".into(),
                version: "1".into(),
            },
            body: AnnotationBody::Verdict {
                outcome: Outcome::Pass,
                strength: Strength::Judged,
                evidence: serde_json::Value::Null,
            },
        })
        .unwrap();
    let set = ctx
        .experiences()
        .put_set(&ExperienceSet {
            name: "one dialogue".into(),
            members: vec![id],
        })
        .unwrap();
    let request = BuildRequest {
        sets: vec![set],
        view: ViewName::SftFinal,
        strip: None,
        min_strength: Some(Strength::Judged),
        system_prompt: None,
        export_only: false,
        limit: None,
        voice: VoiceBuild::default(),
    };
    let retrieval = Retrieval {
        library: &library,
        embedder: &Concepts,
        passages: 2,
        rerank: None,
    };
    let read = |built: &splinter_pipelines::datasets::Built| -> serde_json::Value {
        let text = std::fs::read_to_string(&built.path).unwrap();
        serde_json::from_str(text.lines().next().unwrap()).unwrap()
    };
    let plain = read(&build_with(&ctx, &request, None).unwrap());
    let share = PassageShare {
        records: 1.0,
        with_evidence: 0.0,
        abstain: 1.0,
        unsupported: 0.0,
    };
    let abstaining = read(
        &build_with(
            &ctx,
            &request,
            Some(&Passages {
                retrieval: &retrieval,
                share,
                abstainer: Some(&Writer),
            }),
        )
        .unwrap(),
    );
    assert!(abstaining["messages"][2]["content"]
        .as_str()
        .unwrap()
        .starts_with("Miss: "));
    assert!(plain["metadata"]["group"].is_string());
    assert_eq!(abstaining["metadata"]["group"], plain["metadata"]["group"]);
    assert_eq!(
        abstaining["metadata"]["experiences"],
        plain["metadata"]["experiences"]
    );
}

/// The model that writes abstentions is the person under their prompt, and
/// is sent back for a reply that states a number nobody showed it.
#[test]
fn the_model_writes_the_abstention_as_the_person_and_is_corrected() {
    use splinter_agent::solve::Model;
    use splinter_agent::CancelToken;
    use splinter_pipelines::abstain::ModelAbstainer;
    use std::sync::{Arc, Mutex};

    let replies = Arc::new(Mutex::new(vec![
        r#"{"reply": "The writings before me do not establish this, though I wrote of it in 1787 and will say no more than they do on the matter."}"#,
        r#"{"reply": "The writings before me do not establish my view on this matter, though they touch on the education of the young, and I will not guess beyond them."}"#,
    ]));
    let queue = replies.clone();
    let writer = Scripted::new(move |_| queue.lock().unwrap().remove(0).to_string());
    let (ctx, _scratch, library, _record) = fixture("raft-model");
    let model = Model::new(Arc::new(writer.clone()), "scripted/writer");
    let abstainer = ModelAbstainer::new(
        &ctx,
        model,
        "You are the Writer.".into(),
        None,
        CancelToken::new(),
    );
    let shown: Vec<&Passage> = library.passages().iter().take(2).collect();
    let reply = abstainer
        .abstention("What of the tariff?", &shown, Abstention::Miss)
        .unwrap();
    assert!(!reply.contains("1787"), "{reply}");
    assert!(
        replies.lock().unwrap().is_empty(),
        "the first was sent back"
    );
    let systems = writer.systems.lock().unwrap();
    assert!(systems
        .iter()
        .flatten()
        .any(|s| s.contains("You are the Writer.")));
}
