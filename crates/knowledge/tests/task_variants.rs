// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model-driven task generation that turns
// any source into verifiable training tasks, for its clients. If your team
// needs expertise in synthetic task generation or verifier design, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: the generator model writes differently worded questions about the
//! fact a task asks, and code decides which are admitted. A variant keeps
//! the original's kind, evidence, environment, reference and concepts, so
//! the same verifiers grade it; it asks in other words (not a repeat of the
//! original or of a sibling), stands on its own, still names the subject
//! the original is about, cites a section it was shown and states the
//! original's reference. One request shows the one section the task's
//! evidence falls in and the subject, and nothing else of the source.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use common::{about, entry, generator, reply, stored, Scratch, Scripted};
use splinter_core::experience::{Environment, Privileged, PrivilegedKind, Task};
use splinter_knowledge::tasks::{can_vary, Catalogue, GenerateError, Rejection};

const MANUAL: &str = "# Frobnicator manual

The Frobnicator is a bench power controller.

## Console

The console UART of the Frobnicator runs at 115200 baud with eight data bits.

## Power

The Frobnicator draws 40 mA when idle and 900 mA at full load.
";

const QUESTION: &str = "At what baud rate does the Frobnicator console UART run?";

/// What [`QUESTION`] is about.
const SUBJECT: &str = "Frobnicator";

/// The recall task the variants are written for, generated the way `tasks`
/// generates it.
async fn original(
    model: &std::sync::Arc<Scripted>,
    store: &splinter_record::sources::SourceStore,
    source: &splinter_knowledge::tasks::SourceText,
) -> Task {
    let report = generator(model.clone(), store.clone(), vec![])
        .generate(source, &[Catalogue::builtin().get("recall").unwrap()])
        .await
        .unwrap();
    report.admitted[0].task.clone()
}

fn first_reply() -> String {
    reply(vec![about(
        entry(QUESTION, "115200 baud", 1, Some("runs at 115200 baud")),
        SUBJECT,
    )])
}

#[tokio::test]
async fn variants_keep_the_fact_and_change_the_words() {
    let scratch = Scratch::new("variants-admitted");
    let (store, _, source) = stored(&scratch, MANUAL);
    let model = Scripted::new(vec![
        first_reply(),
        reply(vec![
            entry(
                "Which baud rate is the Frobnicator's console UART set to?",
                "115200 baud",
                0,
                None,
            ),
            entry(
                "What speed, in baud, does the Frobnicator console serial line use?",
                "115200 BAUD.",
                0,
                None,
            ),
        ]),
    ]);
    let original = original(&model, &store, &source).await;

    let report = generator(model.clone(), store, vec![])
        .variants(&original, SUBJECT, 3)
        .await
        .unwrap();
    assert_eq!(report.admitted.len(), 2, "{report:#?}");
    for variant in &report.admitted {
        let task = &variant.task;
        assert_ne!(task.task.id, original.task.id);
        assert_ne!(task.instruction, original.instruction);
        assert_eq!(task.task.kind, original.task.kind);
        assert_eq!(task.evidence, original.evidence);
        assert_eq!(task.environment, Environment::closed_book());
        assert_eq!(task.privileged, original.privileged);
        assert_eq!(task.concepts, original.concepts);
        assert_eq!(variant.subject.as_deref(), Some(SUBJECT));
    }

    // One request, showing the section the evidence falls in and not the
    // others; it names the question and the answer.
    let seen = model.seen.lock().unwrap();
    let request = format!("{:?}", seen.last().unwrap().messages);
    assert_eq!(seen.len(), 2);
    assert!(
        request.contains("115200 baud with eight data bits"),
        "{request}"
    );
    assert!(!request.contains("900 mA at full load"), "{request}");
    assert!(request.contains(QUESTION), "{request}");
    assert!(request.contains("names its subject"), "{request}");
}

#[tokio::test]
async fn a_variant_that_is_not_a_new_wording_of_the_same_fact_is_rejected() {
    let scratch = Scratch::new("variants-rejected");
    let (store, _, source) = stored(&scratch, MANUAL);
    let model = Scripted::new(vec![
        first_reply(),
        reply(vec![
            entry(QUESTION, "115200 baud", 0, None),
            entry(
                "How much current does the Frobnicator draw when idle?",
                "40 mA",
                0,
                None,
            ),
            entry(
                "According to the passage above, what is the console baud rate?",
                "115200 baud",
                0,
                None,
            ),
            entry(
                "Which baud rate is the console UART set to?",
                "115200 baud",
                7,
                None,
            ),
            entry(
                "Which baud rate is the Frobnicator console UART set to?",
                "115200 baud",
                0,
                None,
            ),
            entry(
                "Which baud rate is the Frobnicator console UART set to, exactly?",
                "115200 baud",
                0,
                None,
            ),
            // Which console? The rewording dropped the board.
            entry(
                "What baud rate does the console UART have?",
                "115200 baud",
                0,
                None,
            ),
            entry(
                "Tell me the console UART baud rate of the Frobnicator.",
                "115200 baud",
                0,
                None,
            ),
        ]),
    ]);
    let original = original(&model, &store, &source).await;

    let report = generator(model, store, vec![])
        .variants(&original, SUBJECT, 7)
        .await
        .unwrap();
    assert_eq!(report.admitted.len(), 1, "{report:#?}");
    assert_eq!(report.count(Rejection::Duplicate), 1, "the original again");
    assert_eq!(report.count(Rejection::ReferenceChanged), 1, "another fact");
    assert_eq!(report.count(Rejection::NotSelfContained), 1, "{report:#?}");
    assert_eq!(report.count(Rejection::UnknownSection), 1, "{report:#?}");
    assert_eq!(
        report.count(Rejection::NearDuplicate),
        1,
        "a sibling nearly repeated"
    );
    assert_eq!(report.count(Rejection::NoSubject), 1, "{report:#?}");
    assert_eq!(
        report.count(Rejection::OverCount),
        1,
        "seven were asked for"
    );
    assert_eq!(
        report.admitted[0].task.instruction,
        "Which baud rate is the Frobnicator console UART set to?"
    );
}

#[tokio::test]
async fn only_a_closed_book_text_fact_with_evidence_can_be_varied() {
    let scratch = Scratch::new("variants-eligible");
    let (store, _, source) = stored(&scratch, MANUAL);
    let model = Scripted::new(vec![first_reply()]);
    let original = original(&model, &store, &source).await;
    assert_eq!(can_vary(&original), Ok(()));

    let reference = Privileged {
        kind: PrivilegedKind::Reference,
        content: "115200 baud".into(),
        span: None,
    };
    let task = |kind: &str, environment: Environment, evidence, privileged| {
        Task::new(kind, evidence, environment, QUESTION, privileged).unwrap()
    };
    let no_evidence = task(
        "recall",
        Environment::closed_book(),
        vec![],
        vec![reference.clone()],
    );
    assert!(can_vary(&no_evidence).is_err());
    let no_reference = task(
        "recall",
        Environment::closed_book(),
        original.evidence.clone(),
        vec![],
    );
    assert!(can_vary(&no_reference).is_err());
    let computed = task(
        "predict",
        Environment::closed_book(),
        original.evidence.clone(),
        vec![reference.clone()],
    );
    assert!(
        can_vary(&computed).is_err(),
        "the instruction carries material"
    );
    let unknown = task(
        "denoise",
        Environment::closed_book(),
        original.evidence.clone(),
        vec![reference],
    );
    assert!(can_vary(&unknown).is_err());

    let refused = generator(Scripted::new(vec![]), store.clone(), vec![])
        .variants(&no_evidence, SUBJECT, 3)
        .await
        .unwrap_err();
    assert!(
        matches!(refused, GenerateError::NotVaryable { .. }),
        "{refused}"
    );
    // A subject the question does not name cannot be kept.
    let refused = generator(Scripted::new(vec![]), store, vec![])
        .variants(&original, "Widget", 3)
        .await
        .unwrap_err();
    assert!(
        matches!(
            refused,
            GenerateError::NotVaryable {
                reason: "no_subject",
                ..
            }
        ),
        "{refused}"
    );
}
