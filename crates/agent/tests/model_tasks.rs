// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model-driven task generation that turns
// any source into verifiable training tasks, for its clients. If your team
// needs expertise in synthetic task generation or verifier design, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: a generator model proposes tasks of a kind from a source's
//! sections, and code decides which are admitted. A task's evidence must
//! resolve to the source's bytes and support its reference (or, for a
//! computed answer, its reference must pass its own checks in the sandbox
//! and its generated tests must survive mutation validation); its
//! instruction must stand on its own and, for a question answered from the
//! source, name its subject - what it is about, as the source or a cited
//! section names it - so that it has one answer; two answers to one
//! question of one subject are both refused; a malformed reply is sent back for
//! one correction, and one still malformed is rejected; duplicates are
//! rejected and counted by reason; a kind is data, so a new one needs no
//! generator code.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use common::{about, entry, generator, python, reply, stored, Scratch, Scripted, PART};
use serde_json::json;
use splinter_core::clock::FixedClock;
use splinter_core::experience::{Environment, PrivilegedKind};
use splinter_core::kinds::{EXECUTABLE_CHECK, GENERATED_TEST};
use splinter_core::source::{CapturedSource, Origin, PartContent, Revision};
use splinter_knowledge::tasks::{Catalogue, Rejection, SourceText, TaskKind};
use splinter_store::sources::SourceStore;
use splinter_store::StateRoot;
use sven_sdk::model::ResponseFormat;

const MANUAL: &str = "# Frobnicator manual

The Frobnicator is a bench power controller.

## Console

The console UART of the Frobnicator runs at 115200 baud with eight data bits.

## Power

The Frobnicator draws 40 mA when idle and 900 mA at full load.
";

const CLAMP_SPEC: &str = "# Numeric helpers

## clamp

clamp(x, lo, hi) returns lo when x is below lo, hi when x is above hi, and x otherwise.
";

const CLAMP: &str = "def clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
";

const STRONG_TEST: &str = "assert clamp(-1, 0, 10) == 0
assert clamp(11, 0, 10) == 10
assert clamp(5, 0, 10) == 5
assert clamp(0, 0, 10) == 0
assert clamp(10, 0, 10) == 10
";

fn kind(name: &str) -> TaskKind {
    Catalogue::builtin().get(name).unwrap().clone()
}

#[tokio::test]
async fn a_grounded_recall_task_is_admitted_and_an_ungrounded_one_rejected() {
    let scratch = Scratch::new("model-tasks-recall");
    let (store, id, source) = stored(&scratch, MANUAL);
    let model = Scripted::new(vec![reply(vec![
        about(
            entry(
                "At what baud rate does the Frobnicator console UART run?",
                "115200 baud",
                1,
                Some("runs at 115200 baud"),
            ),
            "Frobnicator",
        ),
        about(
            entry(
                "How much current does the Frobnicator draw in standby?",
                "12 mA over USB-C",
                2,
                None,
            ),
            "Frobnicator",
        ),
    ])]);
    let generator = generator(model, store.clone(), vec![]);
    let report = generator
        .generate(&source, &[&kind("recall")])
        .await
        .unwrap();

    assert_eq!(report.admitted.len(), 1, "{report:#?}");
    assert_eq!(report.count(Rejection::Ungrounded), 1, "{report:#?}");
    let counts = &report.per_kind["recall"];
    assert_eq!((counts.admitted, counts.rejected), (1, 1));

    let task = &report.admitted[0].task;
    assert_eq!(task.task.kind, "recall");
    assert_eq!(task.environment, Environment::closed_book());
    assert_eq!(task.evidence.len(), 1);
    let span = &task.evidence[0];
    let part = span.part.as_ref().unwrap();
    assert_eq!((&part.source, part.name.as_str()), (&id, PART));
    let bytes = store.read_span(span).unwrap();
    assert_eq!(bytes, b"runs at 115200 baud");
    // The span lies inside the Console section, and nowhere else.
    let console = MANUAL.find("## Console").unwrap() as u64;
    let power = MANUAL.find("## Power").unwrap() as u64;
    assert!(console <= span.start && span.end <= power, "{span:?}");
    let reference: Vec<&str> = task
        .privileged
        .iter()
        .filter(|p| p.kind == PrivilegedKind::Reference)
        .map(|p| p.content.as_str())
        .collect();
    assert_eq!(reference, ["115200 baud"]);
    assert!(report.admitted[0]
        .generator
        .ends_with("scripted/generator-1"));
    assert_eq!(report.admitted[0].subject.as_deref(), Some("Frobnicator"));
}

#[tokio::test]
async fn an_instruction_that_needs_the_hidden_passage_is_rejected() {
    let scratch = Scratch::new("model-tasks-contained");
    let (store, _, source) = stored(&scratch, MANUAL);
    let model = Scripted::new(vec![reply(vec![entry(
        "According to the passage above, at what baud rate does the console run?",
        "115200 baud",
        1,
        None,
    )])]);
    let report = generator(model, store, vec![])
        .generate(&source, &[&kind("recall")])
        .await
        .unwrap();
    assert!(report.admitted.is_empty(), "{report:#?}");
    assert_eq!(report.count(Rejection::NotSelfContained), 1, "{report:#?}");
}

fn construct(
    instruction: &str,
    reference: &str,
    checks: &[&str],
    tests: &[&str],
) -> serde_json::Value {
    let mut task = entry(instruction, reference, 1, None);
    let as_checks = |codes: &[&str]| -> Vec<serde_json::Value> {
        codes
            .iter()
            .map(|code| json!({ "code": code, "exit_code": 0 }))
            .collect()
    };
    task["checks"] = json!(as_checks(checks));
    task["tests"] = json!(as_checks(tests));
    task
}

#[tokio::test]
async fn a_construct_task_is_admitted_only_when_its_reference_passes_and_its_tests_kill_mutants() {
    let scratch = Scratch::new("model-tasks-construct");
    let (store, _, source) = stored(&scratch, CLAMP_SPEC);
    let env = python(&scratch);
    let model = Scripted::new(vec![reply(vec![
        construct(
            "Write a Python function clamp(x, lo, hi) that limits x to the closed range from lo to hi.",
            CLAMP,
            &["assert clamp(-5, 0, 3) == 0\nassert clamp(7, 0, 3) == 3\n"],
            &[STRONG_TEST],
        ),
        construct(
            "Implement clamp(x, lo, hi) in Python so that values outside [lo, hi] are pulled to the nearest bound.",
            "def clamp(x, lo, hi):\n    return x\n",
            &["assert clamp(-5, 0, 3) == 0\n"],
            &[],
        ),
        construct(
            "In Python, define clamp(x, lo, hi): below lo gives lo, above hi gives hi, otherwise x itself.",
            CLAMP,
            &["assert clamp(1, 0, 3) == 1\n"],
            &["assert True\n"],
        ),
    ])]);
    let report = generator(model, store, vec![env])
        .generate(&source, &[&kind("construct")])
        .await
        .unwrap();

    assert_eq!(report.admitted.len(), 1, "{report:#?}");
    assert_eq!(report.count(Rejection::ChecksFailed), 1, "{report:#?}");
    assert_eq!(report.count(Rejection::TestNotAdmitted), 1, "{report:#?}");

    let task = &report.admitted[0].task;
    assert_eq!(task.environment.kind, "runtime:python3");
    let kinds: Vec<&PrivilegedKind> = task.privileged.iter().map(|p| &p.kind).collect();
    assert!(kinds.contains(&&PrivilegedKind::Other(EXECUTABLE_CHECK.into())));
    assert!(kinds.contains(&&PrivilegedKind::Other(GENERATED_TEST.into())));
    assert!(kinds.contains(&&PrivilegedKind::Reference));
}

#[tokio::test]
async fn a_predicted_output_is_admitted_only_when_running_the_shown_code_produces_it() {
    let scratch = Scratch::new("model-tasks-predict");
    let (store, _, source) = stored(&scratch, CLAMP_SPEC);
    let env = python(&scratch);
    let predict = |code: &str, output: &str| {
        let mut task = entry(
            &format!("What does this Python program print?\n\n{code}"),
            output,
            1,
            None,
        );
        task["material"] = json!(code);
        task
    };
    let model = Scripted::new(vec![reply(vec![
        predict("print(min(max(12, 0), 10))", "10"),
        predict("print(min(max(-4, 0), 10) + 1)", "0"),
    ])]);
    let report = generator(model, store, vec![env])
        .generate(&source, &[&kind("predict")])
        .await
        .unwrap();
    assert_eq!(report.admitted.len(), 1, "{report:#?}");
    assert_eq!(report.count(Rejection::ChecksFailed), 1, "{report:#?}");
    assert_eq!(
        report.admitted[0].task.environment,
        Environment::closed_book()
    );
}

#[tokio::test]
async fn malformed_replies_are_rejected_without_panicking() {
    let scratch = Scratch::new("model-tasks-malformed");
    let (store, _, source) = stored(&scratch, MANUAL);
    let recall = kind("recall");
    let replies = [
        "Sure! Here are some tasks.".to_string(),
        "{\"tasks\": [{\"instruction\": \"What is the baud rate?\"}]}".to_string(),
        "{\"tasks\": [".to_string(),
        json!({ "tasks": [{
            "instruction": "At what baud rate does the Frobnicator console UART run?",
            "reference": "115200 baud",
            "evidence": [{ "section": 1 }],
            "confidence": 0.9,
        }]})
        .to_string(),
        String::new(),
    ];
    let count = replies.len();
    // Each is sent back once for correction and comes back as malformed.
    let twice = replies
        .iter()
        .flat_map(|r| [r.clone(), r.clone()])
        .collect();
    let model = Scripted::new(twice);
    let generator = generator(model, store, vec![]);
    let kinds = vec![&recall; count];
    let report = generator.generate(&source, &kinds).await.unwrap();
    assert!(report.admitted.is_empty(), "{report:#?}");
    assert_eq!(report.count(Rejection::Malformed), count, "{report:#?}");
}

#[tokio::test]
async fn a_malformed_reply_corrected_on_request_is_admitted() {
    let scratch = Scratch::new("model-tasks-corrected");
    let (store, _, source) = stored(&scratch, MANUAL);
    let recall = kind("recall");
    let model = Scripted::new(vec![
        "Sure! Here are some tasks.".to_string(),
        reply(vec![about(
            entry(
                "At what baud rate does the Frobnicator console UART run?",
                "115200 baud",
                1,
                Some("115200 baud"),
            ),
            "Frobnicator",
        )]),
    ]);
    let generator = generator(model.clone(), store, vec![]);
    let report = generator.generate(&source, &[&recall]).await.unwrap();
    assert_eq!(report.admitted.len(), 1, "{report:#?}");
    assert_eq!(report.count(Rejection::Malformed), 0, "{report:#?}");
    assert_eq!(model.seen.lock().unwrap().len(), 2, "one correction round");
}

/// A local model has no constrained decoding: it learns the reply's shape
/// only from the prompt, so the prompt carries the schema the provider is
/// asked to constrain to.
#[tokio::test]
async fn the_model_is_shown_the_reply_schema_in_its_prompt() {
    let scratch = Scratch::new("model-tasks-schema");
    let (store, _, source) = stored(&scratch, MANUAL);
    let model = Scripted::new(vec![reply(vec![])]);
    let generator = generator(model.clone(), store, vec![]);
    generator
        .generate(&source, &[&kind("recall")])
        .await
        .unwrap();
    let seen = model.seen.lock().unwrap();
    let request = &seen[0];
    let Some(ResponseFormat::JsonSchema { schema, .. }) = &request.response_format else {
        panic!(
            "a typed call names its schema: {:?}",
            request.response_format
        );
    };
    let prompt = format!("{:?}", request.messages);
    let shown = format!("{:?}", schema.to_string());
    assert!(
        prompt.contains(shown.trim_matches('"')),
        "the prompt shows the schema {schema}: {prompt}"
    );
}

#[tokio::test]
async fn duplicates_within_a_batch_are_dropped() {
    let scratch = Scratch::new("model-tasks-dedup");
    let (store, _, source) = stored(&scratch, MANUAL);
    let question = "At what baud rate does the Frobnicator console UART run?";
    let model = Scripted::new(vec![reply(
        [
            entry(question, "115200 baud", 1, None),
            entry(question, "115200 baud", 1, None),
            // The same answer, more briefly: a repeat, not a contradiction.
            entry(
                "  at what BAUD rate does the Frobnicator console UART   run? ",
                "115200",
                1,
                None,
            ),
            entry(
                "At what baud rate does the Frobnicator console UART run, exactly?",
                "115200 baud",
                1,
                None,
            ),
            entry(
                "How much current does the Frobnicator draw when idle?",
                "40 mA",
                2,
                None,
            ),
        ]
        .into_iter()
        .map(|task| about(task, "Frobnicator"))
        .collect(),
    )]);
    let report = generator(model, store, vec![])
        .generate(&source, &[&kind("recall")])
        .await
        .unwrap();
    assert_eq!(report.admitted.len(), 2, "{report:#?}");
    assert_eq!(
        report.count(Rejection::Duplicate) + report.count(Rejection::NearDuplicate),
        3,
        "{report:#?}"
    );
    assert_eq!(report.count(Rejection::NearDuplicate), 1, "{report:#?}");
    assert_eq!(report.per_kind["recall"].admitted, 2);
}

#[tokio::test]
async fn a_kind_defined_as_data_at_runtime_generates_tasks() {
    let scratch = Scratch::new("model-tasks-new-kind");
    let (store, _, source) = stored(&scratch, MANUAL);
    let definition: TaskKind = serde_json::from_value(json!({
        "name": "quantity",
        "brief": "Write up to {count} questions, each asking for one measured quantity the \
                  section states, with its unit.",
        "answer": "text",
        "verifiers": ["formal"],
    }))
    .unwrap();
    let mut catalogue = Catalogue::builtin();
    catalogue.insert(definition).unwrap();
    let model = Scripted::new(vec![reply(vec![about(
        entry(
            "How much current does the Frobnicator draw at full load?",
            "900 mA",
            2,
            None,
        ),
        "Frobnicator",
    )])]);
    let generator = generator(model.clone(), store, vec![]);
    let report = generator
        .generate(&source, &[catalogue.get("quantity").unwrap()])
        .await
        .unwrap();
    assert_eq!(report.admitted.len(), 1, "{report:#?}");
    assert_eq!(report.admitted[0].task.task.kind, "quantity");
    let prompt = format!("{:?}", model.seen.lock().unwrap()[0].messages);
    assert!(
        prompt.contains("one measured quantity the section states"),
        "the kind's brief reaches the model"
    );
}

/// The baud-rate question as the model writes it, naming `subject`.
fn baud(instruction: &str, subject: Option<&str>) -> serde_json::Value {
    let task = entry(instruction, "115200 baud", 1, Some("runs at 115200 baud"));
    match subject {
        Some(subject) => about(task, subject),
        None => task,
    }
}

#[tokio::test]
async fn a_question_answered_from_the_source_must_name_a_subject_the_source_names() {
    let scratch = Scratch::new("model-tasks-subject");
    let (store, _, source) = stored(&scratch, MANUAL);
    let model = Scripted::new(vec![reply(vec![
        // Which console? Every board has its own baud rate.
        baud("At what baud rate does the console UART run?", None),
        baud("At what baud rate does the console UART run?", Some("  ")),
        baud(
            "At what baud rate does the console UART run?",
            Some("Frobnicator"),
        ),
        baud(
            "At what baud rate does the Nucleo-64 console UART run?",
            Some("Nucleo-64"),
        ),
        baud(
            "At what baud rate does the console UART of the Frobnicator run?",
            Some("the"),
        ),
        baud(
            "At what baud rate does the console UART of the  frobnicator run?",
            Some("Frobnicator"),
        ),
    ])]);
    let report = generator(model, store, vec![])
        .generate(&source, &[&kind("recall")])
        .await
        .unwrap();
    assert_eq!(report.count(Rejection::NoSubject), 5, "{report:#?}");
    assert_eq!(report.admitted.len(), 1, "{report:#?}");
    assert_eq!(report.admitted[0].subject.as_deref(), Some("Frobnicator"));
    let details: Vec<&str> = report
        .rejections
        .iter()
        .map(|r| r.detail.as_str())
        .collect();
    assert!(details[0].contains("no subject"), "{details:?}");
    assert!(details[3].contains("Nucleo-64"), "{details:?}");
    // A refusal shows the instruction it refused.
    assert!(
        details[3].contains("(proposed: \"At what baud rate"),
        "{details:?}"
    );

    // Code shown in the instruction carries what its answer depends on.
    let scratch = Scratch::new("model-tasks-subject-code");
    let (store, _, source) = stored(&scratch, CLAMP_SPEC);
    let mut task = entry(
        "What does this Python program print?\n\nprint(min(max(12, 0), 10))",
        "10",
        1,
        None,
    );
    task["material"] = json!("print(min(max(12, 0), 10))");
    let model = Scripted::new(vec![reply(vec![task])]);
    let report = generator(model, store, vec![python(&scratch)])
        .generate(&source, &[&kind("predict")])
        .await
        .unwrap();
    assert_eq!(report.admitted.len(), 1, "{report:#?}");
    assert_eq!(report.admitted[0].subject, None);
}

#[tokio::test]
async fn the_generator_is_shown_what_the_source_is_and_may_name_it_as_the_subject() {
    let scratch = Scratch::new("model-tasks-identity");
    let store = SourceStore::new(&splinter_store::workspace::Workspace::at(&StateRoot::new(
        scratch.0.join("state"),
    )));
    let manual =
        "# Quark Q1 reference manual\n\n## Console\n\nThe console UART runs at 57600 baud.\n";
    let captured = CapturedSource::new(
        Origin::Repository {
            path: "/work/quark-q1".into(),
            revision: Some(Revision {
                commit: Some("3f2a9c1".into()),
                dirty: false,
            }),
            skipped: vec![],
        },
        vec![PartContent {
            name: "guide/manual.md".into(),
            media_type: "text/markdown".into(),
            bytes: manual.as_bytes().to_vec(),
        }],
        &FixedClock::new("2026-09-30T08:00:00.000Z"),
    )
    .unwrap();
    let id = store.put_source(&captured).unwrap();
    // Only the console section: it never says which board it is about.
    let shown = SourceText::load(&store, &id, "guide/manual.md")
        .unwrap()
        .select(&[1])
        .unwrap();
    let model = Scripted::new(vec![reply(vec![about(
        entry(
            "At what baud rate does the console UART of the Quark Q1 run?",
            "57600 baud",
            0,
            None,
        ),
        "Quark Q1",
    )])]);
    let report = generator(model.clone(), store, vec![])
        .generate(&shown, &[&kind("recall")])
        .await
        .unwrap();
    assert_eq!(report.admitted.len(), 1, "{report:#?}");
    let prompt = format!("{:?}", model.seen.lock().unwrap()[0].messages);
    for named in [
        "Quark Q1 reference manual",
        "quark-q1",
        "guide/manual.md",
        "3f2a9c1",
    ] {
        assert!(prompt.contains(named), "{named} is shown: {prompt}");
    }
}

#[tokio::test]
async fn two_answers_to_one_question_of_one_subject_are_both_refused() {
    let scratch = Scratch::new("model-tasks-contradiction");
    let manual = "# Frobnicator manual

## Ports

The console UART of the Frobnicator runs at 115200 baud. Its debug UART runs at 9600 baud.

## Power

The Frobnicator draws 40 mA when idle.
";
    let (store, _, source) = stored(&scratch, manual);
    let question = "At what baud rate does the Frobnicator UART run?";
    let model = Scripted::new(vec![reply(
        [
            entry(question, "115200 baud", 1, None),
            entry(question, "9600 baud", 1, None),
            entry(
                "How much current does the Frobnicator draw when idle?",
                "40 mA",
                2,
                None,
            ),
        ]
        .into_iter()
        .map(|task| about(task, "Frobnicator"))
        .collect(),
    )]);
    let report = generator(model, store, vec![])
        .generate(&source, &[&kind("recall")])
        .await
        .unwrap();
    assert_eq!(report.count(Rejection::Contradiction), 2, "{report:#?}");
    assert_eq!(report.admitted.len(), 1, "{report:#?}");
    assert!(report.admitted[0].task.instruction.contains("idle"));
    assert_eq!(report.per_kind["recall"].admitted, 1);
    assert_eq!(report.per_kind["recall"].rejected, 2);
}

#[test]
fn the_builtin_catalogue_ships_every_kind_and_refuses_an_inconsistent_one() {
    let catalogue = Catalogue::builtin();
    for name in [
        "recall",
        "explain",
        "predict",
        "construct",
        "debug",
        "counterexample",
        "transform",
        "classify",
        "retrieve",
        "multi-turn",
        "combine",
    ] {
        let kind = catalogue.get(name).unwrap();
        kind.validate().unwrap();
    }
    assert_eq!(catalogue.get("combine").unwrap().min_sections, 2);

    // A computed answer with nothing to check it by is not a kind.
    let unchecked: TaskKind = serde_json::from_value(json!({
        "name": "unchecked",
        "brief": "Write code.",
        "answer": "program",
        "runtime": "python3",
        "verifiers": ["executable"],
    }))
    .unwrap();
    assert!(Catalogue::builtin().insert(unchecked).is_err());
}
