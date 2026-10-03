// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements task generation that teaches a model what a
// person advised, grounded in what they wrote, for its clients. If your team
// needs expertise in turning a person's writings into verifiable training
// tasks, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: a person's own advice is a kind of task. The instruction is a
//! predicament put to the person, which names no document (it is addressed to
//! the person, so there is no subject to name); its reference is the advice,
//! and the reference must be a passage of the cited section word for word, so a
//! paraphrase the model wrote is refused and the advice a task teaches is the
//! person's own. Only the sections that read as advice are shown to the model
//! for the kind.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use common::{entry, generator, reply, stored, Scratch, Scripted};
use splinter_knowledge::advice::{advice_cues, advice_sections};
use splinter_knowledge::tasks::{Catalogue, Focus, Rejection, VerifierKind};

const LETTER: &str = "# To a young man

## Business

I have received your favour of the tenth and enclose the bill of lading for the hogsheads of tobacco shipped on the brig Eliza, which should reach Havre within the month.

## Study

I advise you to fix a habit of study every morning before you do anything else. Never let a day pass without reading something of history or ethics, and always write down what you have read, for the memory fails what the pen has not fixed.

## Weather

The weather here has been mild and the roads are passable.
";

const ADVICE: &str = "I advise you to fix a habit of study every morning before you do anything else. Never let a day pass without reading something of history or ethics, and always write down what you have read, for the memory fails what the pen has not fixed.";

const SITUATION: &str = "I am nineteen and have finished my first year at college, but I find my mornings slip away in idleness; what would you advise me to do about my habits?";

fn advise() -> splinter_knowledge::tasks::TaskKind {
    Catalogue::builtin().get("advise").unwrap().clone()
}

#[test]
fn the_advise_kind_is_a_closed_book_text_task_graded_by_quotation() {
    let kind = advise();
    kind.validate().unwrap();
    assert!(kind.verifiers.contains(&VerifierKind::Quotation));
    assert_eq!(kind.focus, Some(Focus::Advice));
    assert!(kind.reference_verbatim);
    assert!(
        !kind.names_subject(),
        "it is addressed to the person; no document is named"
    );
    assert!(
        !kind.exact_answer(),
        "advice may be worded differently and still agree"
    );
}

#[tokio::test]
async fn a_situation_answered_by_a_verbatim_passage_is_admitted_without_naming_a_subject() {
    let scratch = Scratch::new("advice-admitted");
    let (store, _, source) = stored(&scratch, LETTER);
    let model = Scripted::new(vec![reply(vec![entry(SITUATION, ADVICE, 2, Some(ADVICE))])]);
    let report = generator(model, store.clone(), vec![])
        .generate(&source, &[&advise()])
        .await
        .unwrap();
    assert_eq!(report.admitted.len(), 1, "{report:#?}");
    let task = &report.admitted[0].task;
    assert_eq!(task.task.kind, "advise");
    assert_eq!(task.instruction, SITUATION);
    let reference = task
        .privileged
        .iter()
        .find(|p| p.kind == splinter_core::experience::PrivilegedKind::Reference)
        .unwrap();
    assert_eq!(reference.content, ADVICE);
}

#[tokio::test]
async fn a_paraphrase_is_not_the_persons_advice_and_is_refused() {
    let scratch = Scratch::new("advice-paraphrase");
    let (store, _, source) = stored(&scratch, LETTER);
    let paraphrase = "Make studying every morning a habit, read history or ethics daily and write down what you read, because writing helps memory.";
    let model = Scripted::new(vec![reply(vec![entry(
        SITUATION,
        paraphrase,
        2,
        Some(ADVICE),
    )])]);
    let report = generator(model, store, vec![])
        .generate(&source, &[&advise()])
        .await
        .unwrap();
    assert!(report.admitted.is_empty(), "{report:#?}");
    assert_eq!(report.count(Rejection::NotQuoted), 1, "{report:#?}");
}

#[test]
fn only_sections_that_read_as_advice_are_offered() {
    let scratch = Scratch::new("advice-focus");
    let (_, _, source) = stored(&scratch, LETTER);
    let chosen = advice_sections(&source);
    assert_eq!(chosen.len(), 1, "{chosen:?}");
    let text = source.section_text(chosen[0]).unwrap();
    assert!(text.contains("I advise you to fix a habit of study"));
    assert!(advice_cues(ADVICE) >= 3);
    assert_eq!(advice_cues("The weather here has been mild."), 0);
}
