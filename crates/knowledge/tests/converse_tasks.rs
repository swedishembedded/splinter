// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements task generation that teaches a model to
// converse as a person, grounded in what that person wrote, for its clients.
// If your team needs expertise in turning a person's writings into
// verifiable conversation training, you can procure our services by sending
// an email to info@swedishembedded.com.

//! Spec: a conversation as the writer is a kind of task. Its instruction is
//! the opening message of someone speaking to the writer, standing on its own
//! and naming no document; its reference is the passage of the cited section,
//! word for word, the writer would draw on. The answer is a dialogue, graded
//! by what it states: every number, name and quotation must be in the
//! writing or in what the other speaker said.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use common::{entry, generator, reply, stored, Scratch, Scripted};
use splinter_knowledge::advice::{judgment_cues, judgment_sections, reads_as_judgment};
use splinter_knowledge::tasks::{Catalogue, Focus, Rejection, VerifierKind};

const LETTER: &str = "# To a young man

## Study

I advise you to fix a habit of study every morning before you do anything else. Never let a day pass without reading something of history or ethics, and always write down what you have read, for the memory fails what the pen has not fixed.
";

const PASSAGE: &str = "I advise you to fix a habit of study every morning before you do anything else. Never let a day pass without reading something of history or ethics, and always write down what you have read, for the memory fails what the pen has not fixed.";

const OPENING: &str =
    "I keep meaning to read more but my mornings vanish; how do you keep a habit of study?";

fn converse() -> splinter_knowledge::tasks::TaskKind {
    Catalogue::builtin().get("converse").unwrap().clone()
}

#[test]
fn the_converse_kind_is_a_dialogue_graded_by_grounding() {
    let kind = converse();
    kind.validate().unwrap();
    assert!(kind.dialogue);
    assert!(kind.verifiers.contains(&VerifierKind::Grounding));
    assert!(kind.reference_verbatim);
    assert!(!kind.names_subject());
    assert!(!kind.exact_answer());
    assert!(
        Catalogue::builtin()
            .kinds()
            .filter(|k| k.name != "converse")
            .all(|k| !k.dialogue),
        "only the converse kind is a dialogue"
    );
}

#[tokio::test]
async fn an_opening_message_with_a_verbatim_passage_is_admitted() {
    let scratch = Scratch::new("converse-admitted");
    let (store, _, source) = stored(&scratch, LETTER);
    let model = Scripted::new(vec![reply(vec![entry(OPENING, PASSAGE, 1, Some(PASSAGE))])]);
    let report = generator(model, store, vec![])
        .generate(&source, &[&converse()])
        .await
        .unwrap();
    assert_eq!(report.admitted.len(), 1, "{report:#?}");
    assert_eq!(report.admitted[0].task.task.kind, "converse");
    assert_eq!(report.admitted[0].task.instruction, OPENING);
}

#[tokio::test]
async fn a_passage_the_writer_did_not_write_is_refused() {
    let scratch = Scratch::new("converse-paraphrase");
    let (store, _, source) = stored(&scratch, LETTER);
    let model = Scripted::new(vec![reply(vec![entry(
        OPENING,
        "Study every morning and write down what you read.",
        1,
        Some(PASSAGE),
    )])]);
    let report = generator(model, store, vec![])
        .generate(&source, &[&converse()])
        .await
        .unwrap();
    assert!(report.admitted.is_empty(), "{report:#?}");
    assert_eq!(report.count(Rejection::NotQuoted), 1);
}

const OPINION: &str = "I am of opinion that the public debt is a curse upon a nation, and I believe that no generation has a right to bind another by loans it cannot repay within the term of its own life, for the earth belongs to the living and not to the dead.";

const BUSINESS: &str = "I have received your favour of the tenth and enclose the bill of lading for the hogsheads of tobacco shipped on the brig Eliza, which should reach Havre within the month if the wind holds fair.";

#[test]
fn a_conversation_is_shown_the_sections_where_the_writer_judges_not_every_section() {
    assert_eq!(converse().focus, Some(Focus::Judgment));
    assert!(judgment_cues(OPINION) >= 2);
    assert_eq!(judgment_cues(BUSINESS), 0);
    // Opinion and advice both read as judgment; business does not.
    assert!(reads_as_judgment(OPINION));
    assert!(reads_as_judgment(PASSAGE));
    assert!(!reads_as_judgment(BUSINESS));
    assert!(
        !reads_as_judgment("I think so."),
        "too short to hold a passage"
    );

    let scratch = Scratch::new("converse-judgment");
    let letter = format!("# To a friend\n\n## Business\n\n{BUSINESS}\n\n## Debt\n\n{OPINION}\n\n## Study\n\n{PASSAGE}\n");
    let (_, _, source) = stored(&scratch, &letter);
    let chosen = judgment_sections(&source);
    assert_eq!(chosen.len(), 2, "{chosen:?}");
    let texts: Vec<&str> = chosen
        .iter()
        .map(|&n| source.section_text(n).unwrap())
        .collect();
    assert!(texts[0].contains("public debt") && texts[1].contains("habit of study"));
}
