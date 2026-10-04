// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verifiers that grade a model's answers
// against source material without trusting the model, for its clients. If
// your team needs expertise in telling what a persona model actually found in
// the record from what it made up, you can procure our services by sending an
// email to info@swedishembedded.com.

//! The grounding block of an answer, and whether the answer keeps its word.
//!
//! An answer that applies his practice to a situation he never met ends in a
//! block that says, line by line, what each claim rests on: his own words, an
//! inference from them, a fact the situation supplied, a transfer of his
//! method, or a guess. Code reads the block. A quotation labelled as his must
//! be in the document it cites, a modern fact must come from what the
//! situation supplied, nothing labelled historical may use a word from after
//! his death, and an answer without the block fails. The model's own claim to
//! be grounded is not evidence; the lookup is.

use std::collections::HashSet;

use splinter_sdk::measure::verifiers::quotation::{quotations, words};

use crate::curate::Document;
use crate::principles::occurs;

/// Words a quotation must run to be checked as one.
const MIN_QUOTE_WORDS: usize = 6;
/// Words a quotation in the body of an answer must run to be a claim to a passage.
const MIN_BODY_QUOTE_WORDS: usize = 8;

/// What a line of the grounding block says it rests on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Label {
    /// His own words, quoted from a document.
    SourceDirect,
    /// An inference from his documents.
    SourceInferred,
    /// A fact about the world of the situation, given to the model.
    ModernObservation,
    /// His method carried to a problem he did not meet.
    PersonaTransfer,
    /// A guess.
    Speculation,
}

impl Label {
    fn parse(text: &str) -> Option<Label> {
        match text.trim().to_uppercase().as_str() {
            "SOURCE_DIRECT" => Some(Label::SourceDirect),
            "SOURCE_INFERRED" => Some(Label::SourceInferred),
            "MODERN_OBSERVATION" => Some(Label::ModernObservation),
            "PERSONA_TRANSFER" => Some(Label::PersonaTransfer),
            "SPECULATION" => Some(Label::Speculation),
            _ => None,
        }
    }
}

/// One line of the grounding block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Item {
    /// `None` when the line names no label this sample knows.
    pub label: Option<Label>,
    /// The label as written, so a wrong one can be named.
    pub raw_label: String,
    /// The line after its label.
    pub text: String,
    /// The words in double quotes, if any.
    pub quote: Option<String>,
    /// The document id in square brackets, if any.
    pub doc: Option<String>,
}

/// An answer split into what it says and what it says it rests on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Grounded {
    pub body: String,
    /// `None` when the answer has no grounding block.
    pub items: Option<Vec<Item>>,
}

/// Split an answer at its `Grounding:` line.
pub fn parse(answer: &str) -> Grounded {
    let lines: Vec<&str> = answer.lines().collect();
    let Some(at) = lines.iter().position(|l| {
        l.trim()
            .trim_end_matches(':')
            .trim()
            .eq_ignore_ascii_case("grounding")
    }) else {
        return Grounded {
            body: answer.trim().to_string(),
            items: None,
        };
    };
    let items = lines[at + 1..].iter().filter_map(|l| item_of(l)).collect();
    Grounded {
        body: lines[..at].join("\n").trim().to_string(),
        items: Some(items),
    }
}

/// One bullet line of the block, if it is one.
fn item_of(line: &str) -> Option<Item> {
    let line = line.trim().strip_prefix(['-', '*'])?.trim();
    let (raw_label, text) = line
        .split_once(':')
        .map_or((String::new(), line), |(l, t)| {
            (l.trim().to_string(), t.trim())
        });
    Some(Item {
        label: Label::parse(&raw_label),
        raw_label,
        text: text.to_string(),
        quote: quoted(text),
        doc: bracketed(text),
    })
}

/// The first passage in double quotes, straight or curly.
fn quoted(text: &str) -> Option<String> {
    let start = text.find(['"', '\u{201c}'])?;
    let rest = &text[start..];
    let rest = &rest[rest.chars().next()?.len_utf8()..];
    let end = rest.find(['"', '\u{201d}'])?;
    Some(rest[..end].trim().to_string())
}

/// The last passage in square brackets.
fn bracketed(text: &str) -> Option<String> {
    let end = text.rfind(']')?;
    let start = text[..end].rfind('[')?;
    Some(text[start + 1..end].trim().to_string())
}

/// How an answer broke its word.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Violation {
    /// There is no grounding block.
    MissingBlock,
    /// A line names a label that is not one of the five.
    UnknownLabel(String),
    /// A line labelled as his own words has no quotation, no document, or a
    /// quotation too short to check.
    UncitedSource(String),
    /// A quotation is not in the document it cites, or the document is not one
    /// the model may draw on.
    FabricatedQuote(String),
    /// A line labelled as a fact of the situation is not in what was supplied.
    UnsupportedFact(String),
    /// A line labelled as his, or a quotation in the answer, uses a word from
    /// after his death.
    Anachronism(String),
}

/// Words from after his death, which nothing labelled as his may use.
const ANACHRONISMS: [&str; 20] = [
    "internet",
    "computer",
    "computers",
    "software",
    "hardware",
    "email",
    "smartphone",
    "database",
    "algorithm",
    "startup",
    "website",
    "online",
    "digital",
    "laptop",
    "wifi",
    "blockchain",
    "server",
    "telegraph",
    "automobile",
    "television",
];

/// Share of a fact's words that must be in one observation for it to count as
/// supplied, and the fewest words a fact must have to be checked.
const FACT_SHARE: f64 = 0.8;
const MIN_FACT_WORDS: usize = 3;

/// Check an answer's grounding. `docs` are the documents it may cite, by id;
/// `observations` are the facts the situation supplied.
pub fn verify(grounded: &Grounded, docs: &[&Document], observations: &[String]) -> Vec<Violation> {
    let Some(items) = &grounded.items else {
        return vec![Violation::MissingBlock];
    };
    let mut found = Vec::new();

    for quote in quotations(&grounded.body, MIN_BODY_QUOTE_WORDS) {
        let quote_words = words(&quote);
        if !docs.iter().any(|d| occurs(&words(&d.body), &quote_words)) {
            found.push(Violation::FabricatedQuote(quote));
        }
    }
    for item in items {
        match item.label {
            None => found.push(Violation::UnknownLabel(item.raw_label.clone())),
            Some(Label::SourceDirect) => found.extend(check_source(item, docs)),
            Some(Label::SourceInferred) => {
                if let Some(word) = anachronism(&item.text) {
                    found.push(Violation::Anachronism(word));
                }
            }
            Some(Label::ModernObservation) => {
                if !supplied(&item.text, observations) {
                    found.push(Violation::UnsupportedFact(item.text.clone()));
                }
            }
            Some(Label::PersonaTransfer | Label::Speculation) => {}
        }
    }
    found
}

fn check_source(item: &Item, docs: &[&Document]) -> Option<Violation> {
    let (Some(quote), Some(id)) = (&item.quote, &item.doc) else {
        return Some(Violation::UncitedSource(item.text.clone()));
    };
    let quote_words = words(quote);
    if quote_words.len() < MIN_QUOTE_WORDS {
        return Some(Violation::UncitedSource(item.text.clone()));
    }
    let found = docs
        .iter()
        .find(|d| &d.id == id)
        .is_some_and(|d| occurs(&words(&d.body), &quote_words));
    (!found).then(|| Violation::FabricatedQuote(quote.clone()))
}

fn anachronism(text: &str) -> Option<String> {
    words(text)
        .into_iter()
        .find(|w| ANACHRONISMS.contains(&w.as_str()))
}

/// Whether `fact` is in one of the observations: nearly all its words are
/// there, and every number in it is.
fn supplied(fact: &str, observations: &[String]) -> bool {
    let fact = words(fact);
    if fact.len() < MIN_FACT_WORDS {
        return false;
    }
    observations.iter().any(|obs| {
        let seen: HashSet<String> = words(obs).into_iter().collect();
        let present = fact.iter().filter(|w| seen.contains(*w)).count();
        let numbers_ok = fact
            .iter()
            .filter(|w| w.chars().any(|c| c.is_ascii_digit()))
            .all(|w| seen.contains(w));
        numbers_ok && present as f64 >= FACT_SHARE * fact.len() as f64
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{Authorship, Date, Period};

    fn doc(id: &str, body: &str) -> Document {
        Document {
            schema: crate::curate::SCHEMA,
            id: id.into(),
            source_id: "cushing-1".into(),
            heading: "TO A.".into(),
            recipient: Some("A".into()),
            note: "[MS.]".into(),
            date: Date {
                year: 1770,
                month: None,
                day: None,
            },
            period: Period::of(1770),
            authorship: Authorship::DraftInHand,
            authorship_confidence: 0.8,
            authorship_basis: "test".into(),
            temporal_holdout: false,
            body: body.into(),
        }
    }

    const WORDS_OF_HIS: &str = "Let the Committee write to every Town, that the Sense of the People may be known before the Assembly meets again.";

    fn docs() -> Vec<Document> {
        vec![doc("d1", WORDS_OF_HIS)]
    }

    fn check(answer: &str, observations: &[&str]) -> Vec<Violation> {
        let d = docs();
        let refs: Vec<&Document> = d.iter().collect();
        let obs: Vec<String> = observations.iter().map(|s| s.to_string()).collect();
        verify(&parse(answer), &refs, &obs)
    }

    #[test]
    fn the_block_after_the_grounding_line_is_split_into_labelled_items() {
        let g = parse("Write to every team lead first.\n\nGrounding:\n- SOURCE_DIRECT: \"write to every Town\" [d1]\n- PERSONA_TRANSFER: carry the committee method to the teams\n");
        assert_eq!(g.body, "Write to every team lead first.");
        let items = g.items.unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].label, Some(Label::SourceDirect));
        assert_eq!(items[0].quote.as_deref(), Some("write to every Town"));
        assert_eq!(items[0].doc.as_deref(), Some("d1"));
        assert_eq!(items[1].label, Some(Label::PersonaTransfer));
        assert_eq!(items[1].quote, None);
    }

    #[test]
    fn an_answer_with_no_block_has_none_and_fails() {
        assert_eq!(parse("Just advice.").items, None);
        assert_eq!(check("Just advice.", &[]), vec![Violation::MissingBlock]);
    }

    #[test]
    fn labels_and_the_grounding_line_are_read_whatever_their_case() {
        let g = parse("Advice.\n\ngrounding\n- Persona_Transfer: carry it over\n");
        assert_eq!(g.items.unwrap()[0].label, Some(Label::PersonaTransfer));
    }

    #[test]
    fn a_quotation_in_the_cited_document_keeps_its_word() {
        let answer = "Advice.\n\nGrounding:\n- SOURCE_DIRECT: \"write to every Town, that the Sense of the People may be known\" [d1]\n";
        assert_eq!(check(answer, &[]), Vec::new());
    }

    #[test]
    fn a_quotation_not_in_the_document_is_fabricated() {
        let answer = "Advice.\n\nGrounding:\n- SOURCE_DIRECT: \"liberty is the first gift of nature to every citizen\" [d1]\n";
        assert!(matches!(
            check(answer, &[])[..],
            [Violation::FabricatedQuote(_)]
        ));
    }

    #[test]
    fn a_real_quotation_cited_to_the_wrong_document_is_fabricated() {
        let d = [
            doc("d1", "some other words entirely about taxes and duties"),
            doc("d2", WORDS_OF_HIS),
        ];
        let refs: Vec<&Document> = d.iter().collect();
        let answer = "A.\n\nGrounding:\n- SOURCE_DIRECT: \"write to every Town, that the Sense of the People\" [d1]\n";
        assert!(matches!(
            verify(&parse(answer), &refs, &[])[..],
            [Violation::FabricatedQuote(_)]
        ));
    }

    #[test]
    fn a_document_the_model_may_not_draw_on_cannot_be_cited() {
        let answer = "A.\n\nGrounding:\n- SOURCE_DIRECT: \"write to every Town, that the Sense of the People\" [d9]\n";
        assert!(matches!(
            check(answer, &[])[..],
            [Violation::FabricatedQuote(_)]
        ));
    }

    #[test]
    fn his_own_words_need_a_quotation_a_document_and_enough_words() {
        for line in [
            "- SOURCE_DIRECT: he always wrote to every town",
            "- SOURCE_DIRECT: \"write to every Town\" [d1]",
            "- SOURCE_DIRECT: \"write to every Town, that the Sense of the People\"",
        ] {
            let answer = format!("A.\n\nGrounding:\n{line}\n");
            assert!(
                matches!(check(&answer, &[])[..], [Violation::UncitedSource(_)]),
                "{line}"
            );
        }
    }

    #[test]
    fn a_fact_of_the_situation_must_come_from_what_was_supplied() {
        let answer = "A.\n\nGrounding:\n- MODERN_OBSERVATION: the organisation has 800 employees across 5 offices\n";
        assert_eq!(
            check(
                answer,
                &["The organisation has 800 employees across 5 offices and two data centres."]
            ),
            Vec::new()
        );
        assert!(matches!(
            check(answer, &["The organisation is small."])[..],
            [Violation::UnsupportedFact(_)]
        ));
        assert!(matches!(
            check(answer, &[])[..],
            [Violation::UnsupportedFact(_)]
        ));
    }

    #[test]
    fn a_word_from_after_his_death_in_a_line_labelled_as_his_is_an_anachronism() {
        let answer = "A.\n\nGrounding:\n- SOURCE_INFERRED: he would have used the internet to write to every Town\n";
        assert!(matches!(
            check(answer, &[])[..],
            [Violation::Anachronism(_)]
        ));
        let transfer = "A.\n\nGrounding:\n- PERSONA_TRANSFER: carry the committee method to the software teams\n";
        assert_eq!(
            check(transfer, &[]),
            Vec::new(),
            "transfer is where modern things belong"
        );
    }

    #[test]
    fn a_label_that_is_none_of_the_five_is_named() {
        let answer = "A.\n\nGrounding:\n- FACT: it is so\n";
        assert_eq!(
            check(answer, &[]),
            vec![Violation::UnknownLabel("FACT".into())]
        );
    }

    #[test]
    fn a_long_quotation_in_the_body_that_is_in_no_document_is_fabricated_whatever_the_block_says() {
        let answer = "As I wrote, \"the safety of the republic rests upon the vigilance of every citizen\" always.\n\nGrounding:\n- PERSONA_TRANSFER: method\n";
        assert!(matches!(
            check(answer, &[])[..],
            [Violation::FabricatedQuote(_)]
        ));
        let real = "As I wrote, \"write to every Town, that the Sense of the People may be known\" always.\n\nGrounding:\n- PERSONA_TRANSFER: method\n";
        assert_eq!(check(real, &[]), Vec::new());
    }

    #[test]
    fn an_answer_that_keeps_every_word_has_no_violations() {
        let answer = "Write to every team lead first, then gather what they say.\n\nGrounding:\n- SOURCE_DIRECT: \"Let the Committee write to every Town\" [d1]\n- SOURCE_INFERRED: he gathered opinion before acting\n- MODERN_OBSERVATION: there are twelve teams\n- PERSONA_TRANSFER: the committee method applies to the teams\n- SPECULATION: it may take two weeks\n";
        assert_eq!(
            check(answer, &["There are twelve teams in the organisation."]),
            Vec::new()
        );
    }
}
