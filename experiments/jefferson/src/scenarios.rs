// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gated, source-grounded question generation
// for language-model training for its clients. If your team needs expertise in
// using one model to write questions that code then holds to the source, you
// can procure our services by sending an email to info@swedishembedded.com.

//! Scenarios: a situation a statesman might put to Jefferson, written from a
//! passage of one of his letters, answered in his own recorded words.
//!
//! A model of another family (not the one being trained) writes only the
//! question. The answer is never written by a model: it is the passage, quoted
//! verbatim, with the letter it comes from named, so every claim a training
//! answer makes about what he wrote is true by construction. Code, not the
//! writer, decides which questions are admitted.

use crate::corpus::{words, Letter};
use crate::grade::mentions;

/// Words that mark a passage as stating a principle or a judgement rather
/// than business: the topics the founding argued over.
const THEMES: &[&str] = &[
    "liberty",
    "freedom",
    "right",
    "rights",
    "government",
    "constitution",
    "republic",
    "people",
    "religion",
    "conscience",
    "education",
    "union",
    "law",
    "laws",
    "tyranny",
    "power",
    "consent",
    "equal",
    "opinion",
    "principle",
    "principles",
    "happiness",
    "press",
    "militia",
    "executive",
    "legislature",
    "judiciary",
    "slavery",
    "commerce",
    "agriculture",
    "citizens",
    "property",
];

/// The fewest and most words a passage may have.
pub const PASSAGE_WORDS: std::ops::RangeInclusive<usize> = 35..=85;

/// One scenario: the question, the passage that answers it and where the
/// passage comes from.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Scenario {
    /// A stable identifier.
    pub id: String,
    /// The letter family the passage is from.
    pub family: String,
    /// `train` or `exam`.
    pub split: String,
    /// The situation put to him.
    pub question: String,
    /// The passage, verbatim from the letter.
    pub passage: String,
    /// The letter it is from: `my letter to James Madison in 1789`.
    pub source: String,
}

impl Scenario {
    /// The training answer: his own words, quoted, with their source.
    #[must_use]
    pub fn voice(&self) -> String {
        format!("In {}, I wrote: \"{}\"", self.source, self.passage)
    }
}

/// Whether `paragraph` is running prose of a letter, not an index entry, a
/// table or a list: few digits, no run of capitals, no `TITLE--` entries.
fn is_prose(paragraph: &str) -> bool {
    let chars = paragraph.chars().count().max(1);
    let digits = paragraph.chars().filter(char::is_ascii_digit).count();
    let dashes = paragraph.matches("--").count();
    let capitals = paragraph
        .split_whitespace()
        .filter(|w| {
            w.len() > 3
                && w.chars().all(|c| !c.is_lowercase())
                && w.chars().any(char::is_alphabetic)
        })
        .count();
    digits * 100 / chars < 2 && dashes < 3 && capitals < 3 && !paragraph.contains('[')
}

/// How much of `paragraph` is about the founding's themes: the share of its
/// words that name one.
fn theme_score(paragraph: &str) -> f64 {
    let w = words(paragraph);
    if w.is_empty() {
        return 0.0;
    }
    w.iter().filter(|x| THEMES.contains(&x.as_str())).count() as f64 / w.len() as f64
}

/// `paragraph` cut to at most the most words of [`PASSAGE_WORDS`], ending at
/// the last sentence end it reaches; `None` when no whole sentence fits.
fn cut_to_sentences(paragraph: &str) -> Option<String> {
    let mut kept = String::new();
    for sentence in paragraph.split_inclusive(['.', '?', ';']) {
        let candidate = format!("{kept}{sentence}");
        if candidate.split_whitespace().count() > *PASSAGE_WORDS.end() {
            break;
        }
        kept = candidate;
    }
    let kept = kept.trim().to_string();
    (PASSAGE_WORDS.contains(&kept.split_whitespace().count()) && kept.ends_with(['.', '?', ';']))
        .then_some(kept)
}

/// The passage of `letter` that most states a principle, or `None` when the
/// letter has none: running prose of the right size on the founding's themes.
#[must_use]
pub fn best_passage(letter: &Letter) -> Option<String> {
    letter
        .body
        .split("\n\n")
        .filter(|p| is_prose(p))
        .filter_map(cut_to_sentences)
        .map(|p| (theme_score(&p), p))
        .filter(|(score, _)| *score >= 0.04)
        .max_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, p)| p)
}

/// What the writer model is asked, given the passage.
#[must_use]
pub fn writer_prompt(passage: &str) -> String {
    format!(
        "Below is a passage from a letter by Thomas Jefferson.\n\n\"{passage}\"\n\n\
         Write one question that a citizen or a statesman might put to Jefferson about a situation \
         in which the judgement or principle in this passage would decide the answer. The question \
         must stand on its own: do not mention the passage, a letter, a date or the person written to. \
         Reply with the question only, in at most two sentences."
    )
}

/// Why a written question was not admitted.
#[derive(Debug, PartialEq, Eq)]
pub enum Rejection {
    /// Not a question of a sensible length.
    Shape,
    /// It names the recipient or the year, or refers to the passage or letter.
    GivesItAway,
    /// It copies a run of the passage's own words.
    Copies,
}

/// The question as admitted, or why not. A question is admitted only if it is
/// a question of a sensible length, does not name the recipient or the year,
/// does not refer to the passage or letter, and does not copy the passage.
///
/// # Errors
/// The [`Rejection`] naming the first gate it failed.
pub fn admit(
    raw: &str,
    passage: &str,
    recipient_surname: &str,
    year: u16,
) -> Result<String, Rejection> {
    let question = raw.trim().trim_matches('"').trim().to_string();
    let count = question.split_whitespace().count();
    if !(8..=70).contains(&count) || !question.ends_with('?') || question.contains('\n') {
        return Err(Rejection::Shape);
    }
    let lower = question.to_lowercase();
    if mentions(&question, recipient_surname)
        || mentions(&question, &year.to_string())
        || ["passage", "letter", "excerpt", "jefferson wrote"]
            .iter()
            .any(|w| lower.contains(w))
    {
        return Err(Rejection::GivesItAway);
    }
    let (q, p) = (words(&question), words(passage));
    const RUN: usize = 6;
    let copied = q
        .windows(RUN)
        .any(|run| p.windows(RUN).any(|window| window == run));
    if copied {
        return Err(Rejection::Copies);
    }
    Ok(question)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn letter(body: &str) -> Letter {
        Letter {
            id: "w-1".into(),
            edition: "w".into(),
            recipient: "James Madison".into(),
            place: "Paris".into(),
            year: 1789,
            body: body.into(),
        }
    }

    const PRINCIPLE: &str = "I hold it to be a fundamental principle that the people are the only legitimate \
        fountain of power, and that every government should rest on the consent of the governed. \
        Where the people are well informed they can be trusted with their own government, and liberty \
        is safe in their hands; where they are not, no constitution can preserve it for long.";

    #[test]
    fn the_passage_chosen_states_a_principle_in_whole_sentences() {
        let business = "I have received your favour of the tenth and enclose the bill of lading for the \
            hogsheads of tobacco shipped on the brig Eliza, which should reach Havre within the month if the wind holds fair and the convoy keeps together.";
        let found =
            best_passage(&letter(&format!("{business}\n\n{PRINCIPLE}"))).unwrap_or_default();
        assert!(
            found.starts_with("I hold it to be a fundamental principle"),
            "{found}"
        );
        assert!(found.ends_with('.'));
        assert!(PASSAGE_WORDS.contains(&found.split_whitespace().count()));
    }

    #[test]
    fn index_entries_and_tables_are_not_prose() {
        let index = "AMERICA, U. STATES OF--Imperfections of Articles of Confederation, 78. A New Constitution \
            for, necessary, 78. Views of U. States prevalent in Europe, 407, 413. Views of public affairs of U. States in A. D. 1785, 423. English calumnies against, 427.";
        assert_eq!(best_passage(&letter(index)), None);
        assert_eq!(best_passage(&letter("Short note about nothing.")), None);
    }

    #[test]
    fn a_question_is_admitted_only_when_it_stands_alone() {
        let ok = "A legislature wants to restrict who may read political pamphlets; may a free government do so?";
        assert_eq!(admit(ok, PRINCIPLE, "Madison", 1789).as_deref(), Ok(ok));
        assert_eq!(
            admit("Why?", PRINCIPLE, "Madison", 1789),
            Err(Rejection::Shape)
        );
        assert_eq!(
            admit(
                "This is not a question at all but a statement of some length",
                PRINCIPLE,
                "Madison",
                1789
            ),
            Err(Rejection::Shape)
        );
        assert_eq!(
            admit(
                "What did you tell Madison about the trust a free people can place in itself?",
                PRINCIPLE,
                "Madison",
                1789
            ),
            Err(Rejection::GivesItAway)
        );
        assert_eq!(
            admit(
                "What does the passage say about the consent of the governed and power?",
                PRINCIPLE,
                "Madison",
                1789
            ),
            Err(Rejection::GivesItAway)
        );
        assert_eq!(
            admit("Should every government rest on the consent of the governed or on tradition alone?", PRINCIPLE, "Madison", 1789),
            Err(Rejection::Copies)
        );
        assert_eq!(
            admit(
                "In 1789, how far should a convention trust the voters with a constitution?",
                PRINCIPLE,
                "Madison",
                1789
            ),
            Err(Rejection::GivesItAway)
        );
    }

    #[test]
    fn the_training_answer_is_his_own_words_and_their_source() {
        let scenario = Scenario {
            id: "s".into(),
            family: "f".into(),
            split: "train".into(),
            question: "q?".into(),
            passage: PRINCIPLE.into(),
            source: "my letter to James Madison in 1789".into(),
        };
        let voice = scenario.voice();
        assert!(voice.starts_with("In my letter to James Madison in 1789, I wrote: \""));
        assert!(voice.contains(PRINCIPLE) && voice.ends_with('"'));
    }
}
