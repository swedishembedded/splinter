// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gated, source-grounded question generation
// for language-model training for its clients. If your team needs expertise in
// using one model to write questions that code then holds to the source, you
// can procure our services by sending an email to info@swedishembedded.com.

//! Scenarios: a predicament put to Jefferson, written from a passage of one of
//! his letters in which he advises someone, answered in his own recorded words.
//!
//! A model of another family (not the one being trained) writes only the
//! question. The answer is never written by a model: it is the passage, quoted
//! verbatim, with the letter it comes from named, so every claim a training
//! answer makes about what he wrote is true by construction. Code, not the
//! writer, decides which questions are admitted.

use splinter_sdk::knowledge::advice::{advice_cues, is_prose};

use crate::corpus::{words, Letter};
use splinter_sdk::measure::verifiers::answer::mentions;

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
        format!("In {}, I advised: \"{}\"", self.source, self.passage)
    }
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

/// The passages of `letter` that read as advice, strongest first: running
/// prose of the right size, in whole sentences, with at least one advice cue.
#[must_use]
pub fn advice_passages(letter: &Letter) -> Vec<String> {
    let mut found: Vec<(usize, String)> = letter
        .body
        .split("\n\n")
        .filter(|p| is_prose(p))
        .filter_map(cut_to_sentences)
        .map(|p| (advice_cues(&p), p))
        .filter(|(cues, _)| *cues >= 1)
        .collect();
    found.sort_by_key(|(cues, _)| std::cmp::Reverse(*cues));
    found.into_iter().map(|(_, p)| p).collect()
}

/// What the classifier model is asked: whether the passage is advice.
#[must_use]
pub fn classifier_prompt(passage: &str) -> String {
    format!(
        "Below is a passage from a letter.\n\n\"{passage}\"\n\n\
         Is the writer giving the person he writes to concrete advice or counsel: telling \
         them what to do, what to avoid or what to value? Reply with one word: YES or NO."
    )
}

/// What the writer model is asked, given the passage.
#[must_use]
pub fn writer_prompt(passage: &str) -> String {
    format!(
        "Below is a passage from a letter by Thomas Jefferson in which he advises someone.\n\n\"{passage}\"\n\n\
         Write the message that person might have sent to ask for that advice: two to four sentences \
         describing a concrete predicament in their own words, ending by asking what Jefferson would \
         advise. It must stand on its own: do not mention the passage, a letter, a date or the person \
         written to, and do not give the advice. Reply with the message only."
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
    let lower = question.to_lowercase();
    let asks = [
        "advise",
        "advice",
        "counsel",
        "what should",
        "what would you",
    ]
    .iter()
    .any(|w| lower.contains(w));
    if !(12..=120).contains(&count) || !question.ends_with('?') || !asks {
        return Err(Rejection::Shape);
    }
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
            recipient: "Peter Carr".into(),
            place: "Paris".into(),
            year: 1785,
            body: body.into(),
        }
    }

    const ADVICE: &str = "I advise you to fix a habit of study every morning before you do anything else. \
        Never let a day pass without reading something of history or ethics, and always write down \
        what you have read, for the memory fails what the pen has not fixed. Do not spend your evenings \
        in idle company, and be careful of debt, which is the surest enemy of a young man's independence.";

    #[test]
    fn advice_is_found_by_its_cues_in_whole_sentences() {
        let business = "I have received your favour of the tenth and enclose the bill of lading for the \
            hogsheads of tobacco shipped on the brig Eliza, which should reach Havre within the month if the wind holds fair and the convoy keeps together.";
        assert_eq!(advice_cues(business), 0);
        let found = advice_passages(&letter(&format!("{business}\n\n{ADVICE}")));
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(found[0].starts_with("I advise you to fix a habit of study"));
        assert!(found[0].ends_with('.'));
        assert!(PASSAGE_WORDS.contains(&found[0].split_whitespace().count()));
        assert!(advice_cues(&found[0]) >= 4);
    }

    #[test]
    fn index_entries_and_tables_are_not_advice() {
        let index = "AMERICA, U. STATES OF--Imperfections of Articles of Confederation, 78. A New Constitution \
            for, necessary, 78. Views of U. States prevalent in Europe, 407, 413. Never to be forgotten, always, 423. English calumnies against, 427.";
        assert!(advice_passages(&letter(index)).is_empty());
        assert!(advice_passages(&letter("Short note about nothing.")).is_empty());
    }

    #[test]
    fn the_classifier_and_the_judge_are_read_the_same_way() {
        assert!(classifier_prompt(ADVICE).contains("YES or NO"));
        assert!(writer_prompt(ADVICE).contains("do not give the advice"));
    }

    #[test]
    fn a_question_is_admitted_only_when_it_asks_for_advice_and_stands_alone() {
        let ok = "I am nineteen and have finished my first year at college, but I find my mornings slip away \
            in idleness and my evenings in company; what would you advise me to do about my habits?";
        assert_eq!(admit(ok, ADVICE, "Carr", 1785).as_deref(), Ok(ok));
        assert_eq!(
            admit("What now?", ADVICE, "Carr", 1785),
            Err(Rejection::Shape)
        );
        let no_ask = "I am nineteen and have finished my first year at college, and I find my mornings slip away in idleness and evenings in company, is that normal?";
        assert_eq!(admit(no_ask, ADVICE, "Carr", 1785), Err(Rejection::Shape));
        assert_eq!(
            admit("My cousin Carr and I are at college and our mornings are idle, so what would you advise us both to do now?", ADVICE, "Carr", 1785),
            Err(Rejection::GivesItAway)
        );
        assert_eq!(
            admit("Does the passage mean I should study every morning? What would you advise me about my idle hours at college?", ADVICE, "Carr", 1785),
            Err(Rejection::GivesItAway)
        );
        assert_eq!(
            admit("Should I fix a habit of study every morning before I do anything else? What would you advise a young man at college today?", ADVICE, "Carr", 1785),
            Err(Rejection::Copies)
        );
        assert_eq!(
            admit("In 1785 I was a student with idle mornings and I would like to know what you would advise me to do about it now?", ADVICE, "Carr", 1785),
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
            passage: ADVICE.into(),
            source: "my letter to Peter Carr in 1785".into(),
        };
        let voice = scenario.voice();
        assert!(voice.starts_with("In my letter to Peter Carr in 1785, I advised: \""));
        assert!(voice.contains(ADVICE) && voice.ends_with('"'));
    }
}
