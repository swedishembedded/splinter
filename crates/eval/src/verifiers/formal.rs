// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verifiers that grade agent answers against
// references the agent never saw, for its clients. If your team needs
// expertise in verifier design, you can procure our services by sending an
// email to info@swedishembedded.com.

//! The formal verifier: the answer against the task's one reference, under
//! a configured [`Normalisation`].

use serde_json::json;
use splinter_core::annotation::{Producer, Strength};
use splinter_core::digest::Digest;
use splinter_core::experience::{Experience, Task};

use super::normalise::Normalisation;
use super::{single_reference, Finding, Verifier, VerifyError};

/// Passes an answer iff it equals the task's one
/// [`Reference`](splinter_core::experience::PrivilegedKind::Reference)
/// once both are normalised; fails a different answer or none. Abstains on
/// a task without exactly one reference, or of a kind it is not for.
#[derive(Clone, Debug)]
pub struct ExactMatchVerifier {
    producer: Producer,
    normalisation: Normalisation,
    kind: Option<String>,
}

impl ExactMatchVerifier {
    /// A verifier for tasks of every kind, its verdicts carrying `producer`.
    #[must_use]
    pub fn new(producer: Producer, normalisation: Normalisation) -> Self {
        Self {
            producer,
            normalisation,
            kind: None,
        }
    }

    /// The same verifier, applying only to tasks of `kind`.
    #[must_use]
    pub fn for_kind(mut self, kind: &str) -> Self {
        self.kind = Some(kind.to_string());
        self
    }
}

impl Verifier for ExactMatchVerifier {
    fn producer(&self) -> Producer {
        self.producer.clone()
    }

    fn strength(&self) -> Strength {
        Strength::Formal
    }

    /// The evidence names the normalisation and the digests of the two
    /// normalised texts, never the reference itself.
    fn verify(&self, task: &Task, exp: &Experience) -> Result<Finding, VerifyError> {
        if let Some(kind) = &self.kind {
            if task.task.kind != *kind {
                return Ok(Finding::abstain(
                    "the verifier is for another task kind",
                    json!({ "for": kind, "task": task.task.kind }),
                ));
            }
        }
        let reference = match single_reference(task) {
            Ok(reference) => self.normalisation.apply(reference),
            Err(abstain) => return Ok(abstain),
        };
        let output = exp
            .final_output
            .as_deref()
            .map(|o| self.normalisation.apply(o));
        let passed = output.as_deref() == Some(reference.as_str());
        Ok(Finding::decided(
            passed,
            json!({
                "comparison": "exact after normalisation",
                "normalisation": self.normalisation,
                "reference": Digest::of(reference.as_bytes()),
                "output": output.map(|o| Digest::of(o.as_bytes())),
            }),
        ))
    }
}

/// How many times the reference's length, plus [`STATED_SLACK`], an answer
/// may run and still count as stating the reference rather than listing
/// everything in the hope that it is among them.
const STATED_FACTOR: usize = 4;

/// Characters an answer may add around the reference, beyond
/// [`STATED_FACTOR`] times its length: the words of a sentence.
const STATED_SLACK: usize = 60;

/// Passes an answer iff it states the task's one
/// [`Reference`](splinter_core::experience::PrivilegedKind::Reference):
/// the normalised reference occurs in the normalised answer with no letter
/// or digit directly before or after it, and the answer is at most
/// `STATED_FACTOR` times the reference's length plus `STATED_SLACK`
/// characters long. For the short facts a recall task asks for, where a
/// model answers in a sentence.
///
/// Normalisation here also folds sub- and superscript digits to plain ones
/// (a subscripted `H2O` is `H2O`) and takes a whole number written in words and in digits
/// as one (`eight` is `8`).
///
/// An answer that states the reference but runs past the length bound is
/// neither right nor wrong by code: the reference may be the point of a long
/// explanation or one item of a list, which only a judge can tell apart, so
/// the verifier abstains and leaves the answer to one. An answer that does
/// not state the reference fails whatever its length, and so does none;
/// the verifier abstains on a task without exactly one reference.
#[derive(Clone, Debug)]
pub struct StatedReferenceVerifier {
    producer: Producer,
    normalisation: Normalisation,
}

impl StatedReferenceVerifier {
    /// A verifier for tasks of every kind, its verdicts carrying `producer`.
    #[must_use]
    pub fn new(producer: Producer, normalisation: Normalisation) -> Self {
        Self {
            producer,
            normalisation,
        }
    }
}

/// How an answer stands to a reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stating {
    /// The reference is stated in a bounded answer.
    Stated,
    /// The reference does not occur in the answer.
    Absent,
    /// The reference occurs in an answer longer than the bound.
    TooLong,
}

/// The plain digit a sub- or superscript digit stands for.
fn plain_digit(c: char) -> Option<char> {
    let digit = match c {
        '\u{2080}'..='\u{2089}' => u32::from(c) - 0x2080,
        '\u{2074}'..='\u{2079}' => u32::from(c) - 0x2070,
        '\u{2070}' => 0,
        '\u{00B9}' => 1,
        '\u{00B2}' => 2,
        '\u{00B3}' => 3,
        _ => return None,
    };
    char::from_digit(digit, 10)
}

/// `text` with sub- and superscript digits as plain digits.
fn fold_digits(text: &str) -> String {
    text.chars().map(|c| plain_digit(c).unwrap_or(c)).collect()
}

/// The whole numbers a word names, below one hundred and a hundred itself.
const NUMBER_WORDS: &[(&str, u32)] = &[
    ("zero", 0),
    ("one", 1),
    ("two", 2),
    ("three", 3),
    ("four", 4),
    ("five", 5),
    ("six", 6),
    ("seven", 7),
    ("eight", 8),
    ("nine", 9),
    ("ten", 10),
    ("eleven", 11),
    ("twelve", 12),
    ("thirteen", 13),
    ("fourteen", 14),
    ("fifteen", 15),
    ("sixteen", 16),
    ("seventeen", 17),
    ("eighteen", 18),
    ("nineteen", 19),
    ("twenty", 20),
    ("thirty", 30),
    ("forty", 40),
    ("fifty", 50),
    ("sixty", 60),
    ("seventy", 70),
    ("eighty", 80),
    ("ninety", 90),
    ("hundred", 100),
];

/// The other way `reference` is written when it is one whole number: the
/// digits of a number word, or the word of digits; `None` for anything else.
fn other_numeral(reference: &str) -> Vec<String> {
    let word_of = |n: u32| {
        NUMBER_WORDS
            .iter()
            .find(|(_, value)| *value == n)
            .map(|(word, _)| (*word).to_string())
    };
    if let Ok(n) = reference.parse::<u32>() {
        if let Some(word) = word_of(n) {
            return vec![word];
        }
        // 21 to 99 as tens and units: forty-two, forty two.
        if (21..100).contains(&n) {
            if let (Some(tens), Some(units)) = (word_of(n / 10 * 10), word_of(n % 10)) {
                return vec![format!("{tens}-{units}"), format!("{tens} {units}")];
            }
        }
        return Vec::new();
    }
    let compound = reference.split(['-', ' ']).collect::<Vec<_>>();
    let value = |word: &str| {
        NUMBER_WORDS
            .iter()
            .find(|(w, _)| *w == word)
            .map(|(_, v)| *v)
    };
    match compound.as_slice() {
        [word] => value(word).map(|v| v.to_string()).into_iter().collect(),
        [tens, units] => value(tens)
            .zip(value(units))
            .filter(|(t, u)| (20..100).contains(t) && *t % 10 == 0 && *u < 10)
            .map(|(t, u)| (t + u).to_string())
            .into_iter()
            .collect(),
        _ => Vec::new(),
    }
}

/// Whether `reference` occurs in `answer` with no letter or digit
/// adjoining it.
fn occurs(answer: &str, reference: &str) -> bool {
    answer.match_indices(reference).any(|(start, found)| {
        let before = answer[..start].chars().next_back();
        let after = answer[start + found.len()..].chars().next();
        !before.is_some_and(char::is_alphanumeric) && !after.is_some_and(char::is_alphanumeric)
    })
}

/// How `answer` stands to `reference`; see [`StatedReferenceVerifier`].
fn states(answer: &str, reference: &str) -> Stating {
    let (answer, reference) = (fold_digits(answer), fold_digits(reference));
    if reference.is_empty() {
        return Stating::Absent;
    }
    let found = occurs(&answer, &reference)
        || other_numeral(&reference)
            .iter()
            .any(|other| occurs(&answer, other));
    let bound = STATED_FACTOR * reference.chars().count() + STATED_SLACK;
    match (found, answer.chars().count() > bound) {
        (false, _) => Stating::Absent,
        (true, false) => Stating::Stated,
        (true, true) => Stating::TooLong,
    }
}

impl Verifier for StatedReferenceVerifier {
    fn producer(&self) -> Producer {
        self.producer.clone()
    }

    fn strength(&self) -> Strength {
        Strength::Formal
    }

    /// The evidence names the comparison and the digests of the two
    /// normalised texts, never the reference itself.
    fn verify(&self, task: &Task, exp: &Experience) -> Result<Finding, VerifyError> {
        let reference = match single_reference(task) {
            Ok(reference) => self.normalisation.apply(reference),
            Err(abstain) => return Ok(abstain),
        };
        let output = exp
            .final_output
            .as_deref()
            .map(|o| self.normalisation.apply(o));
        let stating = output.as_deref().map(|o| states(o, &reference));
        let evidence = json!({
            "comparison": "reference stated in a bounded answer",
            "normalisation": self.normalisation,
            "reference": Digest::of(reference.as_bytes()),
            "output": output.map(|o| Digest::of(o.as_bytes())),
        });
        Ok(match stating {
            Some(Stating::TooLong) => Finding::abstain(
                "the answer states the reference but is longer than a statement of it: a judge decides",
                evidence,
            ),
            other => Finding::decided(other == Some(Stating::Stated), evidence),
        })
    }
}
