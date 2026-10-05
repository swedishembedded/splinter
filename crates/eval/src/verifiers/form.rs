// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements retention checks that show whether teaching
// a model something new cost it what it could do before, for its clients. If
// your team needs expertise in catching eroded reasoning or format-following
// before a fine-tune is released, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Verifiers that read the form of an answer, not its words: the last number
//! it states, and how many lines it has.
//!
//! A fine-tune on one person's writing can erode what trivia recall cannot
//! show: working a sum through to a number, and following a format. These
//! grade such tasks by code alone, so an anchor suite can hold a release to
//! them with no judge. Neither reads the answer's prose, so both abstain on
//! a task whose reference is not a number.

use serde_json::json;
use splinter_core::annotation::{Producer, Strength};
use splinter_core::experience::{Experience, Task};

use super::{single_reference, Finding, Verifier, VerifyError};

/// Two numbers closer than this are one number: the slack a decimal written
/// two ways needs, and no more.
const NUMBER_TOLERANCE: f64 = 1e-9;

/// The numbers `text` states, in order, thousands separators read and a
/// leading minus kept.
#[must_use]
pub fn numbers(text: &str) -> Vec<f64> {
    let chars: Vec<char> = text.chars().collect();
    let mut found = Vec::new();
    let mut current = String::new();
    for (i, &c) in chars.iter().enumerate() {
        let next_is_digit = chars.get(i + 1).is_some_and(char::is_ascii_digit);
        let starts_negative = c == '-' && current.is_empty() && next_is_digit;
        let continues =
            c.is_ascii_digit() || ((c == '.' || c == ',') && !current.is_empty() && next_is_digit);
        if starts_negative || continues {
            if c != ',' {
                current.push(c);
            }
        } else if !current.is_empty() {
            found.extend(current.parse::<f64>());
            current.clear();
        }
    }
    if !current.is_empty() {
        found.extend(current.parse::<f64>());
    }
    found
}

/// Whether the last number `answer` states is `reference`, a number.
/// `None` when the reference is not a number: there is nothing to compare.
#[must_use]
pub fn final_number_is(answer: &str, reference: &str) -> Option<bool> {
    let wanted = reference.trim().parse::<f64>().ok()?;
    Some(
        numbers(answer)
            .last()
            .is_some_and(|stated| (stated - wanted).abs() < NUMBER_TOLERANCE),
    )
}

/// Whether `answer` has exactly as many non-empty lines as `reference`
/// says. `None` when the reference is not a count.
#[must_use]
pub fn line_count_is(answer: &str, reference: &str) -> Option<bool> {
    let wanted = reference.trim().parse::<usize>().ok()?;
    Some(answer.lines().filter(|l| !l.trim().is_empty()).count() == wanted)
}

/// Grades an answer by the last number it states: a worked sum is right when
/// it ends in the right number, whatever it says on the way.
#[derive(Clone, Debug)]
pub struct FinalNumberVerifier {
    producer: Producer,
}

impl FinalNumberVerifier {
    /// A verifier whose verdicts name `producer`.
    #[must_use]
    pub fn new(producer: Producer) -> Self {
        Self { producer }
    }
}

impl Verifier for FinalNumberVerifier {
    fn producer(&self) -> Producer {
        self.producer.clone()
    }

    fn strength(&self) -> Strength {
        Strength::Formal
    }

    fn verify(&self, task: &Task, exp: &Experience) -> Result<Finding, VerifyError> {
        let reference = match single_reference(task) {
            Ok(reference) => reference,
            Err(abstain) => return Ok(abstain),
        };
        let Some(answer) = exp.final_output.as_deref() else {
            return Ok(Finding::decided(
                false,
                json!({ "comparison": "no answer" }),
            ));
        };
        match final_number_is(answer, reference) {
            Some(passed) => Ok(Finding::decided(
                passed,
                json!({ "comparison": "last number stated", "numbers_stated": numbers(answer).len() }),
            )),
            None => Ok(Finding::abstain("the reference is not a number", json!({}))),
        }
    }
}

/// Grades an answer by its line count: a request for exactly so many lines
/// is met when the answer has that many non-empty lines and nothing else.
#[derive(Clone, Debug)]
pub struct LineCountVerifier {
    producer: Producer,
}

impl LineCountVerifier {
    /// A verifier whose verdicts name `producer`.
    #[must_use]
    pub fn new(producer: Producer) -> Self {
        Self { producer }
    }
}

impl Verifier for LineCountVerifier {
    fn producer(&self) -> Producer {
        self.producer.clone()
    }

    fn strength(&self) -> Strength {
        Strength::Formal
    }

    fn verify(&self, task: &Task, exp: &Experience) -> Result<Finding, VerifyError> {
        let reference = match single_reference(task) {
            Ok(reference) => reference,
            Err(abstain) => return Ok(abstain),
        };
        let Some(answer) = exp.final_output.as_deref() else {
            return Ok(Finding::decided(
                false,
                json!({ "comparison": "no answer" }),
            ));
        };
        match line_count_is(answer, reference) {
            Some(passed) => Ok(Finding::decided(
                passed,
                json!({
                    "comparison": "non-empty lines",
                    "lines": answer.lines().filter(|l| !l.trim().is_empty()).count(),
                }),
            )),
            None => Ok(Finding::abstain(
                "the reference is not a line count",
                json!({}),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_arithmetic_answer_is_right_when_the_last_number_it_states_is_the_reference() {
        assert_eq!(
            final_number_is(
                "5 dollars is 500 cents, minus 180 leaves 320.\nThe answer is 320",
                "320"
            ),
            Some(true)
        );
        assert_eq!(
            final_number_is("So the change is 320.0 cents.", "320"),
            Some(true)
        );
        assert_eq!(final_number_is("The total is 1,250.", "1250"), Some(true));
        assert_eq!(final_number_is("That gives -4.", "-4"), Some(true));
        assert_eq!(
            final_number_is("The answer is 320, not 180", "320"),
            Some(false),
            "the last number is what counts"
        );
        assert_eq!(final_number_is("I cannot tell.", "320"), Some(false));
        assert_eq!(
            final_number_is("320", "three hundred"),
            None,
            "a reference that is not a number cannot be compared"
        );
    }

    #[test]
    fn a_format_answer_has_exactly_the_lines_asked_for() {
        assert_eq!(line_count_is("apple\nbanana\n\ncherry\n", "3"), Some(true));
        assert_eq!(
            line_count_is("Here you go:\napple\nbanana\ncherry", "3"),
            Some(false)
        );
        assert_eq!(line_count_is("apple", "three"), None);
    }

    #[test]
    fn numbers_are_read_in_order_with_separators_and_signs() {
        assert_eq!(
            numbers("12 pencils at 0.12 make 1,440 or -3"),
            vec![12.0, 0.12, 1440.0, -3.0]
        );
        assert!(numbers("no number here").is_empty());
        assert_eq!(numbers("9:15 and 3.50"), vec![9.0, 15.0, 3.5]);
    }
}
