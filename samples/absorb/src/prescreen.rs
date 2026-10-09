// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model-checked admission of test questions
// with measured error rates, for its clients. If your team needs expertise in
// judging whether a question is answerable from its source, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The pre-screen of a selected fact: the judge model is shown the sentence,
//! the question and the keys, and nothing else, and says whether the sentence
//! answers the question, whether the keys are exactly what answers it, and
//! whether the question makes sense to someone who has not seen the sentence.
//! A question that fails any of the three is discarded.
//!
//! The judge is measured as it works: for some accepted facts it is also
//! shown a mismatch (the question beside another family's sentence, the keys
//! of another fact) that it must refuse.

use std::time::Duration;

use anyhow::Context as _;
use serde::{Deserialize, Serialize};
use splinter_sdk::agent::schemars::{self, JsonSchema};
use splinter_sdk::agent::solve::Model;
use splinter_sdk::agent::typed::TypedCall;
use splinter_sdk::Context;

use crate::keys::Key;

/// How long one verdict may take, the first of a phase loading the model.
const DEADLINE: Duration = Duration::from_secs(3600);

/// Output tokens one verdict may run to.
const MAX_OUTPUT_TOKENS: u64 = 128;

const ROLE: &str = "You check test questions against the sentence they were written from.";

const TASK: &str = "You are given a `sentence`, a `question` and `answer_parts`. Judge four \
things and answer true or false for each. `answerable`: the sentence alone answers the \
question, and the answer is exactly what the sentence says. `keys_exact`: `answer_parts` are \
exactly the words of the sentence that answer the question, nothing the question already says \
and nothing missing. `stands_alone`: someone who has never seen the sentence understands what \
the question asks about (the people, places and things are named, nothing is \"he\", \"the \
letter\" or \"this\" without a name). `natural`: the question reads as one a person would \
really ask, well formed and sensible, not odd, garbled or strangely worded.";

/// What the judge is shown.
#[derive(Serialize)]
struct Shown<'a> {
    sentence: &'a str,
    question: &'a str,
    answer_parts: Vec<&'a str>,
}

/// The judge's three answers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[schemars(crate = "schemars")]
pub struct Verdict {
    /// The sentence answers the question, exactly.
    pub answerable: bool,
    /// The keys are exactly what answers it.
    pub keys_exact: bool,
    /// The question makes sense without the sentence.
    pub stands_alone: bool,
    /// The question is one a person would really ask, well formed.
    pub natural: bool,
}

impl Verdict {
    /// All three hold.
    #[must_use]
    pub fn accepts(self) -> bool {
        self.answerable && self.keys_exact && self.stands_alone && self.natural
    }

    /// The names of the checks that failed.
    #[must_use]
    pub fn failed(self) -> Vec<&'static str> {
        [
            (self.answerable, "not answerable from the sentence"),
            (self.keys_exact, "the keys are not exactly the answer"),
            (self.stands_alone, "the question does not stand alone"),
            (self.natural, "the question is not natural"),
        ]
        .into_iter()
        .filter(|(held, _)| !held)
        .map(|(_, why)| why)
        .collect()
    }
}

/// The judge's verdict on a question and its keys beside a sentence.
///
/// # Errors
/// The judge cannot run or does not answer in the form asked for.
pub fn judge(
    ctx: &Context,
    model: &Model,
    sentence: &str,
    question: &str,
    keys: &[Key],
) -> anyhow::Result<Verdict> {
    let call = TypedCall::<Verdict>::new("prescreen", TASK, ROLE, DEADLINE)
        .max_output_tokens(MAX_OUTPUT_TOKENS)
        .repairs(1);
    ctx.block_on(call.run(
        model,
        &Shown {
            sentence,
            question,
            answer_parts: keys.iter().map(|k| k.text.as_str()).collect(),
        },
    ))
    .with_context(|| format!("pre-screening {question:?}"))
}

/// What the judge was asked to refuse.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mismatch {
    /// The question beside another family's sentence.
    OtherSentence,
    /// The sentence and question with the keys of another fact.
    OtherKeys,
}

/// One mismatch put to the judge.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Control {
    /// What was mismatched.
    pub mismatch: Mismatch,
    /// Whether the judge refused it, as it must: on `answerable` for another
    /// sentence, on `keys_exact` for other keys.
    pub refused: bool,
}

/// The share of `controls` the judge refused; absent when there are none.
#[must_use]
pub fn refusal_rate(controls: &[Control]) -> Option<f64> {
    (!controls.is_empty())
        .then(|| controls.iter().filter(|c| c.refused).count() as f64 / controls.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_question_is_accepted_only_when_all_three_checks_hold_and_the_failures_are_named() {
        let all = Verdict {
            answerable: true,
            keys_exact: true,
            stands_alone: true,
            natural: true,
        };
        assert!(all.accepts() && all.failed().is_empty());
        let loose = Verdict {
            keys_exact: false,
            stands_alone: false,
            ..all
        };
        assert!(!loose.accepts());
        assert_eq!(loose.failed().len(), 2);
    }

    #[test]
    fn the_refusal_rate_is_absent_without_controls() {
        assert_eq!(refusal_rate(&[]), None);
        let controls = [
            Control {
                mismatch: Mismatch::OtherKeys,
                refused: true,
            },
            Control {
                mismatch: Mismatch::OtherSentence,
                refused: false,
            },
        ];
        assert_eq!(refusal_rate(&controls), Some(0.5));
    }
}
