// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements calibrated model judges for open-ended
// answers, for its clients. If your team needs expertise in using a model as
// a judge only as far as controls with known answers show it can be trusted,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! A judge for what code cannot grade: whether a reply does what the real
//! letter does.
//!
//! The judge is asked, for each item of the rubric, whether a text does it. It
//! is believed only as far as controls show: shown the real letter it must say
//! yes to most items, shown a different letter on the same rubric it must say
//! yes to few. A judge that cannot tell them apart grades nothing, and the
//! report says no claim is made.

use splinter_sdk::agent::schemars::JsonSchema;
use splinter_sdk::agent::typed::TypedCall;

use crate::helper::Helper;
use crate::reconstruct::Briefing;

/// The share of items the real letter must be credited with, and the most a
/// different letter may be credited with, for the judge's verdicts to count.
pub const REAL_LETTER_AT_LEAST: f64 = 0.8;
pub const OTHER_LETTER_AT_MOST: f64 = 0.3;
/// The most a fluent reply written from the situation alone, never from the
/// letter, may be credited with: a judge that gives it much cannot tell
/// knowing what he did from writing sensibly.
pub const GENERIC_REPLY_AT_MOST: f64 = 0.5;
/// Words of any text the judge is shown, so a longer reply is not credited for
/// its length and a reply is held to the length of a real letter's target.
pub const JUDGED_WORDS: usize = 350;

/// What the judge returns.
#[derive(Debug, serde::Deserialize, serde::Serialize, JsonSchema)]
#[schemars(crate = "splinter_sdk::agent::schemars")]
struct Verdict {
    /// One entry per item, in order: whether the text does it.
    covered: Vec<bool>,
}

const ROLE: &str = "a careful reader who judges only from the text in front of them";

const TASK: &str = "You are given a list of things a letter might do, and a text. For each item, in order, say whether the text does it: true only if the text plainly does it, false otherwise. Judge from the text alone. Do not credit length, polish or style, and do not credit a thing the text only gestures at.";

/// What the judge is shown: the items, never the quotations that ground them.
#[derive(serde::Serialize)]
struct Question<'a> {
    items: Vec<&'a str>,
    text: &'a str,
}

/// What the judge may write across every attempt, repairs included.
const OUTPUT_TOKENS: u64 = 900;
const REPAIRS: u32 = 2;

/// For each rubric item of `briefing`, whether `text` does it.
///
/// # Errors
/// The helper could not be reached, or did not return one answer per item.
pub fn cover(helper: &Helper, briefing: &Briefing, text: &str) -> anyhow::Result<Vec<bool>> {
    let items = briefing.key_points.len();
    let call =
        TypedCall::<Verdict>::new("judge_coverage", TASK, ROLE, crate::helper::CALL_DEADLINE)
            .max_output_tokens(OUTPUT_TOKENS)
            .repairs(REPAIRS)
            .postcondition(move |v| {
                (v.covered.len() == items)
                    .then_some(())
                    .ok_or_else(|| format!("give exactly {items} answers, one per item, in order"))
            });
    let question = Question {
        items: briefing
            .key_points
            .iter()
            .map(|k| k.point.as_str())
            .collect(),
        text: &judged(text),
    };
    Ok(helper.call(call, &question)?.covered)
}

/// The first [`JUDGED_WORDS`] words of `text`.
fn judged(text: &str) -> String {
    text.split_whitespace()
        .take(JUDGED_WORDS)
        .collect::<Vec<_>>()
        .join(" ")
}

/// What the generic reply is written as.
#[derive(Debug, serde::Deserialize, serde::Serialize, JsonSchema)]
#[schemars(crate = "splinter_sdk::agent::schemars")]
struct Generic {
    /// The reply, as a letter.
    letter: String,
}

const GENERIC_TASK: &str = "You are given the situation a statesman of the 1770s was writing about. Write the reply a thoughtful, well-read correspondent of the period might send: fluent, in the first person, in about three hundred words. You have not seen what he actually wrote; write only what the situation itself calls for.";

/// A fluent reply to the situation of `briefing`, written without the letter
/// or the rubric: the control that shows whether the judge credits writing
/// well instead of doing what he did.
///
/// # Errors
/// The helper could not be reached.
pub fn generic_reply(helper: &Helper, briefing: &Briefing) -> anyhow::Result<String> {
    #[derive(serde::Serialize)]
    struct Situation<'a> {
        situation: &'a str,
    }
    let call = TypedCall::<Generic>::new(
        "generic_reply",
        GENERIC_TASK,
        ROLE,
        crate::helper::CALL_DEADLINE,
    )
    .max_output_tokens(OUTPUT_TOKENS)
    .repairs(REPAIRS);
    Ok(helper
        .call(
            call,
            &Situation {
                situation: &briefing.situation,
            },
        )?
        .letter)
}

/// The share of items done.
pub fn share(covered: &[bool]) -> f64 {
    if covered.is_empty() {
        return 0.0;
    }
    covered.iter().filter(|c| **c).count() as f64 / covered.len() as f64
}

/// What the controls showed.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Calibration {
    /// Mean share of items credited to the real letter.
    pub real_letter: f64,
    /// Mean share credited to a different letter on the same items.
    pub other_letter: f64,
    /// Mean share credited to a fluent reply written from the situation
    /// alone; `None` for a calibration made without that control, which is
    /// not enough.
    #[serde(default)]
    pub generic_reply: Option<f64>,
    /// How many briefings the controls were run on.
    pub briefings: usize,
}

impl Calibration {
    /// Whether the judge told the real letter from another well enough that
    /// its verdicts on replies mean anything.
    pub fn passes(&self) -> bool {
        self.real_letter >= REAL_LETTER_AT_LEAST
            && self.other_letter <= OTHER_LETTER_AT_MOST
            && self
                .generic_reply
                .is_some_and(|g| g <= GENERIC_REPLY_AT_MOST)
    }
}

/// What the judge is shown for one briefing: the real letter, another letter,
/// and a fluent reply that never saw the letter.
#[derive(Clone, Copy, Debug)]
pub struct Control<'a> {
    pub briefing: &'a Briefing,
    pub real: &'a str,
    pub other: &'a str,
    pub generic: &'a str,
}

/// Run the controls on each briefing.
///
/// # Errors
/// The helper could not be reached, or there is nothing to calibrate on.
pub fn calibrate(helper: &Helper, controls: &[Control<'_>]) -> anyhow::Result<Calibration> {
    anyhow::ensure!(
        !controls.is_empty(),
        "there is nothing to calibrate the judge on"
    );
    let (mut real, mut other, mut generic) = (0.0, 0.0, 0.0);
    for c in controls {
        real += share(&cover(helper, c.briefing, c.real)?);
        other += share(&cover(helper, c.briefing, c.other)?);
        generic += share(&cover(helper, c.briefing, c.generic)?);
    }
    let n = controls.len() as f64;
    Ok(Calibration {
        real_letter: real / n,
        other_letter: other / n,
        generic_reply: Some(generic / n),
        briefings: controls.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reconstruct::KeyPoint;
    use crate::testkit::{helper, responding};

    const REAL: &str = "REAL LETTER: the Towns ought not to wait upon the Ministry, each should choose a Committee, and send me the number acted.";
    const OTHER: &str =
        "OTHER LETTER: I send my duty to your family and hope the winter is mild at Plymouth.";

    fn briefing() -> Briefing {
        let point = |p: &str| KeyPoint {
            point: p.into(),
            quote: "unused here".into(),
        };
        Briefing {
            id: "recon-1".into(),
            doc_id: "l1".into(),
            situation: "s".into(),
            key_points: vec![
                point("advises not waiting"),
                point("proposes committees"),
                point("asks how many towns acted"),
            ],
        }
    }

    fn verdict(covered: &[bool]) -> String {
        serde_json::json!({"covered": covered}).to_string()
    }

    #[test]
    fn the_judge_is_shown_each_item_and_the_text_and_answers_one_verdict_per_item() {
        let (h, provider) = helper(&[&verdict(&[true, false, true])]);
        let covered = cover(&h, &briefing(), REAL).unwrap();
        assert_eq!(covered, [true, false, true]);
        let sent = provider.sent();
        assert!(
            sent.contains("advises not waiting") && sent.contains("REAL LETTER"),
            "{sent}"
        );
        assert!(
            !sent.contains("unused here"),
            "the judge is not shown the quotations, only what to look for"
        );
    }

    #[test]
    fn a_verdict_with_the_wrong_number_of_answers_is_sent_back_and_never_accepted_as_it_is() {
        let (h, _) = helper(&[&verdict(&[true, false])]);
        assert!(cover(&h, &briefing(), REAL).is_err());
        let (h, _) = helper(&[&verdict(&[true]), &verdict(&[true, true, false])]);
        assert_eq!(cover(&h, &briefing(), REAL).unwrap(), [true, true, false]);
    }

    #[test]
    fn the_share_is_the_fraction_of_items_done() {
        assert!((share(&[true, false, true, true]) - 0.75).abs() < 1e-12);
        assert!((share(&[]) - 0.0).abs() < 1e-12);
    }

    fn judging(says_yes: impl Fn(&str) -> bool + Send + Sync + 'static) -> Helper {
        responding(move |asked| verdict(&[says_yes(asked); 3])).0
    }

    const GENERIC: &str = "GENERIC LETTER: I have your letter and agree that the matter is grave and ought to be handled with wisdom and firmness.";

    fn control(b: &Briefing) -> Control<'_> {
        Control {
            briefing: b,
            real: REAL,
            other: OTHER,
            generic: GENERIC,
        }
    }

    #[test]
    fn a_judge_that_credits_the_real_letter_and_neither_another_nor_a_generic_one_passes() {
        let b = briefing();
        let h = judging(|asked| asked.contains("REAL LETTER"));
        let cal = calibrate(&h, &[control(&b), control(&b)]).unwrap();
        assert_eq!(
            (
                cal.real_letter,
                cal.other_letter,
                cal.generic_reply,
                cal.briefings
            ),
            (1.0, 0.0, Some(0.0), 2)
        );
        assert!(cal.passes());
    }

    #[test]
    fn a_judge_that_credits_a_fluent_generic_reply_cannot_tell_knowing_from_writing_well() {
        let b = briefing();
        let h = judging(|asked| asked.contains("REAL LETTER") || asked.contains("GENERIC LETTER"));
        let cal = calibrate(&h, &[control(&b)]).unwrap();
        assert!(
            cal.real_letter >= REAL_LETTER_AT_LEAST && cal.other_letter <= OTHER_LETTER_AT_MOST
        );
        assert!(!cal.passes(), "{cal:?}");
        let old = Calibration {
            generic_reply: None,
            ..cal
        };
        assert!(
            !old.passes(),
            "a calibration made without the control is not enough"
        );
    }

    #[test]
    fn the_judge_is_shown_no_more_of_any_text_than_the_same_number_of_words() {
        let long = vec!["word"; JUDGED_WORDS * 3].join(" ");
        let (h, provider) = helper(&[&verdict(&[true, true, true])]);
        cover(&h, &briefing(), &long).unwrap();
        let sent = provider.sent();
        assert_eq!(
            sent.matches("word").count(),
            JUDGED_WORDS,
            "a longer reply is not credited for its length"
        );
    }

    #[test]
    fn a_generic_reply_is_written_from_the_situation_alone() {
        let b = briefing();
        let (h, provider) =
            helper(&[&serde_json::json!({"letter": "I have your letter."}).to_string()]);
        assert_eq!(generic_reply(&h, &b).unwrap(), "I have your letter.");
        let sent = provider.sent();
        assert!(
            !sent.contains("advises not waiting"),
            "it has not read the rubric: {sent}"
        );
    }

    #[test]
    fn a_judge_that_says_yes_to_everything_or_no_to_everything_makes_no_claim() {
        let b = briefing();
        assert!(
            !calibrate(&judging(|_| true), &[control(&b)])
                .unwrap()
                .passes(),
            "it cannot tell the letters apart"
        );
        assert!(
            !calibrate(&judging(|_| false), &[control(&b)])
                .unwrap()
                .passes(),
            "it does not recognise the real letter"
        );
    }

    #[test]
    fn calibrating_on_nothing_is_an_error_not_a_pass() {
        let (h, _) = helper(&[&verdict(&[true, true, true])]);
        assert!(calibrate(&h, &[]).is_err());
    }
}
