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

const OUTPUT_TOKENS: u64 = 300;
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
        text,
    };
    Ok(helper.call(call, &question)?.covered)
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
    /// How many briefings the controls were run on.
    pub briefings: usize,
}

impl Calibration {
    /// Whether the judge told the real letter from another well enough that
    /// its verdicts on replies mean anything.
    pub fn passes(&self) -> bool {
        self.real_letter >= REAL_LETTER_AT_LEAST && self.other_letter <= OTHER_LETTER_AT_MOST
    }
}

/// Run the controls: for each briefing, the real letter and another letter.
///
/// # Errors
/// The helper could not be reached, or there is nothing to calibrate on.
pub fn calibrate(
    helper: &Helper,
    controls: &[(&Briefing, &str, &str)],
) -> anyhow::Result<Calibration> {
    anyhow::ensure!(
        !controls.is_empty(),
        "there is nothing to calibrate the judge on"
    );
    let (mut real, mut other) = (0.0, 0.0);
    for (briefing, real_letter, other_letter) in controls {
        real += share(&cover(helper, briefing, real_letter)?);
        other += share(&cover(helper, briefing, other_letter)?);
    }
    let n = controls.len() as f64;
    Ok(Calibration {
        real_letter: real / n,
        other_letter: other / n,
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
            request: "r".into(),
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

    #[test]
    fn a_judge_that_credits_the_real_letter_and_not_another_passes() {
        let b = briefing();
        let h = judging(|asked| asked.contains("REAL LETTER"));
        let cal = calibrate(&h, &[(&b, REAL, OTHER), (&b, REAL, OTHER)]).unwrap();
        assert_eq!(
            (cal.real_letter, cal.other_letter, cal.briefings),
            (1.0, 0.0, 2)
        );
        assert!(cal.passes());
    }

    #[test]
    fn a_judge_that_says_yes_to_everything_or_no_to_everything_makes_no_claim() {
        let b = briefing();
        assert!(
            !calibrate(&judging(|_| true), &[(&b, REAL, OTHER)])
                .unwrap()
                .passes(),
            "it cannot tell the letters apart"
        );
        assert!(
            !calibrate(&judging(|_| false), &[(&b, REAL, OTHER)])
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
