// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measured validation of continual learning
// from a user's own agent sessions for its clients. If your team needs
// expertise in proving that a model absorbed what its user taught it without
// forgetting what it knew, you can procure our services by sending an email
// to info@swedishembedded.com.

//! One evaluation pass over the release in effect after a night.
//!
//! Loading a base takes minutes, so the pass has two phases and each model
//! is loaded once: the policy answers every item, then the judge rules on
//! all the answers code could not decide.

use serde::{Deserialize, Serialize};

use super::grade::{draft, finish, ItemResult, VerdictRequest};
use super::items::Item;
use crate::facts::JudgeCalibration;

/// What the models do for an evaluation: each call loads its model once.
pub trait Models {
    /// The answers of the release in effect (greedy, under the persona) to
    /// `questions`, in order; `None` where a run gave no answer.
    ///
    /// # Errors
    /// The policy cannot be loaded or run.
    fn answer_all(&self, questions: &[String]) -> anyhow::Result<Answered>;

    /// The judge's verdict on each request, in order; `None` where it
    /// abstained.
    ///
    /// # Errors
    /// The judge cannot be loaded, is not precise enough to grade with, or
    /// cannot run.
    fn judge_all(&self, requests: &[VerdictRequest]) -> anyhow::Result<Judged>;
}

/// The policy's answers.
#[derive(Clone, Debug)]
pub struct Answered {
    /// One per question.
    pub texts: Vec<Option<String>>,
    /// The identity the answers are recorded under.
    pub policy: String,
    /// Seconds the phase took, the load included.
    pub seconds: f64,
}

/// The judge's verdicts.
#[derive(Clone, Debug)]
pub struct Judged {
    /// One per request.
    pub verdicts: Vec<Option<bool>>,
    /// What the judge was measured to be before it ruled.
    pub calibration: JudgeCalibration,
    /// Seconds the phase took, the load and the calibration included.
    pub seconds: f64,
}

/// The mean log-probability of a fixed text, against the base.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LogProb {
    /// Of the release in effect.
    pub release: f64,
    /// Of the base.
    pub base: f64,
    /// Tokens scored.
    pub tokens: usize,
}

/// Seconds spent per phase.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Timings {
    /// The policy's answers, its load included.
    pub answer_seconds: f64,
    /// The judge's rulings, its load and calibration included.
    pub judge_seconds: Option<f64>,
}

/// Everything one night's evaluation found.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Evaluation {
    /// The night; 0 is the base before any session.
    pub night: usize,
    /// The release in effect after the night; absent for the base.
    pub release: Option<String>,
    /// The policy's identity.
    pub policy: String,
    /// The hash of the pre-registration this evaluation was made under.
    pub prereg_hash: String,
    /// One result per item, in the order of the item list.
    pub results: Vec<ItemResult>,
    /// The judge's calibration; absent when no answer needed the judge.
    pub judge: Option<JudgeCalibration>,
    /// Seconds per phase.
    pub timings: Timings,
    /// The mean log-probability of the fixed text; absent when not measured.
    pub logprob: Option<LogProb>,
}

/// Evaluates the release in effect on every item: the policy answers all of
/// them, code reads the answers, and the judge rules on the rest, all at once.
///
/// # Errors
/// A model fails, or answers fewer items than were asked.
pub fn evaluate(
    items: &[Item],
    models: &dyn Models,
    night: usize,
    release: Option<String>,
    prereg_hash: &str,
) -> anyhow::Result<Evaluation> {
    let questions: Vec<String> = items.iter().map(|i| i.question.clone()).collect();
    let answered = models.answer_all(&questions)?;
    anyhow::ensure!(
        answered.texts.len() == items.len(),
        "the policy gave {} answers to {} questions",
        answered.texts.len(),
        items.len()
    );
    let drafts: Vec<_> = items
        .iter()
        .zip(&answered.texts)
        .map(|(item, text)| draft(item, text.as_deref()))
        .collect();
    let requests: Vec<VerdictRequest> =
        drafts.iter().filter_map(|d| d.request().cloned()).collect();
    let (verdicts, judge, judge_seconds) = if requests.is_empty() {
        (Vec::new(), None, None)
    } else {
        let judged = models.judge_all(&requests)?;
        anyhow::ensure!(
            judged.verdicts.len() == requests.len(),
            "the judge ruled on {} of {} requests",
            judged.verdicts.len(),
            requests.len()
        );
        (
            judged.verdicts,
            Some(judged.calibration),
            Some(judged.seconds),
        )
    };
    let mut verdicts = verdicts.into_iter();
    let results = items
        .iter()
        .zip(drafts)
        .map(|(item, draft)| {
            let verdict = draft.request().and_then(|_| verdicts.next().flatten());
            finish(item, draft, verdict)
        })
        .collect();
    Ok(Evaluation {
        night,
        release,
        policy: answered.policy,
        prereg_hash: prereg_hash.to_string(),
        results,
        judge,
        timings: Timings {
            answer_seconds: answered.seconds,
            judge_seconds,
        },
        logprob: None,
    })
}
