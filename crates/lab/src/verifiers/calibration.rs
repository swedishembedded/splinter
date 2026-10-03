// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements calibrated model judges whose verdicts
// carry the precision they were measured at, for its clients. If your team
// needs expertise in LLM-as-judge evaluation or reward modelling, you can
// procure our services by sending an email to info@swedishembedded.com.

//! A judge measured against labelled answers, and a judge that speaks only
//! where that measurement says it is precise enough.
//!
//! [`calibrate`] runs a judged verifier over experiences whose correct
//! outcome is known and records how often its passes and its fails were
//! right. [`CalibratedJudge`] wraps the same verifier: it passes on a
//! verdict only when the calibration's precision for that verdict reaches
//! its threshold (by default [`DEFAULT_MIN_PRECISION`]) and abstains
//! otherwise, and every verdict names the calibration it was gated by.

use serde::{Deserialize, Serialize};
use serde_json::json;
use splinter_core::annotation::{AnnotationBody, Outcome, Producer, Strength};
use splinter_core::digest::{canonical_json, Digest};
use splinter_core::experience::{Experience, ExperienceError, ExperienceId, Task};

use super::{annotation, Finding, Verifier, VerifyError};

/// The precision a calibrated judge needs by default for a verdict to
/// stand.
pub const DEFAULT_MIN_PRECISION: f64 = 0.9;

/// How a judge's verdicts compared with known outcomes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Calibration {
    /// The digest of what was measured: the judge's producer, and each
    /// labelled experience with its label and the judge's verdict.
    pub id: Digest,
    /// The judge measured.
    pub producer: Producer,
    /// The labelled experiences it was measured on.
    pub n: usize,
    /// Of the judge's passes, the share labelled pass; absent when it
    /// passed none.
    pub precision_pass: Option<f64>,
    /// Of the judge's fails, the share labelled fail; absent when it
    /// failed none.
    pub precision_fail: Option<f64>,
    /// The share of experiences it abstained on; absent when `n` is zero.
    pub abstain_rate: Option<f64>,
}

/// One labelled experience as the calibration's id covers it.
#[derive(Serialize)]
struct Measured {
    experience: ExperienceId,
    label: Outcome,
    judged: Outcome,
}

/// `numerator / denominator`, absent when the denominator is zero.
fn share(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator > 0).then(|| numerator as f64 / denominator as f64)
}

/// Measures `judge` on `labelled`: each experience of its task with the
/// outcome it should get, pass or fail.
pub fn calibrate(
    judge: &dyn Verifier,
    labelled: &[(Task, Experience, Outcome)],
) -> Result<Calibration, VerifyError> {
    let mut measured = Vec::with_capacity(labelled.len());
    for (index, (task, exp, label)) in labelled.iter().enumerate() {
        if *label == Outcome::Abstain {
            return Err(VerifyError::Label { index });
        }
        let note = annotation(judge, task, exp)?;
        let AnnotationBody::Verdict { outcome, .. } = note.body else {
            unreachable!("annotation() builds a verdict");
        };
        measured.push(Measured {
            experience: note.experience,
            label: *label,
            judged: outcome,
        });
    }
    let count = |judged: Outcome, label: Option<Outcome>| {
        measured
            .iter()
            .filter(|m| m.judged == judged && label.is_none_or(|l| m.label == l))
            .count()
    };
    let producer = judge.producer();
    let id = Digest::of(
        &canonical_json(&json!({ "producer": producer, "measured": measured }))
            .map_err(ExperienceError::from)?,
    );
    Ok(Calibration {
        id,
        producer,
        n: measured.len(),
        precision_pass: share(
            count(Outcome::Pass, Some(Outcome::Pass)),
            count(Outcome::Pass, None),
        ),
        precision_fail: share(
            count(Outcome::Fail, Some(Outcome::Fail)),
            count(Outcome::Fail, None),
        ),
        abstain_rate: share(count(Outcome::Abstain, None), measured.len()),
    })
}

/// A judged verifier gated by its calibration: a verdict whose measured
/// precision is below the threshold, or was not measured, becomes an
/// abstention. The evidence names the calibration, the precision it
/// applied, the judge's own verdict, and the judge's evidence.
#[derive(Debug)]
pub struct CalibratedJudge<V> {
    inner: V,
    calibration: Calibration,
    min_precision: f64,
}

impl<V: Verifier> CalibratedJudge<V> {
    /// `inner` gated by `calibration` at `min_precision` (in `[0, 1]`).
    /// Refused when the calibration measured a different producer.
    pub fn new(
        inner: V,
        calibration: Calibration,
        min_precision: f64,
    ) -> Result<Self, VerifyError> {
        let producer = inner.producer();
        if calibration.producer != producer {
            return Err(VerifyError::CalibrationMismatch {
                measured: format!(
                    "{}@{}",
                    calibration.producer.name, calibration.producer.version
                ),
                verifier: format!("{}@{}", producer.name, producer.version),
            });
        }
        if !(0.0..=1.0).contains(&min_precision) {
            return Err(VerifyError::Parameter {
                name: "min_precision",
                reason: format!("{min_precision} is not in [0, 1]"),
            });
        }
        Ok(Self {
            inner,
            calibration,
            min_precision,
        })
    }

    /// The calibration verdicts are gated by.
    #[must_use]
    pub fn calibration(&self) -> &Calibration {
        &self.calibration
    }
}

impl<V: Verifier> Verifier for CalibratedJudge<V> {
    fn producer(&self) -> Producer {
        let inner = self.inner.producer();
        Producer {
            name: format!("{}+calibrated", inner.name),
            version: inner.version,
        }
    }

    fn strength(&self) -> Strength {
        self.inner.strength()
    }

    fn verify(&self, task: &Task, exp: &Experience) -> Result<Finding, VerifyError> {
        let judged = self.inner.verify(task, exp)?;
        let precision = match judged.outcome {
            Outcome::Pass => self.calibration.precision_pass,
            Outcome::Fail => self.calibration.precision_fail,
            Outcome::Abstain => None,
        };
        let mut evidence = json!({
            "calibration": self.calibration.id,
            "min_precision": self.min_precision,
            "precision": precision,
            "judged": judged.outcome,
            "judge": judged.evidence,
        });
        let stands = precision.is_some_and(|p| p >= self.min_precision);
        if judged.outcome != Outcome::Abstain && !stands {
            evidence["abstained"] = json!("the judge's measured precision is below the threshold");
        }
        Ok(Finding {
            outcome: if stands {
                judged.outcome
            } else {
                Outcome::Abstain
            },
            evidence,
        })
    }
}
