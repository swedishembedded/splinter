// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training audits that tell a fine-tune that
// learned from one that memorised, for its clients. If your team needs
// expertise in judging small-model fine-tunes, you can procure our services
// by sending an email to info@swedishembedded.com.

//! The memorisation verdict of a supervised training run, computed from its
//! curve so that no run is judged by an ad-hoc script.
//!
//! The run is judged on the **kept** evaluation: the one with the lowest
//! monitoring loss, not the last step. It is **memorised** when either
//!
//! * (a) the kept monitoring loss is not below the base model's monitoring
//!   loss on the same records: the run improved nothing held out; or
//! * (b) the kept training loss (the mean over the steps of that
//!   evaluation's window) is below [`MEMORISATION_GAP_RATIO`] times the kept
//!   monitoring loss: the rows it trained on became far more predictable
//!   than unseen ones.
//!
//! Otherwise it is **not memorised**. A curve without evaluations, or a run
//! without a base score, is **not judged**, with the reason. The verdict is
//! reported; it never stops or alters a run.

use serde::{Deserialize, Serialize};
use splinter_core::training::TrainingCurve;

/// The default ratio of training loss to monitoring loss at the kept
/// evaluation below which a run counts as memorised: a training loss under
/// half the loss on unseen rows.
pub const MEMORISATION_GAP_RATIO: f32 = 0.5;

/// What the rule decided.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// The run fits its training rows without a held-out gain to show for it.
    Memorised,
    /// A held-out gain, and no large gap between training and monitoring.
    NotMemorised,
    /// The curve or the base score needed to decide is missing.
    NotJudged,
}

/// A verdict with the numbers it used.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Memorisation {
    /// The decision.
    pub verdict: Verdict,
    /// Why it was decided so: the rule that fired, or what was missing.
    pub reason: String,
    /// The training-loss to monitoring-loss ratio the rule used.
    pub ratio: f32,
    /// Step of the kept evaluation; `None` when there was none.
    pub kept_step: Option<u32>,
    /// Monitoring loss at the kept evaluation.
    pub monitor_loss: Option<f32>,
    /// Mean training loss over the kept evaluation's window.
    pub train_loss: Option<f32>,
    /// The base model's loss on the same records (the candidate's base score).
    pub base_loss: Option<f32>,
}

/// Judges `curve` against the base model's loss on the same records,
/// memorised when the training loss at the kept evaluation is below `ratio`
/// times its monitoring loss ([`MEMORISATION_GAP_RATIO`] by default).
#[must_use]
pub fn judge(curve: &TrainingCurve, base_loss: Option<f32>, ratio: f32) -> Memorisation {
    let kept = curve.best();
    let result = |verdict, reason: String| Memorisation {
        verdict,
        reason,
        ratio,
        kept_step: kept.map(|p| p.step),
        monitor_loss: kept.map(|p| p.monitor_loss),
        train_loss: kept.map(|p| p.train_loss),
        base_loss,
    };
    let Some(kept) = kept else {
        return result(Verdict::NotJudged, "no monitoring evaluations".into());
    };
    let Some(base) = base_loss else {
        return result(Verdict::NotJudged, "no base score".into());
    };
    if kept.monitor_loss >= base {
        return result(
            Verdict::Memorised,
            "no held-out improvement over the base".into(),
        );
    }
    if kept.train_loss < ratio * kept.monitor_loss {
        return result(
            Verdict::Memorised,
            format!("training loss below {ratio} x the monitoring loss"),
        );
    }
    result(
        Verdict::NotMemorised,
        "held-out improvement, small gap".into(),
    )
}

impl Memorisation {
    /// The one-line report:
    /// `memorisation: <verdict> (<reason>): kept step S, monitor M vs base B, train T`.
    #[must_use]
    pub fn line(&self) -> String {
        let label = match self.verdict {
            Verdict::Memorised => "memorised".to_string(),
            Verdict::NotMemorised => "not memorised".to_string(),
            Verdict::NotJudged => format!("not judged ({})", self.reason),
        };
        let number = |v: Option<f32>| v.map_or("n/a".to_string(), |v| format!("{v:.4}"));
        let step = self.kept_step.map_or("n/a".to_string(), |s| s.to_string());
        format!(
            "memorisation: {label}: kept step {step}, monitor {} vs base {}, train {}",
            number(self.monitor_loss),
            number(self.base_loss),
            number(self.train_loss)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use splinter_core::training::{CurvePoint, Selection};

    fn curve(points: &[(u32, f32, f32)], selected_step: u32) -> TrainingCurve {
        let mut curve = TrainingCurve::unmonitored(100, 100);
        curve.eval_every = 5;
        curve.selected_step = selected_step;
        curve.selection = Selection::BestMonitorLoss;
        curve.points = points
            .iter()
            .map(|&(step, train_loss, monitor_loss)| CurvePoint {
                step,
                train_loss,
                monitor_loss,
            })
            .collect();
        curve
    }

    fn verdict(curve: &TrainingCurve, base: Option<f32>) -> Verdict {
        judge(curve, base, MEMORISATION_GAP_RATIO).verdict
    }

    #[test]
    fn an_improvement_with_a_small_gap_is_not_memorised() {
        let c = curve(&[(5, 0.10, 0.09), (10, 0.07, 0.08)], 10);
        assert_eq!(verdict(&c, Some(0.11)), Verdict::NotMemorised);
    }

    #[test]
    fn no_improvement_over_the_base_is_memorised() {
        let c = curve(&[(5, 0.10, 0.12), (10, 0.09, 0.115)], 10);
        assert_eq!(verdict(&c, Some(0.11)), Verdict::Memorised);
        // Equal to the base is not below it.
        assert_eq!(verdict(&c, Some(0.115)), Verdict::Memorised);
    }

    #[test]
    fn a_large_gap_is_memorised_even_with_an_improvement() {
        let c = curve(&[(5, 0.10, 0.09), (10, 0.03, 0.08)], 10);
        assert_eq!(verdict(&c, Some(0.11)), Verdict::Memorised);
        // Exactly half is not below half.
        let c = curve(&[(5, 0.04, 0.08)], 5);
        assert_eq!(verdict(&c, Some(0.11)), Verdict::NotMemorised);
    }

    #[test]
    fn the_kept_evaluation_is_the_best_monitoring_loss_not_the_last() {
        // The last step memorised (train 0.01 vs 0.30) but the kept one did not.
        let c = curve(&[(5, 0.09, 0.08), (10, 0.01, 0.30)], 10);
        let m = judge(&c, Some(0.11), MEMORISATION_GAP_RATIO);
        assert_eq!((m.verdict, m.kept_step), (Verdict::NotMemorised, Some(5)));
    }

    #[test]
    fn missing_evaluations_or_base_are_not_judged_with_a_reason() {
        let none = curve(&[], 100);
        let m = judge(&none, Some(0.11), MEMORISATION_GAP_RATIO);
        assert_eq!(m.verdict, Verdict::NotJudged);
        assert!(m.reason.contains("no monitoring evaluations"), "{m:?}");
        let c = curve(&[(5, 0.09, 0.08)], 5);
        let m = judge(&c, None, MEMORISATION_GAP_RATIO);
        assert_eq!(m.verdict, Verdict::NotJudged);
        assert!(m.reason.contains("no base score"), "{m:?}");
        assert_eq!(
            m.line(),
            "memorisation: not judged (no base score): kept step 5, monitor 0.0800 vs base n/a, train 0.0900"
        );
    }

    #[test]
    fn the_line_names_the_numbers() {
        let c = curve(&[(30, 0.0576, 0.0829)], 30);
        let m = judge(&c, Some(0.1101), MEMORISATION_GAP_RATIO);
        assert_eq!(
            m.line(),
            "memorisation: not memorised: kept step 30, monitor 0.0829 vs base 0.1101, train 0.0576"
        );
    }

    /// CE8 `train-a25-r4`: the kept step 30 gains on the base and its
    /// training loss is 69% of the monitoring loss.
    #[test]
    fn ce8_a25_r4_is_not_memorised() {
        let c = curve(
            &[
                (5, 0.11039801, 0.09632168),
                (10, 0.09478252, 0.0916431),
                (15, 0.08578847, 0.09031377),
                (20, 0.094272874, 0.085774526),
                (25, 0.09391949, 0.0838136),
                (30, 0.057563324, 0.082928784),
                (33, 0.03456582, 0.08317255),
            ],
            30,
        );
        let m = judge(&c, Some(0.110074736), MEMORISATION_GAP_RATIO);
        assert_eq!((m.verdict, m.kept_step), (Verdict::NotMemorised, Some(30)));
    }

    /// CE8 `train-a22-bf16-r3`: it overfit after step 27, but the kept
    /// step 27 has train 0.0840 against monitor 0.0839.
    #[test]
    fn ce8_a22_bf16_r3_is_not_memorised() {
        let c = curve(
            &[
                (9, 0.10759897, 0.0955458),
                (18, 0.08776096, 0.08679464),
                (27, 0.0839881, 0.083882),
                (36, 0.042064197, 0.097105175),
                (45, 0.03994061, 0.087140344),
                (51, 0.032169115, 0.08669395),
            ],
            27,
        );
        let m = judge(&c, Some(0.110074736), MEMORISATION_GAP_RATIO);
        assert_eq!((m.verdict, m.kept_step), (Verdict::NotMemorised, Some(27)));
    }
}
