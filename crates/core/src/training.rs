// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The record of how a candidate was trained: what it trained on, what it
//! replayed, what was measured on held-out records before and after, and
//! the curve of a run that watched a monitoring set as it trained
//! ([`TrainingCurve`]): which step's adapter it exported and why, and what
//! the curve warns of. Release manifests carry it.

use serde::{Deserialize, Serialize};

use crate::dataset::DatasetId;
use crate::digest::Digest;
use crate::release::ReleaseId;
use crate::terms::Terms;

/// One held-out score: teacher-forced loss and token accuracy over the
/// supervised positions of the held-out records.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct HeldOutScore {
    /// Mean per-token cross-entropy over the supervised positions; lower is
    /// better. `None` when no position was scored.
    pub loss: Option<f32>,
    /// Fraction of supervised positions, 0.0-1.0, where the greedy argmax
    /// matched the true next token; `None` when no position was scored.
    pub token_accuracy: Option<f64>,
    /// Supervised token positions the two numbers above were computed over.
    pub positions: usize,
    /// Held-out records scored.
    pub records: usize,
    /// Held-out records skipped (too long for the scoring row, or not
    /// encodable).
    pub skipped: usize,
}

/// brain's preference score of a tuned adapter against its reference on a
/// set of pairs.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PreferenceScore {
    /// Fraction, 0.0-1.0, of scored pairs where the tuned model prefers the
    /// chosen answer more than the reference does; `None` when no pair was
    /// scored.
    pub accuracy: Option<f32>,
    /// Mean over scored pairs of the reference-normalised log-probability
    /// margin of chosen over rejected, in nats (without `beta`); `None`
    /// when no pair was scored.
    pub mean_margin: Option<f32>,
    /// Pairs scored.
    pub pairs: usize,
    /// Pairs skipped because a candidate does not fit the model's context.
    pub skipped: usize,
}

/// How a candidate was trained, decided by its datasets' objective.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Regime {
    /// Supervised fine-tuning on chat records (`generic-messages-v2`); the
    /// regime of a record that names none.
    #[default]
    Sft,
    /// Direct preference optimisation on chosen/rejected pairs
    /// (`generic-preference-v1`).
    Dpo,
}

/// Where the replayed records came from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReplaySource {
    /// The earlier release.
    pub release: ReleaseId,
    /// Its datasets.
    pub datasets: Vec<DatasetId>,
    /// Its records that were trained on, which the sample is drawn from.
    pub available: usize,
    /// Of those, the ones replayed.
    pub sampled: usize,
}

/// The earlier records replayed beside the new ones.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReplaySample {
    /// The fraction of each release's training records drawn.
    pub fraction: f64,
    /// The draw's seed.
    pub seed: u64,
    /// Each earlier release trained on chat records, oldest first.
    pub sources: Vec<ReplaySource>,
    /// Records replayed in all.
    pub records: usize,
    /// The digest of the replay file; `None` when nothing was drawn.
    pub digest: Option<Digest>,
}

/// A preference candidate's measurements: brain's preference score of the
/// adapter against the reference it was trained against.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PreferenceSummary {
    /// The DPO temperature it trained with.
    pub beta: f32,
    /// The digest of the adapter the reference carried (the one
    /// continued); `None` when the reference was the base alone.
    pub reference_adapter: Option<String>,
    /// On the pairs trained on; `None` when not measured.
    pub train_score: Option<PreferenceScore>,
    /// On the held-out pairs; `None` when not measured.
    pub held_out_score: Option<PreferenceScore>,
}

/// Which step's adapter a candidate carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Selection {
    /// The last step trained: nothing was monitored, or the run was asked
    /// to keep its last step.
    LastStep,
    /// The evaluation with the lowest monitoring loss.
    BestMonitorLoss,
}

/// One evaluation of the monitoring set during training.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CurvePoint {
    /// Steps completed when it was taken.
    pub step: u32,
    /// The mean training loss of the steps since the previous evaluation:
    /// the training side of the curve over the same stretch.
    pub train_loss: f32,
    /// The mean per-token loss over the monitoring records.
    pub monitor_loss: f32,
}

/// How a supervised run went as it trained: the monitoring curve, the step
/// whose adapter it exported, and whether it stopped before its budget.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrainingCurve {
    /// The step budget: the most steps the run was allowed.
    pub steps: u32,
    /// Steps trained.
    pub steps_completed: u32,
    /// Steps between evaluations; 0 when nothing was monitored.
    pub eval_every: u32,
    /// Evaluations without improvement the run was allowed before stopping;
    /// 0 when it was never to stop early.
    pub patience: u32,
    /// Records in the monitoring set, carved from the training families;
    /// 0 when nothing was monitored.
    pub monitor_records: usize,
    /// Every evaluation, in order.
    pub points: Vec<CurvePoint>,
    /// The step whose adapter the candidate carries.
    pub selected_step: u32,
    /// Why that step.
    pub selection: Selection,
    /// True when the patience ran out before the budget did.
    pub stopped_early: bool,
}

/// The generalisation gap at the selected step, as a share of the
/// monitoring loss, above which the curve warns: a quarter means the
/// training rows were a quarter again more predictable than unseen ones, a
/// gap the held-out score of a fine-tune on a few hundred answers crosses
/// well before its training loss stops falling.
pub const LARGE_GAP_SHARE: f32 = 0.25;

impl TrainingCurve {
    /// A run that monitored nothing: it trained `steps_completed` of `steps`
    /// and carries its last step.
    #[must_use]
    pub fn unmonitored(steps: u32, steps_completed: u32) -> Self {
        Self {
            steps,
            steps_completed,
            eval_every: 0,
            patience: 0,
            monitor_records: 0,
            points: Vec::new(),
            selected_step: steps_completed,
            selection: Selection::LastStep,
            stopped_early: false,
        }
    }

    /// The evaluation at the selected step, when one was taken there.
    #[must_use]
    pub fn selected(&self) -> Option<&CurvePoint> {
        self.points.iter().find(|p| p.step == self.selected_step)
    }

    /// The evaluation with the lowest monitoring loss.
    #[must_use]
    pub fn best(&self) -> Option<&CurvePoint> {
        self.points
            .iter()
            .min_by(|a, b| a.monitor_loss.total_cmp(&b.monitor_loss))
    }

    /// The generalisation gap at the selected step: its monitoring loss
    /// less the training loss of the interval before it. `None` when no
    /// evaluation was taken there.
    #[must_use]
    pub fn generalisation_gap(&self) -> Option<f32> {
        self.selected().map(|p| p.monitor_loss - p.train_loss)
    }

    /// What the curve warns of about the adapter the candidate carries: a
    /// large generalisation gap at the selected step
    /// ([`LARGE_GAP_SHARE`]); an exported last step that is not the
    /// curve's best (the monitoring loss rose after it); a best that is the
    /// last evaluation, where the loss may still have been falling when the
    /// budget ran out. Empty when nothing was monitored or nothing is amiss.
    #[must_use]
    pub fn warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        let (Some(selected), Some(best), Some(last)) =
            (self.selected(), self.best(), self.points.last())
        else {
            return warnings;
        };
        let gap = selected.monitor_loss - selected.train_loss;
        if selected.monitor_loss > 0.0 && gap / selected.monitor_loss > LARGE_GAP_SHARE {
            warnings.push(format!(
                "the selected step {} still has a large generalisation gap: training loss {:.3} \
                 against monitoring loss {:.3}, a gap of {:.3} ({:.0}% of the monitoring loss; \
                 more than {:.0}% warns)",
                selected.step,
                selected.train_loss,
                selected.monitor_loss,
                gap,
                100.0 * gap / selected.monitor_loss,
                100.0 * LARGE_GAP_SHARE
            ));
        }
        if best.step != self.selected_step && best.monitor_loss < selected.monitor_loss {
            warnings.push(format!(
                "the monitoring loss rose before the end: its best was {:.3} at step {}, and the \
                 candidate carries step {} at {:.3}",
                best.monitor_loss, best.step, selected.step, selected.monitor_loss
            ));
        }
        if best.step == last.step && self.points.len() > 1 && !self.stopped_early {
            warnings.push(format!(
                "the monitoring loss was still falling when the budget of {} steps ran out (best \
                 {:.3} at the last evaluation, step {}): more steps may help",
                self.steps, best.monitor_loss, best.step
            ));
        }
        warnings
    }
}

/// How a candidate was trained, as its release records it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrainingSummary {
    /// The reference it was trained from.
    pub from: String,
    /// Optimizer steps.
    pub steps: u32,
    /// LoRA rank asked for.
    pub rank: u32,
    /// Records in the new datasets.
    pub records: usize,
    /// How it was trained.
    #[serde(default)]
    pub regime: Regime,
    /// The base on the held-out records; `None` when not measured (the
    /// preference regime measures preferences instead).
    #[serde(default)]
    pub base_score: Option<HeldOutScore>,
    /// The base with the adapter on the same records; `None` when not
    /// measured.
    #[serde(default)]
    pub tuned_score: Option<HeldOutScore>,
    /// The preference measurements; `None` for the supervised regime.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preference: Option<PreferenceSummary>,
    /// The monitoring curve and the step selected; `None` for the
    /// preference regime and for a candidate recorded before curves were.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub curve: Option<TrainingCurve>,
    /// brain's own training record of the adapter.
    pub record: serde_json::Value,
    /// The terms of what it trained on: its datasets' and those of the
    /// release it continued, combined; `None` when none were stated, which
    /// is unknown.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terms: Option<Terms>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn curve(points: &[(u32, f32, f32)], selected_step: u32, stopped_early: bool) -> TrainingCurve {
        TrainingCurve {
            steps: 80,
            steps_completed: points.last().map_or(0, |p| p.0),
            eval_every: 10,
            patience: 3,
            monitor_records: 12,
            points: points
                .iter()
                .map(|&(step, train_loss, monitor_loss)| CurvePoint {
                    step,
                    train_loss,
                    monitor_loss,
                })
                .collect(),
            selected_step,
            selection: Selection::BestMonitorLoss,
            stopped_early,
        }
    }

    /// The curve warns of what the selected adapter still shows, and only
    /// that: a gap past a quarter of the monitoring loss; a carried step
    /// the monitoring loss rose after; a best at the end of the budget.
    #[test]
    fn the_curve_warns_of_a_large_gap_a_rise_after_the_carried_step_and_a_best_at_the_end() {
        // Overfitting as measured: the best monitoring loss at step 30, the
        // last step carrying a 1.5 nat gap.
        let points = [
            (10, 2.4, 2.0),
            (20, 1.6, 1.5),
            (30, 1.2, 1.3),
            (40, 0.9, 1.35),
            (50, 0.7, 1.42),
            (60, 0.5, 1.5),
        ];
        let best = curve(&points, 30, true);
        assert_eq!(best.best().map(|p| p.step), Some(30));
        assert!((best.generalisation_gap().unwrap() - 0.1).abs() < 1e-6);
        assert!(best.warnings().is_empty(), "{:?}", best.warnings());

        let last = TrainingCurve {
            selection: Selection::LastStep,
            ..curve(&points, 60, false)
        };
        let warnings = last.warnings();
        assert_eq!(warnings.len(), 2, "{warnings:?}");
        assert!(
            warnings[0].contains("large generalisation gap") && warnings[0].contains("1.000"),
            "{}",
            warnings[0]
        );
        assert!(
            warnings[1].contains("rose before the end") && warnings[1].contains("step 30"),
            "{}",
            warnings[1]
        );

        let still_falling = curve(&[(10, 2.4, 2.0), (20, 1.6, 1.5), (30, 1.4, 1.3)], 30, false);
        let warnings = still_falling.warnings();
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(warnings[0].contains("still falling"), "{}", warnings[0]);

        let unmonitored = TrainingCurve::unmonitored(40, 40);
        assert_eq!(
            (unmonitored.selected_step, unmonitored.selection),
            (40, Selection::LastStep)
        );
        assert!(unmonitored.warnings().is_empty() && unmonitored.generalisation_gap().is_none());
    }
}
