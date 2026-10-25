// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What `train` and `rehearse` print as text: a candidate with what it
//! trained on, replayed and rehearsed, its curve and its scores; and a
//! rehearsal set with what was admitted, refused and answered.

use std::fmt::Write as _;

use serde::Serialize;
use splinter_sdk::measure::memorisation::{judge, Memorisation, MEMORISATION_GAP_RATIO};
use splinter_sdk::rehearsal::Rehearsed;
use splinter_sdk::runs::Recorded;
use splinter_sdk::train::Candidate;
use splinter_sdk::vocabulary::training::{Selection, TrainingCurve};

use crate::output::Report;

/// What `train` reports: the candidate and, beside its fields, the
/// memorisation verdict of its curve (`memorisation`). The verdict is
/// reported only; it never stops or alters the run.
#[derive(Serialize)]
pub struct Trained {
    #[serde(flatten)]
    candidate: Recorded<Candidate>,
    memorisation: Memorisation,
}

impl Trained {
    /// `candidate` judged at `ratio` (the default when `None`) on the kept
    /// evaluation of its curve against the base score. A candidate with no
    /// curve (the preference regime) is not judged.
    pub fn new(candidate: Recorded<Candidate>, ratio: Option<f32>) -> Self {
        let ratio = ratio.unwrap_or(MEMORISATION_GAP_RATIO);
        let base_loss = candidate.report.base_score.and_then(|s| s.loss);
        let memorisation = match &candidate.report.curve {
            Some(curve) => judge(curve, base_loss, ratio),
            None => judge(
                &TrainingCurve::unmonitored(candidate.report.steps, candidate.report.steps),
                base_loss,
                ratio,
            ),
        };
        Self {
            candidate,
            memorisation,
        }
    }
}

impl Report for Trained {
    fn human(&self) -> String {
        format!("{}{}\n", self.candidate.human(), self.memorisation.line())
    }
}

fn loss(value: Option<f32>) -> String {
    value.map_or("not measured".into(), |v| format!("{v:.4}"))
}

impl Report for Candidate {
    fn human(&self) -> String {
        let parent = self
            .parent
            .as_ref()
            .map_or("no release".to_string(), |id| format!("release {id}"));
        let replay = self.replay.as_ref().map_or("none".to_string(), |r| {
            format!(
                "{} record(s) from {} earlier release(s), fraction {}",
                r.records,
                r.sources.len(),
                r.fraction
            )
        });
        let measured = match &self.preference {
            Some(preference) => {
                let (accuracy, margin) = preference
                    .held_out_score
                    .as_ref()
                    .map_or((None, None), |s| (s.accuracy, s.mean_margin));
                let reference = preference
                    .reference_adapter
                    .as_deref()
                    .map_or("the base".to_string(), |a| format!("adapter {a}"));
                format!(
                    "held-out preference against {reference}: accuracy {}, mean margin {} nats (beta {})",
                    loss(accuracy),
                    loss(margin),
                    preference.beta
                )
            }
            None => format!(
                "held-out loss: base {}, candidate {}",
                loss(self.base_score.and_then(|s| s.loss)),
                loss(self.tuned_score.and_then(|s| s.loss))
            ),
        };
        let regime = self.regime.label();
        let rehearsed = self.rehearsal.as_ref().map_or("none".to_string(), |r| {
            format!(
                "{} record(s) of {} at {} of the draws, {} monitored",
                r.trained, r.dataset, r.share, r.monitored
            )
        });
        let mut out = format!(
            "candidate {} trained by {regime} from {} ({parent}) on {} record(s)\n  replayed: {replay}\n  rehearsed: {rehearsed}\n  adapter: {} ({})\n  {measured}\n",
            self.candidate,
            self.from,
            self.records,
            self.adapter.display(),
            self.adapter_digest,
        );
        if let Some(curve) = &self.curve {
            out.push_str(&curve_lines(curve));
        }
        if !self.binary.is_empty() {
            let _ = writeln!(
                out,
                "  binary: {} ({}, {})",
                self.binary.path.as_deref().unwrap_or("unknown path"),
                self.binary
                    .digest
                    .as_ref()
                    .map_or_else(|| "unknown digest".to_string(), ToString::to_string),
                self.binary.version.as_deref().unwrap_or("unknown version")
            );
        }
        for warning in self.warnings() {
            let _ = writeln!(out, "  warning: {warning}");
        }
        let _ = writeln!(
            out,
            "  not released: `splinter release {}` runs the gate",
            self.candidate
        );
        out
    }
}

/// A training curve as text: the steps trained of the budget, the step
/// carried and why, the gap at it, and every evaluation.
fn curve_lines(curve: &TrainingCurve) -> String {
    let mut out = String::new();
    let stopped = if curve.stopped_early {
        format!(
            ", stopped early (no improvement for {} evaluations)",
            curve.patience
        )
    } else {
        String::new()
    };
    let why = match curve.selection {
        Selection::LastStep => "the last step",
        Selection::BestMonitorLoss => "the best monitoring loss",
        Selection::DevSuite => "the best answers on the dev suite",
    };
    let _ = writeln!(
        out,
        "  steps: {} of a budget of {}{stopped}; carries step {} ({why})",
        curve.steps_completed, curve.steps, curve.selected_step
    );
    if let Some(speed) = &curve.speed {
        let _ = writeln!(
            out,
            "  speed: {:.0} tokens/s ({} row tokens in {:.1} s over {} steps, from the first step report to the last); \
             {:.1} s before the first step report (model load, base scoring, one step), {:.1} s after the last (scoring, export)",
            speed.tokens_per_second,
            speed.row_tokens,
            speed.seconds,
            speed.steps_timed,
            speed.lead_seconds,
            speed.tail_seconds
        );
    }
    if curve.eval_every == 0 {
        let _ = writeln!(out, "  monitoring: none");
        return out;
    }
    let _ = write!(
        out,
        "  monitoring: {} record(s) every {} step(s)",
        curve.monitor_records, curve.eval_every
    );
    if let (Some(selected), Some(gap)) = (curve.selected(), curve.generalisation_gap()) {
        let _ = write!(
            out,
            "; at step {}: training loss {}, monitoring loss {}, gap {} ({:.0}%)",
            selected.step,
            loss(Some(selected.train_loss)),
            loss(Some(selected.monitor_loss)),
            loss(Some(gap)),
            100.0 * gap / selected.monitor_loss
        );
    }
    out.push('\n');
    let points: Vec<String> = curve
        .points
        .iter()
        .map(|p| format!("{} {:.3}/{:.3}", p.step, p.train_loss, p.monitor_loss))
        .collect();
    if !points.is_empty() {
        let _ = writeln!(out, "  curve (step train/monitor): {}", points.join(", "));
    }
    out
}

impl Report for Rehearsed {
    fn human(&self) -> String {
        let anchor = self.anchor.as_ref().map_or_else(
            || "no anchor suite is frozen, so nothing was kept out".to_string(),
            |(version, digest)| format!("anchor suite version {version} ({digest}) kept out"),
        );
        let mut out = format!(
            "rehearsal task set {}: {} task(s) for {} - {} by code, {} general by the base; {} \
             refused as anchor tasks, {} repeats, {} request(s) failed; {anchor}\n",
            self.task_set,
            self.tasks,
            self.base,
            self.form_tasks,
            self.general_tasks,
            self.excluded_as_anchor,
            self.repeated,
            self.failed_requests,
        );
        let _ = writeln!(
            out,
            "  answered {} of {}; graded: {} right, {} wrong, {} ungraded",
            self.solved.answered,
            self.solved.solved,
            self.verified.passed,
            self.verified.failed,
            self.verified.undecided
        );
        let _ = writeln!(
            out,
            "  dataset {}: {} record(s) (`train <DATASET-ID>... --rehearsal {}`)",
            self.dataset.dataset, self.dataset.records, self.dataset.dataset
        );
        if let Some(why) = &self.stopped {
            let _ = writeln!(out, "  stopped: {why}");
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use splinter_sdk::vocabulary::training::TrainingSpeed;

    /// The report names the training speed, and what it leaves out.
    #[test]
    fn the_curve_text_reports_tokens_per_second_and_the_time_around_the_steps() {
        let mut curve = TrainingCurve::unmonitored(10, 10);
        curve.speed = TrainingSpeed::measured(9, 4, 1000, 18.0, 40.0, 7.0);
        let text = curve_lines(&curve);
        // perf-number: the fixture of a unit test, not a measurement
        assert!(text.contains("speed: 2000 tokens/s"), "{text}");
        assert!(
            // perf-number: the fixture of a unit test, not a measurement
            text.contains("36000 row tokens in 18.0 s over 9 steps"),
            "{text}"
        );
        assert!(
            // perf-number: the fixture of a unit test, not a measurement
            text.contains("40.0 s before the first step report"),
            "{text}"
        );
        curve.speed = None;
        assert!(
            !curve_lines(&curve).contains("speed:"),
            "a run with no timed interval prints no speed"
        );
    }
}
