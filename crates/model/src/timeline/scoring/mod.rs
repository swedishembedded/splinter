// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements honest head-to-head evaluation of
// time-to-event risk models on held-out participants, for its clients. If
// your team needs expertise in discrimination, calibration and paired
// comparison of competing-risk predictions, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Two timeline models scored on the same held-out subjects.
//!
//! Every number is brain's own arithmetic ([`survival`](super::survival)):
//! Uno's concordance, the time-dependent AUC, the IPCW Brier score and its
//! integral over a grid, calibration at a horizon against the Aalen-Johansen
//! observed risk, and the held-out event likelihood. Each outcome code is
//! scored with the other absorbing codes competing, and an optional all-cause
//! view is the union of the absorbing ones. The candidate-minus-champion
//! differences carry percentile intervals from a bootstrap that resamples
//! whole participants' groups (the keyed group id of the record), never single
//! records, so a household or site counts once.
//!
//! A horizon at which an outcome has fewer than `min_events` events in the
//! scored set is ABSENT from every table: not measured is never reported as
//! zero, and a requirement on it then fails as unmeasured.
//!
//! * [`Predicted`] - one model's predictions for the subjects ([`predict_arm`]).
//! * [`compare`] - two arms' predictions scored and differenced ([`Comparison`]).
//! * [`Comparison::evidence`] - the numbers under the names the release gate
//!   reads ([`splinter_eval::timeline_metrics`]).

mod arm;
mod difference;

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use splinter_eval::metric_gate::Evidence;
use splinter_eval::timeline_metrics::{
    arm_metric, arm_nll, diff_metric, subgroup_diff, subgroup_events, Arm, NLL_DIFF,
};

use super::training::TimelineError;
use super::{Subject, TimelineModel};

pub use arm::{predict_arm, Predicted, View, ViewKind, CALIBRATION_GROUPS, IBS_POINTS};
pub use difference::compare;

/// A percentile interval.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Interval {
    /// The statistic on the whole set.
    pub estimate: f64,
    /// Lower bound.
    pub lo: f64,
    /// Upper bound.
    pub hi: f64,
}

/// Which subjects a named subgroup holds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "rule", rename_all = "snake_case")]
pub enum SubgroupRule {
    /// A categorical variable at a level, as last observed at or before entry.
    Category {
        /// The variable.
        var: String,
        /// The level.
        level: String,
    },
    /// A numeric variable at or above a value at entry.
    AtLeast {
        /// The variable.
        var: String,
        /// The threshold.
        value: f64,
    },
    /// A numeric variable below a value at entry.
    Below {
        /// The variable.
        var: String,
        /// The threshold.
        value: f64,
    },
    /// Records of one source (a cohort cycle, a site).
    Source(String),
}

/// A named slice of the held-out subjects no outcome may regress on.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Subgroup {
    /// The name its numbers travel under.
    pub name: String,
    /// Who is in it.
    pub rule: SubgroupRule,
}

impl Subgroup {
    /// Whether `subject` belongs: a subject without the variable does not.
    #[must_use]
    pub fn holds(&self, subject: &Subject) -> bool {
        use super::Value;
        let latest = |var: &str| {
            subject
                .observations
                .iter()
                .filter(|o| o.var == var && o.t <= subject.entry)
                .max_by(|a, b| a.t.total_cmp(&b.t))
                .map(|o| &o.value)
        };
        match &self.rule {
            SubgroupRule::Category { var, level } => {
                matches!(latest(var), Some(Value::Category(c)) if c == level)
            }
            SubgroupRule::AtLeast { var, value } => {
                matches!(latest(var), Some(Value::Number(v)) if v >= value)
            }
            SubgroupRule::Below { var, value } => {
                matches!(latest(var), Some(Value::Number(v)) if v < value)
            }
            SubgroupRule::Source(source) => &subject.source == source,
        }
    }
}

/// How two arms are scored.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScoreSpec {
    /// The outcome codes both models predict.
    pub codes: Vec<String>,
    /// Which of them end follow-up (they compete with each other).
    pub absorbing: Vec<String>,
    /// When set, an extra view of the union of the absorbing codes under
    /// this name.
    pub all_cause: Option<String>,
    /// The horizons metrics are reported at, increasing, inside both models'
    /// last knot.
    pub horizons: Vec<f64>,
    /// The fewest events of an outcome by a horizon for it to be measured.
    pub min_events: usize,
    /// Bootstrap resamples of the differences.
    pub bootstrap_reps: usize,
    /// The interval level, in (0, 1).
    pub level: f64,
    /// The bootstrap's seed.
    pub seed: u64,
    /// Subgroups the integrated Brier difference is also reported on.
    pub subgroups: Vec<Subgroup>,
}

impl ScoreSpec {
    /// A spec over `codes` of which `absorbing` compete, at `horizons`, with
    /// brain's ten risk groups and twenty point grid ([`CALIBRATION_GROUPS`],
    /// [`IBS_POINTS`]), 200 resamples at level 0.95 and at least ten events.
    pub fn new<C: Into<String>, A: Into<String>>(
        codes: impl IntoIterator<Item = C>,
        absorbing: impl IntoIterator<Item = A>,
        horizons: Vec<f64>,
    ) -> Self {
        Self {
            codes: codes.into_iter().map(Into::into).collect(),
            absorbing: absorbing.into_iter().map(Into::into).collect(),
            all_cause: None,
            horizons,
            min_events: 10,
            bootstrap_reps: 200,
            level: 0.95,
            seed: 1,
            subgroups: Vec::new(),
        }
    }

    /// The last horizon: the end of the integration grid.
    #[must_use]
    pub fn last_horizon(&self) -> f64 {
        self.horizons.last().copied().unwrap_or(0.0)
    }

    pub(crate) fn check(&self, models: &[&TimelineModel]) -> Result<(), TimelineError> {
        let bad = |why: String| Err(TimelineError::Request(format!("scoring: {why}")));
        if self.codes.is_empty() || self.horizons.is_empty() {
            return bad("name at least one outcome code and one horizon".into());
        }
        if let Some(a) = self.absorbing.iter().find(|a| !self.codes.contains(a)) {
            return bad(format!(
                "absorbing code {a:?} is not one of {:?}",
                self.codes
            ));
        }
        if self.horizons.iter().any(|h| !(h.is_finite() && *h > 0.0))
            || self.horizons.windows(2).any(|w| w[1] <= w[0])
        {
            return bad(format!(
                "horizons {:?} must be positive and increasing",
                self.horizons
            ));
        }
        if self.bootstrap_reps == 0 || !(self.level > 0.0 && self.level < 1.0) {
            return bad("reps need to be at least 1 and the level inside (0, 1)".into());
        }
        for (arm, model) in ["champion", "candidate"].iter().zip(models) {
            if let Some(c) = self.codes.iter().find(|c| !model.codes().contains(c)) {
                return bad(format!(
                    "the {arm} does not predict {c:?}; it predicts {:?}",
                    model.codes()
                ));
            }
            let last = model.config().knots.last().copied().map_or(0.0, f64::from);
            if self.last_horizon() > last + 1e-9 {
                return bad(format!(
                    "horizon {} is past the {arm}'s last knot {last}: it says nothing later",
                    self.last_horizon()
                ));
            }
        }
        Ok(())
    }
}

/// One arm's metrics at one horizon.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HorizonScores {
    /// The horizon.
    pub horizon: f64,
    /// Events of the outcome by the horizon in the scored set.
    pub events: usize,
    /// Uno's concordance truncated at the horizon.
    pub uno_c: Option<f64>,
    /// The time-dependent AUC.
    pub auc: Option<f64>,
    /// The IPCW Brier score.
    pub brier: Option<f64>,
    /// The calibration at the horizon, judged on the risk `calibration_basis`
    /// names; absent when the model declared the horizon uncalibrated.
    pub calibration: Option<Calibration>,
    /// Which risk the calibration numbers were measured on.
    #[serde(default)]
    pub calibration_basis: CalibrationBasis,
}

/// Which risk a horizon's calibration is judged on. Discrimination and error
/// are always the model's raw risk, so two models are compared on what they
/// predict; calibration is judged on the risk the model is served as.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationBasis {
    /// The model's raw cumulative incidence: it has no calibration at this
    /// code and horizon, or none at all.
    #[default]
    Raw,
    /// The Venn-Abers calibrated risk brain serves at this code and horizon.
    Calibrated,
    /// brain declared this code and horizon uncalibrated (too few validation
    /// events): nothing is judged, and a requirement on it fails as
    /// unmeasured, never as zero.
    Uncalibrated,
}

/// Calibration at a horizon.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Calibration {
    /// Recalibration slope.
    pub slope: f64,
    /// Recalibration intercept.
    pub intercept: f64,
    /// Observed (Aalen-Johansen) over expected.
    pub oe: f64,
    /// Expected calibration error over the risk groups.
    pub ece: f64,
}

/// One arm's metrics for one view (an outcome code or the all-cause union).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ViewScores {
    /// The integrated Brier score over the grid; absent below `min_events`.
    pub ibs: Option<f64>,
    /// The horizons that were measured, in order.
    pub horizons: Vec<HorizonScores>,
}

/// One arm's scores.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArmScores {
    /// The weighted mean held-out event negative log-likelihood.
    pub event_nll: f64,
    /// Scores by view name.
    pub views: BTreeMap<String, ViewScores>,
}

/// The candidate-minus-champion intervals of one view.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ViewDifferences {
    /// The integrated Brier difference (negative: the candidate is better).
    pub ibs: Option<Interval>,
    /// The differences at each measured horizon.
    pub horizons: Vec<HorizonDifferences>,
}

/// The candidate-minus-champion intervals at a horizon.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HorizonDifferences {
    /// The horizon.
    pub horizon: f64,
    /// The Brier difference (negative: the candidate is better).
    pub brier: Option<Interval>,
    /// The Uno concordance difference (positive: the candidate is better).
    pub uno_c: Option<Interval>,
    /// The AUC difference (positive: the candidate is better).
    pub auc: Option<Interval>,
}

/// A subgroup's comparison on one view.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SubgroupDifference {
    /// Subjects in the subgroup.
    pub subjects: usize,
    /// Events of the view's outcome by the last horizon among them.
    pub events: usize,
    /// The integrated Brier difference, absent below `min_events`.
    pub ibs: Option<f64>,
}

/// Two arms scored on the same subjects.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Comparison {
    /// The subjects both were scored on, in order.
    pub units: Vec<String>,
    /// The champion's scores.
    pub champion: ArmScores,
    /// The candidate's scores.
    pub candidate: ArmScores,
    /// Held-out event likelihood difference (negative: the candidate is better).
    pub event_nll: Option<Interval>,
    /// Differences by view name.
    pub differences: BTreeMap<String, ViewDifferences>,
    /// Subgroup comparisons by subgroup name, then view name.
    pub subgroups: BTreeMap<String, BTreeMap<String, SubgroupDifference>>,
}

impl Comparison {
    /// Every measured number under the names the release gate reads. A number
    /// that was not measured has no entry.
    #[must_use]
    pub fn evidence(&self) -> Evidence {
        let mut e = Evidence::default();
        for (arm, scores) in [
            (Arm::Champion, &self.champion),
            (Arm::Candidate, &self.candidate),
        ] {
            e.values.insert(arm_nll(arm), scores.event_nll);
            for (view, s) in &scores.views {
                if let Some(ibs) = s.ibs {
                    e.values.insert(arm_metric(arm, view, None, "ibs"), ibs);
                }
                for h in &s.horizons {
                    let at = Some(h.horizon);
                    let mut put = |metric: &str, value: Option<f64>| {
                        if let Some(v) = value {
                            e.values.insert(arm_metric(arm, view, at, metric), v);
                        }
                    };
                    put("uno_c", h.uno_c);
                    put("auc", h.auc);
                    put("brier", h.brier);
                    if let Some(c) = &h.calibration {
                        put("slope", Some(c.slope));
                        put("intercept", Some(c.intercept));
                        put("oe", Some(c.oe));
                        put("ece", Some(c.ece));
                    }
                }
            }
        }
        for (view, d) in &self.differences {
            put_interval(&mut e, diff_metric(view, None, "ibs"), d.ibs.as_ref());
            for h in &d.horizons {
                let at = Some(h.horizon);
                put_interval(&mut e, diff_metric(view, at, "brier"), h.brier.as_ref());
                put_interval(&mut e, diff_metric(view, at, "uno_c"), h.uno_c.as_ref());
                put_interval(&mut e, diff_metric(view, at, "auc"), h.auc.as_ref());
            }
        }
        put_interval(&mut e, NLL_DIFF.into(), self.event_nll.as_ref());
        for (group, views) in &self.subgroups {
            for (view, d) in views {
                e.values
                    .insert(subgroup_events(group, view), d.events as f64);
                if let Some(ibs) = d.ibs {
                    e.values.insert(subgroup_diff(group, view), ibs);
                }
            }
        }
        e
    }
}

fn put_interval(e: &mut Evidence, name: String, interval: Option<&Interval>) {
    if let Some(i) = interval {
        e.values.insert(format!("{name}:estimate"), i.estimate);
        e.intervals.insert(name, (i.lo, i.hi));
    }
}
