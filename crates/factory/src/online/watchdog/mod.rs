// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training loops that notice a model moving the
// wrong way and roll it back on evidence, for its clients. If your team needs
// expertise in safe continuous training, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The watchdog of an online run: it detects when training moves the policy
//! the wrong way and rolls it back, in three layers.
//!
//! 1. **Invariants** ([`signals::Bounds`]): fixed registered bounds on the
//!    training's own health. A new breach raises an alarm, pauses the intake
//!    of new groups (rollouts already running finish) and asks for an
//!    evaluation; intake resumes once an evaluation rules (or at once when
//!    there is no candidate to evaluate). It never rolls back by itself.
//! 2. **Drift** ([`signals::Drift`]): detectors on the change of each
//!    version's behaviour from the promoted versions'. A detector that
//!    crosses raises an alarm and asks for an evaluation at once.
//! 3. **Decisions** ([`sequential`]): only a paired evaluation decides. The
//!    newest published version (a frozen copy, by its adapter's digest) and
//!    the anchor (the last promoted version, initially the one the epoch
//!    started from) each attempt the same tasks of a fresh draw of the
//!    held-out monitor pool ([`monitor`]); an anytime-valid test on the
//!    discordant pairs rolls back on evidence of a regression larger than
//!    `delta`, and promotes on evidence the candidate is not worse by
//!    `delta`, after the sealed confirm set shows no regression; otherwise
//!    training continues. Evaluations run every `evaluate_every` published
//!    versions, and at once on a drift alarm.
//!
//! A rollback ends the run's epoch: the stage stops intake, the groups of
//! the epoch not yet committed are rolled back (never trained), and a new
//! learner continues from the anchor's adapter with fresh optimiser state in
//! the next epoch; the stage keeps an audit of why ([`Audit`]).

pub mod monitor;
pub mod sequential;
pub mod signals;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use self::monitor::MonitorPool;
use self::sequential::{Comparison, Decision, TestSettings};
use self::signals::{Bounds, Drift, DriftAlarm, DriftSettings, Signal};
use super::health::{StepHealth, VersionHealth, VersionSeries};
use super::seams::LearnerTelemetry;

/// The watchdog's registered settings (`train.online`'s `watchdog`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WatchdogSettings {
    /// The invariants' bounds.
    pub bounds: Bounds,
    /// The drift detectors' parameters.
    pub drift: DriftSettings,
    /// The paired test's parameters.
    pub test: TestSettings,
    /// Published versions between two scheduled evaluations.
    pub evaluate_every: u32,
    /// Monitor tasks a look draws.
    pub monitor_size: u32,
    /// Looks a monitor task may be drawn in per epoch.
    pub max_uses: u32,
    /// Rollbacks after which the run halts.
    pub max_rollbacks: u32,
    /// The factor a rollback multiplies the learner's learning rate by
    /// (`params.lr`), in `(0, 1]`.
    pub cooldown: f64,
}

impl WatchdogSettings {
    /// Checks the settings.
    ///
    /// # Errors
    /// A setting out of range.
    pub fn validate(&self) -> Result<(), String> {
        self.test.validate()?;
        if self.evaluate_every == 0 || self.monitor_size == 0 || self.max_uses == 0 {
            return Err("evaluate_every, monitor_size and max_uses are at least 1".into());
        }
        if !(self.cooldown > 0.0 && self.cooldown <= 1.0) {
            return Err(format!("cooldown {} is not in (0, 1]", self.cooldown));
        }
        if !(self.drift.slack >= 0.0 && self.drift.threshold > 0.0) {
            return Err("drift slack is at least 0 and its threshold above 0".into());
        }
        Ok(())
    }
}

/// A policy version as evaluations address it: its serving-local version
/// and its adapter's digest, by which a frozen copy is served.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Version {
    /// The serving-local version.
    pub version: u64,
    /// The adapter's digest.
    pub digest: String,
}

/// One task attempted by both policies of a comparison.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pair {
    /// The task.
    pub task: String,
    /// Whether the candidate passed it.
    pub candidate: bool,
    /// Whether the anchor passed it.
    pub anchor: bool,
}

/// What grades a candidate against the anchor on tasks, on capacity of its
/// own, never seen by the task selector.
pub trait Evaluator {
    /// Start: each of `tasks` attempted once by `candidate` and once by
    /// `anchor`.
    fn start(&mut self, candidate: &Version, anchor: &Version, tasks: Vec<String>);

    /// The pairs graded since the last poll.
    fn poll(&mut self) -> Vec<Pair>;

    /// Whether every pair started has been graded (or failed: a pair whose
    /// infrastructure failed is never graded, and is not evidence).
    fn idle(&self) -> bool;
}

/// What the run is to do.
#[derive(Clone, Debug, PartialEq)]
pub enum Verdict {
    /// Go on.
    Continue,
    /// Admit no new group: an invariant is breached.
    Pause,
    /// Roll back to the anchor.
    Rollback(Box<Audit>),
}

/// Why a run rolled back: one record in the group log and the report.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Audit {
    /// What asked for the evaluation that decided: a drift alarm or the
    /// schedule.
    pub trigger: Value,
    /// The comparison that decided.
    pub evidence: Comparison,
    /// The version it rolls back from.
    pub candidate: Version,
    /// The version it rolls back to.
    pub anchor: Version,
    /// Every alarm of the epoch, in order.
    pub alarms: Vec<Value>,
}

/// An evaluation in progress.
#[derive(Debug)]
struct Pending {
    candidate: Version,
    comparison: Comparison,
    trigger: Value,
    /// It reads the sealed confirm set before a promotion.
    confirming: bool,
}

/// The watchdog of one epoch of a run.
pub struct Watchdog<'a> {
    settings: WatchdogSettings,
    evaluator: Box<dyn Evaluator + 'a>,
    pool: MonitorPool,
    confirm: Vec<String>,
    drift: Drift,
    anchor: Option<Version>,
    comparisons: u32,
    pending: Option<Pending>,
    last_evaluated: u64,
    observed: u64,
    alarms: Vec<Value>,
    trigger: Option<Value>,
    promotions: Vec<Version>,
    /// A version promoted whose values are to become the drift reference.
    promoted: Option<u64>,
    /// An invariant was breached and no evaluation has ruled on the policy
    /// since: intake holds.
    unacknowledged: bool,
}

impl<'a> Watchdog<'a> {
    /// The watchdog of an epoch whose anchor is `anchor` (the version it
    /// started from, once published: `None` until then), evaluating with
    /// `evaluator` on `monitor` tasks, with `confirm` tasks sealed for
    /// promotions; its `comparisons` index continues the run's.
    #[must_use]
    pub fn new(
        settings: WatchdogSettings,
        evaluator: Box<dyn Evaluator + 'a>,
        monitor: MonitorPool,
        confirm: Vec<String>,
        anchor: Option<Version>,
        comparisons: u32,
    ) -> Self {
        Self {
            drift: Drift::new(settings.drift),
            settings,
            evaluator,
            pool: monitor,
            confirm,
            anchor,
            comparisons,
            pending: None,
            last_evaluated: 0,
            observed: 0,
            alarms: Vec::new(),
            trigger: None,
            promotions: Vec::new(),
            promoted: None,
            unacknowledged: false,
        }
    }

    /// Comparisons the run has started, this epoch's included.
    #[must_use]
    pub fn comparisons(&self) -> u32 {
        self.comparisons
    }

    /// The anchor.
    #[must_use]
    pub fn anchor(&self) -> Option<&Version> {
        self.anchor.as_ref()
    }

    /// What the watchdog saw this epoch, for the run's report.
    #[must_use]
    pub fn report(&self) -> Value {
        json!({
            "anchor": self.anchor,
            "comparisons": self.comparisons,
            "promotions": self.promotions,
            "alarms": self.alarms,
            "drift": self.drift,
        })
    }

    /// Records an alarm; whether it is new.
    fn alarm(&mut self, kind: &str, detail: Value) -> bool {
        let alarm = json!({"alarm": kind, "detail": detail});
        let new = !self.alarms.contains(&alarm);
        if new {
            self.alarms.push(alarm);
        }
        new
    }

    /// The signals' values of `version`.
    fn values(
        version: u64,
        health: &VersionHealth,
        steps: &[StepHealth],
    ) -> Vec<(Signal, Option<f64>)> {
        let entropies: Vec<f64> = steps
            .iter()
            .filter(|s| s.version == version)
            .filter_map(|s| s.entropy)
            .collect();
        #[allow(clippy::cast_precision_loss)]
        let entropy =
            (!entropies.is_empty()).then(|| entropies.iter().sum::<f64>() / entropies.len() as f64);
        Signal::ALL
            .into_iter()
            .map(|s| (s, s.value(health, entropy)))
            .collect()
    }

    /// The run's verdict after a turn, from the learner's telemetry and the
    /// rollouts' per-version series.
    pub fn verdict(&mut self, telemetry: &LearnerTelemetry, series: &VersionSeries) -> Verdict {
        let Some(latest) = telemetry.published.last() else {
            return Verdict::Continue;
        };
        let latest = Version {
            version: latest.version,
            digest: latest.digest.clone(),
        };
        if self.anchor.is_none() {
            self.anchor = Some(latest.clone());
            self.last_evaluated = latest.version;
        }
        // Layer 1: invariants.
        let bounds = self.settings.bounds;
        let window = usize::try_from(bounds.window).unwrap_or(usize::MAX);
        let recent = &telemetry.step_health[telemetry.step_health.len().saturating_sub(window)..];
        let mut breaches = bounds.steps(recent, telemetry.skipped_steps.len());
        if let Some((version, health)) = series.versions.iter().next_back() {
            breaches.extend(bounds.version(*version, health));
        }
        for breach in &breaches {
            if self.alarm("invariant", json!(breach)) {
                // Intake holds until an evaluation rules on the policy.
                self.unacknowledged = true;
                self.trigger.get_or_insert(json!({"invariant": breach}));
            }
        }
        // Layer 2: drift, once per version with enough rollouts read, from
        // the promoted versions' reference.
        if let Some(version) = self.promoted.take() {
            match series.versions.get(&version) {
                Some(health) => {
                    self.drift
                        .promoted(&Self::values(version, health, &telemetry.step_health))
                }
                None => self.promoted = Some(version),
            }
        }
        for (version, health) in &series.versions {
            if *version <= self.observed || health.rollouts < u64::from(bounds.min_rollouts) {
                continue;
            }
            self.observed = *version;
            let values = Self::values(*version, health, &telemetry.step_health);
            if self.anchor.as_ref().is_some_and(|a| a.version == *version) {
                self.drift.promoted(&values);
                continue;
            }
            let alarms: Vec<DriftAlarm> = self.drift.observe(*version, &values);
            for alarm in alarms {
                let detail = json!(alarm);
                self.alarm("drift", detail.clone());
                self.trigger.get_or_insert(json!({"drift": detail}));
            }
        }
        // Layer 3: the paired evaluation decides.
        if let Some(verdict) = self.evaluate(&latest) {
            return verdict;
        }
        // A breach with no candidate to evaluate (the anchor is the newest
        // version) is recorded and lets intake go on: holding it would hold
        // the training that could publish one.
        if self.pending.is_none()
            && self
                .anchor
                .as_ref()
                .is_some_and(|a| a.version == latest.version)
        {
            self.unacknowledged = false;
            self.trigger = None;
        }
        if self.unacknowledged {
            Verdict::Pause
        } else {
            Verdict::Continue
        }
    }

    fn evaluate(&mut self, latest: &Version) -> Option<Verdict> {
        let anchor = self.anchor.clone()?;
        if self.pending.is_none() && latest.version != anchor.version {
            let due =
                latest.version >= self.last_evaluated + u64::from(self.settings.evaluate_every);
            if due || self.trigger.is_some() {
                let tasks = self
                    .pool
                    .draw(usize::try_from(self.settings.monitor_size).unwrap_or(usize::MAX));
                if !tasks.is_empty() {
                    let trigger = self
                        .trigger
                        .take()
                        .unwrap_or_else(|| json!({"schedule": latest.version}));
                    self.evaluator.start(latest, &anchor, tasks);
                    self.pending = Some(Pending {
                        candidate: latest.clone(),
                        comparison: Comparison::new(self.settings.test, self.comparisons),
                        trigger,
                        confirming: false,
                    });
                    self.comparisons += 1;
                    self.last_evaluated = latest.version;
                }
            }
        }
        let pending = self.pending.as_mut()?;
        for pair in self.evaluator.poll() {
            pending.comparison.read(pair.candidate, pair.anchor);
        }
        if pending.comparison.decision == Decision::Undecided && self.evaluator.idle() {
            pending.comparison.close();
        }
        match (pending.comparison.decision, pending.confirming) {
            (Decision::Undecided, _) => None,
            (Decision::Regressed, _) => {
                let pending = self.pending.take()?;
                Some(Verdict::Rollback(Box::new(Audit {
                    trigger: pending.trigger,
                    evidence: pending.comparison,
                    candidate: pending.candidate,
                    anchor,
                    alarms: self.alarms.clone(),
                })))
            }
            (Decision::NonInferior, false) => {
                // Before a promotion the sealed confirm set is read once.
                let candidate = pending.candidate.clone();
                let trigger = pending.trigger.clone();
                self.evaluator
                    .start(&candidate, &anchor, self.confirm.clone());
                self.pending = Some(Pending {
                    candidate,
                    comparison: Comparison::new(self.settings.test, self.comparisons),
                    trigger,
                    confirming: true,
                });
                self.comparisons += 1;
                None
            }
            (Decision::NonInferior | Decision::Inconclusive, true) => {
                // The confirm set showed no regression: the candidate is the anchor.
                self.unacknowledged = false;
                let pending = self.pending.take()?;
                self.anchor = Some(pending.candidate.clone());
                self.promoted = Some(pending.candidate.version);
                self.promotions.push(pending.candidate);
                None
            }
            (Decision::Inconclusive, false) => {
                self.unacknowledged = false;
                self.pending = None;
                None
            }
        }
    }
}

#[cfg(test)]
mod tests;
