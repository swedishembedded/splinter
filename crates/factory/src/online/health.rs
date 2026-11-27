// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements online reinforcement-learning runs whose
// training health is measured while they run, for its clients. If your team
// needs expertise in observable asynchronous policy optimisation, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The training health of an online run, typed: what the learner reports of
//! each optimiser step ([`StepHealth`], read from brain's step frames) and
//! what the rollouts each policy version decoded came to
//! ([`VersionHealth`]). A figure that was not measured is `None`, never zero.
//!
//! Per version, every signal is defined from the records alone, the same
//! for every task and verifier: a rollout belongs to the oldest version
//! among its requests; a group to the oldest version among its rollouts.
//!
//! * **zero-variance share** - scored groups whose rewards do not differ,
//!   over scored groups with a reward;
//! * **truncation rate** - finished rollouts a limit ended, over finished
//!   rollouts;
//! * **infrastructure failure rate** - rollout attempts the infrastructure
//!   failed, over attempts (finished, failed);
//! * **false-completion rate** - scored rollouts whose agent ended the run
//!   itself, no limit ending it, that the verifier failed, over scored
//!   rollouts whose behaviour was read;
//! * **repeated-call rate** - tool calls identical (tool and arguments) to
//!   an earlier call of the same rollout, over tool calls.

use std::collections::BTreeMap;

use serde::Serialize;
use serde_json::Value;
use splinter_core::factory::group::RolloutOutcome;
use splinter_core::rewards::differ;

/// One optimiser step as the learner reported it.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct StepHealth {
    /// Its place among the learner's steps, from 1.
    pub step: u64,
    /// The learner's published version when it was reported.
    pub version: u64,
    /// The KL estimate to the behaviour policy.
    pub kl: Option<f64>,
    /// The policy's entropy estimate.
    pub entropy: Option<f64>,
    /// Tokens whose ratio the clip cut.
    pub clip_fraction: Option<f64>,
    /// Of those, cut above.
    pub clip_fraction_high: Option<f64>,
    /// Of those, cut below.
    pub clip_fraction_low: Option<f64>,
    /// The smallest importance ratio.
    pub ratio_min: Option<f64>,
    /// The largest importance ratio.
    pub ratio_max: Option<f64>,
    /// The mean importance ratio.
    pub ratio_mean: Option<f64>,
    /// The gradient's norm.
    pub grad_norm: Option<f64>,
    /// The loss.
    pub loss: Option<f64>,
    /// The mean advantage.
    pub mean_advantage: Option<f64>,
    /// The mean staleness of what it trained, in versions.
    pub staleness_mean: Option<f64>,
    /// The largest staleness it trained.
    pub staleness_max: Option<f64>,
    /// Policy tokens it trained.
    pub policy_tokens: Option<u64>,
    /// How long the learner waited for groups before it.
    pub learner_wait_seconds: Option<f64>,
    /// Groups queued.
    pub queue_groups: Option<f64>,
    /// Tokens queued.
    pub queue_tokens: Option<f64>,
    /// How long its groups waited in the queue.
    pub time_in_queue_seconds: Option<f64>,
}

/// Whether every figure of a step is a finite number (an unreported figure
/// is not one that failed).
fn finite(value: Option<f64>) -> bool {
    value.is_none_or(f64::is_finite)
}

impl StepHealth {
    /// The step `pg` reports (brain's frame `pg` object), the `step`-th of
    /// the learner's, while it serves `version`.
    #[must_use]
    pub fn from_pg(pg: &Value, step: u64, version: u64) -> Self {
        let f = |key: &str| pg.get(key).and_then(Value::as_f64);
        Self {
            step,
            version,
            kl: f("kl"),
            entropy: f("entropy_estimate"),
            clip_fraction: f("clip_fraction"),
            clip_fraction_high: f("clip_fraction_high"),
            clip_fraction_low: f("clip_fraction_low"),
            ratio_min: f("ratio_min"),
            ratio_max: f("ratio_max"),
            ratio_mean: f("ratio_mean"),
            grad_norm: f("grad_norm"),
            loss: f("loss"),
            mean_advantage: f("mean_advantage"),
            staleness_mean: f("staleness_mean"),
            staleness_max: f("staleness_max"),
            policy_tokens: pg.get("policy_tokens").and_then(|v| {
                v.as_u64().or_else(|| {
                    let x = v.as_f64()?;
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    (x >= 0.0 && x.fract() == 0.0).then_some(x as u64)
                })
            }),
            learner_wait_seconds: f("learner_wait_seconds"),
            queue_groups: f("queue_groups"),
            queue_tokens: f("queue_tokens"),
            time_in_queue_seconds: f("time_in_queue_seconds"),
        }
    }

    /// Whether every figure it reported is finite.
    #[must_use]
    pub fn finite(&self) -> bool {
        [
            self.kl,
            self.entropy,
            self.clip_fraction,
            self.ratio_min,
            self.ratio_max,
            self.ratio_mean,
            self.grad_norm,
            self.loss,
        ]
        .into_iter()
        .all(finite)
    }
}

/// What the rollouts one policy version decoded came to.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct VersionHealth {
    /// Rollouts that finished.
    pub rollouts: u64,
    /// Rollout attempts the infrastructure failed.
    pub infra_failures: u64,
    /// Finished rollouts a limit ended.
    pub truncated: u64,
    /// Scored rollouts (with a reward).
    pub scored: u64,
    /// Their rewards' sum.
    pub reward_sum: f64,
    /// Scored rollouts whose behaviour was read.
    pub behaviour_read: u64,
    /// Of those, false completions.
    pub false_completions: u64,
    /// Tool calls made.
    pub tool_calls: u64,
    /// Of those, repeats within their rollout.
    pub repeated_calls: u64,
    /// Scored groups with a reward.
    pub groups: u64,
    /// Of those, groups whose rewards do not differ.
    pub zero_variance_groups: u64,
}

#[allow(clippy::cast_precision_loss)]
fn rate(part: u64, whole: u64) -> Option<f64> {
    (whole > 0).then(|| part as f64 / whole as f64)
}

impl VersionHealth {
    /// The mean reward of its scored rollouts.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn reward_mean(&self) -> Option<f64> {
        (self.scored > 0).then(|| self.reward_sum / self.scored as f64)
    }

    /// The share of its groups whose rewards do not differ.
    #[must_use]
    pub fn zero_variance_share(&self) -> Option<f64> {
        rate(self.zero_variance_groups, self.groups)
    }

    /// The share of its finished rollouts a limit ended.
    #[must_use]
    pub fn truncation_rate(&self) -> Option<f64> {
        rate(self.truncated, self.rollouts)
    }

    /// The share of its rollout attempts the infrastructure failed.
    #[must_use]
    pub fn infra_failure_rate(&self) -> Option<f64> {
        rate(self.infra_failures, self.rollouts + self.infra_failures)
    }

    /// The share of its scored rollouts that were false completions.
    #[must_use]
    pub fn false_completion_rate(&self) -> Option<f64> {
        rate(self.false_completions, self.behaviour_read)
    }

    /// The share of its tool calls that repeat an earlier call of their
    /// rollout.
    #[must_use]
    pub fn repeated_call_rate(&self) -> Option<f64> {
        rate(self.repeated_calls, self.tool_calls)
    }

    /// Its rates and counts, as a report carries them.
    #[must_use]
    pub fn summary(&self) -> Value {
        serde_json::json!({
            "rollouts": self.rollouts,
            "groups": self.groups,
            "reward_mean": self.reward_mean(),
            "zero_variance_share": self.zero_variance_share(),
            "truncation_rate": self.truncation_rate(),
            "infra_failure_rate": self.infra_failure_rate(),
            "false_completion_rate": self.false_completion_rate(),
            "repeated_call_rate": self.repeated_call_rate(),
        })
    }

    fn add_rollout(&mut self, outcome: &RolloutOutcome) {
        self.rollouts += 1;
        self.truncated += u64::from(outcome.truncation.is_some());
        if let Some(reward) = outcome.reward {
            self.scored += 1;
            self.reward_sum += reward;
        }
        if let Some(false_completion) = outcome.false_completion() {
            self.behaviour_read += 1;
            self.false_completions += u64::from(false_completion);
        }
        if let Some(behaviour) = outcome.behaviour {
            self.tool_calls += u64::from(behaviour.tool_calls);
            self.repeated_calls += u64::from(behaviour.repeated_calls);
        }
    }
}

/// The per-version series of an online run, fed one scored group at a time.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct VersionSeries {
    /// Each version's health, by version.
    pub versions: BTreeMap<u64, VersionHealth>,
}

impl VersionSeries {
    /// A scored group: its finished rollouts' outcomes and how many of its
    /// rollout attempts the infrastructure failed.
    pub fn add_group(&mut self, outcomes: &[&RolloutOutcome], infra_failures: u64) {
        let Some(version) = outcomes.iter().map(|o| o.span.oldest()).min() else {
            return;
        };
        let health = self.versions.entry(version).or_default();
        health.infra_failures += infra_failures;
        for outcome in outcomes {
            health.add_rollout(outcome);
        }
        let rewards: Vec<f64> = outcomes.iter().filter_map(|o| o.reward).collect();
        if !rewards.is_empty() {
            health.groups += 1;
            health.zero_variance_groups += u64::from(!differ(rewards.iter().copied()));
        }
    }

    /// Every version's summary, by version.
    #[must_use]
    pub fn summary(&self) -> BTreeMap<u64, Value> {
        self.versions
            .iter()
            .map(|(v, h)| (*v, h.summary()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use splinter_core::factory::group::{Behaviour, Truncation, VersionSpan};

    #[test]
    fn a_step_frame_is_read_into_typed_figures_and_an_absent_one_is_none() {
        let pg = json!({"kl": 0.01, "entropy_estimate": 1.5, "ratio_max": 3.0, "policy_tokens": 640.0,
                        "grad_norm": f64::MAX});
        let step = StepHealth::from_pg(&pg, 3, 2);
        assert_eq!((step.step, step.version), (3, 2));
        assert_eq!(step.kl, Some(0.01));
        assert_eq!(step.entropy, Some(1.5));
        assert_eq!(step.policy_tokens, Some(640));
        assert_eq!(step.clip_fraction, None, "unreported is absent");
        assert!(step.finite());
    }

    fn outcome(
        version: u64,
        reward: f64,
        stopped: bool,
        calls: u32,
        repeats: u32,
    ) -> RolloutOutcome {
        RolloutOutcome {
            reward: Some(reward),
            tokens: 1,
            span: VersionSpan::at(version),
            exact: true,
            truncated: false,
            truncation: None,
            behaviour: Some(Behaviour {
                stopped,
                tool_calls: calls,
                repeated_calls: repeats,
            }),
            session: None,
        }
    }

    #[test]
    fn each_versions_signals_are_rates_over_what_it_decoded() {
        let mut series = VersionSeries::default();
        let a = outcome(1, 0.0, true, 4, 2);
        let b = outcome(1, 1.0, true, 2, 0);
        series.add_group(&[&a, &b], 1);
        let c = outcome(1, 0.0, false, 0, 0);
        let mut d = outcome(1, 0.0, true, 0, 0);
        d.truncation = Some(Truncation::Turns);
        series.add_group(&[&c, &d], 0);
        let e = outcome(2, 1.0, true, 1, 0);
        series.add_group(&[&e, &e], 0);
        let one = &series.versions[&1];
        assert_eq!(one.reward_mean(), Some(0.25));
        assert_eq!(one.zero_variance_share(), Some(0.5));
        assert_eq!(one.truncation_rate(), Some(0.25));
        assert_eq!(one.infra_failure_rate(), Some(0.2));
        assert_eq!(
            one.false_completion_rate(),
            Some(0.25),
            "a: ended itself and failed; c called nothing but did not end itself; d a limit ended"
        );
        assert_eq!(one.repeated_call_rate(), Some(2.0 / 6.0));
        assert_eq!(series.versions[&2].zero_variance_share(), Some(1.0));
        assert_eq!(series.versions[&2].false_completion_rate(), Some(0.0));
    }
}
