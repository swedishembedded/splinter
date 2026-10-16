// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measurements of what an agent really
// completed, and at what cost, for its clients. If your team needs expertise
// in benchmarking coding agents, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Competence: what an agent completes, at what cost, without a person.
//!
//! [`outcome_of`] derives a [`TaskOutcome`] from an experience, the
//! annotations about it and its attempt report. [`competence`] then reports,
//! over many attempts, three measures kept apart (there is no scalar):
//!
//! * quality `Q` = verified / attempted;
//! * cost `C` = total cost / verified, once per cost measure (seconds,
//!   tokens, tool calls), so the raw totals stay visible;
//! * autonomy `A` = verified without a person / attempted.
//!
//! An attempt that failed for the environment ([`FailureClass::Environment`])
//! says nothing about the agent: it leaves every denominator and is counted.
//! A cost that one attempt did not record has no total, and the attempts
//! missing it are counted, so an absent measurement never reads as zero.
//! Each estimate carries a percentile interval from a bootstrap that
//! resamples families, because attempts at tasks of one family are one unit
//! of evidence.

use std::collections::BTreeMap;

use atif::StepOrigin;
use serde::{Deserialize, Serialize};
use splinter_core::annotation::{Annotation, AnnotationBody};
use splinter_core::experience::{Experience, ExperienceError};
use splinter_core::outcome::{FailureClass, TaskOutcome};
use splinter_core::rng::SplitMix64;
use splinter_store::decision::decide;

/// Bootstrap resamples per interval.
const RESAMPLES: usize = 1000;

/// The share of resampled values an interval holds.
pub const INTERVAL_LEVEL: f64 = 0.95;

/// The outcome of `experience` under `notes`, the annotations about it.
///
/// Where the attempt report is silent the trajectory answers (token totals,
/// the count of requested tool calls); where neither does, the value is
/// absent. When `notes` hold several attempt reports the last one wins.
pub fn outcome_of(
    experience: &Experience,
    notes: &[Annotation],
) -> Result<TaskOutcome, ExperienceError> {
    let report = notes
        .iter()
        .rev()
        .find_map(|n| match &n.body {
            AnnotationBody::AttemptReport(report) => Some(report.clone()),
            _ => None,
        })
        .unwrap_or_default();
    let decision = decide(notes);
    let (prompt_tokens, completion_tokens) = token_totals(experience);
    Ok(TaskOutcome {
        task_id: experience.task.id.clone(),
        experience_id: experience.id()?,
        verified_success: decision.is_some_and(|d| d.passed),
        verification_strength: decision.map(|d| d.strength),
        submitted: report.submitted.unwrap_or_else(|| {
            experience
                .final_output
                .as_ref()
                .is_some_and(|o| !o.is_empty())
        }),
        wall_seconds: report.wall_seconds,
        gpu_seconds: report.gpu_seconds,
        prompt_tokens,
        completion_tokens,
        tool_calls: report
            .tool_calls
            .or_else(|| requested_tool_calls(experience)),
        tool_errors: report.tool_errors,
        attempted_actions: report.attempted_actions,
        unauthorised_actions: report.unauthorised_actions,
        failure: report.failure,
        human_minutes: report.human_minutes,
    })
}

/// The trajectory's token totals: its final metrics, else the sum of the
/// steps that report any; absent when nothing does.
fn token_totals(experience: &Experience) -> (Option<u64>, Option<u64>) {
    let trajectory = &experience.trajectory;
    let summed = |pick: fn(&atif::StepMetrics) -> Option<u64>| {
        trajectory
            .steps
            .iter()
            .filter_map(|s| s.metrics.as_ref().and_then(pick))
            .reduce(|a, b| a + b)
    };
    let totals = trajectory.final_metrics.as_ref();
    (
        totals
            .and_then(|m| m.total_prompt_tokens)
            .or_else(|| summed(|m| m.prompt_tokens)),
        totals
            .and_then(|m| m.total_completion_tokens)
            .or_else(|| summed(|m| m.completion_tokens)),
    )
}

/// Tool calls the model requested, when the trajectory holds an agent turn.
fn requested_tool_calls(experience: &Experience) -> Option<u32> {
    let mut agent_turns = experience
        .trajectory
        .steps
        .iter()
        .filter(|s| s.source == StepOrigin::Agent)
        .peekable();
    agent_turns.peek()?;
    let calls: usize = agent_turns
        .map(|s| s.tool_calls.as_ref().map_or(0, Vec::len))
        .sum();
    u32::try_from(calls).ok()
}

/// One attempt and the family of its task, the unit of resampling.
#[derive(Clone, Debug, PartialEq)]
pub struct Attempt {
    /// The family the task belongs to.
    pub family: String,
    /// What the attempt came to.
    pub outcome: TaskOutcome,
}

/// A measure with its interval. Each part is absent when it cannot be
/// computed (no attempts, nothing verified, a cost not recorded).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Estimate {
    /// The value over all attempts.
    pub value: Option<f64>,
    /// The lower end of the interval.
    pub low: Option<f64>,
    /// The upper end of the interval.
    pub high: Option<f64>,
}

/// The raw counts the measures rest on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Counts {
    /// Attempts counted (environment failures left out).
    pub attempted: usize,
    /// Of those, the ones verified.
    pub verified: usize,
    /// Of those, the ones verified with no person in the loop.
    pub verified_without_human: usize,
    /// Attempts left out because the environment failed.
    pub excluded_environment: usize,
    /// Families among the attempts counted.
    pub families: usize,
}

/// One cost measure over the attempts counted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CostMeasure {
    /// The total over the attempts counted; `None` when any of them did not
    /// record it, or none was counted.
    pub total: Option<f64>,
    /// Attempts counted that did not record it.
    pub missing: usize,
    /// The total over the number verified.
    pub per_verified: Estimate,
}

/// The cost measures, each kept raw.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Costs {
    /// Wall-clock seconds.
    pub wall_seconds: CostMeasure,
    /// Device-busy seconds.
    pub gpu_seconds: CostMeasure,
    /// Prompt tokens.
    pub prompt_tokens: CostMeasure,
    /// Completion tokens.
    pub completion_tokens: CostMeasure,
    /// Tool calls requested.
    pub tool_calls: CostMeasure,
}

/// Quality, cost and autonomy over a set of attempts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Competence {
    /// What the measures rest on.
    pub counts: Counts,
    /// `Q`: verified / attempted.
    pub quality: Estimate,
    /// `C`: cost per verified, by measure.
    pub cost: Costs,
    /// `A`: verified without a person / attempted.
    pub autonomy: Estimate,
}

type Pick = fn(&TaskOutcome) -> Option<f64>;

const PICKS: [Pick; 5] = [
    |o| o.wall_seconds,
    |o| o.gpu_seconds,
    |o| o.prompt_tokens.map(|t| t as f64),
    |o| o.completion_tokens.map(|t| t as f64),
    |o| o.tool_calls.map(f64::from),
];

fn without_human(o: &TaskOutcome) -> bool {
    o.verified_success && o.human_minutes.is_none_or(|m| m == 0.0)
}

fn counted(o: &TaskOutcome) -> bool {
    o.failure != Some(FailureClass::Environment)
}

/// The shares and per-verified costs over `pool`, the outcomes counted.
fn measures(pool: &[&TaskOutcome]) -> ([Option<f64>; 2], [Option<f64>; 5]) {
    let attempted = pool.len() as f64;
    let verified = pool.iter().filter(|o| o.verified_success).count() as f64;
    let human_free = pool.iter().filter(|o| without_human(o)).count() as f64;
    let share = |n: f64| (attempted > 0.0).then(|| n / attempted);
    let per_verified = PICKS.map(|pick| {
        let costs: Option<Vec<f64>> = pool.iter().map(|o| pick(o)).collect();
        costs
            .filter(|_| verified > 0.0)
            .map(|c| c.iter().sum::<f64>() / verified)
    });
    ([share(verified), share(human_free)], per_verified)
}

/// Quality, cost and autonomy over `attempts`, with intervals resampled
/// from `seed` (the same attempts and seed give the same report).
#[must_use]
pub fn competence(attempts: &[Attempt], seed: u64) -> Competence {
    let excluded = attempts.iter().filter(|a| !counted(&a.outcome)).count();
    let mut families: BTreeMap<&str, Vec<&TaskOutcome>> = BTreeMap::new();
    for attempt in attempts.iter().filter(|a| counted(&a.outcome)) {
        families
            .entry(attempt.family.as_str())
            .or_default()
            .push(&attempt.outcome);
    }
    let groups: Vec<Vec<&TaskOutcome>> = families.into_values().collect();
    let pool: Vec<&TaskOutcome> = groups.iter().flatten().copied().collect();
    let (shares, per_verified) = measures(&pool);

    let mut rng = SplitMix64::new(seed);
    let mut share_draws: [Vec<f64>; 2] = Default::default();
    let mut cost_draws: [Vec<f64>; 5] = Default::default();
    if !groups.is_empty() {
        for _ in 0..RESAMPLES {
            let drawn: Vec<&TaskOutcome> = (0..groups.len())
                .flat_map(|_| groups[rng.below(groups.len())].iter().copied())
                .collect();
            let (s, c) = measures(&drawn);
            for (draws, value) in share_draws.iter_mut().zip(s) {
                draws.extend(value);
            }
            for (draws, value) in cost_draws.iter_mut().zip(c) {
                draws.extend(value);
            }
        }
    }
    let estimate = |value: Option<f64>, mut draws: Vec<f64>| {
        let (low, high) = interval(value, &mut draws);
        Estimate { value, low, high }
    };
    let mut measure = |at: usize| {
        let costs: Vec<Option<f64>> = pool.iter().map(|o| PICKS[at](o)).collect();
        CostMeasure {
            total: costs
                .iter()
                .copied()
                .sum::<Option<f64>>()
                .filter(|_| !pool.is_empty()),
            missing: costs.iter().filter(|c| c.is_none()).count(),
            per_verified: estimate(per_verified[at], std::mem::take(&mut cost_draws[at])),
        }
    };
    Competence {
        counts: Counts {
            attempted: pool.len(),
            verified: pool.iter().filter(|o| o.verified_success).count(),
            verified_without_human: pool.iter().filter(|o| without_human(o)).count(),
            excluded_environment: excluded,
            families: groups.len(),
        },
        quality: estimate(shares[0], std::mem::take(&mut share_draws[0])),
        autonomy: estimate(shares[1], std::mem::take(&mut share_draws[1])),
        cost: Costs {
            wall_seconds: measure(0),
            gpu_seconds: measure(1),
            prompt_tokens: measure(2),
            completion_tokens: measure(3),
            tool_calls: measure(4),
        },
    }
}

/// The percentile interval of `draws`, widened to hold `value`; absent when
/// there is no value or no draw.
fn interval(value: Option<f64>, draws: &mut [f64]) -> (Option<f64>, Option<f64>) {
    let Some(value) = value else {
        return (None, None);
    };
    if draws.is_empty() {
        return (None, None);
    }
    draws.sort_by(f64::total_cmp);
    let tail = (1.0 - INTERVAL_LEVEL) / 2.0;
    let at = |q: f64| draws[((q * (draws.len() - 1) as f64).round() as usize).min(draws.len() - 1)];
    (Some(at(tail).min(value)), Some(at(1.0 - tail).max(value)))
}
