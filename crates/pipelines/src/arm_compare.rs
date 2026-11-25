// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements paired comparisons of agent models whose
// statistics are proven against exact references, for its clients. If your
// team needs expertise in telling a real gain of a fine-tuned agent from the
// luck of a small exam, you can procure our services by sending an email to
// info@swedishembedded.com.

//! `experiences compare-arms`: graded trials of several arms on the same
//! tasks, each arm against a reference arm.
//!
//! A trial is one attempt of one arm at one task, read from a Harbor job
//! directory (one `result.json` per trial) or from a JSONL file of
//! `{"task", "family"?, "arm", "attempt"?, "pass"? | "reward"?, "status"?}`
//! lines. Its status is one of:
//!
//! * **graded**: the verifier passed it (reward exactly 1, or `pass`) or
//!   failed it;
//! * **timeout**: the agent ran out of time and the verifier still graded
//!   what it left; it is graded (a failure unless it passed) and also counted
//!   apart;
//! * **infrastructure error**: no grade (a Harbor exception other than an
//!   agent timeout, or no reward); it is in no denominator and counted.
//!
//! Per arm: pass@1 (the mean over tasks of the task's pass rate) and pass@k
//! for each k asked, by the unbiased estimator `1 - C(n - c, k) / C(n, k)`
//! averaged over the tasks with at least k graded attempts (and how many
//! those were). Per arm against the reference, over the tasks graded in both:
//! the mean of the per-task differences in pass rate, its percentile
//! bootstrap interval over tasks, the interval resampling whole families
//! when tasks name one, and the exact sign test over tasks (a task counts as
//! better, worse or tied by its rate difference; ties dropped), one-sided
//! (the arm is better) and two-sided (twice the smaller tail, at most one),
//! each Holm-adjusted across the arms compared.
//!
//! Every number is computed by the release statistics
//! ([`splinter_model::stats`]) and the family bootstrap of the powered exam;
//! this module only counts and pairs.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use splinter_core::rng::Seed;
use splinter_model::stats::{bootstrap_interval, holm, one_sided_p, pass_at_k, two_sided_p};
use splinter_orchestrator::error::OrchestratorError;

use crate::powered::analysis::Resampled;

/// The format tag of an [`ArmComparison`].
pub const ARM_COMPARISON_FORMAT: &str = "splinter.arm-comparison.v1";

/// Resamples an interval is made from when the request does not say.
pub const DEFAULT_RESAMPLES: usize = 10_000;

/// The share of resampled means an interval holds when the request does not
/// say.
pub const DEFAULT_LEVEL: f64 = 0.95;

/// The Harbor exception of a trial the agent's time limit stopped.
const AGENT_TIMEOUT: &str = "AgentTimeoutError";

/// What became of one trial.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    /// Graded by the verifier.
    Graded,
    /// Stopped by the agent's time limit, then graded.
    Timeout,
    /// Not graded: the infrastructure failed.
    InfraError,
}

/// One trial.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Trial {
    /// The arm.
    pub arm: String,
    /// The task.
    pub task: String,
    /// The task's family, when it names one.
    pub family: Option<String>,
    /// What became of it.
    pub status: Status,
    /// Whether the verifier passed it; false when it was not graded.
    pub passed: bool,
}

fn refused(reason: impl Into<String>) -> OrchestratorError {
    OrchestratorError::Refused(reason.into())
}

/// A JSONL line of a trials file.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Line {
    task: String,
    arm: String,
    #[serde(default)]
    family: Option<String>,
    /// Which attempt of the task it was: accepted for the record, not used
    /// (attempts of one task are exchangeable).
    #[serde(default, rename = "attempt")]
    _attempt: Option<Value>,
    #[serde(default)]
    pass: Option<bool>,
    #[serde(default)]
    reward: Option<f64>,
    #[serde(default)]
    status: Option<Status>,
}

/// The trials of a JSONL file. A line that is graded or a timeout must say
/// whether it passed (`pass`, or a `reward` in `[0, 1]` of which exactly 1
/// passes); a line names no arm or task it cannot be paired by.
///
/// # Errors
/// The file cannot be read, or a line is malformed, naming the line.
pub fn read_trials_jsonl(path: &Path) -> Result<Vec<Trial>, OrchestratorError> {
    let text =
        std::fs::read_to_string(path).map_err(|e| refused(format!("{}: {e}", path.display())))?;
    let mut trials = Vec::new();
    for (at, raw) in text
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
    {
        let place = || format!("{} line {}", path.display(), at + 1);
        let line: Line =
            serde_json::from_str(raw).map_err(|e| refused(format!("{}: {e}", place())))?;
        let status = line.status.unwrap_or(Status::Graded);
        let passed = match (line.pass, line.reward) {
            (_, _) if status == Status::InfraError => false,
            (Some(pass), None) => pass,
            (None, Some(reward)) if (0.0..=1.0).contains(&reward) => {
                (reward - 1.0).abs() < f64::EPSILON
            }
            (None, Some(reward)) => {
                return Err(refused(format!(
                    "{}: reward {reward} is not in [0, 1]",
                    place()
                )))
            }
            (Some(_), Some(_)) => {
                return Err(refused(format!(
                    "{}: give pass or reward, not both",
                    place()
                )))
            }
            (None, None) => {
                return Err(refused(format!(
                    "{}: a graded trial says whether it passed (pass or reward)",
                    place()
                )))
            }
        };
        if line.task.is_empty() || line.arm.is_empty() {
            return Err(refused(format!(
                "{}: the task and the arm are named",
                place()
            )));
        }
        trials.push(Trial {
            arm: line.arm,
            task: line.task,
            family: line.family,
            status,
            passed,
        });
    }
    Ok(trials)
}

/// The trials of a Harbor job directory as `arm`: each subdirectory holding a
/// `result.json` is one trial of the task its `task_name` names. A reward of
/// exactly 1 passes; an agent timeout with a reward is a graded timeout; any
/// other exception, or no reward, is an infrastructure error.
///
/// # Errors
/// The directory cannot be read, a result cannot be parsed, or the job holds
/// no trial.
pub fn read_harbor_job(arm: &str, job: &Path) -> Result<Vec<Trial>, OrchestratorError> {
    let unreadable = |e: std::io::Error| refused(format!("{}: {e}", job.display()));
    let mut dirs: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(job).map_err(unreadable)? {
        let path = entry.map_err(unreadable)?.path();
        if path.join("result.json").is_file() {
            dirs.push(path);
        }
    }
    dirs.sort();
    let mut trials = Vec::new();
    for dir in dirs {
        let path = dir.join("result.json");
        let text = std::fs::read_to_string(&path)
            .map_err(|e| refused(format!("{}: {e}", path.display())))?;
        let result: Value =
            serde_json::from_str(&text).map_err(|e| refused(format!("{}: {e}", path.display())))?;
        let task = result
            .get("task_name")
            .and_then(Value::as_str)
            .ok_or_else(|| refused(format!("{}: no task_name", path.display())))?;
        if is_human_guided(&result) {
            return Err(refused(format!(
                "{}: a human-guided trial (a person typed into it): its pass is not the agent's, \
                 so it belongs to no arm",
                path.display()
            )));
        }
        let exception = result
            .pointer("/exception_info/exception_type")
            .and_then(Value::as_str);
        let reward = result
            .pointer("/verifier_result/rewards/reward")
            .and_then(Value::as_f64);
        let status = match (exception, reward) {
            (_, None) => Status::InfraError,
            (None, Some(_)) => Status::Graded,
            (Some(AGENT_TIMEOUT), Some(_)) => Status::Timeout,
            (Some(_), Some(_)) => Status::InfraError,
        };
        trials.push(Trial {
            arm: arm.to_string(),
            task: task.to_string(),
            family: None,
            status,
            passed: status != Status::InfraError
                && reward.is_some_and(|r| (r - 1.0).abs() < f64::EPSILON),
        });
    }
    if trials.is_empty() {
        return Err(refused(format!(
            "{} holds no trial (no subdirectory with a result.json)",
            job.display()
        )));
    }
    Ok(trials)
}

/// Whether a Harbor trial's `result.json` is of a human-guided trial: bench's
/// play mode runs the agent with `human_guided` set, and a person typed into it.
#[must_use]
pub fn is_human_guided(result: &Value) -> bool {
    result
        .pointer("/config/agent/kwargs/human_guided")
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

/// What a comparison is asked.
#[derive(Clone, Debug)]
pub struct CompareRequest {
    /// The arm every other arm is compared with.
    pub reference: String,
    /// The k of each pass@k reported (pass@1 is always reported).
    pub k: Vec<usize>,
    /// Resamples per interval.
    pub resamples: usize,
    /// The share of resampled means an interval holds, in `(0, 1)`.
    pub level: f64,
    /// The seed every interval is resampled from.
    pub seed: Seed,
    /// Each task's family, beside any a trial names; a task named in both
    /// must agree.
    pub families: BTreeMap<String, String>,
}

/// pass@k of an arm.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PassAtK {
    /// k.
    pub k: usize,
    /// The mean unbiased estimate over the tasks with at least k graded
    /// attempts; `None` when no task has.
    pub value: Option<f64>,
    /// The tasks it is the mean over.
    pub tasks: usize,
}

/// One arm over its own trials.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArmSummary {
    /// The arm.
    pub arm: String,
    /// Trials read.
    pub trials: usize,
    /// Of those, graded (timeouts included).
    pub graded: usize,
    /// Of the graded, passed.
    pub passed: usize,
    /// Graded trials the agent's time limit stopped.
    pub timeouts: usize,
    /// Of those, passed.
    pub timeouts_passed: usize,
    /// Trials the infrastructure stopped: in no denominator.
    pub infra_errors: usize,
    /// Tasks with a graded trial.
    pub tasks: usize,
    /// pass@1, then each k asked.
    pub pass_at_k: Vec<PassAtK>,
}

/// An interval of a mean difference.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Bounds {
    /// Its lower end.
    pub low: f64,
    /// Its upper end.
    pub high: f64,
}

/// One arm against the reference over the tasks graded in both.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Contrast {
    /// The arm.
    pub arm: String,
    /// The reference arm.
    pub reference: String,
    /// Tasks graded in both.
    pub paired_tasks: usize,
    /// The arm's pass@1 over those tasks.
    pub arm_pass_at_1: Option<f64>,
    /// The reference's pass@1 over those tasks.
    pub reference_pass_at_1: Option<f64>,
    /// The mean over those tasks of the arm's pass rate less the reference's.
    pub difference: Option<f64>,
    /// Its percentile bootstrap interval over tasks; `None` with fewer than
    /// two tasks.
    pub task_interval: Option<Bounds>,
    /// Its interval resampling whole families (every task in one); `None`
    /// unless every paired task names a family and there are two or more.
    pub family_interval: Option<Bounds>,
    /// Families the paired tasks fall in; `None` when not every one names one.
    pub families: Option<usize>,
    /// Tasks the arm did better on.
    pub better: usize,
    /// Tasks it did worse on.
    pub worse: usize,
    /// Tasks it tied on: dropped from the sign test.
    pub ties: usize,
    /// The exact one-sided sign test that the arm is better.
    pub p_one_sided: f64,
    /// The exact two-sided sign test: twice the smaller tail, at most one.
    pub p_two_sided: f64,
    /// `p_one_sided` Holm-adjusted across the arms compared.
    pub p_one_sided_holm: f64,
    /// `p_two_sided` Holm-adjusted across the arms compared.
    pub p_two_sided_holm: f64,
}

/// What `experiences compare-arms` reports.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArmComparison {
    /// Always [`ARM_COMPARISON_FORMAT`].
    pub format: String,
    /// The reference arm.
    pub reference: String,
    /// The seed the intervals were resampled from.
    pub seed: Seed,
    /// Resamples per interval.
    pub resamples: usize,
    /// The share of resampled means an interval holds.
    pub level: f64,
    /// How the sign test is read.
    pub sign_test: String,
    /// Each arm, in name order.
    pub arms: Vec<ArmSummary>,
    /// Each arm but the reference against it, in name order.
    pub contrasts: Vec<Contrast>,
}

/// The sign test's convention, as the report states it.
pub const SIGN_TEST_CONVENTION: &str = "exact binomial over tasks whose pass rates differ (ties \
dropped); one-sided: P(X >= better), X ~ Binomial(better + worse, 1/2); two-sided: \
min(1, 2 * min(P(X >= better), P(X <= better))); Holm across the arms compared";

/// Per arm, per task: the graded verdicts.
type Graded = BTreeMap<String, BTreeMap<String, Vec<bool>>>;

fn rate(verdicts: &[bool]) -> f64 {
    verdicts.iter().filter(|p| **p).count() as f64 / verdicts.len() as f64
}

fn mean(values: impl Iterator<Item = f64>) -> Option<f64> {
    let (sum, n) = values.fold((0.0, 0usize), |(s, n), v| (s + v, n + 1));
    (n > 0).then(|| sum / n as f64)
}

fn summary(
    arm: &str,
    trials: &[&Trial],
    graded: &BTreeMap<String, Vec<bool>>,
    k: &[usize],
) -> ArmSummary {
    let count = |f: &dyn Fn(&Trial) -> bool| trials.iter().filter(|t| f(t)).count();
    let pass_at = |k: usize| {
        let estimates: Vec<f64> = graded
            .values()
            .filter_map(|v| pass_at_k(v.len(), v.iter().filter(|p| **p).count(), k))
            .collect();
        PassAtK {
            k,
            value: mean(estimates.iter().copied()),
            tasks: estimates.len(),
        }
    };
    ArmSummary {
        arm: arm.to_string(),
        trials: trials.len(),
        graded: count(&|t| t.status != Status::InfraError),
        passed: count(&|t| t.status != Status::InfraError && t.passed),
        timeouts: count(&|t| t.status == Status::Timeout),
        timeouts_passed: count(&|t| t.status == Status::Timeout && t.passed),
        infra_errors: count(&|t| t.status == Status::InfraError),
        tasks: graded.len(),
        pass_at_k: std::iter::once(1)
            .chain(k.iter().copied().filter(|k| *k != 1))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .map(pass_at)
            .collect(),
    }
}

/// `arm` against `reference` over the tasks graded in both, before Holm.
fn contrast(
    arm: &str,
    reference: &str,
    graded: &Graded,
    families: &BTreeMap<String, String>,
    request: &CompareRequest,
) -> Contrast {
    let (ours, theirs) = (&graded[arm], &graded[reference]);
    let paired: Vec<(&String, f64, f64)> = ours
        .iter()
        .filter_map(|(task, v)| theirs.get(task).map(|w| (task, rate(v), rate(w))))
        .collect();
    let differences: Vec<f64> = paired.iter().map(|(_, a, b)| a - b).collect();
    let better = differences.iter().filter(|d| **d > 0.0).count();
    let worse = differences.iter().filter(|d| **d < 0.0).count();
    let task_interval = bootstrap_interval(
        &differences,
        request.resamples,
        request.level,
        request.seed.value(),
    )
    .map(|i| Bounds {
        low: i.low,
        high: i.high,
    });
    let named: Option<Vec<(&str, f64)>> = paired
        .iter()
        .zip(&differences)
        .map(|((task, _, _), d)| families.get(*task).map(|f| (f.as_str(), *d)))
        .collect();
    let (family_interval, family_count) = match named {
        Some(named) if !named.is_empty() => {
            let mut by_family: BTreeMap<&str, (f64, usize)> = BTreeMap::new();
            for (family, d) in named {
                let entry = by_family.entry(family).or_default();
                entry.0 += d;
                entry.1 += 1;
            }
            let count = by_family.len();
            let tail = (1.0 - request.level) / 2.0;
            let interval = Resampled::of(
                &by_family.into_values().collect::<Vec<_>>(),
                request.resamples,
                request.seed,
            )
            .map(|r| Bounds {
                low: r.quantile(tail),
                high: r.quantile(1.0 - tail),
            });
            (interval, Some(count))
        }
        _ => (None, None),
    };
    Contrast {
        arm: arm.to_string(),
        reference: reference.to_string(),
        paired_tasks: paired.len(),
        arm_pass_at_1: mean(paired.iter().map(|(_, a, _)| *a)),
        reference_pass_at_1: mean(paired.iter().map(|(_, _, b)| *b)),
        difference: mean(differences.iter().copied()),
        task_interval,
        family_interval,
        families: family_count,
        better,
        worse,
        ties: differences.len() - better - worse,
        p_one_sided: one_sided_p(better, worse),
        p_two_sided: two_sided_p(better, worse),
        p_one_sided_holm: 1.0,
        p_two_sided_holm: 1.0,
    }
}

/// The comparison of `trials` asked by `request`.
///
/// # Errors
/// The reference arm has no trial, there is no other arm, a level or a
/// resample count is unusable, a k is zero, or a task is given two families.
pub fn compare_arms(
    trials: &[Trial],
    request: &CompareRequest,
) -> Result<ArmComparison, OrchestratorError> {
    if !(request.level > 0.0 && request.level < 1.0) || request.resamples == 0 {
        return Err(refused(format!(
            "an interval needs a level in (0, 1) and resamples; {} and {} given",
            request.level, request.resamples
        )));
    }
    if request.k.contains(&0) {
        return Err(refused("pass@k needs k of at least 1"));
    }
    let mut families = request.families.clone();
    for trial in trials {
        if let Some(family) = &trial.family {
            match families.get(&trial.task) {
                Some(known) if known != family => {
                    return Err(refused(format!(
                        "task {} is given two families, {known} and {family}",
                        trial.task
                    )))
                }
                _ => {
                    families.insert(trial.task.clone(), family.clone());
                }
            }
        }
    }
    let mut by_arm: BTreeMap<&str, Vec<&Trial>> = BTreeMap::new();
    let mut graded: Graded = BTreeMap::new();
    for trial in trials {
        by_arm.entry(&trial.arm).or_default().push(trial);
        let tasks = graded.entry(trial.arm.clone()).or_default();
        if trial.status != Status::InfraError {
            tasks
                .entry(trial.task.clone())
                .or_default()
                .push(trial.passed);
        }
    }
    if !by_arm.contains_key(request.reference.as_str()) {
        return Err(refused(format!(
            "the reference arm {} has no trial; arms read: {}",
            request.reference,
            by_arm.keys().copied().collect::<Vec<_>>().join(", ")
        )));
    }
    if by_arm.len() < 2 {
        return Err(refused("a comparison needs an arm beside the reference"));
    }
    let arms = by_arm
        .iter()
        .map(|(arm, trials)| summary(arm, trials, &graded[*arm], &request.k))
        .collect();
    let mut contrasts: Vec<Contrast> = by_arm
        .keys()
        .filter(|arm| **arm != request.reference)
        .map(|arm| contrast(arm, &request.reference, &graded, &families, request))
        .collect();
    let one: Vec<f64> = contrasts.iter().map(|c| c.p_one_sided).collect();
    let two: Vec<f64> = contrasts.iter().map(|c| c.p_two_sided).collect();
    for ((c, one), two) in contrasts.iter_mut().zip(holm(&one)).zip(holm(&two)) {
        c.p_one_sided_holm = one;
        c.p_two_sided_holm = two;
    }
    Ok(ArmComparison {
        format: ARM_COMPARISON_FORMAT.into(),
        reference: request.reference.clone(),
        seed: request.seed,
        resamples: request.resamples,
        level: request.level,
        sign_test: SIGN_TEST_CONVENTION.into(),
        arms,
        contrasts,
    })
}

#[cfg(test)]
#[path = "arm_compare_tests.rs"]
mod tests;
