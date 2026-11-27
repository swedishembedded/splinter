// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning loops whose every generation can be
// explained from its own records, for its clients. If your team needs
// expertise in observing and steering agent training loops, you can procure
// our services by sending an email to info@swedishembedded.com.

//! What a generation's candidate measured against its baseline: every
//! evaluation of it, labelled by the purpose its release gave it (or as a
//! measurement only, when no release reads it) and by the task set it ran
//! on, with both arms' pass rates, the paired sign test, the interval of the
//! per-task difference, what the infrastructure left uncounted, how each
//! arm's attempts failed, and whether its tasks are known to be solvable.
//!
//! The interval is a paired percentile bootstrap of the mean per-task
//! difference in pass rate (candidate minus baseline) over the tasks both
//! arms have a counted attempt at, tasks resampled; its seed is the
//! observation's.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use splinter_core::factory::product::TrialResult;
use splinter_eval::pilot::Share;
use splinter_model::stats::{bootstrap_interval, Interval};

use super::conduct::{Conduct, Trace, Witness};
use super::data::tasks_of;
use super::records::{Product, Reference, RunRecords};
use super::system::{pace, Pace, Utilisation};

/// The resamples of an evaluation's interval.
pub const BOOTSTRAP_ITERATIONS: usize = 10_000;
/// The level of an evaluation's interval.
pub const INTERVAL_LEVEL: f64 = 0.95;

/// How many of a task set's tasks are known to be solvable.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Solvability {
    /// Tasks whose reference solution passed.
    pub witnessed: usize,
    /// Tasks whose reference solution ran and failed.
    pub refuted: usize,
    /// Tasks whose reference solution never ran.
    pub unknown: usize,
}

impl Solvability {
    /// The solvability of `tasks` under `witnesses`.
    #[must_use]
    pub fn of<'a>(
        tasks: impl IntoIterator<Item = &'a String>,
        witnesses: &BTreeMap<String, Witness>,
    ) -> Self {
        let mut out = Self::default();
        for task in tasks {
            match witnesses.get(task).copied().unwrap_or(Witness::Unknown) {
                Witness::Witnessed => out.witnessed += 1,
                Witness::Refuted => out.refuted += 1,
                Witness::Unknown => out.unknown += 1,
            }
        }
        out
    }
}

/// The tasks an evaluation ran on.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TaskSet {
    /// What they are, relative to the generation: `own held-out`,
    /// `held-out of generation N`, `recipe input NAME`, `tasks of STAGE` or
    /// `the plan's tasks`.
    pub label: String,
    /// The stage that pinned them, if one did.
    pub stage: Option<String>,
    /// How many.
    pub tasks: usize,
    /// Whether each is known to be solvable.
    pub solvability: Solvability,
}

/// One arm of an evaluation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Arm {
    /// The model it ran.
    pub model: Option<String>,
    /// Its counted attempts and passes, with the Wilson interval.
    pub pass_rate: Option<Share>,
    /// Its attempts.
    pub trials: Option<u64>,
    /// How its attempts went, when its trials could be read.
    pub conduct: Option<Conduct>,
    /// Its pace.
    pub pace: Option<Pace>,
}

/// One evaluation of a generation's candidate.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Evaluation {
    /// The evaluation stage.
    pub stage: String,
    /// The candidate's training stage.
    pub candidate: String,
    /// The purpose its release gave it (`target`, `transfer`, `retention`);
    /// `None` for a measurement no release reads.
    pub purpose: Option<String>,
    /// Whether no release reads it.
    pub measurement_only: bool,
    /// The tasks it ran on.
    pub task_set: TaskSet,
    /// Its decision (`improved`, `inconclusive`, ...).
    pub decision: Option<String>,
    /// The baseline arm.
    pub baseline: Arm,
    /// The candidate arm.
    pub candidate_arm: Arm,
    /// Paired slots the candidate won.
    pub wins: Option<u64>,
    /// Paired slots it lost.
    pub losses: Option<u64>,
    /// Paired slots.
    pub pairs: Option<u64>,
    /// The sign test's one-sided p.
    pub p: Option<f64>,
    /// The candidate's pass rate minus the baseline's.
    pub difference: Option<f64>,
    /// The interval of the mean per-task difference.
    pub interval: Option<Interval>,
    /// Tasks the interval rests on.
    pub interval_tasks: usize,
    /// Slots either arm never counted.
    pub unpaired_slots: Option<u64>,
    /// Of the slots, the share either arm never counted.
    pub uncounted_share: Option<f64>,
}

/// The pass rate per task of counted attempts.
fn task_rates(trials: &[TrialResult]) -> BTreeMap<String, f64> {
    let mut by: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for t in trials.iter().filter(|t| t.grade.counted) {
        let e = by.entry(&t.task).or_default();
        e.0 += 1;
        e.1 += usize::from(t.grade.passed == Some(true));
    }
    by.into_iter()
        .map(|(k, (n, p))| (k.to_owned(), p as f64 / n as f64))
        .collect()
}

/// How the attempts of `product` went.
pub(super) fn conduct_of(
    product: &Product,
    trials: &[TrialResult],
    witnesses: &BTreeMap<String, Witness>,
) -> Conduct {
    let mut conduct = Conduct::default();
    for trial in trials {
        let trace = trial
            .trace
            .as_ref()
            .and_then(|t| product.json::<Value>(&t.path))
            .map(|v| Trace::of(&v));
        let witness = witnesses
            .get(&trial.task)
            .copied()
            .unwrap_or(Witness::Unknown);
        conduct.add(trial, trace, witness);
    }
    conduct.finished()
}

/// Everything an evaluation's reading needs from the generation it is of.
pub(super) struct Context<'a> {
    pub(super) records: &'a RunRecords,
    pub(super) witnesses: &'a BTreeMap<String, Witness>,
    pub(super) utilisation: Option<&'a Utilisation>,
    pub(super) seed: u64,
    /// Each generation's dataset tasks collection, in order.
    pub(super) collections: &'a [Option<Reference>],
    /// The generation read, from 0.
    pub(super) generation: usize,
}

/// The collection a tasks stage pinned its tasks from, and whether it kept
/// the held-out side.
fn pinned_from(records: &RunRecords, tasks: &str) -> (Option<Reference>, bool) {
    let held = records
        .param(tasks, "holdout")
        .and_then(|h| h["side"].as_str())
        == Some("held");
    (
        records.references(tasks, "collection").into_iter().next(),
        held,
    )
}

/// The collection the dataset `id` takes its tasks from.
pub(super) fn dataset_collection(records: &RunRecords, id: &str) -> Option<Reference> {
    let tasks = records.upstream(id, "tasks").into_iter().next()?;
    pinned_from(records, &tasks).0
}

fn label(cx: &Context<'_>, tasks: Option<&str>) -> String {
    let Some(tasks) = tasks else {
        return "the plan's tasks".to_owned();
    };
    let (collection, held) = pinned_from(cx.records, tasks);
    if held {
        if let Some(g) = cx
            .collections
            .iter()
            .position(|c| c.is_some() && *c == collection)
        {
            return if g == cx.generation {
                "own held-out".to_owned()
            } else {
                format!("held-out of generation {}", g + 1)
            };
        }
    }
    match collection {
        Some(Reference::Input(name)) => format!("recipe input {name}"),
        _ => format!("tasks of {tasks}"),
    }
}

fn arm(cx: &Context<'_>, name: &str, result: &Value, key: &str) -> (Arm, Vec<TrialResult>) {
    let side = &result[key];
    let product = side["trials_product"]
        .as_str()
        .and_then(|d| cx.records.product_by_manifest(d));
    let trials = product.as_ref().map(Product::trials).unwrap_or_default();
    let counted = side["counted"].as_u64().map(|n| n as usize);
    let passed = side["passed"].as_u64().map(|n| n as usize);
    (
        Arm {
            model: side["model"].as_str().map(str::to_owned),
            pass_rate: counted.zip(passed).and_then(|(n, k)| Share::wilson(k, n)),
            trials: side["trials"].as_u64(),
            conduct: product
                .as_ref()
                .filter(|_| !trials.is_empty())
                .map(|p| conduct_of(p, &trials, cx.witnesses)),
            pace: (!trials.is_empty()).then(|| pace(name, &trials, cx.utilisation)),
        },
        trials,
    )
}

/// The evaluation `id` of the candidate trained by `candidate`, with the
/// purpose its release gave it.
pub(super) fn observe(
    cx: &Context<'_>,
    id: &str,
    candidate: &str,
    purpose: Option<String>,
) -> Option<Evaluation> {
    let product = cx.records.product(id)?;
    let result: Value = product.json("result.json").unwrap_or(Value::Null);
    let stage: Value = product.json("stage.json").unwrap_or(Value::Null);
    let tasks_stage = cx.records.upstream(id, "tasks").into_iter().next();
    let task_ids: BTreeSet<String> = tasks_stage
        .as_deref()
        .and_then(|t| cx.records.product(t))
        .map(|p| tasks_of(&p))
        .filter(|t| !t.is_empty())
        .unwrap_or_else(|| {
            stage["parameters"]["development"]["tasks"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|t| t[0].as_str().map(str::to_owned))
                .collect()
        });
    let (baseline, base_trials) = arm(cx, &format!("{id}/baseline"), &result, "baseline");
    let (candidate_arm, cand_trials) = arm(cx, &format!("{id}/candidate"), &result, "candidate");
    let (base_rates, cand_rates) = (task_rates(&base_trials), task_rates(&cand_trials));
    let differences: Vec<f64> = cand_rates
        .iter()
        .filter_map(|(task, c)| base_rates.get(task).map(|b| c - b))
        .collect();
    let stats = &result["statistics"];
    let pairs = stats["pairs"].as_u64();
    let unpaired = stats["unpaired_slots"].as_u64();
    let measurement_only = purpose.is_none();
    Some(Evaluation {
        stage: id.to_owned(),
        candidate: candidate.to_owned(),
        purpose,
        measurement_only,
        task_set: TaskSet {
            label: label(cx, tasks_stage.as_deref()),
            stage: tasks_stage.clone(),
            tasks: task_ids.len(),
            solvability: Solvability::of(&task_ids, cx.witnesses),
        },
        decision: result["decision"].as_str().map(str::to_owned),
        baseline,
        candidate_arm,
        wins: stats["wins"].as_u64(),
        losses: stats["losses"].as_u64(),
        pairs,
        p: stats["p"].as_f64(),
        difference: stats["difference"].as_f64(),
        interval: bootstrap_interval(&differences, BOOTSTRAP_ITERATIONS, INTERVAL_LEVEL, cx.seed),
        interval_tasks: differences.len(),
        unpaired_slots: unpaired,
        uncounted_share: pairs
            .zip(unpaired)
            .and_then(|(p, u)| (p + u > 0).then(|| u as f64 / (p + u) as f64)),
    })
}
