// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning loops whose every generation can be
// explained from its own records, for its clients. If your team needs
// expertise in observing and steering agent training loops, you can procure
// our services by sending an email to info@swedishembedded.com.

//! What a generation's dataset is made of: whose records (the target's own
//! policy or another, by records and supervised characters, in training and
//! in validation), how hard its new tasks were for each policy that attempted
//! them, and how its records spread over the frontier of the target's own
//! policy (tasks it fails, passes only sometimes, or already passes).
//!
//! A record is the target's *own* when the model that made it is the
//! dataset's target or an adapter served on it (its name the target's
//! followed by `:`); every other policy is a teacher.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use splinter_core::chat::WireMessage;
use splinter_core::factory::product::TrialResult;
use splinter_eval::pilot::Share;

use crate::run::dataset::measure;

use super::records::{Product, RunRecords};

/// The policy a record or an attempt came from, relative to a target.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Policy {
    /// The target itself or an adapter served on it (the student, the
    /// champion).
    Own,
    /// Any other model (a teacher).
    Other,
}

impl Policy {
    /// The policy of `model` relative to `target`.
    #[must_use]
    pub fn of(model: &str, target: &str) -> Self {
        if model == target
            || model
                .strip_prefix(target)
                .is_some_and(|r| r.starts_with(':'))
        {
            Self::Own
        } else {
            Self::Other
        }
    }
}

/// Records and supervised characters of a part of a dataset.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Amount {
    /// Records.
    pub records: usize,
    /// Characters of the supervised messages (text, reasoning, calls).
    pub supervised_characters: u64,
}

impl Amount {
    fn add(&mut self, characters: u64) {
        self.records += 1;
        self.supervised_characters += characters;
    }
}

/// One attempt stage's records in the dataset.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Source {
    /// The attempt stage.
    pub stage: String,
    /// The models its records came from.
    pub models: Vec<String>,
    /// Whose they are.
    pub policy: Policy,
    /// Its training records.
    pub train: Amount,
    /// Its validation records.
    pub validation: Amount,
    /// Its counted attempts at the dataset's new tasks, and how many passed.
    pub pass_rate: Option<Share>,
    /// The mean over those tasks of each task's pass rate.
    pub mean_task_pass_rate: Option<f64>,
    /// The tasks it attempted (counted).
    pub tasks_attempted: usize,
}

/// The dataset's records by the class of their task on the frontier of the
/// target's own policy, from its own attempts at those tasks.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Frontier {
    /// The attempt stages of the target's own policy it is read from.
    pub from: Vec<String>,
    /// Tasks it failed every counted attempt at.
    pub failed_tasks: usize,
    /// Tasks it passed some but not all counted attempts at.
    pub sometimes_tasks: usize,
    /// Tasks it passed every counted attempt at.
    pub passed_tasks: usize,
    /// Records of tasks it failed, by policy of the record.
    pub records_failed: BTreeMap<Policy, usize>,
    /// Records of tasks it passed only sometimes, by policy of the record.
    pub records_sometimes: BTreeMap<Policy, usize>,
    /// Records of tasks it already passed, by policy of the record.
    pub records_passed: BTreeMap<Policy, usize>,
    /// Records of tasks it has no counted attempt at.
    pub records_unattempted: usize,
    /// Of the records of attempted tasks, the share from tasks it does not
    /// yet always pass (failed or sometimes): the frontier share.
    pub frontier_share: Option<f64>,
}

/// What a generation's dataset is made of.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Data {
    /// The dataset stage.
    pub stage: String,
    /// The model it trains.
    pub target: String,
    /// Training records, all sources.
    pub train: Amount,
    /// Validation records, all sources.
    pub validation: Amount,
    /// By attempt stage.
    pub sources: Vec<Source>,
    /// Of the training records, the target's own share by records.
    pub own_share_records: Option<f64>,
    /// Of the training supervised characters, the target's own share.
    pub own_share_characters: Option<f64>,
    /// Of the validation records, the target's own share by records: the
    /// policy checkpoint selection follows.
    pub validation_own_share_records: Option<f64>,
    /// Of the validation supervised characters, the target's own share.
    pub validation_own_share_characters: Option<f64>,
    /// Distinct supervised action sequences over every record.
    pub action_sequences: usize,
    /// Supervised tokens: the records hold characters, not the trainer's
    /// tokens, so this is absent unless a record states it.
    pub supervised_tokens: Option<u64>,
    /// The new tasks: those the dataset's tasks stage names, else those its
    /// sources attempted.
    pub new_tasks: usize,
    /// The frontier of the target's own policy, when it attempted the tasks.
    pub frontier: Option<Frontier>,
    /// Replayed datasets, with their training records when they state them.
    pub replay: Vec<(String, Option<usize>)>,
}

/// A line of a chat dataset, as much of it as is measured here.
#[derive(Deserialize)]
struct Line {
    messages: Vec<WireMessage>,
    #[serde(default)]
    metadata: Value,
}

fn share(part: f64, whole: f64) -> Option<f64> {
    (whole > 0.0).then(|| part / whole)
}

/// The tasks a tasks product names.
pub(super) fn tasks_of(product: &Product) -> BTreeSet<String> {
    product
        .json::<Value>("collection.json")
        .and_then(|c| c["tasks"].as_array().cloned())
        .unwrap_or_default()
        .iter()
        .filter_map(|t| t["id"].as_str().map(str::to_owned))
        .collect()
}

/// Per task: counted attempts and passes.
fn per_task(
    trials: &[TrialResult],
    within: Option<&BTreeSet<String>>,
) -> BTreeMap<String, (usize, usize)> {
    let mut out: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for t in trials.iter().filter(|t| t.grade.counted) {
        if within.is_some_and(|w| !w.contains(&t.task)) {
            continue;
        }
        let e = out.entry(t.task.clone()).or_default();
        e.0 += 1;
        e.1 += usize::from(t.grade.passed == Some(true));
    }
    out
}

/// The training records of the dataset `id`, when it states them.
pub(super) fn train_records(records: &RunRecords, id: &str) -> Option<usize> {
    let product = records.product(id)?;
    product
        .json::<Value>("composition.json")
        .and_then(|c| c["train_records"].as_u64())
        .map(|n| n as usize)
        .or_else(|| {
            product
                .file("train.jsonl")
                .and_then(|f| std::fs::read_to_string(f).ok())
                .map(|t| t.lines().count())
        })
}

/// The data state of the dataset stage `id`, replaying `replay`.
pub(super) fn observe(records: &RunRecords, id: &str, replay: &[String]) -> Option<Data> {
    let product = records.product(id)?;
    let dataset: Value = product.json("dataset.json")?;
    let target = dataset["target"].as_str()?.to_owned();
    let sources_ids = records.upstream(id, "attempts");
    let within = records
        .upstream(id, "tasks")
        .first()
        .and_then(|t| records.product(t))
        .map(|p| tasks_of(&p));

    // Attempts of every source, by stage.
    let attempts: BTreeMap<String, Vec<TrialResult>> = sources_ids
        .iter()
        .map(|s| {
            (
                s.clone(),
                records.product(s).map(|p| p.trials()).unwrap_or_default(),
            )
        })
        .collect();
    let new_tasks: BTreeSet<String> = within.clone().unwrap_or_else(|| {
        attempts
            .values()
            .flatten()
            .map(|t| t.task.clone())
            .collect()
    });

    // The records, by source and split.
    let mut sources: BTreeMap<String, Source> = BTreeMap::new();
    let mut records_by_task: Vec<(String, Policy)> = Vec::new();
    let mut sequences = BTreeSet::new();
    let (mut train, mut validation) = (Amount::default(), Amount::default());
    let mut own = (Amount::default(), Amount::default());
    for (split, file) in [("train", "train.jsonl"), ("validation", "validation.jsonl")] {
        for line in product.lines::<Line>(file) {
            let m = measure(&line.messages);
            sequences.insert(m.action_sequence);
            let trial = &line.metadata["trial"];
            let stage = trial["pass"].as_str().unwrap_or("").to_owned();
            let model = trial["model"].as_str().unwrap_or("").to_owned();
            let policy = Policy::of(&model, &target);
            let task = line.metadata["group"].as_str().unwrap_or("").to_owned();
            records_by_task.push((task, policy));
            let source = sources.entry(stage.clone()).or_insert_with(|| Source {
                stage,
                models: Vec::new(),
                policy,
                train: Amount::default(),
                validation: Amount::default(),
                pass_rate: None,
                mean_task_pass_rate: None,
                tasks_attempted: 0,
            });
            if !source.models.contains(&model) {
                source.models.push(model);
            }
            let (all, mine, part) = if split == "train" {
                (&mut train, &mut own.0, &mut source.train)
            } else {
                (&mut validation, &mut own.1, &mut source.validation)
            };
            all.add(m.supervised_characters);
            part.add(m.supervised_characters);
            if policy == Policy::Own {
                mine.add(m.supervised_characters);
            }
        }
    }

    // Difficulty: each source's counted attempts at the new tasks.
    let mut own_tasks: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut own_from = Vec::new();
    for (stage, trials) in &attempts {
        let tasks = per_task(trials, Some(&new_tasks));
        let (n, k) = tasks.values().fold((0, 0), |a, v| (a.0 + v.0, a.1 + v.1));
        let mean = (!tasks.is_empty()).then(|| {
            tasks
                .values()
                .map(|(n, k)| *k as f64 / *n as f64)
                .sum::<f64>()
                / tasks.len() as f64
        });
        let policy = trials
            .iter()
            .find_map(|t| t.model.as_deref())
            .map(|m| Policy::of(m, &target));
        let source = sources.entry(stage.clone()).or_insert_with(|| Source {
            stage: stage.clone(),
            models: Vec::new(),
            policy: policy.unwrap_or(Policy::Other),
            train: Amount::default(),
            validation: Amount::default(),
            pass_rate: None,
            mean_task_pass_rate: None,
            tasks_attempted: 0,
        });
        source.pass_rate = Share::wilson(k, n);
        source.mean_task_pass_rate = mean;
        source.tasks_attempted = tasks.len();
        if policy == Some(Policy::Own) {
            own_from.push(stage.clone());
            for (task, (n, k)) in tasks {
                let e = own_tasks.entry(task).or_default();
                e.0 += n;
                e.1 += k;
            }
        }
    }

    let frontier = (!own_from.is_empty()).then(|| {
        let mut f = Frontier {
            from: own_from,
            ..Frontier::default()
        };
        for (n, k) in own_tasks.values() {
            match (*k, *n) {
                (0, _) => f.failed_tasks += 1,
                (k, n) if k == n => f.passed_tasks += 1,
                _ => f.sometimes_tasks += 1,
            }
        }
        for (task, policy) in &records_by_task {
            let bucket = match own_tasks.get(task) {
                None => {
                    f.records_unattempted += 1;
                    continue;
                }
                Some((_, 0)) => &mut f.records_failed,
                Some((n, k)) if k == n => &mut f.records_passed,
                Some(_) => &mut f.records_sometimes,
            };
            *bucket.entry(*policy).or_default() += 1;
        }
        let sum = |m: &BTreeMap<Policy, usize>| m.values().sum::<usize>() as f64;
        let attempted = sum(&f.records_failed) + sum(&f.records_sometimes) + sum(&f.records_passed);
        f.frontier_share = share(
            sum(&f.records_failed) + sum(&f.records_sometimes),
            attempted,
        );
        f
    });

    let mut sources: Vec<Source> = sources
        .into_values()
        .filter(|s| !s.stage.is_empty())
        .collect();
    sources.sort_by_key(|s| {
        sources_ids
            .iter()
            .position(|id| *id == s.stage)
            .unwrap_or(usize::MAX)
    });
    Some(Data {
        stage: id.to_owned(),
        target,
        own_share_records: share(own.0.records as f64, train.records as f64),
        own_share_characters: share(
            own.0.supervised_characters as f64,
            train.supervised_characters as f64,
        ),
        validation_own_share_records: share(own.1.records as f64, validation.records as f64),
        validation_own_share_characters: share(
            own.1.supervised_characters as f64,
            validation.supervised_characters as f64,
        ),
        train,
        validation,
        sources,
        action_sequences: sequences.len(),
        supervised_tokens: None,
        new_tasks: new_tasks.len(),
        frontier,
        replay: replay
            .iter()
            .map(|r| (r.clone(), train_records(records, r)))
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_adapter_served_on_the_target_is_its_own_policy() {
        let target = "brain:Qwen/Qwen3-8B";
        assert_eq!(Policy::of(target, target), Policy::Own);
        assert_eq!(
            Policy::of("brain:Qwen/Qwen3-8B:splinter:candidate:lora-1", target),
            Policy::Own
        );
        assert_eq!(Policy::of("brain:Qwen/Qwen3-8B-FP8", target), Policy::Other);
        assert_eq!(
            Policy::of("brain:Qwen/Qwen3.8-27B-FP8", target),
            Policy::Other
        );
    }
}
