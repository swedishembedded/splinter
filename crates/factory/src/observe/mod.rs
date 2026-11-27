// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning loops whose every generation can be
// explained from its own records, for its clients. If your team needs
// expertise in observing and steering agent training loops, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The state of a generational learning loop at every generation, read from
//! a run's records alone and never changing them.
//!
//! A loop is a control system: to explain why a generation did or did not
//! improve, its state is measured where it is decided. A *generation* is a
//! release stage (`promotion`) of the recipe; the evaluations it reads name
//! its candidates; each candidate's training stage names its dataset and
//! replay; the dataset's `attempts` and `tasks` name where its records and
//! new tasks came from. Every relation is the recipe graph's (`recipe.yaml`
//! of the run), never a stage's name. Per generation the state is:
//!
//! - **data** ([`data`]): records by source policy (the target's own or a
//!   teacher's) by record and supervised characters, in training and in
//!   validation; the frontier share; each policy's pass rate on the new
//!   tasks; distinct action sequences;
//! - **training** ([`training`]): the budget actually trained, in epochs at
//!   the selected step, steps, tokens and the validation curve's ends;
//! - **capability** ([`capability`]): every evaluation of the candidate with
//!   its purpose, task set, decision, sign test and interval, and the
//!   release's decision and why;
//! - **conduct and system** ([`conduct`], [`system`]): how every attempt
//!   stage's and arm's failures divide into interface, policy and
//!   infrastructure classes, how the agents' claims of completion agreed
//!   with the grader, whether each task set is known to be solvable, and the
//!   pace of every attempt stage;
//! - **online training** ([`online`]): each online training stage's health,
//!   per optimiser step and per policy version, live while it runs.

pub mod capability;
pub mod conduct;
pub mod data;
pub mod online;
pub mod probe;
mod records;
pub mod system;
pub mod training;

pub use splinter_eval::pilot::Share;
pub use splinter_model::stats::Interval;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::Value;

use self::capability::{conduct_of, dataset_collection, Context, Evaluation, Solvability};
use self::conduct::{Conduct, Witness};
use self::data::{tasks_of, Data};
use self::records::RunRecords;
use self::system::{pace, CardUse, Pace, Utilisation};
use self::training::Training;

/// The operation of a release stage.
const PROMOTION: &str = "promotion";
/// The operation of an evaluation stage.
const EVALUATION: &str = "evaluation";
/// The operation of an attempt stage.
const ATTEMPTS: &str = "attempt.collect";
/// The operation of a tasks stage.
const TASKS: &str = "tasks.pin";

/// Why a run could not be observed.
#[derive(Debug, thiserror::Error)]
pub enum ObserveError {
    /// A record the observation needs could not be read.
    #[error("{path}: {reason}")]
    Read {
        /// The file.
        path: PathBuf,
        /// Why.
        reason: String,
    },
}

/// What an observation reads beyond the run's records.
#[derive(Clone, Debug, Default)]
pub struct Options {
    /// The seed of the evaluations' bootstrap intervals.
    pub seed: u64,
    /// A card utilisation log to join by time.
    pub utilisation: Option<Utilisation>,
}

/// One candidate of a generation: its training and its dataset.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Candidate {
    /// The training stage.
    pub stage: String,
    /// The training it had.
    pub training: Option<Training>,
    /// The cards' use while it trained, when a utilisation log is given.
    pub cards: Vec<CardUse>,
    /// What its dataset is made of.
    pub data: Option<Data>,
}

/// The release of a generation and why it decided as it did.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Release {
    /// The release stage.
    pub stage: String,
    /// The candidate it promoted, if any.
    pub promoted: Option<String>,
    /// The champion the run leaves in use.
    pub champion: Option<String>,
    /// The purposes a promotion required.
    pub requires: Vec<String>,
    /// Each candidate's reading, as one line.
    pub why: Vec<String>,
}

/// One generation's state.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Generation {
    /// Its number in the run, from 1.
    pub number: usize,
    /// Its release.
    pub release: Release,
    /// Its candidates.
    pub candidates: Vec<Candidate>,
    /// Every evaluation of its candidates.
    pub evaluations: Vec<Evaluation>,
}

/// One attempt stage's conduct and pace.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AttemptStage {
    /// The stage.
    pub stage: String,
    /// The models its attempts ran.
    pub models: Vec<String>,
    /// Whether it runs the tasks' reference solutions (no model).
    pub reference: bool,
    /// How its attempts went.
    pub conduct: Conduct,
    /// Its pace.
    pub pace: Pace,
    /// Whether its tasks are known to be solvable.
    pub solvability: Solvability,
}

/// One run's state at every generation.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RunObservation {
    /// The run directory.
    pub run: PathBuf,
    /// The seed of its intervals.
    pub seed: u64,
    /// Its generations, in recipe order.
    pub generations: Vec<Generation>,
    /// Its attempt stages, in recipe order.
    pub attempts: Vec<AttemptStage>,
    /// Whether every task its reference solutions ran on is solvable.
    pub solvability: Solvability,
    /// Its online training stages' health, finished or running.
    pub online: Vec<online::OnlineStage>,
}

/// The tasks' witnesses: for each task a stage that runs no model attempted
/// (its reference solution), witnessed when one counted attempt passed,
/// refuted when every counted attempt failed.
fn witnesses(records: &RunRecords) -> (BTreeMap<String, Witness>, Vec<String>) {
    let mut out: BTreeMap<String, Witness> = BTreeMap::new();
    let mut stages = Vec::new();
    for stage in records.stages().iter().filter(|s| s.uses == ATTEMPTS) {
        let Some(product) = records.product(&stage.id) else {
            continue;
        };
        let runs_no_model = product
            .json::<Value>("stage.json")
            .is_some_and(|s| s["parameters"]["model"].is_null());
        if !runs_no_model {
            continue;
        }
        stages.push(stage.id.clone());
        for trial in product.trials().iter().filter(|t| t.grade.counted) {
            let passed = trial.grade.passed == Some(true);
            let entry = out.entry(trial.task.clone()).or_insert(Witness::Refuted);
            if passed {
                *entry = Witness::Witnessed;
            }
        }
    }
    (out, stages)
}

/// The evaluations a release stage reads, in its recipe order, with the
/// purpose it gave each (its recorded `purposes`, else the evaluation's own
/// `purpose`, else `target`).
fn released(records: &RunRecords, release: &str) -> Vec<(String, String)> {
    let evaluations = records.upstream(release, "evaluations");
    let recorded: Vec<String> = records
        .product(release)
        .and_then(|p| p.json::<Value>("stage.json"))
        .and_then(|s| s["parameters"]["purposes"].as_array().cloned())
        .unwrap_or_default()
        .iter()
        .filter_map(|p| p.as_str().map(str::to_owned))
        .collect();
    evaluations
        .into_iter()
        .enumerate()
        .map(|(i, e)| {
            let purpose = recorded.get(i).cloned().unwrap_or_else(|| {
                records
                    .param(&e, "purpose")
                    .and_then(Value::as_str)
                    .unwrap_or("target")
                    .to_owned()
            });
            (e, purpose)
        })
        .collect()
}

fn release_of(records: &RunRecords, id: &str) -> Release {
    let release: Value = records
        .product(id)
        .and_then(|p| p.json("release.json"))
        .unwrap_or(Value::Null);
    let why = release["candidates"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|c| {
            let gates: Vec<String> = c["gates"]
                .as_object()
                .into_iter()
                .flatten()
                .map(|(k, v)| format!("{k} {}", v.as_str().unwrap_or("?")))
                .collect();
            format!(
                "{} ({}): {}, difference {}; gates {}; eligible {}; divergence {}",
                c["candidate"].as_str().unwrap_or("?"),
                c["purpose"].as_str().unwrap_or("?"),
                c["decision"].as_str().unwrap_or("?"),
                c["difference"]
                    .as_f64()
                    .map_or("-".to_owned(), |d| format!("{d:+.4}")),
                gates.join(", "),
                c["eligible"],
                c["divergence"].as_str().unwrap_or("?"),
            )
        })
        .collect();
    Release {
        stage: id.to_owned(),
        promoted: release["promoted"].as_str().map(str::to_owned),
        champion: release["champion"]["candidate"].as_str().map(str::to_owned),
        requires: records
            .param(id, "requires")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|r| r.as_str().map(str::to_owned))
            .collect(),
        why,
    }
}

/// The run directory `run` observed.
///
/// # Errors
/// Its recipe cannot be read.
pub fn observe(run: &Path, options: &Options) -> Result<RunObservation, ObserveError> {
    let records = RunRecords::open(run)?;
    let (witnesses, reference_stages) = witnesses(&records);
    let windows = records.stage_windows();
    let utilisation = options.utilisation.as_ref();

    // The generations, in recipe order, and each one's dataset collection.
    let releases: Vec<String> = records
        .stages()
        .iter()
        .filter(|s| s.uses == PROMOTION)
        .map(|s| s.id.clone())
        .collect();
    let candidates_of = |release: &str| -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for (e, _) in released(&records, release) {
            for c in records.upstream(&e, "candidate") {
                if !out.contains(&c) {
                    out.push(c);
                }
            }
        }
        out
    };
    let collections: Vec<_> = releases
        .iter()
        .map(|r| {
            candidates_of(r)
                .first()
                .and_then(|c| records.upstream(c, "dataset").into_iter().next())
                .and_then(|d| dataset_collection(&records, &d))
        })
        .collect();

    let mut generations = Vec::new();
    for (index, release) in releases.iter().enumerate() {
        let cx = Context {
            records: &records,
            witnesses: &witnesses,
            utilisation,
            seed: options.seed,
            collections: &collections,
            generation: index,
        };
        let candidates = candidates_of(release);
        let purposes: BTreeMap<String, String> = released(&records, release).into_iter().collect();
        let evaluations = records
            .stages()
            .iter()
            .filter(|s| s.uses == EVALUATION)
            .filter_map(|s| {
                let candidate = records.upstream(&s.id, "candidate").into_iter().next()?;
                candidates.contains(&candidate).then(|| {
                    capability::observe(&cx, &s.id, &candidate, purposes.get(&s.id).cloned())
                })?
            })
            .collect();
        let candidates = candidates
            .into_iter()
            .map(|stage| {
                let training = training::observe(&records, &stage);
                let data = training
                    .as_ref()
                    .and_then(|t| data::observe(&records, &t.dataset, &t.replay));
                let cards = match (utilisation, records.product(&stage)) {
                    (Some(u), Some(p)) => windows
                        .get(p.manifest.stage.as_str())
                        .map(|(s, e)| u.over(*s as f64, *e as f64))
                        .unwrap_or_default(),
                    _ => Vec::new(),
                };
                Candidate {
                    stage,
                    training,
                    cards,
                    data,
                }
            })
            .collect();
        generations.push(Generation {
            number: index + 1,
            release: release_of(&records, release),
            candidates,
            evaluations,
        });
    }

    let attempts = records
        .stages()
        .iter()
        .filter(|s| s.uses == ATTEMPTS)
        .filter_map(|s| {
            let product = records.product(&s.id)?;
            let trials = product.trials();
            let mut models: Vec<String> = trials.iter().filter_map(|t| t.model.clone()).collect();
            models.sort();
            models.dedup();
            let tasks: Vec<String> = {
                let mut t: Vec<String> = trials.iter().map(|t| t.task.clone()).collect();
                t.sort();
                t.dedup();
                t
            };
            Some(AttemptStage {
                stage: s.id.clone(),
                models,
                reference: reference_stages.contains(&s.id),
                conduct: conduct_of(&product, &trials, &witnesses),
                pace: pace(&s.id, &trials, utilisation),
                solvability: Solvability::of(&tasks, &witnesses),
            })
        })
        .collect();

    let all_tasks: Vec<String> = records
        .stages()
        .iter()
        .filter(|s| s.uses == TASKS)
        .filter_map(|s| records.product(&s.id))
        .flat_map(|p| tasks_of(&p))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    Ok(RunObservation {
        run: run.to_path_buf(),
        seed: options.seed,
        generations,
        attempts,
        solvability: Solvability::of(&all_tasks, &witnesses),
        online: online::observe(&records, run),
    })
}
