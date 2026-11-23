// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements experiment planners that show the work an
// experiment will do before it spends anything, for its clients. If your
// team needs expertise in machine-learning experiment orchestration, you can
// procure our services by sending an email to info@swedishembedded.com.

//! A recipe as a graph of stages, each with what it depends on and the work
//! it will do.
//!
//! The graph is fixed by the recipe: one tasks stage; one stage per rollout
//! pass, after the tasks and after every pass it selects unsolved tasks
//! from; the dataset after its sources; one training stage per variant
//! after the dataset; one evaluation per candidate after its training stage;
//! the release after every evaluation. Stages are listed in an order in
//! which each comes after all it depends on. Their identities are computed
//! when they run, from the products of what they depend on.

use std::collections::BTreeSet;

use serde::Serialize;
use splinter_core::factory::identity::StageKind;

use crate::recipe::Recipe;

/// One planned stage.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PlannedStage {
    /// The stage's key in the run (`rollouts/local`).
    pub key: String,
    /// What it does.
    pub kind: StageKind,
    /// Its name in the recipe.
    pub name: String,
    /// The keys of the stages it consumes.
    pub depends_on: Vec<String>,
    /// What it will do, for a person.
    pub work: String,
}

/// The planned graph, in dependency order.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Plan {
    /// The stages.
    pub stages: Vec<PlannedStage>,
}

/// The key of a stage: `<directory>/<name>`, or the directory alone for the
/// run's single tasks and release stages.
#[must_use]
pub fn stage_key(kind: StageKind, name: &str) -> String {
    match kind {
        StageKind::Tasks | StageKind::Release => kind.directory().to_owned(),
        _ => format!("{}/{name}", kind.directory()),
    }
}

impl Plan {
    /// Plan a validated recipe; `tasks` is the number of tasks the tasks
    /// section selects.
    #[must_use]
    pub fn of(recipe: &Recipe, tasks: usize) -> Self {
        let mut stages = vec![PlannedStage {
            key: stage_key(StageKind::Tasks, ""),
            kind: StageKind::Tasks,
            name: recipe.name.clone(),
            depends_on: Vec::new(),
            work: format!("pin {tasks} tasks of {}", recipe.tasks.collection),
        }];
        for pass in &recipe.rollouts {
            let mut depends_on = vec![stage_key(StageKind::Tasks, "")];
            depends_on.extend(
                pass.tasks
                    .unsolved_by
                    .iter()
                    .map(|p| stage_key(StageKind::Rollout, p)),
            );
            let selected = if !pass.tasks.ids.is_empty() {
                format!("{} named tasks", pass.tasks.ids.len())
            } else if pass.tasks.unsolved_by.is_empty() {
                format!("{tasks} tasks")
            } else {
                format!(
                    "at most {tasks} tasks (those unsolved by {})",
                    pass.tasks.unsolved_by.join(", ")
                )
            };
            let model = pass
                .model
                .as_ref()
                .map_or_else(|| "its own model".to_owned(), ToString::to_string);
            stages.push(PlannedStage {
                key: stage_key(StageKind::Rollout, &pass.name),
                kind: StageKind::Rollout,
                name: pass.name.clone(),
                depends_on,
                work: format!(
                    "{} x {} attempts of {} on {model}, {} at a time",
                    selected, pass.attempts, pass.agent, pass.concurrency
                ),
            });
        }
        if let Some(dataset) = &recipe.dataset {
            stages.push(PlannedStage {
                key: stage_key(StageKind::Dataset, &dataset.name),
                kind: StageKind::Dataset,
                name: dataset.name.clone(),
                depends_on: dataset
                    .sources
                    .iter()
                    .map(|s| stage_key(StageKind::Rollout, s))
                    .collect(),
                work: format!(
                    "{} {} records of {} trials for {}",
                    dataset.projection, dataset.objective, dataset.admission, dataset.target
                ),
            });
            if let Some(training) = &recipe.training {
                for variant in &training.variants {
                    stages.push(PlannedStage {
                        key: stage_key(StageKind::Training, &variant.name),
                        kind: StageKind::Training,
                        name: variant.name.clone(),
                        depends_on: vec![stage_key(StageKind::Dataset, &dataset.name)],
                        work: format!(
                            "{} {} of {} with {}",
                            training.engine, variant.method, training.base, variant.params
                        ),
                    });
                }
            }
        }
        let mut evals = Vec::new();
        if let Some(eval) = &recipe.eval {
            for candidate in &eval.candidates {
                let key = stage_key(StageKind::Eval, candidate);
                evals.push(key.clone());
                stages.push(PlannedStage {
                    key,
                    kind: StageKind::Eval,
                    name: candidate.clone(),
                    depends_on: vec![stage_key(StageKind::Training, candidate)],
                    work: format!("{candidate} against {} under {}", eval.baseline, eval.plan),
                });
            }
        }
        if let Some(release) = &recipe.release {
            stages.push(PlannedStage {
                key: stage_key(StageKind::Release, ""),
                kind: StageKind::Release,
                name: recipe.name.clone(),
                depends_on: evals,
                work: if release.promote {
                    "promote the best improved candidate".into()
                } else {
                    "record the candidates without promoting any".into()
                },
            });
        }
        Self { stages }
    }

    /// The stage with `key`.
    #[must_use]
    pub fn stage(&self, key: &str) -> Option<&PlannedStage> {
        self.stages.iter().find(|s| s.key == key)
    }

    /// Whether every stage comes after all it depends on.
    #[must_use]
    pub fn is_ordered(&self) -> bool {
        let mut seen = BTreeSet::new();
        self.stages.iter().all(|stage| {
            let ok = stage.depends_on.iter().all(|d| seen.contains(d.as_str()));
            seen.insert(stage.key.as_str());
            ok
        })
    }

    /// Every stage that depends on `key`, directly or not.
    #[must_use]
    pub fn downstream(&self, key: &str) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        let mut frontier = vec![key.to_owned()];
        while let Some(at) = frontier.pop() {
            for stage in &self.stages {
                if stage.depends_on.contains(&at) && out.insert(stage.key.clone()) {
                    frontier.push(stage.key.clone());
                }
            }
        }
        out
    }
}
