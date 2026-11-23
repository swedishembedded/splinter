// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible experiments whose every input
// is pinned before work starts, for its clients. If your team needs
// expertise in reproducible machine-learning runs, you can procure our
// services by sending an email to info@swedishembedded.com.

//! A recipe with every file it names read, checked and pinned.
//!
//! [`resolve`] reads the recipe, the collection and its tasks, every
//! environment, eval plan, agent file and prompt, and binds every model to
//! a connection. Every problem is reported at once, so `splinter validate`
//! shows them all. The [`Lock`] records the digest of each input: a resumed
//! run compares the repository against it and refuses to continue on inputs
//! that changed under it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use splinter_core::digest::{digest_of, Digest};
use splinter_core::factory::task::{plain_relative, TaskCollection, TaskRef};

use crate::champion::{self, Champion};
use crate::collection::{self, LoadedCollection};
use crate::connections::Connections;
use crate::environment::Environment;
use crate::eval_plan::EvalPlan;
use crate::plan::Plan;
use crate::recipe::{ModelRef, Recipe, RecipeError};

/// The schema of a [`Lock`].
pub const LOCK_SCHEMA: &str = "splinter.lock/v1";

/// What a run was started on.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Lock {
    /// Always [`LOCK_SCHEMA`].
    pub schema: String,
    /// The recipe's digest.
    pub recipe: Digest,
    /// Every file the recipe names, by repository-relative path.
    pub files: BTreeMap<String, Digest>,
    /// The pinned task collection's digest.
    pub tasks: Digest,
    /// The connections the recipe's models are reached through, as
    /// configured (names of credential variables only); a connection no
    /// model uses is not part of it.
    pub connections: Digest,
}

/// A resolved recipe.
#[derive(Clone, Debug)]
pub struct Resolved {
    /// The repository root every path is relative to.
    pub repository: PathBuf,
    /// The recipe.
    pub recipe: Recipe,
    /// The recipe's text, as read (its digest is the lock's).
    pub recipe_text: String,
    /// The tasks the recipe's tasks section selects.
    pub collection: LoadedCollection,
    /// Environments by path.
    pub environments: BTreeMap<String, Environment>,
    /// The evaluation plan and its own collection, when the recipe has one.
    pub eval: Option<(EvalPlan, Option<LoadedCollection>)>,
    /// The champions the recipe's models name, by reference.
    pub champions: BTreeMap<String, Champion>,
    /// The connections.
    pub connections: Connections,
    /// The lock.
    pub lock: Lock,
    /// The graph of stages.
    pub plan: Plan,
}

fn read(repository: &Path, relative: &str, problems: &mut Vec<String>) -> Option<String> {
    if !plain_relative(relative) {
        problems.push(format!(
            "{relative:?} is not a plain path relative to the repository"
        ));
        return None;
    }
    match std::fs::read_to_string(repository.join(relative)) {
        Ok(text) => Some(text),
        Err(e) => {
            problems.push(format!("{relative}: {e}"));
            None
        }
    }
}

/// Read and check everything `recipe_text` names under `repository`.
///
/// # Errors
/// Every problem found: the recipe, a missing or invalid file, a model with
/// no connection, an environment a pass cannot run in.
pub fn resolve(
    repository: &Path,
    recipe_text: &str,
    connections: Connections,
) -> Result<Resolved, RecipeError> {
    let recipe = Recipe::parse(recipe_text)?;
    let mut problems = match recipe.validate() {
        Ok(()) => Vec::new(),
        Err(RecipeError(problems)) => problems,
    };
    let mut files = BTreeMap::new();
    for file in recipe.referenced_files() {
        if let Some(text) = read(repository, &file, &mut problems) {
            files.insert(file, Digest::of(text.as_bytes()));
        }
    }
    let mut environments = BTreeMap::new();
    let mut environment = |path: &str, problems: &mut Vec<String>| {
        if environments.contains_key(path) {
            return;
        }
        if let Some(text) = read(repository, path, problems) {
            match Environment::parse(&text) {
                Ok(env) => {
                    environments.insert(path.to_owned(), env);
                }
                Err(RecipeError(found)) => {
                    problems.extend(found.into_iter().map(|p| format!("{path}: {p}")))
                }
            }
        }
    };
    for pass in &recipe.rollouts {
        environment(&pass.environment, &mut problems);
        for (name, value) in &pass.environment_variables {
            let literal = !value.starts_with("env:");
            if literal && splinter_core::factory::execution::credential_like(name) {
                problems.push(format!(
                    "rollout {}: {name} looks like a credential; write env:{name} to pass the worker's own variable",
                    pass.name
                ));
            }
        }
    }
    let mut models: Vec<_> = recipe
        .rollouts
        .iter()
        .filter_map(|p| p.model.clone())
        .collect();
    models.extend(recipe.dataset.iter().map(|d| d.target.clone()));
    models.extend(recipe.training.iter().map(|t| t.base.clone()));
    models.extend(recipe.eval.iter().map(|e| e.baseline.clone()));
    // A model several sections name is checked, and reported, once.
    models.sort();
    models.dedup();
    let mut champions = BTreeMap::new();
    for model in &models {
        if champion::is_champion(model) {
            match champion::load(repository, model) {
                Ok(found) => {
                    if let Err(e) = connections.bind(&found.record.base) {
                        problems.push(format!("{model}: {e}"));
                    }
                    files.insert(model.to_string(), found.digest.clone());
                    champions.insert(model.to_string(), found);
                }
                Err(e) => problems.push(e),
            }
        } else if let Err(e) = connections.bind(model) {
            problems.push(e.to_string());
        }
    }
    let trained_from = recipe.training.iter().map(|t| &t.base);
    for model in trained_from.chain(recipe.dataset.iter().map(|d| &d.target)) {
        if champion::is_champion(model) {
            problems.push(format!(
                "{model}: a champion is served, never trained from: a variant trains from a base model, which is also its dataset's target"
            ));
        }
    }
    let loaded = match collection::load(repository, &recipe.tasks.collection) {
        Ok(loaded) => Some(loaded),
        Err(e) => {
            problems.push(e.to_string());
            None
        }
    };
    let mut eval = None;
    if let Some(section) = &recipe.eval {
        if let Some(text) = read(repository, &section.plan, &mut problems) {
            match EvalPlan::parse(&text) {
                Ok(plan) => {
                    if let Some(env) = &plan.environment {
                        environment(env, &mut problems);
                        if let Some(text) = read(repository, env, &mut problems) {
                            files.insert(env.clone(), Digest::of(text.as_bytes()));
                        }
                    }
                    let tasks = match &plan.collection {
                        Some(file) => match collection::load(repository, file) {
                            Ok(loaded) => {
                                if let Some(text) = read(repository, file, &mut problems) {
                                    files.insert(file.clone(), Digest::of(text.as_bytes()));
                                }
                                Some(loaded)
                            }
                            Err(e) => {
                                problems.push(e.to_string());
                                None
                            }
                        },
                        None => None,
                    };
                    eval = Some((plan, tasks));
                }
                Err(RecipeError(found)) => {
                    problems.extend(found.into_iter().map(|p| format!("{}: {p}", section.plan)));
                }
            }
        }
    }
    let Some(mut loaded) = loaded else {
        return Err(RecipeError(problems));
    };
    let split = recipe.tasks.split.clone();
    loaded
        .collection
        .tasks
        .retain(|t| split.is_none() || t.split == split);
    if loaded.collection.tasks.is_empty() {
        problems.push(format!(
            "{} has no task of split {}",
            recipe.tasks.collection,
            split.as_deref().unwrap_or("(any)")
        ));
    }
    for pass in &recipe.rollouts {
        for id in &pass.tasks.ids {
            if loaded.collection.task(id).is_none() {
                problems.push(format!(
                    "rollout {} names task {id}, which the collection does not select",
                    pass.name
                ));
            }
        }
    }
    if let Some(dataset) = &recipe.dataset {
        problems.extend(examination_material(
            &loaded.collection,
            eval.as_ref(),
            &dataset.name,
        ));
    }
    if !problems.is_empty() {
        return Err(RecipeError(problems));
    }
    let lock = Lock {
        schema: LOCK_SCHEMA.into(),
        recipe: Digest::of(recipe_text.as_bytes()),
        files,
        tasks: digest_of(&loaded.collection),
        connections: digest_of(&used_connections(&connections, &models, &champions)),
    };
    let plan = Plan::of(&recipe, loaded.collection.tasks.len());
    Ok(Resolved {
        repository: repository.to_path_buf(),
        recipe,
        recipe_text: recipe_text.to_owned(),
        collection: loaded,
        environments,
        eval,
        champions,
        connections,
        lock,
        plan,
    })
}

/// The connections the recipe's models are reached through (a champion's
/// through its base): what a run's lock pins. Another connection of the
/// machine or the repository, added or changed, changes no run.
fn used_connections<'c>(
    connections: &'c Connections,
    models: &[ModelRef],
    champions: &BTreeMap<String, Champion>,
) -> BTreeMap<&'c str, &'c crate::connections::Connection> {
    let names: std::collections::BTreeSet<&str> = models
        .iter()
        .map(|m| match champions.get(&m.to_string()) {
            Some(champion) => champion.record.base.connection.as_str(),
            None => m.connection.as_str(),
        })
        .collect();
    connections
        .connections
        .iter()
        .filter(|(name, _)| names.contains(name.as_str()))
        .map(|(name, c)| (name.as_str(), c))
        .collect()
}

/// The recipe's tasks that a dataset would learn from but that are
/// examination material: on the `eval` split, or the very task (by content)
/// the eval plan examines under any name. Trials of them could otherwise
/// reach training, and a candidate would be examined on what it was taught.
fn examination_material(
    training: &TaskCollection,
    eval: Option<&(EvalPlan, Option<LoadedCollection>)>,
    dataset: &str,
) -> Vec<String> {
    let examined: Vec<&TaskRef> = match eval {
        Some((plan, Some(loaded))) => loaded
            .collection
            .tasks
            .iter()
            .filter(|t| plan.split.is_none() || t.split == plan.split)
            .collect(),
        _ => Vec::new(),
    };
    let mut problems = Vec::new();
    for task in &training.tasks {
        if task.split.as_deref() == Some("eval") {
            problems.push(format!(
                "{} is examination material (split eval) and cannot reach dataset {dataset}: select the train split",
                task.id
            ));
        } else if let Some(same) = examined.iter().find(|e| e.digest == task.digest) {
            problems.push(format!(
                "{} is the same task as {}, which the eval plan examines; it cannot reach dataset {dataset}",
                task.id, same.id
            ));
        }
    }
    problems
}

impl Lock {
    /// What differs between this lock and `now`, one line each; empty when
    /// the inputs are unchanged.
    #[must_use]
    pub fn changes(&self, now: &Self) -> Vec<String> {
        let mut out = Vec::new();
        if self.recipe != now.recipe {
            out.push("the recipe changed".into());
        }
        if self.tasks != now.tasks {
            out.push("the task collection changed".into());
        }
        if self.connections != now.connections {
            out.push("the connections changed".into());
        }
        for (path, digest) in &self.files {
            match now.files.get(path) {
                Some(d) if d == digest => {}
                Some(_) => out.push(format!("{path} changed")),
                None => out.push(format!("{path} is no longer named")),
            }
        }
        for path in now.files.keys().filter(|p| !self.files.contains_key(*p)) {
            out.push(format!("{path} is newly named"));
        }
        out
    }
}
