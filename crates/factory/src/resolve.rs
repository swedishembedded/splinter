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
use splinter_core::factory::task::plain_relative;

use crate::collection::{self, LoadedCollection};
use crate::connections::Connections;
use crate::environment::Environment;
use crate::eval_plan::EvalPlan;
use crate::plan::Plan;
use crate::recipe::{Recipe, RecipeError};

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
    /// The connections the run binds, as configured (names of credential
    /// variables only).
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
    for model in &models {
        if let Err(e) = connections.bind(model) {
            problems.push(e.to_string());
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
    if !problems.is_empty() {
        return Err(RecipeError(problems));
    }
    let lock = Lock {
        schema: LOCK_SCHEMA.into(),
        recipe: Digest::of(recipe_text.as_bytes()),
        files,
        tasks: digest_of(&loaded.collection),
        connections: digest_of(&connections),
    };
    let plan = Plan::of(&recipe, loaded.collection.tasks.len());
    Ok(Resolved {
        repository: repository.to_path_buf(),
        recipe,
        recipe_text: recipe_text.to_owned(),
        collection: loaded,
        environments,
        eval,
        connections,
        lock,
        plan,
    })
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
