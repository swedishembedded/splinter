// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning factories that take a model from
// agent rollouts to an evaluated, released candidate, for its clients. If
// your team needs expertise in model fine-tuning pipelines, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The learning factory, configured from a [`Config`]: a recipe of the task
//! repository (the working directory) resolved with this machine's
//! connections, and the execution worker and trainer this machine offers.
//!
//! Everything else is [`splinter_factory`]'s, re-exported here.

use std::path::Path;

use splinter_factory::backend::HarborWorker;
use splinter_factory::connections::Connections;
use splinter_factory::resolve::{resolve, Resolved};
use splinter_factory::training::BrainTrainer;
use splinter_orchestrator::config::Config;

pub use splinter_factory::{
    backend, collection, connections, environment, eval_plan, graph, ops, plan, recipe,
    resolve as resolution, run, training, tree,
};

/// Why the factory could not be set up.
#[derive(Debug, thiserror::Error)]
pub enum SetupError {
    /// A file cannot be read.
    #[error("{path}: {message}")]
    Read {
        /// The file.
        path: String,
        /// Why.
        message: String,
    },
    /// The connections file is invalid.
    #[error(transparent)]
    Connections(#[from] splinter_factory::connections::ConnectionError),
    /// The recipe is invalid.
    #[error(transparent)]
    Recipe(#[from] splinter_factory::recipe::RecipeError),
    /// The execution worker cannot be run.
    #[error("{0}")]
    Worker(String),
    /// The run directory refused (it is not a run, or its record is broken).
    #[error(transparent)]
    Run(#[from] splinter_factory::run::RunError),
}

/// The connections a run of the task repository at `repository` uses: the
/// repository's `connections.yaml`, each connection overridden by this
/// machine's file of the same name ([`Connections::layered`]).
///
/// # Errors
/// A file exists and is invalid.
pub fn connections_of(config: &Config, repository: &Path) -> Result<Connections, SetupError> {
    Ok(Connections::layered(
        repository,
        &config.factory.connections,
    )?)
}

/// The recipe at `recipe` (relative to the working directory, the task
/// repository), resolved with this machine's connections.
///
/// # Errors
/// The recipe, a file it names, or the connections are invalid.
pub fn resolve_recipe(config: &Config, recipe: &Path) -> Result<Resolved, SetupError> {
    let path = config.working_dir.join(recipe);
    let text = std::fs::read_to_string(&path).map_err(|e| SetupError::Read {
        path: path.display().to_string(),
        message: e.to_string(),
    })?;
    Ok(resolve(
        &config.working_dir,
        &text,
        connections_of(config, &config.working_dir)?,
    )?)
}

/// The run in `run` resolved again from the repository and recipe it was
/// started with, for a resume.
///
/// # Errors
/// The run's record, or what it names, cannot be read.
pub fn resolve_run(config: &Config, run: &Path) -> Result<Resolved, SetupError> {
    let (repository, text) = splinter_factory::run::started_with(run)?;
    Ok(resolve(
        &repository,
        &text,
        connections_of(config, &repository)?,
    )?)
}

/// The run in `run` resolved again with its recipe's evaluation plan
/// replaced by `plan`: what a fork re-evaluating its candidates runs.
///
/// # Errors
/// As [`resolve_run`], or the recipe evaluates nothing.
pub fn resolve_run_under_plan(
    config: &Config,
    run: &Path,
    plan: &Path,
) -> Result<Resolved, SetupError> {
    let (repository, text) = splinter_factory::run::started_with(run)?;
    let text = splinter_factory::recipe::with_eval_plan(&text, &plan.to_string_lossy())?;
    Ok(resolve(
        &repository,
        &text,
        connections_of(config, &repository)?,
    )?)
}

/// The execution worker this machine offers.
///
/// # Errors
/// None is configured or it cannot be run.
pub fn worker(config: &Config) -> Result<HarborWorker, SetupError> {
    let program = config.factory.harbor_worker.as_ref().ok_or_else(|| {
        SetupError::Worker(
            "no execution worker: install it (`make worker/install` in Splinter's checkout) and put \
             splinter-harbor-worker on PATH, or set SPLINTER_HARBOR_WORKER"
                .into(),
        )
    })?;
    HarborWorker::new(program).map_err(|e| SetupError::Worker(e.to_string()))
}

/// The trainer this machine offers: brain at the configured endpoints, with
/// base names read under the model store. `None` without endpoints.
#[must_use]
pub fn trainer(config: &Config) -> Option<BrainTrainer> {
    config
        .brain
        .clone()
        .map(|endpoints| BrainTrainer::new(endpoints, config.model_store.clone()))
}
