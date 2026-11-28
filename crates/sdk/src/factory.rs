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

use splinter_agent::solve::Model;
use splinter_core::factory::execution::{ModelBinding, Protocol};
use splinter_factory::backend::HarborWorker;
use splinter_factory::connections::Connections;
use splinter_factory::execute::Models;
use splinter_factory::grants::Grants;
use splinter_factory::machine_config::MachineConfig;
use splinter_factory::resolve::{resolve_in, Resolved};
use splinter_factory::training::BrainTrainer;
use splinter_model::selection::{ModelSelection, RemoteModel};
use splinter_orchestrator::config::credential;
use splinter_orchestrator::config::Config;

pub use splinter_factory::{
    backend, collection, connections, environment, eval_plan, execute, graph, knobs,
    machine_config, observe, online, ops, plan, recipe, registry, resolve as resolution, run,
    templates, training, tree,
};

/// The models a run's computed stages call in process: each binding's
/// endpoint and model, with the credential its connection names read from
/// this process's environment.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostModels;

/// The key sent to an endpoint whose connection names no credential, so
/// that sven never falls back to a key it keeps for another endpoint.
const NO_CREDENTIAL: &str = "no-credential";

/// The connection a recipe's `brain:` model role binds to: a brain server, asked through sven's brain driver, which
/// switches the model's reasoning off for a caller that wants an answer (a generic OpenAI client leaves a reasoning
/// model thinking until its output budget is spent, which is most of every reply).
const BRAIN_CONNECTION: &str = "brain";

/// The sven driver that speaks to the endpoint `binding` names.
fn provider_of(binding: &ModelBinding) -> &'static str {
    if binding.connection == BRAIN_CONNECTION && binding.protocol == Protocol::Openai {
        return BRAIN_CONNECTION;
    }
    match binding.protocol {
        Protocol::Openai => "openai",
        Protocol::Anthropic => "anthropic",
    }
}

impl Models for HostModels {
    fn model(&self, binding: &ModelBinding) -> Result<Model, String> {
        let provider = provider_of(binding);
        let api_key = match &binding.token_env {
            Some(name) => credential(name).ok_or_else(|| {
                format!(
                    "connection {} reads its key from {name}, which is not set",
                    binding.connection
                )
            })?,
            None => NO_CREDENTIAL.to_owned(),
        };
        let remote = RemoteModel {
            spec: format!("{provider}/{}", binding.model),
            base_url: binding.url.clone(),
            api_key: Some(api_key),
            ..RemoteModel::default()
        };
        let loaded = ModelSelection::Remote(remote)
            .load_remote()
            .map_err(|e| e.to_string())?;
        Ok(Model::new(loaded.provider(), loaded.identity()))
    }
}

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
/// machine's configuration of the same name ([`Connections::layered`]).
///
/// # Errors
/// A file exists and is invalid.
pub fn connections_of(config: &Config, repository: &Path) -> Result<Connections, SetupError> {
    Ok(Connections::layered(repository, &config.factory.config)?)
}

/// What this machine grants the task repository's operators.
///
/// # Errors
/// The machine configuration is invalid, or absent beside the older
/// `operators.yaml` it replaces.
pub fn grants_of(config: &Config) -> Result<Grants, SetupError> {
    MachineConfig::load(&config.factory.config)
        .map(|machine| Grants::from_config(&machine))
        .map_err(|e| SetupError::Read {
            path: config.factory.config.display().to_string(),
            message: e.to_string(),
        })
}

/// The online rollouts this machine offers an online training stage: practice
/// episodes in `ctx`, each against the version the stage's learner serves,
/// reached on the configured brain endpoint by the id it publishes under
/// (with brain's policy trace on every request). `None` without brain
/// endpoints.
#[must_use]
pub fn online_rollouts<'c>(
    config: &Config,
    ctx: &'c splinter_orchestrator::Context,
) -> Option<splinter_pipelines::episodes::online::FactoryRollouts<'c>> {
    use splinter_pipelines::episodes::online::{FactoryRollouts, OnlinePolicy};
    let brain = config.brain.clone()?;
    // seed: every online stage gives the rollouts its own seed; this one is replaced.
    let episode = splinter_pipelines::episodes::EpisodeSettings::new(1, 0);
    Some(FactoryRollouts::new(ctx, episode, move |served| {
        let url = brain
            .urls()
            .first()
            .ok_or("the brain endpoints name no URL")?
            .clone();
        let remote = RemoteModel {
            // brain's driver asks the server for its policy trace.
            spec: format!("brain/{served}"),
            base_url: Some(url),
            api_key: Some(brain.key().to_string()),
            ..RemoteModel::default()
        };
        let loaded = ModelSelection::Remote(remote)
            .load_remote()
            .map_err(|e| format!("the served policy {served}: {e}"))?;
        Ok(OnlinePolicy::Model(Model::new(
            loaded.provider(),
            loaded.identity(),
        )))
    }))
}

/// A variable of this process's environment, for an operator its machine
/// grants it to (read where the configuration reads the environment).
#[must_use]
pub fn host_variable(name: &str) -> Option<String> {
    credential(name)
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
    Ok(resolve_in(
        &config.working_dir,
        &text,
        connections_of(config, &config.working_dir)?,
        &grants_of(config)?,
    )?)
}

/// The run in `run` resolved again from the repository and recipe it was
/// started with, for a resume.
///
/// # Errors
/// The run's record, or what it names, cannot be read.
pub fn resolve_run(config: &Config, run: &Path) -> Result<Resolved, SetupError> {
    let (repository, text) = splinter_factory::run::started_with(run)?;
    Ok(resolve_in(
        &repository,
        &text,
        connections_of(config, &repository)?,
        &grants_of(config)?,
    )?)
}

/// The recipe `text` resolved in the task repository `repository` with this
/// machine's connections and grants: what a probe of a run's knob runs.
///
/// # Errors
/// The connections or grants cannot be read, or the recipe does not resolve.
pub fn resolve_text(
    config: &Config,
    repository: &Path,
    text: &str,
) -> Result<Resolved, SetupError> {
    Ok(resolve_in(
        repository,
        text,
        connections_of(config, repository)?,
        &grants_of(config)?,
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
    Ok(resolve_in(
        &repository,
        &text,
        connections_of(config, &repository)?,
        &grants_of(config)?,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn binding(connection: &str, protocol: Protocol) -> ModelBinding {
        ModelBinding {
            connection: connection.into(),
            model: "m".into(),
            protocol,
            url: Some("http://127.0.0.1:1/v1".into()),
            token_env: None,
            drop_headers: Vec::new(),
            driver: None,
        }
    }

    #[test]
    fn a_brain_connection_is_asked_through_the_brain_driver_and_others_by_their_protocol() {
        assert_eq!(provider_of(&binding("brain", Protocol::Openai)), "brain");
        assert_eq!(provider_of(&binding("hosted", Protocol::Openai)), "openai");
        assert_eq!(
            provider_of(&binding("hosted", Protocol::Anthropic)),
            "anthropic"
        );
        assert_eq!(
            provider_of(&binding("brain", Protocol::Anthropic)),
            "anthropic"
        );
    }
}
