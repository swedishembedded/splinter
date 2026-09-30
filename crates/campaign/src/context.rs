// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What every command works with: the configuration and state root, the
//! clock that stamps records, the async runtime model runs block on, the
//! models it has loaded, and the environments tasks are solved in.
//!
//! A model is loaded once per [`Context`] and shared by every stage that
//! names it; dropping the context stops a local generation still running
//! before the process exits. A caller that already holds a model - a test's
//! scripted one - hands it in with [`Context::with_model`] instead.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use splinter_agent::solve::Model;
use splinter_policy::{LoadedModel, ModelSelection};
use splinter_sandbox::{
    Limits, ProcessSandbox, ResolvedEnvironment, RuntimeEnvironment, RuntimeRegistry, Sandbox,
    SandboxError,
};
use splinter_store::clock::{Clock, SystemClock};
use splinter_store::experience::Environment;
use splinter_store::experiences::ExperienceStore;
use splinter_store::sources::SourceStore;
use splinter_store::tasks::TaskStore;
use splinter_store::StateRoot;
use splinter_views::DatasetStore;

use crate::config::Config;
use crate::error::CampaignError;
use crate::model_ref::ModelRef;

/// How long a local model may stay silent between two stream chunks: as
/// long as its run lasts. sven's idle limit guards a remote wire going
/// stale; a local provider is silent through its prefill for as long as the
/// prompt takes, and no chunk in between is honest to invent, so the run's
/// own deadline is the bound (a solve caps the limit at its deadline).
const LOCAL_STREAM_IDLE: Duration = Duration::MAX;

/// Receives each stage's summary as it finishes: `(stage, summary)`.
pub type Progress = Box<dyn Fn(&str, &serde_json::Value) + Send + Sync>;

/// A model a context holds, and what keeps a loaded one alive.
struct Held {
    model: Model,
    /// The loaded model, when the context loaded it; dropping it quiesces
    /// a local device.
    _loaded: Option<LoadedModel>,
}

/// See the module documentation.
pub struct Context {
    config: Config,
    allow_remote: bool,
    clock: Box<dyn Clock + Send + Sync>,
    runtime: tokio::runtime::Runtime,
    models: Mutex<HashMap<ModelRef, Held>>,
    environments: Environments,
    progress: Option<Progress>,
}

impl Context {
    /// A context over `config`, with `allow_remote` as the command line's
    /// network opt-in, stamping records with the wall clock.
    pub fn new(config: Config, allow_remote: bool) -> Result<Self, CampaignError> {
        let runtime = tokio::runtime::Runtime::new().map_err(CampaignError::Runtime)?;
        let environments = Environments::process(&config.state_root);
        Ok(Self {
            config,
            allow_remote,
            clock: Box::new(SystemClock),
            runtime,
            models: Mutex::new(HashMap::new()),
            environments,
            progress: None,
        })
    }

    /// The same context answering `reference` with `model` instead of
    /// loading it.
    #[must_use]
    pub fn with_model(self, reference: ModelRef, model: Model) -> Self {
        self.lock_models().insert(
            reference,
            Held {
                model,
                _loaded: None,
            },
        );
        self
    }

    /// The same context reporting each finished stage to `progress`.
    #[must_use]
    pub fn with_progress(mut self, progress: Progress) -> Self {
        self.progress = Some(progress);
        self
    }

    /// The configuration.
    #[must_use]
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// The state root.
    #[must_use]
    pub fn root(&self) -> &StateRoot {
        &self.config.state_root
    }

    /// Whether models reached over the network may be used.
    #[must_use]
    pub fn allow_remote(&self) -> bool {
        self.allow_remote || self.config.allow_remote
    }

    /// The clock records are stamped with.
    #[must_use]
    pub fn clock(&self) -> &dyn Clock {
        self.clock.as_ref()
    }

    /// The source store.
    #[must_use]
    pub fn sources(&self) -> SourceStore {
        SourceStore::open(self.root())
    }

    /// The task store.
    #[must_use]
    pub fn tasks(&self) -> TaskStore {
        TaskStore::open(self.root())
    }

    /// The experience store.
    #[must_use]
    pub fn experiences(&self) -> ExperienceStore {
        ExperienceStore::open(self.root())
    }

    /// The dataset store.
    #[must_use]
    pub fn datasets(&self) -> DatasetStore {
        DatasetStore::open(self.root())
    }

    /// The environments tasks are solved and checked in.
    #[must_use]
    pub fn environments(&self) -> &Environments {
        &self.environments
    }

    /// The selection `reference` resolves to, without loading it: refused
    /// for a remote model without the network opt-in.
    pub fn selection(&self, reference: &ModelRef) -> Result<ModelSelection, CampaignError> {
        Ok(reference.resolve(&self.config, self.allow_remote())?)
    }

    /// The model `reference` names, loaded on first use.
    pub fn model(&self, reference: &ModelRef) -> Result<Model, CampaignError> {
        if let Some(held) = self.lock_models().get(reference) {
            return Ok(held.model.clone());
        }
        let selection = self.selection(reference)?;
        let local = selection.local().is_some();
        let loaded = selection.load().map_err(|e| CampaignError::Model {
            model: reference.to_string(),
            detail: format!("{e:#}"),
        })?;
        let mut model = Model::new(loaded.provider(), loaded.identity());
        if local {
            model = model.with_stream_idle(LOCAL_STREAM_IDLE);
        }
        self.lock_models().insert(
            reference.clone(),
            Held {
                model: model.clone(),
                _loaded: Some(loaded),
            },
        );
        Ok(model)
    }

    /// Runs `future` to completion on the context's runtime. Must not be
    /// called from inside an async task.
    pub(crate) fn block_on<F: std::future::Future>(&self, future: F) -> F::Output {
        self.runtime.block_on(future)
    }

    /// A handle to the context's runtime, for the lower crates that block
    /// on it themselves.
    pub(crate) fn handle(&self) -> tokio::runtime::Handle {
        self.runtime.handle().clone()
    }

    /// Reports a finished stage to the progress receiver, if any.
    pub(crate) fn report_stage(&self, stage: &str, summary: &serde_json::Value) {
        if let Some(progress) = &self.progress {
            progress(stage, summary);
        }
    }

    fn lock_models(&self) -> std::sync::MutexGuard<'_, HashMap<ModelRef, Held>> {
        // A poisoned lock only means a load panicked; the map is still
        // consistent, since an entry is inserted whole.
        self.models
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// The runtimes code runs in, in the process sandbox under the state root,
/// each resolved once.
///
/// The sandbox's environment and limits are fixed - no variable of this
/// process reaches a call - so a runtime resolves to the same environment
/// record in every command, and a task generated in one command is solved
/// and checked in exactly its environment in the next.
pub struct Environments {
    sandbox: Arc<dyn Sandbox>,
    registry: RuntimeRegistry,
    resolved: Mutex<BTreeMap<String, RuntimeEnvironment>>,
}

impl Environments {
    /// The process sandbox under `root`, with the built-in runtimes.
    #[must_use]
    pub fn process(root: &StateRoot) -> Self {
        let sandbox = ProcessSandbox::new(root.sandbox(), Vec::new(), &[], Limits::default());
        Self {
            sandbox: Arc::new(sandbox),
            registry: RuntimeRegistry::builtin(),
            resolved: Mutex::new(BTreeMap::new()),
        }
    }

    /// The runtime `name`, resolved in the sandbox.
    pub fn runtime(&self, name: &str) -> Result<RuntimeEnvironment, SandboxError> {
        let mut resolved = self
            .resolved
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(env) = resolved.get(name) {
            return Ok(env.clone());
        }
        let env = RuntimeEnvironment::new(&self.registry, name, Arc::clone(&self.sandbox))?;
        resolved.insert(name.to_string(), env.clone());
        Ok(env)
    }

    /// Those of the runtimes `names` that resolve here.
    pub fn available<'a>(
        &self,
        names: impl IntoIterator<Item = &'a str>,
    ) -> Vec<RuntimeEnvironment> {
        names
            .into_iter()
            .filter_map(|name| self.runtime(name).ok())
            .collect()
    }

    /// The environment `recorded` names, resolved here. Whether it is
    /// still the same environment is the solve's check, against the
    /// snapshot.
    pub fn for_record(&self, recorded: &Environment) -> Result<ResolvedEnvironment, CampaignError> {
        if recorded.kind == Environment::CLOSED_BOOK {
            return Ok(ResolvedEnvironment::ClosedBook);
        }
        let name = recorded
            .kind
            .strip_prefix(splinter_sandbox::environment::RUNTIME_KIND_PREFIX)
            .ok_or_else(|| {
                CampaignError::Refused(format!(
                    "environment {:?} is neither closed-book nor a runtime",
                    recorded.kind
                ))
            })?;
        Ok(ResolvedEnvironment::Runtime(self.runtime(name)?))
    }
}
