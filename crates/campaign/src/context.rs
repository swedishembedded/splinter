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
//!
//! Local models go on the context's [`Residency`]: one resident copy of
//! each base, shared by every model on it - the policy and the release
//! gate's two arms differ only by adapter, and never hold two bases. Work
//! that needs the device for itself - a fine-tune, which loads its own
//! copy, and the gate's serve check, a separate process - first releases
//! every resident base ([`Context::release_bases`]); the next generation of
//! a model reloads its base.
//!
//! `policy:<alias>` is resolved once per context, on first use: the
//! release the alias points at then is the one every stage of the command
//! uses, however the alias moves meanwhile ([`Context::policy_pin`]). Only
//! the context's own release or rollback moves the pin along.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::Serialize;

use splinter_agent::solve::Model;
use splinter_policy::{LoadedModel, ModelSelection, Residency, Sampling};
use splinter_record::clock::{Clock, SystemClock};
use splinter_record::experience::Environment;
use splinter_record::experiences::ExperienceStore;
use splinter_record::sources::SourceStore;
use splinter_record::tasks::TaskStore;
use splinter_record::workspace::Workspace;
use splinter_record::StateRoot;
use splinter_sandbox::{
    Limits, ProcessSandbox, ResolvedEnvironment, RuntimeEnvironment, RuntimeRegistry, Sandbox,
    SandboxError,
};
use splinter_views::DatasetStore;

use crate::answers::AnswerStore;
use crate::config::Config;
use crate::error::CampaignError;
use crate::model_ref::ModelRef;
use crate::release::{ReleaseId, ReleaseStore};

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
    loaded: Option<LoadedModel>,
}

/// The release a policy alias was resolved to for a context.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PolicyPin {
    /// The alias.
    pub alias: String,
    /// The release it pointed at.
    pub release: ReleaseId,
    /// That release's adapter file.
    pub adapter: PathBuf,
}

/// See the module documentation.
pub struct Context {
    config: Config,
    workspace: Workspace,
    allow_remote: bool,
    clock: Box<dyn Clock + Send + Sync>,
    runtime: tokio::runtime::Runtime,
    models: Mutex<HashMap<ModelRef, Held>>,
    residency: Residency,
    pins: Mutex<BTreeMap<String, Option<PolicyPin>>>,
    environments: Environments,
    progress: Option<Progress>,
}

impl Context {
    /// A context over `config`, with `allow_remote` as the command line's
    /// network opt-in, stamping records with the wall clock.
    pub fn new(config: Config, allow_remote: bool) -> Result<Self, CampaignError> {
        let runtime = tokio::runtime::Runtime::new().map_err(CampaignError::Runtime)?;
        let environments = Environments::process(&config.state_root);
        let workspace = Workspace::at(&config.state_root);
        Ok(Self {
            config,
            workspace,
            allow_remote,
            clock: Box::new(SystemClock),
            runtime,
            models: Mutex::new(HashMap::new()),
            residency: Residency::default(),
            pins: Mutex::new(BTreeMap::new()),
            environments,
            progress: None,
        })
    }

    /// The same context answering `reference` with `model` instead of
    /// loading it.
    #[must_use]
    pub fn with_model(self, reference: ModelRef, model: Model) -> Self {
        self.add_model(reference, model);
        self
    }

    /// Answers `reference` with `model` from now on, instead of loading it.
    pub fn add_model(&self, reference: ModelRef, model: Model) {
        self.lock_models().insert(
            reference,
            Held {
                model,
                loaded: None,
            },
        );
    }

    /// The same context keeping its local models' bases on `residency`
    /// instead of loading them with brain: a spec's scripted bases.
    #[must_use]
    pub fn with_residency(mut self, residency: Residency) -> Self {
        self.residency = residency;
        self
    }

    /// How many bases are resident on the device for this context's
    /// models right now.
    #[must_use]
    pub fn resident_bases(&self) -> usize {
        self.residency.resident()
    }

    /// Frees every resident base for work that needs the device for
    /// itself: each loaded model's running generation is stopped first
    /// (and waited for, as dropping the model does), and the models stay,
    /// their next generation loading the base again.
    pub fn release_bases(&self) {
        for held in self.lock_models().values() {
            if let Some(loaded) = &held.loaded {
                // Best effort, as on drop: a generation outliving the grace
                // is cancelling, and the release waits for it to end.
                let _ = loaded.quiesce();
            }
        }
        self.residency.release_all();
    }

    /// The same context stamping records with `clock`.
    #[must_use]
    pub fn with_clock(mut self, clock: Box<dyn Clock + Send + Sync>) -> Self {
        self.clock = clock;
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

    /// The experience database every store reads and writes.
    #[must_use]
    pub fn workspace(&self) -> &Workspace {
        &self.workspace
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
        SourceStore::new(&self.workspace)
    }

    /// The task store.
    #[must_use]
    pub fn tasks(&self) -> TaskStore {
        TaskStore::new(&self.workspace)
    }

    /// The experience store.
    #[must_use]
    pub fn experiences(&self) -> ExperienceStore {
        ExperienceStore::new(&self.workspace)
    }

    /// The answers `ask` gave.
    #[must_use]
    pub fn answers(&self) -> AnswerStore {
        AnswerStore::new(&self.workspace)
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
    /// for a remote model without the network opt-in, and for a policy
    /// alias other than `default` that points at no release.
    pub fn selection(&self, reference: &ModelRef) -> Result<ModelSelection, CampaignError> {
        let adapter = match reference {
            ModelRef::Policy(alias) => {
                let pin = self.policy_pin(alias)?;
                if pin.is_none() && alias != crate::model_ref::POLICY_DEFAULT {
                    return Err(CampaignError::NotFound {
                        what: "release alias",
                        id: alias.clone(),
                    });
                }
                pin.map(|p| p.adapter)
            }
            _ => None,
        };
        Ok(reference.resolve(&self.config, self.allow_remote(), adapter.as_deref())?)
    }

    /// The release `alias` points at, resolved on this context's first ask
    /// and the same on every later one; `None` when it points at none.
    pub fn policy_pin(&self, alias: &str) -> Result<Option<PolicyPin>, CampaignError> {
        let mut pins = self
            .pins
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(pin) = pins.get(alias) {
            return Ok(pin.clone());
        }
        let store = ReleaseStore::open(self.root());
        let pin = match store.alias(alias)? {
            Some(release) => {
                let stored = store.get(&release)?;
                Some(PolicyPin {
                    alias: alias.to_string(),
                    release,
                    adapter: stored.adapter,
                })
            }
            None => None,
        };
        pins.insert(alias.to_string(), pin.clone());
        Ok(pin)
    }

    /// Drops `alias`'s pin and the policy model this context loaded for
    /// it, so the next use resolves the alias afresh: for the context's own
    /// release or rollback, which moved it.
    pub(crate) fn repin_policy(&self, alias: &str) {
        self.pins
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(alias);
        // A model handed in stays; one this context loaded is dropped,
        // and its base with it once no other model uses it.
        // Dropped outside the lock: dropping quiesces a running generation.
        let reference = ModelRef::Policy(alias.to_string());
        let dropped = {
            let mut models = self.lock_models();
            let loaded = models
                .get(&reference)
                .is_some_and(|held| held.loaded.is_some());
            loaded.then(|| models.remove(&reference))
        };
        drop(dropped);
    }

    /// The model `reference` names, loaded on first use.
    pub fn model(&self, reference: &ModelRef) -> Result<Model, CampaignError> {
        if let Some(held) = self.lock_models().get(reference) {
            return Ok(held.model.clone());
        }
        let selection = self.selection(reference)?;
        let local = selection.local().is_some();
        let loaded = selection
            .load(&self.residency)
            .map_err(|e| CampaignError::Model {
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
                loaded: Some(loaded),
            },
        );
        Ok(model)
    }

    /// The model `reference` names, sampling as `sampling` says; `None`
    /// when its sampling cannot be set here: a model reached over an API
    /// (its server samples) or one handed in rather than loaded.
    pub fn resampled(
        &self,
        reference: &ModelRef,
        sampling: Sampling,
    ) -> Result<Option<Model>, CampaignError> {
        let model = self.model(reference)?;
        let provider = self
            .lock_models()
            .get(reference)
            .and_then(|held| held.loaded.as_ref())
            .and_then(|loaded| loaded.resampled(sampling));
        Ok(provider.map(|provider| Model { provider, ..model }))
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
