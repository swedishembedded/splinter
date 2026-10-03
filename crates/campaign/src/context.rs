// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What a command works with, in two parts.
//!
//! A [`Runtime`] is what a process shares across its commands: the
//! configuration and state root, the clock that stamps records, the async
//! runtime model runs block on, the models loaded so far, and the
//! environments tasks are solved in. A [`Context`] is one command's view of
//! it: the release each policy alias pointed at when the command first asked,
//! and where the command reports its stages. A REPL builds one runtime and a
//! context per sentence, so a model loaded for one sentence is there for the
//! next, and an alias that moved in between is resolved afresh.
//!
//! A model is loaded once per runtime and adapter and shared by every stage
//! that names it; dropping the runtime stops a local generation still
//! running before the process exits. A caller that already holds a model - a
//! test's scripted one - hands it in with [`Context::with_model`] instead.
//!
//! Local models go on the runtime's [`Residency`]: one resident copy of each
//! base, shared by every model on it - the policy and the release gate's two
//! arms differ only by adapter, and never hold two bases. Work that needs the
//! device for itself - a fine-tune, which loads its own copy, and the gate's
//! serve check, a separate process - first releases every resident base
//! ([`Runtime::release_bases`]); the next generation of a model reloads its
//! base.
//!
//! `policy:<alias>` is resolved once per context, on first use: the release
//! the alias points at then is the one every stage of the command uses,
//! however the alias moves meanwhile ([`Context::policy_pin`]). Only the
//! context's own release or rollback moves the pin along.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::Serialize;

use splinter_agent::solve::Model;
use splinter_core::clock::{Clock, SystemClock};
use splinter_core::experience::Environment;
use splinter_data::DatasetStore;
use splinter_model::{LoadedModel, ModelSelection, Residency, Sampling};
use splinter_sandbox::{
    Limits, ProcessSandbox, ResolvedEnvironment, RuntimeEnvironment, RuntimeRegistry, Sandbox,
    SandboxError,
};
use splinter_store::artifacts::ArtifactStore;
use splinter_store::experiences::ExperienceStore;
use splinter_store::sources::SourceStore;
use splinter_store::tasks::TaskStore;
use splinter_store::workspace::Workspace;
use splinter_store::StateRoot;

use crate::answers::AnswerStore;
use crate::config::Config;
use crate::error::CampaignError;
use crate::release::ReleaseStore;
use splinter_core::model_ref::ModelRef;
use splinter_core::release::ReleaseId;

/// How long a local model may stay silent between two stream chunks: as
/// long as its run lasts. sven's idle limit guards a remote wire going
/// stale; a local provider is silent through its prefill for as long as the
/// prompt takes, and no chunk in between is honest to invent, so the run's
/// own deadline is the bound (a solve caps the limit at its deadline).
const LOCAL_STREAM_IDLE: Duration = Duration::MAX;

/// Receives each stage's summary as it finishes: `(stage, summary)`.
pub type Progress = Box<dyn Fn(&str, &serde_json::Value) + Send + Sync>;

/// A loaded model: what a command runs on, and what keeps it alive.
struct Held {
    model: Model,
    /// The loaded model; dropping it quiesces a local device.
    loaded: LoadedModel,
}

/// Where a loaded model is cached: the reference that named it and the
/// adapter it was resolved to, so two commands that pinned different
/// releases of one alias never share a model.
type ModelKey = (ModelRef, Option<PathBuf>);

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

/// The services every command of a process shares: the configuration and
/// state root, the clock, the async runtime model runs block on, the models
/// loaded so far and the bases they sit on, and the environments tasks are
/// solved in. Built once; each command works through its own [`Context`].
pub struct Runtime {
    config: Config,
    workspace: Workspace,
    allow_remote: bool,
    clock: Box<dyn Clock + Send + Sync>,
    runtime: tokio::runtime::Runtime,
    handed_in: Mutex<HashMap<ModelRef, Model>>,
    models: Mutex<HashMap<ModelKey, Held>>,
    residency: Residency,
    environments: Environments,
}

impl Runtime {
    /// A runtime over `config`, with `allow_remote` as the command line's
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
            handed_in: Mutex::new(HashMap::new()),
            models: Mutex::new(HashMap::new()),
            residency: Residency::default(),
            environments,
        })
    }

    /// The same runtime keeping its local models' bases on `residency`
    /// instead of loading them with brain: a spec's scripted bases.
    #[must_use]
    pub fn with_residency(mut self, residency: Residency) -> Self {
        self.residency = residency;
        self
    }

    /// The same runtime stamping records with `clock`.
    #[must_use]
    pub fn with_clock(mut self, clock: Box<dyn Clock + Send + Sync>) -> Self {
        self.clock = clock;
        self
    }

    /// A context for one command on this runtime.
    #[must_use]
    pub fn context(self: &Arc<Self>) -> Context {
        Context::on(Arc::clone(self))
    }

    /// Answers `reference` with `model` from now on, instead of loading it.
    pub fn add_model(&self, reference: ModelRef, model: Model) {
        self.lock_handed_in().insert(reference, model);
    }

    /// How many bases are resident on the device for this runtime's
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
            // Best effort, as on drop: a generation outliving the grace
            // is cancelling, and the release waits for it to end.
            let _ = held.loaded.quiesce();
        }
        self.residency.release_all();
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

    /// The releases and the aliases that point at them.
    #[must_use]
    pub fn releases(&self) -> ReleaseStore {
        ReleaseStore::new(&self.workspace, self.root())
    }

    /// The bulk files tools need: adapters, the records brain trains on.
    #[must_use]
    pub fn artifacts(&self) -> ArtifactStore {
        ArtifactStore::new(&self.workspace, self.root())
    }

    /// The answers `ask` gave.
    #[must_use]
    pub fn answers(&self) -> AnswerStore {
        AnswerStore::new(&self.workspace)
    }

    /// The dataset store.
    #[must_use]
    pub fn datasets(&self) -> DatasetStore {
        DatasetStore::new(&self.workspace, self.root())
    }

    /// The environments tasks are solved and checked in.
    #[must_use]
    pub fn environments(&self) -> &Environments {
        &self.environments
    }

    /// Runs `future` to completion on the runtime. Must not be called from
    /// inside an async task.
    pub(crate) fn block_on<F: std::future::Future>(&self, future: F) -> F::Output {
        self.runtime.block_on(future)
    }

    /// A handle to the runtime, for the lower crates that block on it
    /// themselves.
    pub(crate) fn handle(&self) -> tokio::runtime::Handle {
        self.runtime.handle().clone()
    }

    /// The model `reference` names, resolved to `selection` and `adapter`:
    /// the one handed in for it, else the loaded one, loading it on first
    /// use.
    fn model(
        &self,
        reference: &ModelRef,
        adapter: Option<PathBuf>,
        selection: &ModelSelection,
    ) -> Result<Model, CampaignError> {
        if let Some(model) = self.lock_handed_in().get(reference) {
            return Ok(model.clone());
        }
        let key = (reference.clone(), adapter);
        if let Some(held) = self.lock_models().get(&key) {
            return Ok(held.model.clone());
        }
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
            key,
            Held {
                model: model.clone(),
                loaded,
            },
        );
        Ok(model)
    }

    /// The model `reference` names, sampling as `sampling` says; `None`
    /// when its sampling cannot be set here: a model reached over an API
    /// (its server samples) or one handed in rather than loaded.
    fn resampled(
        &self,
        reference: &ModelRef,
        adapter: Option<PathBuf>,
        selection: &ModelSelection,
        sampling: Sampling,
    ) -> Result<Option<Model>, CampaignError> {
        let model = self.model(reference, adapter.clone(), selection)?;
        let provider = self
            .lock_models()
            .get(&(reference.clone(), adapter))
            .and_then(|held| held.loaded.resampled(sampling));
        Ok(provider.map(|provider| Model { provider, ..model }))
    }

    /// Drops every model loaded for `reference`, whatever adapter it was on:
    /// the models stay gone until next used, and a base goes with its last
    /// model. A model handed in stays.
    fn drop_loaded(&self, reference: &ModelRef) {
        // Dropped outside the lock: dropping quiesces a running generation.
        let dropped: Vec<Held> = {
            let mut models = self.lock_models();
            let keys: Vec<ModelKey> = models
                .keys()
                .filter(|(held, _)| held == reference)
                .cloned()
                .collect();
            keys.iter().filter_map(|key| models.remove(key)).collect()
        };
        drop(dropped);
    }

    fn lock_models(&self) -> std::sync::MutexGuard<'_, HashMap<ModelKey, Held>> {
        // A poisoned lock only means a load panicked; the map is still
        // consistent, since an entry is inserted whole.
        self.models
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn lock_handed_in(&self) -> std::sync::MutexGuard<'_, HashMap<ModelRef, Model>> {
        self.handed_in
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// What one command works with: the shared [`Runtime`], the release each
/// policy alias pointed at when the command first asked, and where the
/// command reports its stages.
///
/// `policy:<alias>` is resolved once per context, on first use: the release
/// the alias points at then is the one every stage of the command uses,
/// however the alias moves meanwhile ([`Context::policy_pin`]). Only the
/// context's own release or rollback moves the pin along. A later command
/// has a context of its own and resolves the alias afresh.
pub struct Context {
    runtime: Arc<Runtime>,
    pins: Mutex<BTreeMap<String, Option<PolicyPin>>>,
    progress: Option<Progress>,
}

impl std::ops::Deref for Context {
    type Target = Runtime;

    fn deref(&self) -> &Runtime {
        &self.runtime
    }
}

impl Context {
    /// A context for one command on `runtime`.
    #[must_use]
    pub fn on(runtime: Arc<Runtime>) -> Self {
        Self {
            runtime,
            pins: Mutex::new(BTreeMap::new()),
            progress: None,
        }
    }

    /// A context over its own runtime on `config`, with `allow_remote` as
    /// the command line's network opt-in, stamping records with the wall
    /// clock: for a process that runs one command.
    pub fn new(config: Config, allow_remote: bool) -> Result<Self, CampaignError> {
        Ok(Self::on(Arc::new(Runtime::new(config, allow_remote)?)))
    }

    /// The runtime this context works on, to build the next command's
    /// context on.
    #[must_use]
    pub fn runtime(&self) -> &Arc<Runtime> {
        &self.runtime
    }

    /// The runtime this context was built on, owned by this context alone:
    /// the builders below run before it is shared.
    fn sole_runtime(&mut self) -> &mut Runtime {
        Arc::get_mut(&mut self.runtime)
            .unwrap_or_else(|| unreachable!("a context is configured before its runtime is shared"))
    }

    /// The same context answering `reference` with `model` instead of
    /// loading it.
    #[must_use]
    pub fn with_model(self, reference: ModelRef, model: Model) -> Self {
        self.add_model(reference, model);
        self
    }

    /// The same context keeping its local models' bases on `residency`
    /// instead of loading them with brain: a spec's scripted bases. Only a
    /// context that owns its runtime alone can change it.
    #[must_use]
    pub fn with_residency(mut self, residency: Residency) -> Self {
        self.sole_runtime().residency = residency;
        self
    }

    /// The same context stamping records with `clock`. Only a context that
    /// owns its runtime alone can change it.
    #[must_use]
    pub fn with_clock(mut self, clock: Box<dyn Clock + Send + Sync>) -> Self {
        self.sole_runtime().clock = clock;
        self
    }

    /// The same context reporting each finished stage to `progress`.
    #[must_use]
    pub fn with_progress(mut self, progress: Progress) -> Self {
        self.progress = Some(progress);
        self
    }

    /// The selection `reference` resolves to, without loading it: refused
    /// for a remote model without the network opt-in, and for a policy
    /// alias other than `default` that points at no release.
    pub fn selection(&self, reference: &ModelRef) -> Result<ModelSelection, CampaignError> {
        Ok(self.resolve(reference)?.1)
    }

    /// The adapter and selection `reference` resolves to under this
    /// context's pins.
    fn resolve(
        &self,
        reference: &ModelRef,
    ) -> Result<(Option<PathBuf>, ModelSelection), CampaignError> {
        let adapter = match reference {
            ModelRef::Policy(alias) => {
                let pin = self.policy_pin(alias)?;
                if pin.is_none() && alias != splinter_core::model_ref::POLICY_DEFAULT {
                    return Err(CampaignError::NotFound {
                        what: "release alias",
                        id: alias.clone(),
                    });
                }
                pin.map(|p| p.adapter)
            }
            _ => None,
        };
        let selection = crate::model_ref::resolve(
            reference,
            self.config(),
            self.allow_remote(),
            adapter.as_deref(),
        )?;
        Ok((adapter, selection))
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
        let store = self.releases();
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

    /// Drops `alias`'s pin and the policy model this context's runtime
    /// loaded for it, so the next use resolves the alias afresh: for the
    /// context's own release or rollback, which moved it.
    pub(crate) fn repin_policy(&self, alias: &str) {
        self.pins
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(alias);
        self.runtime
            .drop_loaded(&ModelRef::Policy(alias.to_string()));
    }

    /// The model `reference` names, loaded on first use.
    pub fn model(&self, reference: &ModelRef) -> Result<Model, CampaignError> {
        let (adapter, selection) = self.resolve(reference)?;
        self.runtime.model(reference, adapter, &selection)
    }

    /// The model `reference` names, sampling as `sampling` says; `None`
    /// when its sampling cannot be set here: a model reached over an API
    /// (its server samples) or one handed in rather than loaded.
    pub fn resampled(
        &self,
        reference: &ModelRef,
        sampling: Sampling,
    ) -> Result<Option<Model>, CampaignError> {
        let (adapter, selection) = self.resolve(reference)?;
        self.runtime
            .resampled(reference, adapter, &selection, sampling)
    }

    /// Reports a finished stage to the progress receiver, if any.
    pub(crate) fn report_stage(&self, stage: &str, summary: &serde_json::Value) {
        if let Some(progress) = &self.progress {
            progress(stage, summary);
        }
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
