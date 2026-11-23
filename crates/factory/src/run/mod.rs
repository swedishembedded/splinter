// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements experiment controllers that resume after
// any crash without losing or repeating committed work, for its clients. If
// your team needs expertise in durable machine-learning orchestration, you
// can procure our services by sending an email to info@swedishembedded.com.

//! The controller: one run directory, one lease, every stage of the plan in
//! order, each skipped when its product is already committed.
//!
//! A run directory holds the recipe and its lock, the run state
//! (`state/controller.db`, `state/events.jsonl`), the content-addressed
//! products and their readable views (`tasks/`, `rollouts/<pass>/`,
//! `datasets/<name>/`, ...), and `work/`, where unfinished stages keep what
//! they have done so far. A stage is done exactly when its product is
//! committed: the product store's commit point decides, and the run state
//! records it.
//!
//! Reopening a run compares the repository against the lock and refuses to
//! continue on changed inputs; a fresh run (or a fork) is how a change is
//! made.

pub mod dataset;
pub mod eval;
pub mod release;
pub mod rollout;
pub mod tasks;
pub mod training;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use splinter_core::clock::{unix_now, SystemClock};
use splinter_core::digest::Digest;
use splinter_core::factory::identity::{StageIdentity, StageKind};
use splinter_store::factory::products::{ProductError, ProductManifest, ProductStore, Staging};
use splinter_store::factory::runstate::{Fence, RunState, RunStateError, StageOutcome, StageState};

use crate::backend::{BackendError, ExecutionBackend};
use crate::plan::PlannedStage;
use crate::resolve::{Lock, Resolved};
use crate::training::Trainer;
use crate::tree::TreeError;

/// The file a run's lock is kept in.
pub const LOCK_FILE: &str = "recipe.lock.json";
/// The file a run's recipe is kept in.
pub const RECIPE_FILE: &str = "recipe.yaml";
/// The run's own manifest: every stage and its product.
pub const RUN_MANIFEST: &str = "manifest.json";

/// Why a run stopped.
#[derive(Debug, thiserror::Error)]
pub enum RunError {
    /// The inputs changed since the run started.
    #[error("the run's inputs changed since it started; start a new run or fork this one:\n  {}", .0.join("\n  "))]
    InputsChanged(Vec<String>),
    /// The run state refused.
    #[error(transparent)]
    State(#[from] RunStateError),
    /// The product store refused.
    #[error(transparent)]
    Product(#[from] ProductError),
    /// The execution backend failed.
    #[error(transparent)]
    Backend(#[from] BackendError),
    /// A tree could not be copied or digested.
    #[error(transparent)]
    Tree(#[from] TreeError),
    /// A stage failed.
    #[error("stage {stage}: {message}")]
    Stage {
        /// The stage key.
        stage: String,
        /// Why.
        message: String,
    },
    /// The run was cancelled; what is committed stays.
    #[error("the run was cancelled; resume it to continue")]
    Cancelled,
    /// A file of the run cannot be written or read.
    #[error("{path}: {message}")]
    Io {
        /// The file.
        path: PathBuf,
        /// What failed.
        message: String,
    },
}

pub(crate) fn io(path: &Path) -> impl Fn(std::io::Error) -> RunError + '_ {
    move |e| RunError::Io {
        path: path.to_path_buf(),
        message: e.to_string(),
    }
}

pub(crate) fn stage_error(stage: &str, message: impl ToString) -> RunError {
    RunError::Stage {
        stage: stage.to_owned(),
        message: message.to_string(),
    }
}

/// What a run uses that is not in its recipe.
#[derive(Clone, Copy)]
pub struct Services<'a> {
    /// Executes rollout attempts.
    pub backend: &'a dyn ExecutionBackend,
    /// Trains and serves candidates; needed when the recipe trains.
    pub trainer: Option<&'a dyn Trainer>,
}

/// One stage as the run recorded it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageEntry {
    /// The stage key.
    pub key: String,
    /// Its kind.
    pub kind: StageKind,
    /// Its identity.
    pub stage: Digest,
    /// Its product manifest's digest.
    pub product: Digest,
    /// Whether this invocation computed it (false: it was already committed).
    pub computed: bool,
}

/// The run's manifest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunManifest {
    /// Always `splinter.run/1`.
    pub schema: String,
    /// The recipe's name.
    pub recipe: String,
    /// The lock the run was started on.
    pub lock: Lock,
    /// Every completed stage, in plan order.
    pub stages: Vec<StageEntry>,
}

/// A controller over one run directory.
pub struct Factory<'a> {
    pub(crate) resolved: &'a Resolved,
    pub(crate) run: PathBuf,
    pub(crate) state: RunState,
    pub(crate) products: ProductStore,
    pub(crate) fence: Fence,
    pub(crate) services: Services<'a>,
    /// Committed products by stage key, as this invocation knows them.
    pub(crate) done: BTreeMap<String, (Digest, ProductManifest)>,
    cancel: Box<dyn Fn() -> bool + 'a>,
}

/// How long a controller's lease lasts without renewal, in seconds.
pub const LEASE_SECONDS: u64 = 600;

impl<'a> Factory<'a> {
    /// Open (or start) the run in `run` for `resolved`, as controller
    /// `owner`.
    ///
    /// # Errors
    /// The run was started on other inputs, another controller holds it, or
    /// it cannot be written.
    pub fn open(
        resolved: &'a Resolved,
        run: &Path,
        services: Services<'a>,
        owner: &str,
    ) -> Result<Self, RunError> {
        std::fs::create_dir_all(run).map_err(io(run))?;
        let lock_path = run.join(LOCK_FILE);
        match std::fs::read_to_string(&lock_path) {
            Ok(text) => {
                let started: Lock = serde_json::from_str(&text).map_err(|e| RunError::Io {
                    path: lock_path.clone(),
                    message: e.to_string(),
                })?;
                let changes = started.changes(&resolved.lock);
                if !changes.is_empty() {
                    return Err(RunError::InputsChanged(changes));
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                write_json(&lock_path, &resolved.lock)?;
                let recipe = serde_yaml::to_string(&resolved.recipe).map_err(|e| RunError::Io {
                    path: run.join(RECIPE_FILE),
                    message: e.to_string(),
                })?;
                std::fs::write(run.join(RECIPE_FILE), recipe)
                    .map_err(io(&run.join(RECIPE_FILE)))?;
            }
            Err(e) => return Err(io(&lock_path)(e)),
        }
        let state = RunState::open(run)?;
        let fence = state.acquire(owner, LEASE_SECONDS, unix_now())?;
        let products = ProductStore::new(run, Box::new(SystemClock));
        products.recover()?;
        Ok(Self {
            resolved,
            run: run.to_path_buf(),
            state,
            products,
            fence,
            services,
            done: BTreeMap::new(),
            cancel: Box::new(|| false),
        })
    }

    /// Stop between stages and between rollout batches once `cancelled`
    /// returns true; what is committed stays, and a resume continues.
    #[must_use]
    pub fn with_cancel(mut self, cancelled: impl Fn() -> bool + 'a) -> Self {
        self.cancel = Box::new(cancelled);
        self
    }

    pub(crate) fn cancelled(&self) -> Result<(), RunError> {
        if (self.cancel)() {
            Err(RunError::Cancelled)
        } else {
            Ok(())
        }
    }

    /// Run every stage of the plan, in order; stop at the first that fails.
    ///
    /// # Errors
    /// The first failing stage, recorded as failed in the run state.
    pub fn run(&mut self) -> Result<RunManifest, RunError> {
        let stages = self.resolved.plan.stages.clone();
        let mut entries = Vec::new();
        for stage in &stages {
            self.cancelled()?;
            let entry = self.stage(stage)?;
            entries.push(entry);
        }
        let manifest = RunManifest {
            schema: "splinter.run/1".into(),
            recipe: self.resolved.recipe.name.clone(),
            lock: self.resolved.lock.clone(),
            stages: entries,
        };
        write_json(&self.run.join(RUN_MANIFEST), &manifest)?;
        Ok(manifest)
    }

    fn stage(&mut self, stage: &PlannedStage) -> Result<StageEntry, RunError> {
        let identity = match stage.kind {
            StageKind::Tasks => tasks::identity(self)?,
            StageKind::Rollout => rollout::identity(self, stage)?,
            StageKind::Dataset => dataset::identity(self, stage)?,
            StageKind::Training => training::identity(self, stage)?,
            StageKind::Eval => eval::identity(self, stage)?,
            StageKind::Release => release::identity(self, stage)?,
        };
        let produce = |factory: &Self, staging: &Staging| match stage.kind {
            StageKind::Tasks => tasks::produce(factory, staging),
            StageKind::Rollout => rollout::produce(factory, stage, &identity, staging),
            StageKind::Dataset => dataset::produce(factory, stage, staging),
            StageKind::Training => training::produce(factory, stage, &identity, staging),
            StageKind::Eval => eval::produce(factory, stage, &identity, staging),
            StageKind::Release => release::produce(factory, stage, staging),
        };
        let (manifest, computed) = self.commit(&stage.key, stage.kind, &identity, &produce)?;
        let id = identity.digest();
        let product = manifest.digest();
        self.done.insert(stage.key.clone(), (id.clone(), manifest));
        Ok(StageEntry {
            key: stage.key.clone(),
            kind: stage.kind,
            stage: id,
            product,
            computed,
        })
    }

    /// The committed product of the stage `identity` viewed at `key`:
    /// reused when the product store holds it, produced into a staging area
    /// and published otherwise. The run state follows; a failed production
    /// is recorded as failed and leaves nothing committed. Returns the
    /// manifest and whether this call produced it.
    pub(crate) fn commit(
        &self,
        key: &str,
        kind: StageKind,
        identity: &StageIdentity,
        produce: &dyn Fn(&Self, &Staging) -> Result<(), RunError>,
    ) -> Result<(ProductManifest, bool), RunError> {
        let id = identity.digest();
        let record =
            self.state
                .upsert_pending(&self.fence, &id, key, &format!("{kind:?}"), unix_now())?;
        if let Ok(manifest) = self.products.manifest(&id) {
            if record.state != StageState::Completed {
                let running = self.to_running(&id, record.version, record.state)?;
                self.state.transition(
                    &self.fence,
                    &id,
                    running,
                    StageState::Completed,
                    StageOutcome::Product(manifest.digest()),
                    unix_now(),
                )?;
            }
            return Ok((manifest, false));
        }
        let version = self.to_running(&id, record.version, record.state)?;
        let staging = self.products.begin_staging()?;
        if let Err(error) = produce(self, &staging)
            .and_then(|()| write_json(&staging.path().join("stage.json"), identity))
        {
            if !matches!(error, RunError::Cancelled) {
                let failed = StageOutcome::Error(error.to_string());
                self.state.transition(
                    &self.fence,
                    &id,
                    version,
                    StageState::Failed,
                    failed,
                    unix_now(),
                )?;
            }
            return Err(error);
        }
        let manifest = self
            .products
            .publish(&id, key, staging, &self.run.join(key))?;
        self.state.transition(
            &self.fence,
            &id,
            version,
            StageState::Completed,
            StageOutcome::Product(manifest.digest()),
            unix_now(),
        )?;
        let work = self.work(key);
        if work.exists() {
            std::fs::remove_dir_all(&work).map_err(io(&work))?;
        }
        Ok((manifest, true))
    }

    /// Move a stage to running from wherever it stands; returns its version.
    fn to_running(&self, id: &Digest, version: u64, state: StageState) -> Result<u64, RunError> {
        match state {
            StageState::Running => Ok(version),
            StageState::Pending | StageState::Failed => Ok(self
                .state
                .transition(
                    &self.fence,
                    id,
                    version,
                    StageState::Running,
                    StageOutcome::None,
                    unix_now(),
                )?
                .version),
            StageState::Completed => Ok(version),
        }
    }

    /// Where an unfinished stage keeps its work.
    #[must_use]
    pub fn work(&self, key: &str) -> PathBuf {
        self.run.join("work").join(key)
    }

    /// The committed product of the stage `key`, with its stage identity.
    pub(crate) fn product(&self, key: &str) -> Result<&(Digest, ProductManifest), RunError> {
        self.done
            .get(key)
            .ok_or_else(|| stage_error(key, "is not completed"))
    }

    /// The readable view of the completed stage `key`.
    #[must_use]
    pub fn view(&self, key: &str) -> PathBuf {
        self.run.join(key)
    }

    /// The implementation string of a stage computed in this process.
    pub(crate) fn implementation(extra: &str) -> String {
        let own = format!("splinter {}", env!("CARGO_PKG_VERSION"));
        if extra.is_empty() {
            own
        } else {
            format!("{own}; {extra}")
        }
    }
}

/// Write `value` as pretty JSON to `path`.
pub(crate) fn write_json(path: &Path, value: &impl Serialize) -> Result<(), RunError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(io(parent))?;
    }
    let text = serde_json::to_string_pretty(value).map_err(|e| RunError::Io {
        path: path.to_path_buf(),
        message: e.to_string(),
    })?;
    std::fs::write(path, format!("{text}\n")).map_err(io(path))
}

/// Read JSON from `path`.
pub(crate) fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, RunError> {
    let text = std::fs::read_to_string(path).map_err(io(path))?;
    serde_json::from_str(&text).map_err(|e| RunError::Io {
        path: path.to_path_buf(),
        message: e.to_string(),
    })
}

/// Write `lines` as JSON lines to `path`.
pub(crate) fn write_lines<T: Serialize>(path: &Path, lines: &[T]) -> Result<(), RunError> {
    let mut text = String::new();
    for line in lines {
        text.push_str(&serde_json::to_string(line).map_err(|e| RunError::Io {
            path: path.to_path_buf(),
            message: e.to_string(),
        })?);
        text.push('\n');
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(io(parent))?;
    }
    std::fs::write(path, text).map_err(io(path))
}

/// Read JSON lines from `path`.
pub(crate) fn read_lines<T: serde::de::DeserializeOwned>(path: &Path) -> Result<Vec<T>, RunError> {
    let text = std::fs::read_to_string(path).map_err(io(path))?;
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            serde_json::from_str(l).map_err(|e| RunError::Io {
                path: path.to_path_buf(),
                message: e.to_string(),
            })
        })
        .collect()
}

/// The identity of a stage with no randomness and no parameters beyond
/// those given.
pub(crate) fn identity_of(
    kind: StageKind,
    name: &str,
    parameters: serde_json::Value,
    inputs: Vec<(String, Digest)>,
    extra: &str,
    seed: Option<u64>,
) -> StageIdentity {
    StageIdentity::new(
        kind,
        name,
        parameters,
        inputs,
        Factory::implementation(extra),
        seed,
    )
}

/// Reuse every product committed in the run `source` in the run `run`
/// (normally a new directory), so the stages a changed recipe leaves
/// unchanged are not computed again. Returns how many products were
/// brought over.
///
/// # Errors
/// A product cannot be imported.
pub fn fork(source: &Path, run: &Path) -> Result<usize, RunError> {
    std::fs::create_dir_all(run).map_err(io(run))?;
    let from = ProductStore::new(source, Box::new(SystemClock));
    let into = ProductStore::new(run, Box::new(SystemClock));
    let mut imported = 0;
    for manifest in from.list()? {
        into.import(&from, &manifest.stage, &run.join(&manifest.name))?;
        imported += 1;
    }
    Ok(imported)
}

/// Check every committed product of the run in `run` against its blobs and
/// its view; one line per problem, empty when the run is intact.
///
/// # Errors
/// The products cannot be listed.
pub fn verify(run: &Path) -> Result<Vec<String>, RunError> {
    let store = ProductStore::new(run, Box::new(SystemClock));
    let mut problems = Vec::new();
    for manifest in store.list()? {
        if let Err(e) = store.verify_product(&manifest.stage, &run.join(&manifest.name)) {
            problems.push(format!("{}: {e}", manifest.name));
        }
    }
    Ok(problems)
}

/// One stage's state as the run state records it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct StageStatus {
    /// The stage key.
    pub key: String,
    /// Its state.
    pub state: String,
    /// Its product, when committed.
    pub product: Option<Digest>,
    /// Why it failed, when it did.
    pub error: Option<String>,
    /// Its rollout slots by state, for a rollout or eval arm.
    pub slots: BTreeMap<String, usize>,
}

/// What the run in `run` has done, stage by stage, from its run state alone.
///
/// # Errors
/// The run state cannot be read.
pub fn status(run: &Path) -> Result<Vec<StageStatus>, RunError> {
    let state = RunState::open(run)?;
    let mut out = Vec::new();
    for record in state.list()? {
        let pass = record.key_pass();
        let mut slots = BTreeMap::new();
        if let Some(pass) = pass {
            for slot in state.list_slots(&pass)? {
                *slots.entry(slot.state.to_string()).or_default() += 1;
            }
        }
        out.push(StageStatus {
            key: record.name.clone(),
            state: record.state.to_string(),
            product: record.product.clone(),
            error: record.error.clone(),
            slots,
        });
    }
    Ok(out)
}

/// The slot namespace of a stage record, for the stages that have slots.
trait SlotNamespace {
    fn key_pass(&self) -> Option<String>;
}

impl SlotNamespace for splinter_store::factory::runstate::StageRecord {
    fn key_pass(&self) -> Option<String> {
        if let Some(pass) = self.name.strip_prefix("rollouts/") {
            return Some(pass.to_owned());
        }
        self.name
            .strip_prefix("eval/_arms/")
            .map(|arm| format!("eval-{arm}"))
    }
}
