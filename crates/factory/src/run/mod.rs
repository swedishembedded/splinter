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

mod compute;
pub mod dataset;
mod directory;
pub mod eval;
pub mod import;
mod pass;
pub mod play;
pub mod release;
pub mod rollout;
mod rollout_report;
pub mod tasks;
pub mod training;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use splinter_core::clock::{unix_now, SystemClock};
use splinter_core::digest::Digest;
use splinter_core::factory::identity::StageIdentity;
use splinter_store::factory::products::{ProductError, ProductManifest, ProductStore, Staging};
use splinter_store::factory::runstate::{Fence, RunState, RunStateError, StageOutcome, StageState};

use crate::backend::{BackendError, ExecutionBackend};
use crate::ops::names;
use crate::plan::PlannedStage;
use crate::resolve::{Lock, Resolved};
use crate::training::Trainer;
use crate::tree::TreeError;

pub use directory::{fork, report_of, status, verify, StageStatus};

/// The file a run's lock is kept in.
pub const LOCK_FILE: &str = "recipe.lock.json";
/// The file a run's recipe is kept in.
pub const RECIPE_FILE: &str = "recipe.yaml";
/// The run's report for a person.
pub const REPORT_FILE: &str = "report.md";
/// Where the run was started from.
pub const ORIGIN_FILE: &str = "origin.json";
/// The file whose presence asks a running controller to stop.
pub const CANCEL_FILE: &str = "state/cancel";

/// Where a run was started from: the repository its recipe's paths are
/// relative to.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunOrigin {
    /// The repository root.
    pub repository: PathBuf,
}

/// The repository and recipe text a run was started with, for a resume.
///
/// # Errors
/// The run's recipe or origin cannot be read.
pub fn started_with(run: &Path) -> Result<(PathBuf, String), RunError> {
    require_run(run)?;
    let origin: RunOrigin = read_json(&run.join(ORIGIN_FILE))?;
    let recipe = run.join(RECIPE_FILE);
    let text = std::fs::read_to_string(&recipe).map_err(io(&recipe))?;
    Ok((origin.repository, text))
}

/// Ask the controller of `run` to stop at its next stage or batch boundary;
/// what is committed stays, and `resume` continues.
///
/// # Errors
/// The request cannot be written.
pub fn request_cancel(run: &Path) -> Result<(), RunError> {
    require_run(run)?;
    let path = run.join(CANCEL_FILE);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(io(parent))?;
    }
    std::fs::write(&path, b"cancel\n").map_err(io(&path))
}

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
    /// The directory is not a factory run.
    #[error("{}", not_a_run_message(.0))]
    NotARun(PathBuf),
    /// A file of the run cannot be written or read.
    #[error("{path}: {message}")]
    Io {
        /// The file.
        path: PathBuf,
        /// What failed.
        message: String,
    },
}

fn not_a_run_message(dir: &Path) -> String {
    if dir.exists() {
        format!(
            "{} is not a factory run: it holds no {LOCK_FILE} (a run is the directory `splinter run --out` wrote)",
            dir.display()
        )
    } else {
        format!(
            "{} does not exist: give the directory `splinter run --out` wrote",
            dir.display()
        )
    }
}

/// Whether `dir` is a factory run directory.
#[must_use]
pub fn is_run(dir: &Path) -> bool {
    dir.join(LOCK_FILE).is_file()
}

/// `Ok` when `dir` is a factory run; never creates anything in it.
///
/// # Errors
/// [`RunError::NotARun`].
pub fn require_run(dir: &Path) -> Result<(), RunError> {
    if is_run(dir) {
        Ok(())
    } else {
        Err(RunError::NotARun(dir.to_path_buf()))
    }
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
    /// Reaches models in process; needed when a computed stage names a
    /// model role.
    pub models: Option<&'a dyn crate::execute::Models>,
}

/// One stage as the run recorded it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageEntry {
    /// The stage key.
    pub key: String,
    /// The operation it ran.
    pub operation: String,
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

/// Where [`Factory::begin`] leaves a stage.
pub(crate) enum Begun {
    /// Its product is committed already.
    Reused(ProductManifest),
    /// It runs: produce into `staging`, then [`Factory::finish`].
    Fresh {
        /// Its run-state version.
        version: u64,
        /// Where to produce.
        staging: Staging,
    },
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
                std::fs::write(run.join(RECIPE_FILE), &resolved.recipe_text)
                    .map_err(io(&run.join(RECIPE_FILE)))?;
                let origin = RunOrigin {
                    repository: resolved.repository.clone(),
                };
                write_json(&run.join(ORIGIN_FILE), &origin)?;
            }
            Err(e) => return Err(io(&lock_path)(e)),
        }
        // A variant the trainer would refuse is refused before any rollout
        // spends anything.
        if let Some(trainer) = services.trainer {
            for stage in resolved.plan.of_operation(crate::ops::names::TRAIN) {
                let params: crate::ops::TrainParams =
                    stage.params().map_err(|e| stage_error(&stage.key, e))?;
                trainer
                    .resolve(&params.method, &params.params)
                    .map_err(|e| stage_error(&stage.key, e))?;
            }
        }
        let state = RunState::open(run)?;
        let fence = match state.acquire(owner, LEASE_SECONDS, unix_now()) {
            Err(RunStateError::LeaseHeld { owner: holder, .. }) if exited_here(owner, &holder) => {
                state.take_over(owner, &holder, LEASE_SECONDS, unix_now())?
            }
            other => other?,
        };
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
        }
        .with_cancel_file())
    }

    /// Stop between stages and between rollout batches once `cancelled`
    /// returns true; what is committed stays, and a resume continues.
    #[must_use]
    pub fn with_cancel(mut self, cancelled: impl Fn() -> bool + 'a) -> Self {
        self.cancel = Box::new(cancelled);
        self
    }

    /// Stop once the run's cancel file appears (`request_cancel`), and
    /// forget an earlier request: opening the run is the resume.
    fn with_cancel_file(self) -> Self {
        let path = self.run.join(CANCEL_FILE);
        let _ = std::fs::remove_file(&path);
        self.with_cancel(move || path.exists())
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
            recipe: self.resolved.name().to_owned(),
            lock: self.resolved.lock.clone(),
            stages: entries,
        };
        write_json(&self.run.join(RUN_MANIFEST), &manifest)?;
        let report = self.run.join(REPORT_FILE);
        std::fs::write(&report, report_of(&self.run, &manifest)).map_err(io(&report))?;
        Ok(manifest)
    }

    fn stage(&mut self, stage: &PlannedStage) -> Result<StageEntry, RunError> {
        let (identity, (manifest, computed)) = match stage.operation.as_str() {
            names::PIN
            | names::ATTEMPTS
            | names::IMPORT
            | names::PROJECT
            | names::TRAIN
            | names::EVALUATION
            | names::PROMOTION => self.built_in(stage)?,
            _ => compute::stage(self, stage)?,
        };
        let id = identity.digest();
        let product = manifest.digest();
        self.done.insert(stage.key.clone(), (id.clone(), manifest));
        Ok(StageEntry {
            key: stage.key.clone(),
            operation: stage.operation.clone(),
            stage: id,
            product,
            computed,
        })
    }

    /// A built-in operation's stage: its identity and committed product.
    fn built_in(
        &self,
        stage: &PlannedStage,
    ) -> Result<(StageIdentity, (ProductManifest, bool)), RunError> {
        let identity = match stage.operation.as_str() {
            names::PIN => tasks::identity(self, stage)?,
            names::ATTEMPTS => rollout::identity(self, stage)?,
            names::IMPORT => import::identity(self, stage)?,
            names::PROJECT => dataset::identity(self, stage)?,
            names::TRAIN => training::identity(self, stage)?,
            names::EVALUATION => eval::identity(self, stage)?,
            _ => release::identity(self, stage)?,
        };
        let produce = |factory: &Self, staging: &Staging| match stage.operation.as_str() {
            names::PIN => tasks::produce(factory, stage, staging),
            names::ATTEMPTS => rollout::produce(factory, stage, &identity, staging),
            names::IMPORT => import::produce(factory, stage, staging),
            names::PROJECT => dataset::produce(factory, stage, staging),
            names::TRAIN => training::produce(factory, stage, &identity, staging),
            names::EVALUATION => eval::produce(factory, stage, &identity, staging),
            _ => release::produce(factory, stage, staging),
        };
        let committed = self.commit(&stage.key, &stage.operation, &identity, &produce)?;
        Ok((identity, committed))
    }

    /// The committed product of the stage `identity` viewed at `key`:
    /// reused when the product store holds it, produced into a staging area
    /// and published otherwise. The run state follows; a failed production
    /// is recorded as failed and leaves nothing committed. Returns the
    /// manifest and whether this call produced it.
    pub(crate) fn commit(
        &self,
        key: &str,
        operation: &str,
        identity: &StageIdentity,
        produce: &dyn Fn(&Self, &Staging) -> Result<(), RunError>,
    ) -> Result<(ProductManifest, bool), RunError> {
        match self.begin(key, operation, identity)? {
            Begun::Reused(manifest) => Ok((manifest, false)),
            Begun::Fresh { version, staging } => {
                let produced = produce(self, &staging);
                self.finish(key, identity, version, staging, produced)
                    .map(|manifest| (manifest, true))
            }
        }
    }

    /// The first half of [`Self::commit`]: the product of `identity`
    /// reused, or the stage recorded as running with an empty staging area
    /// to produce it in.
    pub(crate) fn begin(
        &self,
        key: &str,
        operation: &str,
        identity: &StageIdentity,
    ) -> Result<Begun, RunError> {
        let id = identity.digest();
        let record = self
            .state
            .upsert_pending(&self.fence, &id, key, operation, unix_now())?;
        if let Ok(manifest) = self.products.ensure_view(&id, &self.run.join(key)) {
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
            return Ok(Begun::Reused(manifest));
        }
        let version = self.to_running(&id, record.version, record.state)?;
        let staging = self.products.begin_staging()?;
        Ok(Begun::Fresh { version, staging })
    }

    /// The second half of [`Self::commit`]: what was `produced` into
    /// `staging` published as the product of `identity`, or the stage
    /// recorded as failed (a cancelled one stays running, for the resume).
    pub(crate) fn finish(
        &self,
        key: &str,
        identity: &StageIdentity,
        version: u64,
        staging: Staging,
        produced: Result<(), RunError>,
    ) -> Result<ProductManifest, RunError> {
        let id = identity.digest();
        if let Err(error) =
            produced.and_then(|()| write_json(&staging.path().join("stage.json"), identity))
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
        Ok(manifest)
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

impl Drop for Factory<'_> {
    /// The lease is given up when the controller ends, so the next one
    /// (a resume) need not wait for it to lapse.
    fn drop(&mut self) {
        let _ = self.state.release(&self.fence, unix_now());
    }
}

/// Whether the lease holder `holder` is a process of this host (owners are
/// `<host>:<pid>`) that no longer runs: a crashed controller, whose lease
/// another controller of the same host may take over at once.
fn exited_here(owner: &str, holder: &str) -> bool {
    let (Some((host, _)), Some((holder_host, pid))) =
        (owner.rsplit_once(':'), holder.rsplit_once(':'))
    else {
        return false;
    };
    host == holder_host && pid.parse::<u32>().is_ok() && !Path::new("/proc").join(pid).exists()
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
    operation: &str,
    name: &str,
    parameters: serde_json::Value,
    inputs: Vec<(String, Digest)>,
    extra: &str,
    seed: Option<u64>,
) -> StageIdentity {
    StageIdentity::new(
        operation,
        name,
        parameters,
        inputs,
        Factory::implementation(extra),
        seed,
    )
}
