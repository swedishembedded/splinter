// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements data pipelines that fan work out over large
// corpora and survive crashes without redoing finished items, for its
// clients. If your team needs expertise in resumable, content-addressed data
// factories, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Stages of an operation the controller runs as a computation
//! ([`crate::execute`]): once, or once per item of a collection.
//!
//! A recipe input a computation reads is pinned into the run first, as the
//! product `_inputs/NAME`: an item index and the input's files under
//! `files/`, copied and checked against the digest the lock holds.
//!
//! A fan-out (`foreach`) commits every item as a product of its own,
//! `_items/STAGE/ID`, whose identity names the item's content, not the
//! collection's: a resumed run reuses every item committed before it
//! stopped, and a changed item recomputes only itself. Items run on at most
//! `concurrency` workers; only this thread touches the run state and the
//! product store. An item's product holds `outcome.json` (produced, or
//! rejected and why) and the operation's output under `out/`. The stage's
//! product reduces them, in id order: each produced item's output under its
//! id, and an item index listing every item produced and every item
//! rejected.
//!
//! An item whose operation fails in a way that may pass (a timeout, a
//! dropped or refused connection, a 5xx answer) is asked again, a bounded
//! number of times after growing pauses ([`ItemRetry`]); any other failure,
//! or one that outlasts its retries, fails the item. A failed item stops
//! nothing else: every other item still runs and commits, and the stage
//! then fails naming the failed items, which a resume computes again (and
//! only them). A stage that states `tolerate_failed_items` (a count, or a
//! share of the items) within that bound records its failed items as
//! rejected, with the failure as the reason, instead of failing.

use std::cell::OnceCell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use splinter_agent::solve::Model;
use splinter_core::digest::Digest;
use splinter_core::factory::execution::ModelBinding;
use splinter_core::factory::identity::StageIdentity;
use splinter_core::factory::items::{Item, ItemIndex, Rejection, ITEMS_FILE};
use splinter_store::factory::products::{ProductManifest, Staging};

use super::models::{binding_of, serve};
use super::{io, stage_error, write_json, Begun, Factory, RunError};
use crate::champion::Champion;
use crate::execute::{Call, Execute, Outcome};
use crate::graph::Source;
use crate::plan::PlannedStage;
use crate::tree::{copy_input, entry_digest, link_tree};

/// The operation that pins a recipe input.
const INPUT: &str = "input@1";
/// What a reduction adds to its operation's implementation.
const REDUCTION: &str = "reduce 1";
/// Where an item's product keeps what the operation wrote.
const OUT: &str = "out";
/// Where an item's product says what came of it.
const OUTCOME: &str = "outcome.json";

/// What came of one item, as its product records it.
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "outcome")]
enum ItemOutcome {
    Produced,
    Rejected { reason: String },
}

/// The committed product of the computed stage `stage`, with its identity.
pub(super) fn stage(
    factory: &Factory<'_>,
    stage: &PlannedStage,
) -> Result<(StageIdentity, (ProductManifest, bool)), RunError> {
    let ctx = Context::of(factory, stage)?;
    match &stage.foreach {
        None => {
            let identity = ctx.identity(&stage.key, ctx.bound.digests(None));
            let produce = |_: &Factory<'_>, staging: &Staging| {
                let call = Call {
                    inputs: ctx.bound.paths(None),
                    params: &stage.params,
                    files: &ctx.files,
                    models: ctx.models()?,
                    variables: ctx.factory.services.variables,
                    out: staging.path().to_path_buf(),
                };
                match ctx.executor.execute(&call) {
                    Ok(Outcome::Produced) => Ok(()),
                    Ok(Outcome::Rejected(why)) => Err(stage_error(
                        &stage.key,
                        format!("{} rejected its inputs: {why}", stage.operation),
                    )),
                    Err(why) => Err(stage_error(&stage.key, why)),
                }
            };
            let committed = factory.commit(&stage.key, &stage.operation, &identity, &produce)?;
            Ok((identity, committed))
        }
        Some(collection) => fan_out(&ctx, collection),
    }
}

/// What every computation of one stage shares.
struct Context<'f, 'a> {
    factory: &'f Factory<'a>,
    stage: &'f PlannedStage,
    executor: &'f dyn Execute,
    bound: Bound,
    /// The files the parameters name, checked against the lock.
    files: BTreeMap<String, String>,
    /// Their digests, as identities name them.
    file_digests: Vec<(String, Digest)>,
    /// Each model role's binding, as identities see it, and its champion.
    bindings: BTreeMap<String, (ModelBinding, Option<Champion>)>,
    /// The models, reached once something must be computed.
    models: OnceCell<BTreeMap<String, Model>>,
}

impl<'f, 'a> Context<'f, 'a> {
    fn of(factory: &'f Factory<'a>, stage: &'f PlannedStage) -> Result<Self, RunError> {
        let operation = factory
            .resolved
            .registry
            .get(&stage.operation)
            .ok_or_else(|| stage_error(&stage.key, format!("no operation {}", stage.operation)))?;
        let executor = operation.executor().ok_or_else(|| {
            stage_error(&stage.key, format!("{} does not compute", stage.operation))
        })?;
        let (mut files, mut file_digests) = (BTreeMap::new(), Vec::new());
        for path in operation.files(&stage.params) {
            let pinned = factory
                .resolved
                .lock
                .files
                .get(&path)
                .ok_or_else(|| stage_error(&stage.key, format!("{path} is not in the lock")))?;
            let full = factory.resolved.repository.join(&path);
            let text = std::fs::read_to_string(&full).map_err(io(&full))?;
            if &Digest::of(text.as_bytes()) != pinned {
                return Err(stage_error(
                    &stage.key,
                    format!("{path} changed since the run was resolved"),
                ));
            }
            file_digests.push((format!("file:{path}"), pinned.clone()));
            files.insert(path, text);
        }
        let mut bindings = BTreeMap::new();
        for role in operation.model_roles(&stage.params) {
            let model = factory.resolved.model(&role).ok_or_else(|| {
                stage_error(&stage.key, format!("model role {role} is not bound"))
            })?;
            bindings.insert(role, binding_of(factory, &stage.key, model)?);
        }
        Ok(Self {
            factory,
            stage,
            executor,
            bound: Bound::of(factory, stage)?,
            files,
            file_digests,
            bindings,
            models: OnceCell::new(),
        })
    }

    /// The models of the stage's roles, reached through the host on first
    /// use: a run that reuses every product reaches none.
    fn models(&self) -> Result<&BTreeMap<String, Model>, RunError> {
        if let Some(models) = self.models.get() {
            return Ok(models);
        }
        let mut models = BTreeMap::new();
        for (role, (binding, champion)) in &self.bindings {
            let host = self.factory.services.models.ok_or_else(|| {
                stage_error(
                    &self.stage.key,
                    format!(
                        "model role {role} is called in process, and the run was given no models"
                    ),
                )
            })?;
            let served = serve(self.factory, &self.stage.key, binding, champion.as_ref())?;
            let model = host
                .model(&served)
                .map_err(|e| stage_error(&self.stage.key, format!("model role {role}: {e}")))?;
            models.insert(role.clone(), model);
        }
        Ok(self.models.get_or_init(|| models))
    }

    /// A computation's identity: the operation, its parameters, the models
    /// its roles are bound to, the files it reads, its inputs and its
    /// implementation.
    fn identity(&self, name: &str, mut inputs: Vec<(String, Digest)>) -> StageIdentity {
        let models: BTreeMap<&String, &ModelBinding> = self
            .bindings
            .iter()
            .map(|(role, (b, _))| (role, b))
            .collect();
        inputs.extend(self.file_digests.iter().cloned());
        super::identity_of(
            &self.stage.versioned_operation(),
            name,
            json!({ "params": self.stage.params, "models": models }),
            inputs,
            &self.executor.implementation(),
            None,
        )
    }
}

/// The products a stage's ports are bound to, other than its item.
struct Bound {
    /// Each port's products: the view and the product's digest.
    ports: BTreeMap<String, Vec<Option<(PathBuf, Digest)>>>,
}

impl Bound {
    fn of(factory: &Factory<'_>, stage: &PlannedStage) -> Result<Self, RunError> {
        let mut ports = BTreeMap::new();
        for (port, sources) in &stage.inputs {
            let mut bound = Vec::new();
            for source in sources {
                bound.push(match source {
                    Source::Item => None,
                    other => Some(product_of(factory, other)?),
                });
            }
            ports.insert(port.clone(), bound);
        }
        Ok(Self { ports })
    }

    /// Each port's products, the item (if any) where `$item` is bound.
    fn paths(&self, item: Option<&Path>) -> BTreeMap<String, Vec<PathBuf>> {
        self.ports
            .iter()
            .map(|(port, bound)| {
                let paths = bound
                    .iter()
                    .filter_map(|b| match b {
                        Some((view, _)) => Some(view.clone()),
                        None => item.map(Path::to_path_buf),
                    })
                    .collect();
                (port.clone(), paths)
            })
            .collect()
    }

    /// Each port's digests, as an identity names them.
    fn digests(&self, item: Option<&Digest>) -> Vec<(String, Digest)> {
        let mut out = Vec::new();
        for (port, bound) in &self.ports {
            for (i, b) in bound.iter().enumerate() {
                let digest = match b {
                    Some((_, digest)) => Some(digest.clone()),
                    None => item.cloned(),
                };
                if let Some(digest) = digest {
                    out.push((format!("{port}#{i}"), digest));
                }
            }
        }
        out
    }
}

/// The view and the product digest of a recipe input or a stage's output.
fn product_of(factory: &Factory<'_>, source: &Source) -> Result<(PathBuf, Digest), RunError> {
    match source {
        Source::Input(name) => input(factory, name),
        Source::Stage { stage, .. } => {
            Ok((factory.view(stage), factory.product(stage)?.1.digest()))
        }
        Source::Item => Err(stage_error("$item", "is an item, not a product")),
    }
}

/// The recipe input `name` pinned into the run: its view and product digest.
fn input(factory: &Factory<'_>, name: &str) -> Result<(PathBuf, Digest), RunError> {
    let key = format!("_inputs/{name}");
    let spec = factory
        .resolved
        .compiled
        .inputs
        .get(name)
        .ok_or_else(|| stage_error(&key, "is not an input of the recipe"))?;
    let pinned = factory
        .resolved
        .lock
        .files
        .get(&spec.path)
        .ok_or_else(|| stage_error(&key, format!("{} is not in the lock", spec.path)))?
        .clone();
    let identity = super::identity_of(
        INPUT,
        name,
        Value::Null,
        vec![("files".into(), pinned.clone())],
        "",
        None,
    );
    let source = factory.resolved.repository.join(&spec.path);
    let produce = |_: &Factory<'_>, staging: &Staging| {
        let files = staging.path().join("files");
        copy_input(&source, &files).map_err(|e| stage_error(&key, e))?;
        let copied = entry_digest(&files).map_err(|e| stage_error(&key, e))?;
        if copied != pinned {
            return Err(stage_error(
                &key,
                format!("{} changed since the run was resolved", spec.path),
            ));
        }
        let mut items = Vec::new();
        for entry in std::fs::read_dir(&files).map_err(io(&files))? {
            let id = entry
                .map_err(io(&files))?
                .file_name()
                .to_string_lossy()
                .into_owned();
            items.push(Item {
                path: format!("files/{id}"),
                id,
            });
        }
        let index = ItemIndex::new(items, Vec::new()).map_err(|e| stage_error(&key, e))?;
        write_json(&staging.path().join(ITEMS_FILE), &index)
    };
    let (manifest, _) = factory.commit(&key, INPUT, &identity, &produce)?;
    Ok((factory.view(&key), manifest.digest()))
}

/// One item of a fan-out, before it runs.
struct Pending {
    id: String,
    path: PathBuf,
    key: String,
    identity: StageIdentity,
}

/// Every item of `collection` computed (or reused), then reduced.
fn fan_out(
    ctx: &Context<'_, '_>,
    collection: &Source,
) -> Result<(StageIdentity, (ProductManifest, bool)), RunError> {
    let (factory, stage) = (ctx.factory, ctx.stage);
    let (view, _) = product_of(factory, collection)?;
    let index_path = view.join(ITEMS_FILE);
    let index: ItemIndex = std::fs::read(&index_path)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .ok_or_else(|| {
            stage_error(
                &stage.key,
                "runs once per item of a collection, and what it names lists no items",
            )
        })?;
    index.validate().map_err(|e| stage_error(&stage.key, e))?;
    let mut pending = Vec::new();
    for item in &index.items {
        let path = view.join(&item.path);
        let digest = entry_digest(&path).map_err(|e| stage_error(&stage.key, e))?;
        let name = format!("{}/{}", stage.key, item.id);
        pending.push(Pending {
            id: item.id.clone(),
            key: format!("_items/{name}"),
            identity: ctx.identity(&name, ctx.bound.digests(Some(&digest))),
            path,
        });
    }
    let items = pending.len();
    let (committed, mut failed) = run_items(ctx, pending)?;
    failed.sort_by(|a, b| a.id.cmp(&b.id));
    let tolerated = stage
        .tolerate_failed_items
        .is_some_and(|bound| bound.allows(failed.len(), items));
    if !failed.is_empty() && !tolerated {
        let named: Vec<String> = failed
            .iter()
            .map(|f| format!("{}: {}", f.id, f.reason))
            .collect();
        return Err(stage_error(
            &stage.key,
            format!(
                "{} of {items} items failed (a resume computes only these):\n  {}",
                failed.len(),
                named.join("\n  ")
            ),
        ));
    }
    reduce(factory, stage, ctx.executor, &committed, &failed)
}

/// How a fan-out's item that failed transiently is asked again: up to
/// `attempts` asks in all, the pause before the second `first_pause` and
/// each later one twice the one before. Deterministic: no jitter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ItemRetry {
    /// Asks in all, the first included (at least 1).
    pub attempts: u32,
    /// The pause before the first repeat.
    pub first_pause: Duration,
}

impl Default for ItemRetry {
    /// Five asks, 30 s, 60 s, 120 s and 240 s apart: 7.5 minutes of pauses,
    /// enough for a model server to restart and load its weights.
    fn default() -> Self {
        Self {
            attempts: 5,
            first_pause: Duration::from_secs(30),
        }
    }
}

/// Whether an item's failure may pass when it is asked again: the
/// connection-level failures a model call retries, a timeout, a refused
/// connection or a server error.
fn transient(error: &str) -> bool {
    let text = error.to_lowercase();
    splinter_agent::retry::is_transient_text(&text)
        || [
            "timed out",
            "timeout",
            "no response headers",
            "connection refused",
            "connection aborted",
            "http 500",
            "500 internal server error",
        ]
        .iter()
        .any(|marker| text.contains(marker))
}

/// `execute` until it does not fail transiently, at most `retry.attempts`
/// times; a cancelled run is not asked again. A panic is a failure that
/// asking again does not pass.
fn execute_retried(
    retry: ItemRetry,
    cancelled: &(dyn Fn() -> bool + Send + Sync),
    out: &Path,
    execute: impl Fn() -> Result<Outcome, String>,
) -> Result<Outcome, String> {
    let mut pause = retry.first_pause;
    let mut asked = 1;
    loop {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // Each ask begins from an empty output directory.
            let _ = std::fs::remove_dir_all(out);
            std::fs::create_dir_all(out)
                .map_err(|e| e.to_string())
                .and_then(|()| execute())
        }))
        .unwrap_or_else(|panic| {
            Err(format!(
                "the operation panicked: {}",
                panic_message(&*panic)
            ))
        });
        match result {
            Err(why) if transient(&why) && asked < retry.attempts.max(1) => {
                let until = std::time::Instant::now() + pause;
                while std::time::Instant::now() < until {
                    if cancelled() {
                        return Err(why);
                    }
                    std::thread::sleep(PAUSE_SLICE.min(pause));
                }
                pause = pause.saturating_mul(2);
                asked += 1;
            }
            Err(why) if asked > 1 => return Err(format!("{why} (asked {asked} times)")),
            other => return other,
        }
    }
}

/// How often a pause between asks looks whether the run was cancelled.
const PAUSE_SLICE: Duration = Duration::from_millis(500);

/// A fan-out's items: each committed one by id, and each that failed with
/// its reason.
type ItemsRun = (Vec<(String, ProductManifest)>, Vec<Rejection>);

/// Every item committed, at most `concurrency` computing at once. An item
/// that fails (after its retries) stops nothing: it is recorded as failed
/// in the run state and returned with its reason. Any other error stops new
/// items from starting, lets the running ones finish (and commit), and is
/// the result.
fn run_items(ctx: &Context<'_, '_>, pending: Vec<Pending>) -> Result<ItemsRun, RunError> {
    let (factory, stage, executor) = (ctx.factory, ctx.stage, ctx.executor);
    let workers = stage.concurrency.max(1) as usize;
    let (retry, cancelled) = (factory.item_retry, &*factory.cancel);
    let mut committed = Vec::new();
    let mut failed = Vec::new();
    let mut failure = None;
    std::thread::scope(|scope| -> Result<(), RunError> {
        let (done, results) = mpsc::channel();
        let mut running = BTreeMap::new();
        let mut queue = pending.into_iter();
        loop {
            while failure.is_none() && running.len() < workers {
                if let Err(e) = factory.cancelled() {
                    failure = Some(e);
                    break;
                }
                let Some(item) = queue.next() else {
                    break;
                };
                match factory.begin(&item.key, &stage.operation, &item.identity)? {
                    Begun::Reused(manifest) => committed.push((item.id, manifest)),
                    Begun::Fresh {
                        version,
                        staging,
                        claim,
                    } => {
                        let models = match ctx.models() {
                            Ok(models) => models,
                            Err(e) => {
                                let failed = factory.finish(
                                    &item.key,
                                    &item.identity,
                                    version,
                                    staging,
                                    claim,
                                    Err(e),
                                );
                                failure = failed.err();
                                break;
                            }
                        };
                        let out = staging.path().join(OUT);
                        let call = Call {
                            inputs: ctx.bound.paths(Some(&item.path)),
                            params: &stage.params,
                            files: &ctx.files,
                            models,
                            variables: factory.services.variables,
                            out: out.clone(),
                        };
                        let done = done.clone();
                        let key = item.key.clone();
                        scope.spawn(move || {
                            // A panicking operation is a failed item: its
                            // result is sent like any other, or the loop
                            // below would wait for it for ever.
                            let result =
                                execute_retried(retry, cancelled, &out, || executor.execute(&call));
                            let _ = done.send((key, result));
                        });
                        running.insert(item.key.clone(), (item, version, staging, claim));
                    }
                }
            }
            if running.is_empty() {
                return Ok(());
            }
            let Ok((key, result)) = results.recv() else {
                return Ok(());
            };
            let Some((item, version, staging, claim)) = running.remove(&key) else {
                continue;
            };
            if let Err(reason) = result {
                // Recorded as failed in the run state, so a resume computes
                // it again; what the recording itself returns is this error.
                let _ = factory.finish(
                    &item.key,
                    &item.identity,
                    version,
                    staging,
                    claim,
                    Err(stage_error(&item.key, &reason)),
                );
                failed.push(Rejection {
                    id: item.id,
                    reason,
                });
                continue;
            }
            let produced = record(&item, &staging, result);
            match factory.finish(&item.key, &item.identity, version, staging, claim, produced) {
                Ok(manifest) => committed.push((item.id, manifest)),
                Err(e) => {
                    failure.get_or_insert(e);
                }
            }
        }
    })?;
    match failure {
        Some(e) => Err(e),
        None => Ok((committed, failed)),
    }
}

/// What an item's computation came to, written into its product.
fn record(
    item: &Pending,
    staging: &Staging,
    result: Result<Outcome, String>,
) -> Result<(), RunError> {
    let outcome = match result {
        Ok(Outcome::Produced) => ItemOutcome::Produced,
        Ok(Outcome::Rejected(reason)) => {
            // A rejected item keeps nothing the operation began to write.
            let out = staging.path().join(OUT);
            std::fs::remove_dir_all(&out).map_err(io(&out))?;
            std::fs::create_dir(&out).map_err(io(&out))?;
            ItemOutcome::Rejected { reason }
        }
        Err(why) => return Err(stage_error(&item.key, why)),
    };
    write_json(&staging.path().join(OUTCOME), &outcome)
}

/// The fan-out's own product: every item's output under its id, and the
/// index of what was produced and rejected.
fn reduce(
    factory: &Factory<'_>,
    stage: &PlannedStage,
    executor: &dyn Execute,
    committed: &[(String, ProductManifest)],
    failed: &[Rejection],
) -> Result<(StageIdentity, (ProductManifest, bool)), RunError> {
    let inputs = committed
        .iter()
        .map(|(id, manifest)| (format!("item:{id}"), manifest.digest()))
        .collect();
    let mut parameters = json!({ "params": stage.params });
    if !failed.is_empty() {
        // Tolerated failures are absent items: another product than the
        // complete one, and one a resume that computes them replaces.
        parameters["failed"] = json!(failed.iter().map(|f| &f.id).collect::<Vec<_>>());
    }
    let identity = super::identity_of(
        &stage.versioned_operation(),
        &stage.key,
        parameters,
        inputs,
        &format!("{}; {REDUCTION}", executor.implementation()),
        None,
    );
    let produce = |factory: &Factory<'_>, staging: &Staging| {
        let (mut items, mut rejected) = (Vec::new(), Vec::new());
        for (id, _) in committed {
            let view = factory.view(&format!("_items/{}/{id}", stage.key));
            let outcome_path = view.join(OUTCOME);
            let text = std::fs::read(&outcome_path).map_err(io(&outcome_path))?;
            let outcome: ItemOutcome =
                serde_json::from_slice(&text).map_err(|e| stage_error(&stage.key, e))?;
            match outcome {
                ItemOutcome::Produced => {
                    link_tree(&view.join(OUT), &staging.path().join(id))
                        .map_err(|e| stage_error(&stage.key, e))?;
                    items.push(Item {
                        id: id.clone(),
                        path: id.clone(),
                    });
                }
                ItemOutcome::Rejected { reason } => rejected.push(Rejection {
                    id: id.clone(),
                    reason,
                }),
            }
        }
        rejected.extend(failed.iter().map(|f| Rejection {
            id: f.id.clone(),
            reason: format!("failed: {}", f.reason),
        }));
        let index = ItemIndex::new(items, rejected).map_err(|e| stage_error(&stage.key, e))?;
        write_json(&staging.path().join(ITEMS_FILE), &index)
    };
    let committed = factory.commit(&stage.key, &stage.operation, &identity, &produce)?;
    Ok((identity, committed))
}

/// What a panic said, when it said it with a string.
pub(super) fn panic_message(panic: &(dyn std::any::Any + Send)) -> String {
    panic
        .downcast_ref::<&str>()
        .map(|s| (*s).to_owned())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "without a message".into())
}
