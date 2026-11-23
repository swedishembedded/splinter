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

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use splinter_core::digest::Digest;
use splinter_core::factory::identity::StageIdentity;
use splinter_core::factory::items::{Item, ItemIndex, Rejection, ITEMS_FILE};
use splinter_store::factory::products::{ProductManifest, Staging};

use super::{io, stage_error, write_json, Begun, Factory, RunError};
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
    let executor = factory
        .resolved
        .registry
        .get(&stage.operation)
        .and_then(|o| o.executor())
        .ok_or_else(|| stage_error(&stage.key, format!("no operation {}", stage.operation)))?;
    let bound = Bound::of(factory, stage)?;
    match &stage.foreach {
        None => {
            let identity = identity(factory, stage, executor, &stage.key, bound.digests(None));
            let produce = |_: &Factory<'_>, staging: &Staging| {
                let call = Call {
                    inputs: bound.paths(None),
                    params: &stage.params,
                    out: staging.path().to_path_buf(),
                };
                match executor.execute(&call) {
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
        Some(collection) => fan_out(factory, stage, executor, &bound, collection),
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

/// A computed stage's identity: its operation, parameters, the models its
/// roles are bound to, its inputs and its implementation.
fn identity(
    factory: &Factory<'_>,
    stage: &PlannedStage,
    executor: &dyn Execute,
    name: &str,
    inputs: Vec<(String, Digest)>,
) -> StageIdentity {
    let models: BTreeMap<String, String> = factory
        .resolved
        .registry
        .get(&stage.operation)
        .map(|o| o.model_roles(&stage.params))
        .unwrap_or_default()
        .into_iter()
        .filter_map(|role| {
            let model = factory.resolved.model(&role)?.to_string();
            Some((role, model))
        })
        .collect();
    super::identity_of(
        &stage.versioned_operation(),
        name,
        json!({ "params": stage.params, "models": models }),
        inputs,
        &executor.implementation(),
        None,
    )
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
    factory: &Factory<'_>,
    stage: &PlannedStage,
    executor: &dyn Execute,
    bound: &Bound,
    collection: &Source,
) -> Result<(StageIdentity, (ProductManifest, bool)), RunError> {
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
            identity: identity(
                factory,
                stage,
                executor,
                &name,
                bound.digests(Some(&digest)),
            ),
            path,
        });
    }
    let committed = run_items(factory, stage, executor, bound, pending)?;
    reduce(factory, stage, executor, &committed)
}

/// Every item committed, at most `concurrency` computing at once; the
/// first failure stops new items from starting, lets the running ones
/// finish (and commit), and is the result.
fn run_items(
    factory: &Factory<'_>,
    stage: &PlannedStage,
    executor: &dyn Execute,
    bound: &Bound,
    pending: Vec<Pending>,
) -> Result<Vec<(String, ProductManifest)>, RunError> {
    let workers = stage.concurrency.max(1) as usize;
    let mut committed = Vec::new();
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
                    Begun::Fresh { version, staging } => {
                        let out = staging.path().join(OUT);
                        let call = Call {
                            inputs: bound.paths(Some(&item.path)),
                            params: &stage.params,
                            out: out.clone(),
                        };
                        let done = done.clone();
                        let key = item.key.clone();
                        scope.spawn(move || {
                            let result = std::fs::create_dir_all(&out)
                                .map_err(|e| e.to_string())
                                .and_then(|()| executor.execute(&call));
                            let _ = done.send((key, result));
                        });
                        running.insert(item.key.clone(), (item, version, staging));
                    }
                }
            }
            if running.is_empty() {
                return Ok(());
            }
            let Ok((key, result)) = results.recv() else {
                return Ok(());
            };
            let Some((item, version, staging)) = running.remove(&key) else {
                continue;
            };
            let produced = record(&item, &staging, result);
            match factory.finish(&item.key, &item.identity, version, staging, produced) {
                Ok(manifest) => committed.push((item.id, manifest)),
                Err(e) => {
                    failure.get_or_insert(e);
                }
            }
        }
    })?;
    match failure {
        Some(e) => Err(e),
        None => Ok(committed),
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
) -> Result<(StageIdentity, (ProductManifest, bool)), RunError> {
    let inputs = committed
        .iter()
        .map(|(id, manifest)| (format!("item:{id}"), manifest.digest()))
        .collect();
    let identity = super::identity_of(
        &stage.versioned_operation(),
        &stage.key,
        json!({ "params": stage.params }),
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
        let index = ItemIndex::new(items, rejected).map_err(|e| stage_error(&stage.key, e))?;
        write_json(&staging.path().join(ITEMS_FILE), &index)
    };
    let committed = factory.commit(&stage.key, &stage.operation, &identity, &produce)?;
    Ok((identity, committed))
}
