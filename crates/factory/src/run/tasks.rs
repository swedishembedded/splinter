// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pinned, verifiable task collections for
// agent training and evaluation for its clients. If your team needs
// expertise in benchmark task design, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The tasks stage: the selected tasks copied into the run, so every later
//! stage (and every worker) reads the run's copy, never the repository.
//!
//! Its product holds `collection.json` and each task's directory at its id.

use serde_json::json;
use splinter_core::digest::digest_of;
use splinter_core::factory::identity::StageIdentity;
use splinter_core::factory::task::TaskCollection;
use splinter_store::factory::products::Staging;

use super::{identity_of, stage_error, write_json, Factory, RunError};
use crate::collection::LoadedCollection;
use crate::plan::PlannedStage;
use crate::tree::link_tree;

/// The collection file in the tasks product.
pub const COLLECTION_FILE: &str = "collection.json";

pub(crate) fn identity(
    factory: &Factory<'_>,
    stage: &PlannedStage,
) -> Result<StageIdentity, RunError> {
    let loaded = collection(factory, stage)?;
    Ok(identity_of(
        &stage.versioned_operation(),
        &stage.key,
        json!({ "collection": digest_of(&loaded.collection) }),
        Vec::new(),
        "",
        None,
    ))
}

fn collection<'r>(
    factory: &'r Factory<'_>,
    stage: &PlannedStage,
) -> Result<&'r LoadedCollection, RunError> {
    factory
        .collections
        .get(&stage.key)
        .ok_or_else(|| stage_error(&stage.key, "its collection was not resolved"))
}

/// The collection a stage produced, as the pin `stage` reads it: every task
/// below the producing stage's view, of the pin's split.
///
/// # Errors
/// The pin is bound to no stage, or the product holds no such task.
pub(crate) fn produced_collection(
    factory: &Factory<'_>,
    stage: &PlannedStage,
) -> Result<LoadedCollection, RunError> {
    let source = stage
        .source("collection")
        .ok_or_else(|| stage_error(&stage.key, "its collection was not resolved"))?;
    factory.product(source)?;
    let params: crate::ops::PinParams = stage.params().map_err(|e| stage_error(&stage.key, e))?;
    let mut loaded = crate::collection::scan(&factory.view(source), stage.key.clone(), &[])
        .map_err(|e| stage_error(&stage.key, e))?;
    loaded
        .collection
        .tasks
        .retain(|t| params.selects(&t.id, t.split.as_deref()));
    if loaded.collection.tasks.is_empty() {
        return Err(stage_error(
            &stage.key,
            format!("{source} produced no task of {}", params.selection()),
        ));
    }
    Ok(loaded)
}

pub(crate) fn produce(
    factory: &Factory<'_>,
    stage: &PlannedStage,
    staging: &Staging,
) -> Result<(), RunError> {
    let loaded = collection(factory, stage)?;
    for task in &loaded.collection.tasks {
        link_tree(&loaded.root.join(&task.id), &staging.path().join(&task.id))?;
    }
    write_json(&staging.path().join(COLLECTION_FILE), &loaded.collection)
}

/// The tasks a committed `tasks.pin` view holds.
///
/// # Errors
/// Its collection file cannot be read.
pub fn tasks_of(view: &std::path::Path) -> Result<TaskCollection, RunError> {
    super::read_json(&view.join(COLLECTION_FILE))
}
