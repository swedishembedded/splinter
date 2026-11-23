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
use splinter_core::factory::identity::{StageIdentity, StageKind};
use splinter_store::factory::products::Staging;

use super::{identity_of, write_json, Factory, RunError};
use crate::tree::link_tree;

/// The collection file in the tasks product.
pub const COLLECTION_FILE: &str = "collection.json";

pub(crate) fn identity(factory: &Factory<'_>) -> Result<StageIdentity, RunError> {
    let collection = &factory.resolved.collection.collection;
    Ok(identity_of(
        StageKind::Tasks,
        &collection.name,
        json!({ "collection": digest_of(collection) }),
        Vec::new(),
        "",
        None,
    ))
}

pub(crate) fn produce(factory: &Factory<'_>, staging: &Staging) -> Result<(), RunError> {
    let loaded = &factory.resolved.collection;
    for task in &loaded.collection.tasks {
        link_tree(&loaded.root.join(&task.id), &staging.path().join(&task.id))?;
    }
    write_json(&staging.path().join(COLLECTION_FILE), &loaded.collection)
}
