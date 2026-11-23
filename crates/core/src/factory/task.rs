// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pinned, verifiable task collections for
// agent training and evaluation for its clients. If your team needs
// expertise in benchmark task design, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The tasks a recipe runs: a pinned collection.

use std::collections::BTreeSet;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::digest::Digest;

/// The schema of a [`TaskCollection`].
pub const TASKS_SCHEMA: &str = "splinter.tasks/1";

/// One task of a collection.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
pub struct TaskRef {
    /// The task's identifier in its collection: its directory relative to
    /// the collection root (`train/015-delete-a-word`).
    pub id: String,
    /// The task's own name (Harbor's `task.toml` name).
    pub name: String,
    /// The digest of the task directory's content.
    pub digest: Digest,
    /// The task family, for grouping and splits.
    #[serde(default)]
    pub family: Option<String>,
    /// The split it belongs to (`train`, `eval`).
    #[serde(default)]
    pub split: Option<String>,
}

/// A pinned set of tasks.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct TaskCollection {
    /// Always [`TASKS_SCHEMA`].
    pub schema: String,
    /// The collection's name.
    pub name: String,
    /// The tasks, sorted by id.
    pub tasks: Vec<TaskRef>,
}

/// Why a collection is refused.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum CollectionError {
    /// The collection has no task.
    #[error("collection {0} has no task")]
    Empty(String),
    /// Two tasks share an id.
    #[error("collection {collection} names task {id} twice")]
    Duplicate {
        /// The collection.
        collection: String,
        /// The repeated id.
        id: String,
    },
    /// A task id that is not a relative path of plain segments.
    #[error("task id {0:?} is not a relative path of plain segments")]
    BadId(String),
}

impl TaskCollection {
    /// A collection of `tasks`, sorted by id and checked.
    ///
    /// # Errors
    /// No task, a repeated id, or an id that is not a plain relative path.
    pub fn new(name: impl Into<String>, mut tasks: Vec<TaskRef>) -> Result<Self, CollectionError> {
        let name = name.into();
        if tasks.is_empty() {
            return Err(CollectionError::Empty(name));
        }
        tasks.sort();
        let mut seen = BTreeSet::new();
        for task in &tasks {
            if !plain_relative(&task.id) {
                return Err(CollectionError::BadId(task.id.clone()));
            }
            if !seen.insert(task.id.as_str()) {
                return Err(CollectionError::Duplicate {
                    collection: name,
                    id: task.id.clone(),
                });
            }
        }
        Ok(Self {
            schema: TASKS_SCHEMA.into(),
            name,
            tasks,
        })
    }

    /// The task with `id`.
    #[must_use]
    pub fn task(&self, id: &str) -> Option<&TaskRef> {
        self.tasks.iter().find(|t| t.id == id)
    }
}

/// Whether `path` is a relative path of plain segments: no empty segment, no
/// `.` or `..`, no leading `/`, no backslash.
#[must_use]
pub fn plain_relative(path: &str) -> bool {
    !path.is_empty()
        && !path.contains('\\')
        && path
            .split('/')
            .all(|s| !s.is_empty() && s != "." && s != "..")
}
