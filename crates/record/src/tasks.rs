// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements content-addressed task stores that every
// training example traces back to, for its clients. If your team needs
// expertise in training-data lineage or durable learning state, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The task store: generated tasks under their own address, and named sets
//! of them - how a pipeline stage names the tasks it hands the next one.
//!
//! ```text
//! <root>/tasks/
//!   objects/<hex>.json     one task, its canonical form; written once
//!   sets/<hex>.json        one task set, its canonical form; written once
//! ```
//!
//! A task's file is named by its [`TaskRef::id`](crate::experience::TaskRef),
//! the address of its content, and a read decodes it and checks that the
//! content still addresses it. A set is addressed by the digest of its own
//! canonical form, like an [`ExperienceSet`](crate::experiences::ExperienceSet),
//! and carries how each task was generated, which the task's own address
//! leaves out: the same task generated twice is one task.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::digest::{canonical_json, Digest};
use crate::error::{decode, io, object_digests, read_verified, StoreError};
use crate::experience::Task;
use crate::{write_once, StateRoot};

/// The content address of a [`TaskSet`].
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TaskSetId(pub Digest);

impl std::fmt::Display for TaskSetId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// One task of a set, and how it came to be.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskEntry {
    /// The task's address.
    pub task: Digest,
    /// The generator that produced it, as an experience's provenance names
    /// a generator; `None` when none is known.
    pub generator: Option<String>,
    /// The digest of the prompt its generator was sent, when there was one.
    pub prompt: Option<Digest>,
    /// The task this one is a variant of - the same fact asked in other
    /// words, to be graded by the same verifiers and never trained on;
    /// the task-level form of
    /// [`RelationKind::VariantOf`](crate::annotation::RelationKind::VariantOf).
    /// Left out of the canonical form when `None`, so a set of tasks that
    /// are no variants keeps its address.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub variant_of: Option<Digest>,
    /// What the task's question is about - the product, document, tool,
    /// component or version its instruction names - as its generator
    /// admitted it: what a variant must keep naming, and what two tasks
    /// must share before their answers can contradict each other. `None`
    /// for a task that names no subject; left out of the canonical form
    /// then, so a set recorded before subjects were keeps its address.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subject: Option<String>,
}

/// A named, ordered list of stored tasks.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskSet {
    /// What the set is, for a reader.
    pub name: String,
    /// The tasks, in order; each at most once.
    pub members: Vec<TaskEntry>,
}

/// The task store under one state root.
#[derive(Clone, Debug)]
pub struct TaskStore {
    dir: PathBuf,
}

impl TaskStore {
    /// The store under `root`. Nothing is created until something is
    /// written.
    #[must_use]
    pub fn open(root: &StateRoot) -> Self {
        Self { dir: root.tasks() }
    }

    fn object(&self, id: &Digest) -> PathBuf {
        self.dir.join("objects").join(format!("{}.json", id.hex()))
    }

    fn set(&self, id: &TaskSetId) -> PathBuf {
        self.dir.join("sets").join(format!("{}.json", id.0.hex()))
    }

    /// Stores `task` and returns its address. Write-once: storing it again
    /// is a no-op; an existing file that no longer holds the task it is
    /// named for is reported as corrupt rather than replaced.
    pub fn put(&self, task: &Task) -> Result<Digest, StoreError> {
        task.validate()?;
        let bytes = canonical_json(task).map_err(|source| StoreError::Serialize {
            what: "task",
            source,
        })?;
        let id = task.task.id.clone();
        let path = self.object(&id);
        if !write_once(&path, &bytes).map_err(io(&path))? {
            self.get(&id)?;
        }
        Ok(id)
    }

    /// Whether the store holds the task `id` (without verifying it).
    #[must_use]
    pub fn contains(&self, id: &Digest) -> bool {
        self.object(id).is_file()
    }

    /// The task stored under `id`, verified: it decodes, its content
    /// addresses it, and it re-encodes to exactly its stored form.
    pub fn get(&self, id: &Digest) -> Result<Task, StoreError> {
        let path = self.object(id);
        if !path.is_file() {
            return Err(StoreError::UnknownTask(id.clone()));
        }
        let bytes = std::fs::read(&path).map_err(io(&path))?;
        let task: Task = decode(&path, &bytes)?;
        task.validate()?;
        if task.task.id != *id {
            return Err(StoreError::Corrupt {
                path,
                expected: id.clone(),
                found: task.task.id,
            });
        }
        let canonical = canonical_json(&task).map_err(|source| StoreError::Serialize {
            what: "task",
            source,
        })?;
        if canonical != bytes {
            return Err(StoreError::Undecodable {
                path,
                reason: "it does not re-encode to its stored form".into(),
            });
        }
        Ok(task)
    }

    /// Every stored task's address, in order (without verifying them).
    pub fn list(&self) -> Result<Vec<Digest>, StoreError> {
        object_digests(&self.dir.join("objects"))
    }

    /// Stores `set` and returns its id; write-once like [`Self::put`].
    /// Refused when its name is empty, or a member is unknown or listed
    /// twice.
    pub fn put_set(&self, set: &TaskSet) -> Result<TaskSetId, StoreError> {
        let rejected = |reason: String| StoreError::Rejected {
            what: "task set",
            reason,
        };
        if set.name.trim().is_empty() {
            return Err(rejected("the name is empty".into()));
        }
        let mut seen = std::collections::HashSet::new();
        for member in &set.members {
            if !seen.insert(&member.task) {
                return Err(rejected(format!("{} is listed twice", member.task)));
            }
            if !self.contains(&member.task) {
                return Err(StoreError::UnknownTask(member.task.clone()));
            }
        }
        let bytes = canonical_json(set).map_err(|source| StoreError::Serialize {
            what: "task set",
            source,
        })?;
        let id = TaskSetId(Digest::of(&bytes));
        let path = self.set(&id);
        if !write_once(&path, &bytes).map_err(io(&path))? {
            read_verified(&path, &id.0)?;
        }
        Ok(id)
    }

    /// The set stored under `id`, verified against its address.
    pub fn get_set(&self, id: &TaskSetId) -> Result<TaskSet, StoreError> {
        let path = self.set(id);
        if !path.is_file() {
            return Err(StoreError::UnknownTaskSet(id.clone()));
        }
        let bytes = read_verified(&path, &id.0)?;
        decode(&path, &bytes)
    }

    /// Every stored set's id, in id order (without verifying them).
    pub fn list_sets(&self) -> Result<Vec<TaskSetId>, StoreError> {
        Ok(object_digests(&self.dir.join("sets"))?
            .into_iter()
            .map(TaskSetId)
            .collect())
    }
}
