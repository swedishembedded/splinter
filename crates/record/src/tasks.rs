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
//! A task is an entity keyed by its [`TaskRef::id`](crate::experience::TaskRef),
//! the address of its content, and a read decodes it and checks that the
//! content still addresses it. A set is addressed by the digest of its own
//! canonical form, like an [`ExperienceSet`](crate::experiences::ExperienceSet),
//! and carries how each task was generated, which the task's own address
//! leaves out: the same task generated twice is one task.

use serde::{Deserialize, Serialize};

use splinter_expdb::model::Entity;

use crate::digest::{canonical_json, Digest};
use crate::error::StoreError;
use crate::experience::Task;
use crate::workspace::{content_id, Workspace};

const TASK: &str = "task";
const TASK_SET: &str = "task_set";

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

/// The task store: tasks under their address and the named sets that record
/// how each was generated.
#[derive(Clone, Debug)]
pub struct TaskStore {
    workspace: Workspace,
}

impl TaskStore {
    /// The store over `workspace`.
    #[must_use]
    pub fn new(workspace: &Workspace) -> Self {
        Self {
            workspace: workspace.clone(),
        }
    }

    /// Stores `task` and returns its address. Storing it again writes
    /// nothing; one already stored that no longer holds the task it is
    /// stored for is reported rather than replaced.
    pub fn put(&self, task: &Task) -> Result<Digest, StoreError> {
        task.validate()?;
        let id = task.task.id.clone();
        if self.contains(&id)? {
            self.get(&id)?;
            return Ok(id);
        }
        let value = serde_json::to_value(task).map_err(|source| StoreError::Serialize {
            what: "task",
            source,
        })?;
        let entity = Entity::keyed(TASK, content_id(&id)?, value);
        self.workspace.write(|s| s.put_entity(&entity))?;
        Ok(id)
    }

    /// Whether the store holds the task `id` (without verifying it).
    pub fn contains(&self, id: &Digest) -> Result<bool, StoreError> {
        self.workspace.has(TASK, id)
    }

    /// The task stored under `id`, verified: it decodes, its content
    /// addresses it, and it re-encodes to exactly what was stored.
    pub fn get(&self, id: &Digest) -> Result<Task, StoreError> {
        let entity = self
            .workspace
            .find(TASK, id)?
            .ok_or_else(|| StoreError::UnknownTask(id.clone()))?;
        let stored = canonical_json(&entity.value).map_err(|source| StoreError::Serialize {
            what: "task",
            source,
        })?;
        let task: Task =
            serde_json::from_value(entity.value).map_err(|e| StoreError::UndecodableObject {
                what: format!("task {id}"),
                reason: e.to_string(),
            })?;
        task.validate()?;
        if task.task.id != *id {
            return Err(StoreError::Altered {
                what: format!("task {id}"),
                expected: id.clone(),
                found: task.task.id,
            });
        }
        let canonical = canonical_json(&task).map_err(|source| StoreError::Serialize {
            what: "task",
            source,
        })?;
        if canonical != stored {
            return Err(StoreError::UndecodableObject {
                what: format!("task {id}"),
                reason: "it does not re-encode to its stored form".into(),
            });
        }
        Ok(task)
    }

    /// Every stored task's address, in order (without verifying them).
    pub fn list(&self) -> Result<Vec<Digest>, StoreError> {
        self.workspace.ids_of(TASK)
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
            if !self.contains(&member.task)? {
                return Err(StoreError::UnknownTask(member.task.clone()));
            }
        }
        let bytes = canonical_json(set).map_err(|source| StoreError::Serialize {
            what: "task set",
            source,
        })?;
        let id = TaskSetId(Digest::of(&bytes));
        if !self.workspace.has(TASK_SET, &id.0)? {
            let value = serde_json::to_value(set).map_err(|source| StoreError::Serialize {
                what: "task set",
                source,
            })?;
            let entity = Entity::keyed(TASK_SET, content_id(&id.0)?, value);
            self.workspace.write(|s| s.put_entity(&entity))?;
        }
        Ok(id)
    }

    /// The set stored under `id`, verified against its address.
    pub fn get_set(&self, id: &TaskSetId) -> Result<TaskSet, StoreError> {
        let entity = self
            .workspace
            .find(TASK_SET, &id.0)?
            .ok_or_else(|| StoreError::UnknownTaskSet(id.clone()))?;
        let set: TaskSet =
            serde_json::from_value(entity.value).map_err(|e| StoreError::UndecodableObject {
                what: format!("task set {id}"),
                reason: e.to_string(),
            })?;
        let bytes = canonical_json(&set).map_err(|source| StoreError::Serialize {
            what: "task set",
            source,
        })?;
        let found = Digest::of(&bytes);
        if found != id.0 {
            return Err(StoreError::Altered {
                what: format!("task set {id}"),
                expected: id.0.clone(),
                found,
            });
        }
        Ok(set)
    }

    /// Every stored set's id, in id order (without verifying them).
    pub fn list_sets(&self) -> Result<Vec<TaskSetId>, StoreError> {
        Ok(self
            .workspace
            .ids_of(TASK_SET)?
            .into_iter()
            .map(TaskSetId)
            .collect())
    }
}
