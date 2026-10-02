// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Raw experience: tasks, worlds, what an agent saw, chose and caused.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::blob::BlobRef;
use crate::error::{Error, Result};
use crate::id::{ContentId, RecordId};

fn content_id_of<T: Serialize>(what: &'static str, value: &T) -> Result<ContentId> {
    let bytes = serde_json::to_vec(value).map_err(|source| Error::Encode { what, source })?;
    Ok(ContentId::of(&bytes))
}

/// An application-defined, content-addressed object: a source, a task, a
/// named set, a release. The class tells the application what it is; the
/// database only promises that the same identity is one entity however many
/// writers store it, and that it can be found by id or by class.
///
/// The identity is a hash of class, value and blobs unless the application
/// supplies its own `key`, for an object that already has a content address
/// the rest of the system uses. The database does not check a supplied key
/// against the content: whoever supplies it owns that promise.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entity {
    /// What kind of object this is, in the application's vocabulary.
    pub class: String,
    /// The application's own content address for the object, when it has one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<ContentId>,
    /// The object itself.
    pub value: serde_json::Value,
    /// Large payloads the object refers to, stored in blob packs.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blobs: Vec<BlobRef>,
}

impl Entity {
    /// An entity of `class` with no blobs.
    pub fn new(class: impl Into<String>, value: serde_json::Value) -> Self {
        Self {
            class: class.into(),
            key: None,
            value,
            blobs: Vec::new(),
        }
    }

    /// An entity of `class` that is known by `key`, an address the
    /// application computed.
    pub fn keyed(class: impl Into<String>, key: ContentId, value: serde_json::Value) -> Self {
        Self {
            key: Some(key),
            ..Self::new(class, value)
        }
    }

    /// The same entity referring to `blobs`.
    pub fn with_blobs(mut self, blobs: Vec<BlobRef>) -> Self {
        self.blobs = blobs;
        self
    }

    /// The content id: the application's key, else a function of class,
    /// value and blobs alone.
    pub fn id(&self) -> Result<ContentId> {
        match self.key {
            Some(key) => Ok(key),
            None => content_id_of("entity", self),
        }
    }
}

/// Bytes the record carries itself, or points at in the blob store when they
/// are too large to live in a record block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "storage", rename_all = "snake_case")]
pub enum Content {
    /// Text stored inline.
    Text {
        /// The text.
        text: String,
    },
    /// Bytes in the blob store.
    Blob {
        /// Where they are.
        blob: BlobRef,
    },
}

impl Content {
    /// Inline text.
    pub fn text(text: impl Into<String>) -> Self {
        Content::Text { text: text.into() }
    }
}

/// What was supposed to be achieved, independent of any one instance of it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskDefinition {
    /// A short name.
    pub name: String,
    /// What the task asks for.
    pub description: String,
    /// The area the task belongs to, for balancing and transfer questions.
    pub domain: String,
}

impl TaskDefinition {
    /// The content id every record about this definition refers to.
    pub fn id(&self) -> Result<ContentId> {
        content_id_of("task definition", self)
    }
}

/// One concrete instance of a task: this repository at this commit, this
/// issue. Every attempt against exactly this instance belongs together.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskInstance {
    /// The definition this is an instance of.
    pub definition: ContentId,
    /// The parameters that make it this instance.
    pub params: serde_json::Value,
    /// The environment snapshot the instance starts in.
    pub environment: Option<BlobRef>,
}

impl TaskInstance {
    /// The content id attempts and families refer to.
    pub fn id(&self) -> Result<ContentId> {
        content_id_of("task instance", self)
    }
}

/// How reproducible a state is, strongest first, so that a counterfactual
/// built on it can be trusted accordingly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReproLevel {
    /// Byte-for-byte replayable.
    Exact,
    /// Reconstructable given its seeds.
    Deterministic,
    /// External observations were captured.
    Snapshot,
    /// Some external state is unavailable.
    Approximate,
    /// Depends on the live environment.
    Live,
}

impl ReproLevel {
    /// The weakest of several levels, which bounds how far anything built on
    /// all of them can be trusted. `None` if there are none.
    pub fn weakest(levels: impl IntoIterator<Item = ReproLevel>) -> Option<ReproLevel> {
        levels.into_iter().max()
    }
}

/// A point in the world: everything needed to restore it, as content ids of
/// its parts (file tree, container, conversation, tool state, memory).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct State {
    /// The named parts of the world, each a content id.
    pub parts: BTreeMap<String, ContentId>,
    /// How well the state can be replayed.
    pub repro: ReproLevel,
    /// What cannot be reproduced, so nobody assumes it can.
    pub non_reproducible: Vec<String>,
}

impl State {
    /// A state from its parts.
    pub fn new(parts: BTreeMap<String, ContentId>, repro: ReproLevel) -> Self {
        Self {
            parts,
            repro,
            non_reproducible: Vec::new(),
        }
    }

    /// The content id every record that is "in this state" refers to. Two
    /// worlds with the same parts are the same state, so equal states merge.
    pub fn id(&self) -> Result<ContentId> {
        content_id_of("state", self)
    }
}

/// The key of an episode family: the same instance from the same starting
/// world. Derived from content, so writers on different nodes agree on it
/// without talking to each other.
pub fn family_key(task_instance: &ContentId, initial_state: &ContentId) -> ContentId {
    let mut bytes = Vec::with_capacity(64);
    bytes.extend_from_slice(task_instance.as_bytes());
    bytes.extend_from_slice(initial_state.as_bytes());
    ContentId::of(&bytes)
}

/// All attempts that start from one instance in one world.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpisodeFamily {
    /// The task instance.
    pub task_instance: ContentId,
    /// The world the attempts begin in.
    pub initial_state: ContentId,
}

/// Which policy acted, and how it was configured.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyRef {
    /// The policy or model name.
    pub name: String,
    /// Its version or checkpoint.
    pub version: String,
    /// Sampling and other settings.
    pub params: serde_json::Value,
}

impl PolicyRef {
    /// A policy with no extra settings.
    pub fn new(name: &str, version: &str) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            params: serde_json::Value::Null,
        }
    }
}

/// One attempt at a task instance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Attempt {
    /// The family the attempt belongs to.
    pub family: ContentId,
    /// The policy attempting it.
    pub policy: PolicyRef,
    /// The random seed, when the run was seeded.
    pub seed: Option<u64>,
}

/// What an agent was shown, as distinct from the state of the world.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Observation {
    /// The world the observation was taken from.
    pub state: ContentId,
    /// What the agent saw.
    pub content: Content,
}

/// What an agent did.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Action {
    /// The tool or kind of action.
    pub name: String,
    /// Its arguments.
    pub arguments: serde_json::Value,
}

impl Action {
    /// An action with arguments.
    pub fn new(name: &str, arguments: serde_json::Value) -> Self {
        Self {
            name: name.into(),
            arguments,
        }
    }
}

/// The moment an agent chose among possible behaviours.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Decision {
    /// The world it chose in.
    pub state: ContentId,
    /// What it had been shown.
    pub observation: Option<RecordId>,
    /// Who chose.
    pub policy: PolicyRef,
    /// The context it chose from, when stored apart so siblings can share it.
    pub context: Option<BlobRef>,
    /// What it chose.
    pub action: Action,
    /// The log-probability the policy gave its choice, for on-policy training.
    pub old_logprob: Option<f64>,
    /// The value the policy estimated, for on-policy training.
    pub value_estimate: Option<f64>,
}

/// What a decision caused: the world before and after, with the reward the
/// environment gave, if it gave one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Transition {
    /// The decision that caused it.
    pub decision: RecordId,
    /// The world before.
    pub from: ContentId,
    /// The world after.
    pub to: ContentId,
    /// What the agent saw afterwards.
    pub observation: Option<RecordId>,
    /// The environment's reward, absent when it gave none.
    pub reward: Option<f64>,
    /// Whether the episode ended here.
    pub done: bool,
}

/// How an attempt ended, as the environment reported it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    /// The task was achieved.
    Pass,
    /// The task was not achieved.
    Fail,
    /// The attempt could not be completed (crash, budget, timeout).
    Aborted,
}
