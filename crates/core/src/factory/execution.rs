// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agent execution that is reproducible,
// sandboxed and independent of any one agent, for its clients. If your team
// needs expertise in running coding agents at scale, you can procure our
// services by sending an email to info@swedishembedded.com.

//! What a worker is asked to execute: one attempt at one logical slot.
//!
//! An [`ExecutionSpec`] is immutable and complete: a worker needs nothing
//! else from the controller to run it. It carries no secret: a credential is
//! named by the environment variable the worker reads it from, never by its
//! value, so a specification can be stored, hashed and shown.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::task::{plain_relative, TaskRef};
use super::trial::{RunBudget, TimeoutPolicy};
use crate::digest::Digest;

/// The schema of an [`ExecutionSpec`].
pub const EXECUTION_SCHEMA: &str = "splinter.execution/1";

/// A requested trial: one slot of one task in one pass. It stays the same
/// whichever physical attempt fills it.
#[derive(
    Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
pub struct SlotKey {
    /// The rollout pass.
    pub pass: String,
    /// The task's id in the collection.
    pub task: String,
    /// The slot's index among the task's attempts in the pass, from 0.
    pub slot: u32,
}

impl SlotKey {
    /// Slot `slot` of `task` in `pass`.
    #[must_use]
    pub fn new(pass: impl Into<String>, task: impl Into<String>, slot: u32) -> Self {
        Self {
            pass: pass.into(),
            task: task.into(),
            slot,
        }
    }
}

/// `pass/task/slot`, the slot's readable name.
impl std::fmt::Display for SlotKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}/{}", self.pass, self.task, self.slot)
    }
}

/// A file pinned by content.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
pub struct FileRef {
    /// Where it is, relative to the root its owner names.
    pub path: String,
    /// Its content digest.
    pub digest: Digest,
}

/// One environment variable an agent is given.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EnvValue {
    /// A literal, non-secret value.
    Literal(String),
    /// The value of this variable in the worker's own environment.
    FromEnv(String),
}

/// How an agent is started: its name and its own argv, which Splinter does
/// not interpret.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct AgentInvocation {
    /// The agent adapter (`sven`, `claude-code`, `oracle`, `nop`).
    pub name: String,
    /// The pinned agent version, when the recipe pins one.
    #[serde(default)]
    pub version: Option<String>,
    /// Arguments passed to the agent as they are, never through a shell.
    #[serde(default)]
    pub args: Vec<String>,
    /// Configuration files the arguments refer to, pinned by digest.
    #[serde(default)]
    pub files: Vec<FileRef>,
    /// The environment variables the agent is given; nothing else of the
    /// worker's environment reaches it.
    #[serde(default)]
    pub env: BTreeMap<String, EnvValue>,
    /// A system prompt file appended to the agent's own, when given.
    #[serde(default)]
    pub system_prompt: Option<FileRef>,
}

/// The wire protocol of a model connection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Protocol {
    /// OpenAI-compatible chat completions.
    Openai,
    /// Anthropic messages.
    Anthropic,
}

/// The model an agent is pointed at.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ModelBinding {
    /// The connection's name (`brain`, `hosted`).
    pub connection: String,
    /// The model id the endpoint serves.
    pub model: String,
    /// The wire protocol.
    pub protocol: Protocol,
    /// The endpoint, when the connection has one.
    #[serde(default)]
    pub url: Option<String>,
    /// The worker environment variable holding the credential.
    #[serde(default)]
    pub token_env: Option<String>,
}

impl ModelBinding {
    /// `connection:model`, as a recipe writes it.
    #[must_use]
    pub fn reference(&self) -> String {
        format!("{}:{}", self.connection, self.model)
    }
}

/// What the agent's sandbox may reach on the network.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum NetworkMode {
    /// Only the model gateway.
    #[default]
    ModelGatewayOnly,
    /// Nothing.
    None,
    /// The open network.
    Open,
}

/// A directory made visible inside the sandbox.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Mount {
    /// The pinned content: a product view directory, relative to the run.
    pub source: String,
    /// The digest of the content.
    pub digest: Digest,
    /// Where it appears in the sandbox (absolute).
    pub target: String,
    /// Mounted read-only.
    pub read_only: bool,
}

/// The sandbox the agent runs in.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SandboxSpec {
    /// The execution backend's sandbox provider (`docker`, `userns`,
    /// `proot`).
    pub provider: String,
    /// The image, pinned by digest, when the provider takes one.
    #[serde(default)]
    pub image: Option<String>,
    /// CPUs.
    #[serde(default)]
    pub cpus: Option<u32>,
    /// Memory in MiB.
    #[serde(default)]
    pub memory_mib: Option<u64>,
    /// The agent's wall-clock limit in seconds.
    #[serde(default)]
    pub timeout_seconds: Option<u64>,
    /// What the sandbox may reach.
    #[serde(default)]
    pub network: NetworkMode,
    /// Extra directories.
    #[serde(default)]
    pub mounts: Vec<Mount>,
}

/// One attempt at one slot.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ExecutionSpec {
    /// Always [`EXECUTION_SCHEMA`].
    pub schema: String,
    /// The logical slot this attempt fills.
    pub slot: SlotKey,
    /// The physical attempt's number for the slot, from 0.
    pub attempt: u32,
    /// The task.
    pub task: TaskRef,
    /// The task's directory, relative to the run's pinned task tree.
    pub task_dir: String,
    /// How to start the agent.
    pub agent: AgentInvocation,
    /// The model, unless the agent brings its own.
    #[serde(default)]
    pub model: Option<ModelBinding>,
    /// The sandbox.
    pub sandbox: SandboxSpec,
    /// The attempt's seed.
    pub seed: u64,
    /// The run's per-trial budget.
    #[serde(default)]
    pub budget: Option<RunBudget>,
    /// The task's timeout policy.
    #[serde(default)]
    pub timeout_policy: TimeoutPolicy,
}

/// Why a specification is refused.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SpecError {
    /// The agent has no name.
    #[error("the agent has no name")]
    NoAgent,
    /// A literal environment value under a credential-like name.
    #[error("environment variable {0} looks like a credential; name the variable it is read from instead")]
    LiteralSecret(String),
    /// A path that escapes its root.
    #[error("path {0:?} is not a plain relative path")]
    BadPath(String),
    /// A mount target that is not absolute.
    #[error("mount target {0:?} is not an absolute path")]
    BadTarget(String),
    /// Something else is wrong.
    #[error("{0}")]
    Invalid(String),
}

/// Whether a variable name reads as a credential.
#[must_use]
pub fn credential_like(name: &str) -> bool {
    let upper = name.to_ascii_uppercase();
    ["KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL"]
        .iter()
        .any(|w| upper.contains(w))
}

impl ExecutionSpec {
    /// Check the invariants a worker relies on.
    ///
    /// # Errors
    /// The first violation found.
    pub fn validate(&self) -> Result<(), SpecError> {
        if self.schema != EXECUTION_SCHEMA {
            return Err(SpecError::Invalid(format!(
                "schema {:?} is not {EXECUTION_SCHEMA}",
                self.schema
            )));
        }
        if self.agent.name.trim().is_empty() {
            return Err(SpecError::NoAgent);
        }
        for (name, value) in &self.agent.env {
            if matches!(value, EnvValue::Literal(_)) && credential_like(name) {
                return Err(SpecError::LiteralSecret(name.clone()));
            }
        }
        let files = self
            .agent
            .files
            .iter()
            .chain(self.agent.system_prompt.iter());
        for path in files.map(|f| &f.path).chain([&self.task_dir]) {
            if !plain_relative(path) {
                return Err(SpecError::BadPath(path.clone()));
            }
        }
        for mount in &self.sandbox.mounts {
            if !plain_relative(&mount.source) {
                return Err(SpecError::BadPath(mount.source.clone()));
            }
            if !mount.target.starts_with('/') || mount.target.split('/').any(|s| s == "..") {
                return Err(SpecError::BadTarget(mount.target.clone()));
            }
        }
        if self.task.id != self.slot.task {
            return Err(SpecError::Invalid(format!(
                "slot names task {} and the spec carries task {}",
                self.slot.task, self.task.id
            )));
        }
        Ok(())
    }
}
