// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements sandboxed agent environments whose
// isolation is never silently weakened, for its clients. If your team needs
// expertise in agent sandboxing, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Where an agent runs: an `environment.yaml` of the task repository.
//!
//! ```yaml
//! schema: splinter.environment/v1
//! sandbox:
//!   provider: proot
//!   resources: { cpus: 2, memory_gb: 4, timeout: 10m }
//! mounts:
//!   - { source: knowledge/vim, target: /knowledge/vim, read_only: true }
//! network: { mode: model-gateway-only }
//! ```
//!
//! `proot` translates paths; it is a compatibility sandbox for hosts without
//! user namespaces or containers, never a security boundary: an environment
//! that asks for `isolation: strong` refuses it, and one on it must state
//! that its network is open, since proot cannot restrict it.

use serde::{Deserialize, Serialize};
use splinter_core::factory::execution::NetworkMode;

use crate::recipe::RecipeError;

/// The schema every environment file names.
pub const ENVIRONMENT_SCHEMA: &str = "splinter.environment/v1";

/// The sandbox providers the execution worker offers, and whether each is a
/// security boundary.
pub const PROVIDERS: &[(&str, bool)] = &[("docker", true), ("userns", true), ("proot", false)];

/// Resources of the sandbox.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Resources {
    /// CPUs.
    #[serde(default)]
    pub cpus: Option<u32>,
    /// Memory in GiB.
    #[serde(default)]
    pub memory_gb: Option<u64>,
    /// The agent's wall-clock limit: a whole number of seconds (`s`),
    /// minutes (`m`) or hours (`h`).
    #[serde(default)]
    pub timeout: Option<String>,
}

/// The sandbox section.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sandbox {
    /// The provider.
    pub provider: String,
    /// The image, pinned by digest, when the provider takes one.
    #[serde(default)]
    pub image: Option<String>,
    /// Resources.
    #[serde(default)]
    pub resources: Resources,
}

/// A read-only or writable directory of the repository, visible in the
/// sandbox.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MountSetting {
    /// The directory, relative to the repository root.
    pub source: String,
    /// Where it appears in the sandbox.
    pub target: String,
    /// Read-only.
    #[serde(default = "yes")]
    pub read_only: bool,
}

fn yes() -> bool {
    true
}

/// The network section.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Network {
    /// What the sandbox may reach.
    #[serde(default)]
    pub mode: NetworkMode,
}

/// The isolation an environment demands.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Isolation {
    /// Any provider.
    #[default]
    Any,
    /// Only a provider that is a security boundary.
    Strong,
}

/// An environment file.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Environment {
    /// Always [`ENVIRONMENT_SCHEMA`].
    pub schema: String,
    /// The sandbox.
    pub sandbox: Sandbox,
    /// Extra directories.
    #[serde(default)]
    pub mounts: Vec<MountSetting>,
    /// The network.
    #[serde(default)]
    pub network: Network,
    /// The isolation it demands.
    #[serde(default)]
    pub isolation: Isolation,
}

impl Environment {
    /// Parse and check an environment file.
    ///
    /// # Errors
    /// The wrong shape or schema, an unknown provider, a provider weaker
    /// than the isolation demanded, a bad timeout or mount.
    pub fn parse(text: &str) -> Result<Self, RecipeError> {
        let env: Self = serde_yaml::from_str(text).map_err(|e| RecipeError(vec![e.to_string()]))?;
        let mut problems = Vec::new();
        if env.schema != ENVIRONMENT_SCHEMA {
            problems.push(format!(
                "schema is {:?}; expected {ENVIRONMENT_SCHEMA}",
                env.schema
            ));
        }
        match PROVIDERS
            .iter()
            .find(|(name, _)| *name == env.sandbox.provider)
        {
            None => problems.push(format!(
                "sandbox provider {:?} is not one of {:?}",
                env.sandbox.provider,
                PROVIDERS.iter().map(|(n, _)| *n).collect::<Vec<_>>()
            )),
            Some((name, false)) if env.isolation == Isolation::Strong => problems.push(format!(
                "the environment demands strong isolation and {name} is not a security boundary"
            )),
            Some(("proot", _)) if env.network.mode != NetworkMode::Open => problems.push(
                "proot cannot restrict the network; an environment on proot must say `network: { mode: open }`".into(),
            ),
            Some(_) => {}
        }
        if let Some(timeout) = &env.sandbox.resources.timeout {
            if let Err(problem) = seconds(timeout) {
                problems.push(problem);
            }
        }
        for mount in &env.mounts {
            if !splinter_core::factory::task::plain_relative(&mount.source) {
                problems.push(format!(
                    "mount source {:?} is not a plain relative path",
                    mount.source
                ));
            }
            if !mount.target.starts_with('/') || mount.target.split('/').any(|s| s == "..") {
                problems.push(format!(
                    "mount target {:?} is not an absolute path",
                    mount.target
                ));
            }
        }
        if problems.is_empty() {
            Ok(env)
        } else {
            Err(RecipeError(problems))
        }
    }

    /// The agent's wall-clock limit in seconds.
    #[must_use]
    pub fn timeout_seconds(&self) -> Option<u64> {
        self.sandbox
            .resources
            .timeout
            .as_deref()
            .and_then(|t| seconds(t).ok())
    }
}

/// `90`, `90s`, `10m` or `2h` in seconds.
///
/// # Errors
/// Anything else, or zero.
pub fn seconds(text: &str) -> Result<u64, String> {
    let (number, scale) = match text.trim().char_indices().last() {
        Some((i, 's')) => (&text.trim()[..i], 1),
        Some((i, 'm')) => (&text.trim()[..i], 60),
        Some((i, 'h')) => (&text.trim()[..i], 3600),
        _ => (text.trim(), 1),
    };
    match number.parse::<u64>() {
        Ok(n) if n > 0 => Ok(n * scale),
        _ => Err(format!(
            "timeout {text:?} is not a positive duration such as 90s, 10m or 2h"
        )),
    }
}
