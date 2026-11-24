// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model routing that keeps credentials out
// of experiment files, for its clients. If your team needs expertise in
// running agents against many model providers, you can procure our
// services by sending an email to info@swedishembedded.com.

//! How this installation reaches model providers: the `connections:` section
//! of the machine's `config.yaml` ([`crate::machine_config`]), layered over a
//! task repository's `connections.yaml`.
//!
//! ```yaml
//! connections:
//!   brain:  { protocol: openai, url: "http://127.0.0.1:8990/v1", token: "env:BRAIN_API_KEY" }
//!   hosted: { protocol: openai, url: "https://example.invalid/v1", token: "env:SVEN_KEY" }
//! ```
//!
//! A credential is only ever named by the environment variable that holds
//! it (`env:NAME`); a literal token in the file is refused, so the file can
//! be shared and recorded with a run.
//!
//! A credential is the machine's to give. A task repository's
//! `connections.yaml` may name a connection's credential variable as a
//! shared default, but such a connection is bound only when this machine's
//! file defines the connection too (its URL and its variable then being the
//! machine's): otherwise a repository could have the gateway send any of the
//! controller's variables to an endpoint of its choosing.

use std::collections::BTreeMap;
use std::path::Path;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use splinter_core::factory::execution::{ModelBinding, Protocol, FORWARDED_HEADERS};

use crate::machine_config::MachineConfig;
use crate::recipe::ModelRef;

/// One connection: how a model provider is reached.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Connection {
    /// The wire protocol.
    pub protocol: Protocol,
    /// The endpoint.
    #[serde(default)]
    pub url: Option<String>,
    /// `env:NAME`: the variable holding the credential.
    #[serde(default)]
    #[schemars(regex(pattern = r"^env:[A-Za-z0-9_]+$"))]
    pub token: Option<String>,
    /// Headers of the agent's this endpoint does not take, which the
    /// worker's gateway drops (an endpoint that refuses `anthropic-beta`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub drop_headers: Vec<String>,
}

/// Every connection of an installation.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Connections {
    /// By name.
    #[serde(default)]
    pub connections: BTreeMap<String, Connection>,
    /// The files these were read from, or the ones looked for when none
    /// exists: what an unknown connection's error names. Not part of their
    /// identity.
    #[serde(skip)]
    pub looked_in: Vec<String>,
    /// Connections whose credential only a task repository names, with the
    /// machine file that would have to define them: refused when bound.
    #[serde(skip)]
    pub unvouched: BTreeMap<String, String>,
}

/// The connections file of a task repository: shared defaults (it can hold
/// no credential), each overridden by this machine's connection of the same
/// name.
pub const REPOSITORY_FILE: &str = "connections.yaml";

/// Why connections are refused or a model cannot be bound.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ConnectionError {
    /// The file does not parse.
    #[error("connections: {0}")]
    Parse(String),
    /// A literal credential.
    #[error("connection {0}: a token must be `env:NAME`, never the credential itself")]
    LiteralToken(String),
    /// A connection whose credential only a task repository names.
    #[error(
        "connection {connection} sends the credential in {variable}, which only the task repository names; \
         a credential is the machine's to give: define {connection} in {machine}"
    )]
    Unvouched {
        /// The connection.
        connection: String,
        /// The variable it would send.
        variable: String,
        /// The machine's configuration file.
        machine: String,
    },
    /// A model names a connection that does not exist.
    #[error(
        "model {model} names connection {connection}, which no connections file defines ({looked_in})"
    )]
    Unknown {
        /// The model reference.
        model: String,
        /// The missing connection.
        connection: String,
        /// Where connections were read, or looked for.
        looked_in: String,
    },
}

impl Connections {
    /// Parse and check a connections file.
    ///
    /// # Errors
    /// Not the right shape, or a literal token.
    pub fn parse(text: &str) -> Result<Self, ConnectionError> {
        let parsed: Self =
            serde_yaml::from_str(text).map_err(|e| ConnectionError::Parse(e.to_string()))?;
        Self::validate(&parsed.connections)?;
        Ok(parsed)
    }

    /// Check `connections` by the rules of a connections file: every
    /// dropped header is one the gateway forwards, and every token is
    /// `env:NAME`, never the credential itself.
    ///
    /// # Errors
    /// The first connection that breaks a rule.
    pub fn validate(connections: &BTreeMap<String, Connection>) -> Result<(), ConnectionError> {
        for (name, connection) in connections {
            if let Some(header) = connection
                .drop_headers
                .iter()
                .find(|h| !FORWARDED_HEADERS.contains(&h.as_str()))
            {
                return Err(ConnectionError::Parse(format!(
                    "connection {name} drops {header}, but the gateway forwards only {}",
                    FORWARDED_HEADERS.join(", ")
                )));
            }
            if let Some(token) = &connection.token {
                let variable = token.strip_prefix("env:").unwrap_or_default();
                if variable.is_empty()
                    || !variable
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || c == '_')
                {
                    return Err(ConnectionError::LiteralToken(name.clone()));
                }
            }
        }
        Ok(())
    }

    /// The connections for a run of the task repository at `repository` on
    /// this machine: the repository's [`REPOSITORY_FILE`], each connection
    /// overridden by the one of the same name in the `connections:` section
    /// of `machine` (this machine's `config.yaml`). Either may be missing.
    ///
    /// # Errors
    /// A file exists but cannot be read, or is refused; the machine's file is
    /// absent while the older `connections.yaml` or `operators.yaml` beside
    /// it exist.
    pub fn layered(repository: &Path, machine: &Path) -> Result<Self, ConnectionError> {
        let in_file = |path: &Path, e: &dyn std::fmt::Display| {
            ConnectionError::Parse(format!("{}: {e}", path.display()))
        };
        let repository_file = repository.join(REPOSITORY_FILE);
        let mut layered = Self::default();
        let mut layers = Vec::new();
        match std::fs::read_to_string(&repository_file) {
            Ok(text) => {
                let found = Self::parse(&text).map_err(|e| in_file(&repository_file, &e))?;
                layers.push((repository_file, Some(found.connections), false));
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                layers.push((repository_file, None, false));
            }
            Err(e) => return Err(in_file(&repository_file, &e)),
        }
        let config =
            MachineConfig::read(machine).map_err(|e| ConnectionError::Parse(e.to_string()))?;
        layers.push((machine.to_path_buf(), config.map(|c| c.connections), true));
        for (path, found, by_machine) in layers {
            let Some(found) = found else {
                layered.looked_in.push(format!("no {}", path.display()));
                continue;
            };
            for (name, connection) in &found {
                if by_machine {
                    layered.unvouched.remove(name);
                } else if connection.token.is_some() {
                    layered
                        .unvouched
                        .insert(name.clone(), machine.display().to_string());
                }
            }
            layered.connections.extend(found);
            layered.looked_in.push(format!("read {}", path.display()));
        }
        Ok(layered)
    }

    /// The binding a worker needs for `model`.
    ///
    /// # Errors
    /// The model's connection is not defined.
    pub fn bind(&self, model: &ModelRef) -> Result<ModelBinding, ConnectionError> {
        let connection =
            self.connections
                .get(&model.connection)
                .ok_or_else(|| ConnectionError::Unknown {
                    model: model.to_string(),
                    connection: model.connection.clone(),
                    looked_in: if self.looked_in.is_empty() {
                        "the connections given".into()
                    } else {
                        self.looked_in.join("; ")
                    },
                })?;
        if let Some(machine) = self.unvouched.get(&model.connection) {
            return Err(ConnectionError::Unvouched {
                connection: model.connection.clone(),
                variable: connection
                    .token
                    .as_deref()
                    .and_then(|t| t.strip_prefix("env:"))
                    .unwrap_or_default()
                    .to_owned(),
                machine: machine.clone(),
            });
        }
        Ok(ModelBinding {
            connection: model.connection.clone(),
            model: model.model.clone(),
            protocol: connection.protocol,
            url: connection.url.clone(),
            token_env: connection
                .token
                .as_deref()
                .and_then(|t| t.strip_prefix("env:"))
                .map(str::to_owned),
            drop_headers: connection.drop_headers.clone(),
        })
    }
}
