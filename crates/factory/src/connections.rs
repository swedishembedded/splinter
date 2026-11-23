// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model routing that keeps credentials out
// of experiment files, for its clients. If your team needs expertise in
// running agents against many model providers, you can procure our
// services by sending an email to info@swedishembedded.com.

//! How this installation reaches model providers: the machine-level
//! `connections.yaml`.
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

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use splinter_core::factory::execution::{ModelBinding, Protocol, FORWARDED_HEADERS};

use crate::recipe::ModelRef;

/// The schema-less shape of one connection.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Connection {
    /// The wire protocol.
    pub protocol: Protocol,
    /// The endpoint.
    #[serde(default)]
    pub url: Option<String>,
    /// `env:NAME`: the variable holding the credential.
    #[serde(default)]
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
}

/// Why connections are refused or a model cannot be bound.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ConnectionError {
    /// The file does not parse.
    #[error("connections: {0}")]
    Parse(String),
    /// A literal credential.
    #[error("connection {0}: a token must be `env:NAME`, never the credential itself")]
    LiteralToken(String),
    /// A model names a connection that does not exist.
    #[error("model {model} names connection {connection}, which connections.yaml does not define")]
    Unknown {
        /// The model reference.
        model: String,
        /// The missing connection.
        connection: String,
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
        for (name, connection) in &parsed.connections {
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
        Ok(parsed)
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
                })?;
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
