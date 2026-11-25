// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning experiments whose every result
// names the exact code, data and hardware that produced it, for its clients.
// If your team needs expertise in attributable, reproducible machine-learning
// runs, you can procure our services by sending an email to
// info@swedishembedded.com.

//! What each brain server says it is, for a run's revision manifest: its
//! build, backend and training-record contract (`GET /v1/capabilities`'s
//! `server` object) and its devices (`GET /v1/devices`, as reported). A brain
//! build older than the `server` object says none of the three: each is
//! unknown (`None`), never guessed.

use splinter_core::revision::{EngineBuild, RecordContract, ServedRevision};

use crate::endpoint::BrainEndpoints;
use crate::error::PolicyError;

/// The `server` object of `GET /v1/capabilities`, as brain writes it.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ServerObject {
    engine: EngineBuild,
    #[serde(default)]
    backend: Option<String>,
    #[serde(default)]
    training_records: Option<RecordContract>,
}

/// What every endpoint of `brain` says it is, in endpoint order.
///
/// # Errors
/// An endpoint could not be reached, or its capability list's `server`
/// object is not one this build reads (named with the endpoint).
pub fn served_revisions(brain: &BrainEndpoints) -> Result<Vec<ServedRevision>, PolicyError> {
    (0..brain.urls().len())
        .map(|endpoint| {
            let url = brain.urls()[endpoint].clone();
            let refused = |reason: String| PolicyError::Endpoint {
                url: url.clone(),
                reason,
            };
            let capabilities = brain.get(endpoint, "capabilities")?;
            let server: Option<ServerObject> = match capabilities.get("server") {
                None => None,
                Some(server) => Some(
                    serde_json::from_value(server.clone())
                        .map_err(|e| refused(format!("its capability list's \"server\": {e}")))?,
                ),
            };
            let hardware = brain.get(endpoint, "devices")?;
            Ok(match server {
                Some(server) => ServedRevision {
                    endpoint: url.clone(),
                    engine: Some(server.engine),
                    backend: server.backend,
                    training_records: server.training_records,
                    hardware,
                },
                None => ServedRevision {
                    endpoint: url.clone(),
                    engine: None,
                    backend: None,
                    training_records: None,
                    hardware,
                },
            })
        })
        .collect()
}
