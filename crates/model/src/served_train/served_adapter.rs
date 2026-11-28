// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A published version served on a server other than the one that trained it: brain's `serve_adapter` offers a stored
//! adapter (by its artifact digest) under an id the caller names, as the policy version it names, until
//! `withdraw_adapter` takes it off. An online run's evaluations use it to run on servers of their own.

use serde_json::json;

use super::job::{call, REFUSED_WAIT};
use crate::endpoint::BrainEndpoints;

/// The training service's model, which takes both actions.
const SERVICE: &str = "brain/train";

/// Serves the stored adapter `artifact` on server `index` of `endpoints` under `id` as policy version `version`.
///
/// # Errors
/// The server refused (an id it serves already, an artifact its store does not hold, an adapter whose base it does
/// not serve) or did not answer.
pub fn serve_adapter(
    endpoints: &BrainEndpoints,
    index: usize,
    artifact: &str,
    id: &str,
    version: u64,
) -> Result<(), String> {
    run(
        endpoints,
        index,
        "serve_adapter",
        json!({"adapter_digest": artifact, "id": id, "version": version}),
    )
}

/// Withdraws what [`serve_adapter`] serves under `id` on server `index`; requests already accepted finish.
///
/// # Errors
/// The server refused or did not answer.
pub fn withdraw_adapter(endpoints: &BrainEndpoints, index: usize, id: &str) -> Result<(), String> {
    run(endpoints, index, "withdraw_adapter", json!({"id": id}))
}

fn run(
    endpoints: &BrainEndpoints,
    index: usize,
    action: &str,
    params: serde_json::Value,
) -> Result<(), String> {
    let url = endpoints
        .urls()
        .get(index)
        .ok_or_else(|| format!("{action}: no server {index}"))?;
    let body = json!({"model": SERVICE, "action": action, "params": params});
    call(
        std::slice::from_ref(url),
        endpoints.key(),
        REFUSED_WAIT,
        0,
        &body,
    )
    .map(drop)
    .map_err(|e| format!("{action} on {url}: {e:?}"))
}
