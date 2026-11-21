// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that schedule their work on
// a model service's cards and ask it what it has, for its clients. If your
// team needs expertise in GPU capacity planning around a shared model server,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! The hardware a run's brain server schedules on, as the server reports it.
//!
//! `--device` and `--backend` are brain's grammar and are handed to the
//! server a run starts, never read here. What the cards hold is asked of the
//! server (`GET /v1/devices`, [`devices`]); the VRAM a process was granted
//! (`--vram-budget`) bounds every server it starts ([`set_vram_budget`]).

use std::sync::OnceLock;

use serde::Serialize;

use crate::endpoint::BrainEndpoints;
use crate::error::PolicyError;

/// The grant [`set_vram_budget`] fixed, in GiB.
static VRAM_BUDGET: OnceLock<f64> = OnceLock::new();

/// Cap every server this process starts at `gib` GiB per card, the VRAM it
/// was granted (`--vram-budget`). The first grant stands.
pub fn set_vram_budget(gib: f64) {
    let _ = VRAM_BUDGET.set(gib);
}

/// The grant [`set_vram_budget`] fixed, in GiB; `None` when none was set.
#[must_use]
pub fn vram_budget_gib() -> Option<f64> {
    VRAM_BUDGET.get().copied()
}

/// One GPU's memory right now, as the server accounts it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct GpuCard {
    /// The endpoint whose server schedules it, by its place in
    /// [`BrainEndpoints::urls`]: with a server on every node of an
    /// allocation, the node.
    pub endpoint: usize,
    /// The card's index on its server, the `i` of `--device gpu<i>`.
    pub index: u32,
    /// Its memory in all.
    pub total_bytes: u64,
    /// What no resident model, run or operator reserve holds.
    pub free_bytes: u64,
    /// Whether the server measured the free memory (it always does; kept so
    /// a report can say when it did not).
    pub measured: bool,
}

/// The cards every endpoint of `brain` schedules on (`GET /v1/devices`), in
/// endpoint then index order; the CPU and NPUs are left out.
///
/// # Errors
/// An endpoint could not be reached or answered no device list.
pub fn devices(brain: &BrainEndpoints) -> Result<Vec<GpuCard>, PolicyError> {
    let mut cards = Vec::new();
    for endpoint in 0..brain.urls().len() {
        cards.extend(endpoint_devices(brain, endpoint)?);
    }
    Ok(cards)
}

/// The cards the endpoint at `endpoint` schedules on, in index order.
fn endpoint_devices(brain: &BrainEndpoints, endpoint: usize) -> Result<Vec<GpuCard>, PolicyError> {
    let url = brain.urls()[endpoint].clone();
    let list = brain.get(endpoint, "devices")?;
    let entries = list["data"]
        .as_array()
        .ok_or_else(|| PolicyError::Endpoint {
            url: url.clone(),
            reason: "its device list has no \"data\" array".into(),
        })?;
    let mut cards: Vec<GpuCard> = entries
        .iter()
        .filter_map(|entry| {
            let index = entry["id"].as_str()?.strip_prefix("gpu")?.parse().ok()?;
            let memory = &entry["memory"];
            Some(GpuCard {
                endpoint,
                index,
                total_bytes: memory["total_bytes"].as_u64()?,
                free_bytes: memory["free_bytes"].as_u64()?,
                measured: true,
            })
        })
        .collect();
    cards.sort_by_key(|card| card.index);
    Ok(cards)
}
