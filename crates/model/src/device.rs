// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Which backend drives the hardware, chosen by the caller.

use serde::Serialize;

use crate::error::PolicyError;

/// Drive the hardware with `backend` (`wgpu | vulkan | cuda | cpu`) in every
/// model built after this call. A program takes it as a flag and passes it
/// here before loading anything; nothing is read from the environment.
///
/// # Errors
/// `backend` is not a backend name.
pub fn select_backend(backend: &str) -> Result<(), PolicyError> {
    brain::select_backend(backend).map_err(|e| PolicyError::Load {
        path: backend.into(),
        reason: e.to_string(),
    })
}

/// One GPU's memory right now, as the driver reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct GpuCard {
    /// The card's index, the `i` of `--device gpu<i>`.
    pub index: u32,
    /// The card's size in bytes.
    pub total_bytes: u64,
    /// Bytes free right now, other processes' allocations included.
    pub free_bytes: u64,
    /// False when the free figure could not be read and was assumed, so it
    /// is never presented as a measurement.
    pub measured: bool,
}

/// Every GPU in this machine with the memory it has free, empty when there is
/// none. Reads the driver; loads no model and allocates nothing on a card.
#[must_use]
pub fn gpu_cards() -> Vec<GpuCard> {
    gpu_core::capacity::probe_gpus()
        .into_iter()
        .map(|card| GpuCard {
            index: card.index,
            total_bytes: card.total,
            free_bytes: card.free,
            measured: card.measured,
        })
        .collect()
}
