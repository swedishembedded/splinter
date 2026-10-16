// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Which hardware a process may use, chosen by the caller.
//!
//! A program takes `--device` and `--backend` as flags and passes them here
//! before loading anything; nothing is read from the environment. `--device`
//! is read once, by brain's own grammar (`gpu0`, `gpu0,gpu1`, `cpu0-7`, ...),
//! into the [`Devices`] every model built afterwards is scheduled on.

use std::sync::OnceLock;

use gpu_core::devices::{ComputeSet, Inventory};
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

/// The devices a process may schedule work on: what `--device` asked for,
/// resolved against what the machine has.
#[derive(Clone, Debug)]
pub struct Devices {
    spec: brain::Device,
    set: ComputeSet,
}

impl Devices {
    /// `text`, brain's `--device` grammar (empty for everything), resolved
    /// against `inventory`.
    ///
    /// # Errors
    /// `text` is not in the grammar, or names a device `inventory` lacks.
    pub fn resolve(text: &str, inventory: &Inventory) -> Result<Self, PolicyError> {
        let refuse = |reason: String| PolicyError::Device {
            text: text.to_string(),
            reason,
        };
        let spec = brain::Device::parse(text).map_err(refuse)?;
        let set = spec.resolve(inventory).map_err(refuse)?;
        Ok(Self { spec, set })
    }

    /// The GPUs work may run on, by index.
    #[must_use]
    pub fn gpus(&self) -> &[u32] {
        &self.set.gpus
    }

    /// The CPU cores work may run on.
    #[must_use]
    pub fn cpu_cores(&self) -> &[usize] {
        &self.set.cpu_cores
    }

    /// The value as given, or `all`.
    #[must_use]
    pub fn source(&self) -> &str {
        &self.set.source
    }

    /// The selection as brain's builders take it.
    #[must_use]
    pub fn spec(&self) -> brain::Device {
        self.spec.clone()
    }
}

static SELECTED: OnceLock<Devices> = OnceLock::new();

/// Fixes the devices this process schedules on to what `text` names, for
/// every model built after this call. The same text may be given again; a
/// different one is refused, as the first has already placed models.
///
/// # Errors
/// `text` names no schedulable device, or differs from the one selected.
pub fn select(text: &str) -> Result<&'static Devices, PolicyError> {
    if let Some(chosen) = SELECTED.get() {
        return same_selection(chosen, text);
    }
    let devices = Devices::resolve(text, &Inventory::probe())?;
    gpu_core::devices::publish_compute_set(devices.set.clone());
    devices
        .set
        .apply_backend()
        .map_err(|reason| PolicyError::Device {
            text: text.to_string(),
            reason,
        })?;
    Ok(SELECTED.get_or_init(|| devices))
}

/// The devices this process schedules on: what [`select`] fixed, else
/// everything the machine has.
#[must_use]
pub fn selected() -> &'static Devices {
    SELECTED.get().unwrap_or_else(|| {
        select("").unwrap_or_else(|e| unreachable!("the empty selection always resolves: {e}"))
    })
}

fn same_selection(chosen: &'static Devices, text: &str) -> Result<&'static Devices, PolicyError> {
    let asked = if text.trim().is_empty() {
        "all"
    } else {
        text.trim()
    };
    if chosen.source() == asked {
        Ok(chosen)
    } else {
        Err(PolicyError::Device {
            text: text.to_string(),
            reason: format!("the process already schedules on {}", chosen.source()),
        })
    }
}
