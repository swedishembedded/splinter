// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Which backend drives the hardware, chosen by the caller.

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
