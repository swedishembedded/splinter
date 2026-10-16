// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements controlled placement of model work on named
// GPUs and CPU cores for its clients. If your team needs expertise in
// scheduling learning workloads on shared hardware, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The hardware flags every program that loads a model takes, declared once.
//!
//! A program flattens [`DeviceFlags`] into its command line and calls
//! [`DeviceFlags::apply`] before anything is loaded. `--device` is brain's
//! grammar, read once into the devices the process may schedule on; no
//! environment variable chooses hardware.

use crate::model::device::{select, select_backend, Devices};
use crate::model::PolicyError;

/// `--device` and `--backend`, as a program declares them.
#[derive(clap::Args, Clone, Debug, Default)]
pub struct DeviceFlags {
    /// Which compute may be scheduled: `gpu`, `gpu1`, `gpu0,gpu1`, `cpu0-7`,
    /// `vulkan`, ... in brain's grammar. Without it, every device the
    /// machine has.
    #[arg(long, global = true)]
    pub device: Option<String>,
    /// How the hardware is driven: wgpu, vulkan, cuda or cpu. Without it
    /// brain picks, which on a CUDA host is not CUDA.
    #[arg(long, global = true)]
    pub backend: Option<String>,
}

impl DeviceFlags {
    /// Removes `--device VALUE` and `--backend VALUE` from `args` and returns
    /// them, for a program that reads its own arguments.
    ///
    /// # Errors
    /// A flag has no value after it.
    pub fn take(args: &mut Vec<String>) -> Result<Self, String> {
        let mut take = |name: &str| -> Result<Option<String>, String> {
            let Some(at) = args.iter().position(|a| a == name) else {
                return Ok(None);
            };
            if at + 1 >= args.len() {
                return Err(format!("{name} needs a value"));
            }
            let value = args.remove(at + 1);
            args.remove(at);
            Ok(Some(value))
        };
        Ok(Self {
            device: take("--device")?,
            backend: take("--backend")?,
        })
    }

    /// Fixes the process's backend and schedulable devices, and says which
    /// they are on standard error.
    ///
    /// # Errors
    /// A flag names no backend or device, or the machine lacks it.
    pub fn apply(&self) -> Result<&'static Devices, PolicyError> {
        if let Some(backend) = &self.backend {
            select_backend(backend)?;
        }
        let devices = select(self.device.as_deref().unwrap_or(""))?;
        eprintln!(
            "devices: {} (gpus {:?}, {} cpu cores)",
            devices.source(),
            devices.gpus(),
            devices.cpu_cores().len()
        );
        Ok(devices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(text: &str) -> Vec<String> {
        text.split_whitespace().map(String::from).collect()
    }

    #[test]
    fn the_flags_are_taken_out_wherever_they_stand() {
        let mut args = words("train --device gpu1 --out dir --backend cuda");
        let flags = DeviceFlags::take(&mut args).unwrap();
        assert_eq!(flags.device.as_deref(), Some("gpu1"));
        assert_eq!(flags.backend.as_deref(), Some("cuda"));
        assert_eq!(args, words("train --out dir"));
    }

    #[test]
    fn a_program_without_the_flags_is_left_alone() {
        let mut args = words("train --out dir");
        let flags = DeviceFlags::take(&mut args).unwrap();
        assert!(flags.device.is_none() && flags.backend.is_none());
        assert_eq!(args, words("train --out dir"));
    }

    #[test]
    fn a_flag_without_a_value_is_refused() {
        let mut args = words("train --device");
        assert!(DeviceFlags::take(&mut args).is_err());
    }
}
