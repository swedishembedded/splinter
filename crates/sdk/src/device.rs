// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements controlled placement of model work on named
// GPUs and CPU cores for its clients. If your team needs expertise in
// scheduling learning workloads on shared hardware, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The flags every program that asks a model takes, declared once: where
//! brain is, and on which compute it runs.
//!
//! A program flattens [`DeviceFlags`] into its command line, applies it to
//! its [`Config`] and, where it asks a model directly, takes the brain it is
//! to ask from [`DeviceFlags::connection`]. Splinter reaches brain only
//! through its API: `--brain-endpoint` names a server that is running, and
//! without it the program starts one of its own from `--brain-bin`. `--device`
//! is brain's grammar, handed to that server; no environment variable chooses
//! hardware.

use std::path::PathBuf;

use crate::model::{BrainConnection, BrainEndpoints, ManagedBrain, PolicyError, ServiceSettings};
use crate::Config;

/// `--device`, `--backend`, `--vram-budget` and the flags that name the brain
/// server, as a program declares them.
#[derive(clap::Args, Clone, Debug, Default)]
pub struct DeviceFlags {
    /// Which compute the brain server may schedule: `gpu`, `gpu1`,
    /// `gpu0,gpu1`, `cpu0-7`, `vulkan`, ... in brain's grammar. Without it,
    /// every device the machine has.
    #[arg(long, global = true)]
    pub device: Option<String>,
    /// How the hardware is driven: wgpu, vulkan, cuda or cpu. Without it
    /// brain picks, which on a CUDA host is not CUDA.
    #[arg(long, global = true)]
    pub backend: Option<String>,
    /// The VRAM, in GiB, this process was granted: every card the server
    /// budgets from is capped at it. Without it the whole card is budgeted.
    #[arg(long, global = true)]
    pub vram_budget: Option<f64>,
    /// A running `brain serve --openai` to request every local model from
    /// (its base URL, ending in `/v1`); repeat for more. Without one, a
    /// server is started from `--brain-bin`.
    #[arg(long, global = true)]
    pub brain_endpoint: Vec<String>,
    /// The file holding the endpoints' API key.
    #[arg(long, global = true)]
    pub brain_key_file: Option<PathBuf>,
    /// The `brain` binary a server is started from (default: the first on
    /// `PATH`).
    #[arg(long, global = true)]
    pub brain_bin: Option<PathBuf>,
}

impl DeviceFlags {
    /// Removes `--device VALUE`, `--backend VALUE`, `--vram-budget GIB`,
    /// `--brain-endpoint URL` (any number), `--brain-key-file FILE` and
    /// `--brain-bin FILE` from `args` and returns them, for a program that
    /// reads its own arguments.
    ///
    /// # Errors
    /// A flag has no value after it, or `--vram-budget` is not a positive number.
    pub fn take(args: &mut Vec<String>) -> Result<Self, String> {
        let mut take_all = |name: &str| -> Result<Vec<String>, String> {
            let mut values = Vec::new();
            while let Some(at) = args.iter().position(|a| a == name) {
                if at + 1 >= args.len() {
                    return Err(format!("{name} needs a value"));
                }
                values.push(args.remove(at + 1));
                args.remove(at);
            }
            Ok(values)
        };
        let device = take_all("--device")?.pop();
        let backend = take_all("--backend")?.pop();
        let vram_budget = take_all("--vram-budget")?
            .pop()
            .map(|text| match text.parse::<f64>() {
                Ok(gib) if gib > 0.0 && gib.is_finite() => Ok(gib),
                _ => Err(format!(
                    "--vram-budget needs a positive number of GiB, got {text}"
                )),
            })
            .transpose()?;
        let brain_endpoint = take_all("--brain-endpoint")?;
        let brain_key_file = take_all("--brain-key-file")?.pop().map(PathBuf::from);
        let brain_bin = take_all("--brain-bin")?.pop().map(PathBuf::from);
        Ok(Self {
            device,
            backend,
            vram_budget,
            brain_endpoint,
            brain_key_file,
            brain_bin,
        })
    }

    /// Puts the flags into `config`: the compute and the binary of the server
    /// a run starts, and the endpoints of one it is given.
    ///
    /// # Errors
    /// An endpoint is not a base URL, it has no key file, or the key file
    /// cannot be read.
    pub fn apply_to(&self, config: &mut Config) -> Result<(), PolicyError> {
        config.brain_device.clone_from(&self.device);
        config.brain_backend.clone_from(&self.backend);
        if let Some(gib) = self.vram_budget {
            crate::model::device::set_vram_budget(gib);
        }
        if let Some(binary) = &self.brain_bin {
            config.brain_binary = Some(binary.clone());
        }
        if self.brain_endpoint.is_empty() {
            return Ok(());
        }
        let Some(key_file) = &self.brain_key_file else {
            return Err(PolicyError::Endpoint {
                url: self.brain_endpoint[0].clone(),
                reason: "--brain-endpoint needs --brain-key-file".into(),
            });
        };
        let key = BrainEndpoints::read_key(key_file)?;
        config.brain = Some(BrainEndpoints::new(self.brain_endpoint.clone(), key)?);
        Ok(())
    }

    /// States the flags as the process's brain settings, for a program that
    /// builds its [`Config`] in several places: every [`Config::from_env`]
    /// after carries them.
    ///
    /// # Errors
    /// The flags are refused ([`Self::apply_to`]), or were already stated.
    pub fn apply(&self) -> Result<(), PolicyError> {
        let mut config = Config::from_env();
        self.apply_to(&mut config)?;
        crate::set_brain_defaults(crate::BrainDefaults {
            device: config.brain_device,
            backend: config.brain_backend,
            max_batch: config.brain_max_batch,
            endpoints: config.brain,
            binary: self.brain_bin.clone(),
        })
        .map_err(|e| PolicyError::Endpoint {
            url: String::new(),
            reason: e.into(),
        })
    }

    /// [`Self::connection`] for a program that keeps no [`Config`] of its own:
    /// the machine's defaults (`brain` on `PATH`, brain's model store) with
    /// these flags applied.
    ///
    /// # Errors
    /// The flags are refused ([`Self::apply_to`]) or no server can be reached
    /// or started ([`Self::connection`]).
    pub fn brain(&self) -> Result<BrainConnection, PolicyError> {
        let mut config = Config::from_env();
        self.apply_to(&mut config)?;
        self.connection(&config)
    }

    /// The brain a program that asks a model directly requests it from: the
    /// endpoints `config` names, else a server started for the program's
    /// lifetime ([`Runtime::brain`](crate::Runtime::brain) does the same for
    /// a program that runs on a runtime).
    ///
    /// # Errors
    /// No endpoint is named and no server can be started.
    pub fn connection(&self, config: &Config) -> Result<BrainConnection, PolicyError> {
        if let Some(endpoints) = &config.brain {
            return Ok(BrainConnection::Given(endpoints.clone()));
        }
        let binary = config
            .brain_binary
            .clone()
            .ok_or_else(|| PolicyError::Endpoint {
                url: String::new(),
                reason: "name a brain server with --brain-endpoint, or the `brain` binary a \
                         server is started from with --brain-bin"
                    .into(),
            })?;
        let settings = ServiceSettings {
            binary,
            models: config.model_store.clone(),
            state: config.state_root.brain_service(),
            device: config.brain_device.clone(),
            backend: config.brain_backend.clone(),
            max_batch: config.brain_max_batch,
        };
        ManagedBrain::start(&settings, &crate::agent::CancelToken::default())
            .map(BrainConnection::Owned)
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
    fn the_vram_budget_is_taken_as_gib_and_refused_when_not_a_positive_number() {
        let mut args = words("train --vram-budget 66 --out dir");
        let flags = DeviceFlags::take(&mut args).unwrap();
        assert_eq!(flags.vram_budget, Some(66.0));
        assert_eq!(args, words("train --out dir"));
        for bad in ["0", "-3", "lots", "NaN"] {
            let mut args = words(&format!("train --vram-budget {bad}"));
            assert!(DeviceFlags::take(&mut args).is_err(), "{bad}");
        }
    }

    #[test]
    fn a_program_without_the_flags_is_left_alone() {
        let mut args = words("train --out dir");
        let flags = DeviceFlags::take(&mut args).unwrap();
        assert!(flags.device.is_none() && flags.backend.is_none() && flags.vram_budget.is_none());
        assert!(flags.brain_endpoint.is_empty() && flags.brain_bin.is_none());
        assert_eq!(args, words("train --out dir"));
    }

    #[test]
    fn the_brain_flags_are_taken_and_endpoints_may_repeat() {
        let mut args = words(
            "run --brain-endpoint http://a/v1 --brain-key-file k --brain-endpoint http://b/v1 \
             --brain-bin /bin/brain --x",
        );
        let flags = DeviceFlags::take(&mut args).unwrap();
        assert_eq!(flags.brain_endpoint, ["http://a/v1", "http://b/v1"]);
        assert_eq!(flags.brain_key_file, Some(PathBuf::from("k")));
        assert_eq!(flags.brain_bin, Some(PathBuf::from("/bin/brain")));
        assert_eq!(args, words("run --x"));
    }

    #[test]
    fn endpoints_without_a_key_file_are_refused_and_flags_land_in_the_config() {
        let mut config = Config::from_env();
        let flags = DeviceFlags {
            device: Some("gpu1".into()),
            brain_endpoint: vec!["http://127.0.0.1:1/v1".into()],
            ..DeviceFlags::default()
        };
        let refused = flags.apply_to(&mut config).unwrap_err().to_string();
        assert!(refused.contains("--brain-key-file"), "{refused}");
        assert_eq!(config.brain_device.as_deref(), Some("gpu1"));
        assert!(config.brain.is_none());
    }

    #[test]
    fn a_program_with_no_server_and_no_binary_is_told_what_to_name() {
        let mut config = Config::from_env();
        config.brain = None;
        config.brain_binary = None;
        let refused = DeviceFlags::default()
            .connection(&config)
            .err()
            .expect("refused")
            .to_string();
        assert!(
            refused.contains("--brain-endpoint") && refused.contains("--brain-bin"),
            "{refused}"
        );
    }

    #[test]
    fn a_flag_without_a_value_is_refused() {
        let mut args = words("train --device");
        assert!(DeviceFlags::take(&mut args).is_err());
    }
}
