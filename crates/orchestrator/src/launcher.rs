// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements selectable job launching for self-improving
// systems, for its clients. If your team needs expertise in scheduler
// integration or process supervision, you can procure our services by sending
// an email to info@swedishembedded.com.

//! The launcher the configuration selects: this machine's processes, or a
//! batch system.
//!
//! [`launcher`] is the one place a [`JobLauncher`] is built, so the controller
//! and the SDK never name a concrete launcher. The settings are values: a
//! Slurm configuration that names no program is refused by the setting it is
//! missing, before anything is submitted.

use splinter_sandbox::job::{JobLauncher, LocalLauncher, SlurmLauncher};

use crate::config::{JobSettings, LauncherKind};

/// Why the configured launcher cannot be built.
#[derive(Debug, thiserror::Error)]
pub enum LauncherError {
    /// A setting the selected launcher needs was not given.
    #[error("the slurm job launcher needs {setting}: {purpose}")]
    MissingSetting {
        /// The flag that sets it.
        setting: &'static str,
        /// What that program does.
        purpose: &'static str,
    },
}

/// The launcher `settings` selects.
///
/// The local kind starts each worker on this machine, under the configured
/// supervisor when there is one. The Slurm kind submits the job with the
/// configured `sbatch`, observes it with `squeue` and stops it with `scancel`.
pub fn launcher(settings: &JobSettings) -> Result<Box<dyn JobLauncher>, LauncherError> {
    match settings.launcher {
        LauncherKind::Local => Ok(Box::new(match &settings.supervisor {
            Some(supervisor) => LocalLauncher::supervised(supervisor.clone()),
            None => LocalLauncher::default(),
        })),
        LauncherKind::Slurm => {
            let slurm = &settings.slurm;
            let sbatch = slurm.sbatch.clone().ok_or(LauncherError::MissingSetting {
                setting: "--slurm-sbatch",
                purpose: "the program that submits a job",
            })?;
            let squeue = slurm.squeue.clone().ok_or(LauncherError::MissingSetting {
                setting: "--slurm-squeue",
                purpose: "the program that reports a job's state",
            })?;
            let scancel = slurm.scancel.clone().ok_or(LauncherError::MissingSetting {
                setting: "--slurm-scancel",
                purpose: "the program that stops a job",
            })?;
            Ok(Box::new(
                SlurmLauncher::new(sbatch, squeue, scancel).with_sbatch_args(slurm.args.clone()),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn slurm_settings() -> JobSettings {
        JobSettings {
            launcher: LauncherKind::Slurm,
            slurm: crate::config::SlurmSettings {
                sbatch: Some(PathBuf::from("/scheduler/sbatch")),
                squeue: Some(PathBuf::from("/scheduler/squeue")),
                scancel: Some(PathBuf::from("/scheduler/scancel")),
                args: vec!["--partition=debug".into()],
            },
            ..JobSettings::default()
        }
    }

    #[test]
    fn the_default_settings_launch_locally() {
        let settings = JobSettings::default();
        assert_eq!(launcher(&settings).unwrap().name(), "local");
    }

    #[test]
    fn a_supervisor_keeps_the_launcher_local() {
        let settings = JobSettings {
            supervisor: Some(PathBuf::from("/splinter")),
            ..JobSettings::default()
        };
        assert_eq!(launcher(&settings).unwrap().name(), "local");
    }

    #[test]
    fn a_slurm_configuration_builds_the_slurm_launcher() {
        assert_eq!(launcher(&slurm_settings()).unwrap().name(), "slurm");
    }

    #[test]
    fn a_missing_slurm_program_is_refused_by_its_setting() {
        for (setting, clear) in [
            ("--slurm-sbatch", 0usize),
            ("--slurm-squeue", 1),
            ("--slurm-scancel", 2),
        ] {
            let mut settings = slurm_settings();
            match clear {
                0 => settings.slurm.sbatch = None,
                1 => settings.slurm.squeue = None,
                _ => settings.slurm.scancel = None,
            }
            let Err(error) = launcher(&settings) else {
                panic!("{setting} was not refused");
            };
            let LauncherError::MissingSetting { setting: named, .. } = &error;
            assert_eq!(*named, setting);
            assert!(
                error.to_string().contains(setting),
                "the refusal names {setting}: {error}"
            );
        }
    }
}
