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

use splinter_sandbox::job::{JobLauncher, LocalLauncher, NodeStepLauncher, SlurmLauncher};

use crate::cluster::ClusterSettings;
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
    /// The allocation the step launcher places ranks on cannot be used.
    #[error("the slurm-step job launcher: {0}")]
    Cluster(String),
}

/// The launcher `settings` selects, inside the allocation `cluster`
/// describes.
///
/// The local kind starts each worker on this machine, under the configured
/// supervisor when there is one. The Slurm kind submits the job with the
/// configured `sbatch`, observes it with `squeue` and stops it with `scancel`.
/// The step kind runs every rank as a task of one `srun` step on the
/// allocation's usable nodes. The automatic kind, the default, is the step
/// kind inside an allocation that spans nodes and the local kind otherwise.
pub fn launcher(
    settings: &JobSettings,
    cluster: &ClusterSettings,
) -> Result<Box<dyn JobLauncher>, LauncherError> {
    let spans = || -> Result<Option<Vec<String>>, LauncherError> {
        let usable = cluster
            .usable()
            .map_err(|e| LauncherError::Cluster(e.to_string()))?;
        Ok(usable.filter(|u| u.spans_nodes()).map(|u| u.nodes))
    };
    let kind = match settings.launcher {
        LauncherKind::Auto if spans()?.is_some() => LauncherKind::SlurmStep,
        LauncherKind::Auto => LauncherKind::Local,
        kind => kind,
    };
    match kind {
        LauncherKind::Auto | LauncherKind::Local => Ok(Box::new(match &settings.supervisor {
            Some(supervisor) => LocalLauncher::supervised(supervisor.clone()),
            None => LocalLauncher::default(),
        })),
        LauncherKind::SlurmStep => {
            let nodes = match cluster.usable() {
                Ok(Some(usable)) => usable.nodes,
                Ok(None) => {
                    return Err(LauncherError::Cluster(
                        "no Slurm allocation is in use (none detected, or --cluster off)".into(),
                    ))
                }
                Err(e) => return Err(LauncherError::Cluster(e.to_string())),
            };
            let srun = cluster.srun.clone().ok_or(LauncherError::MissingSetting {
                setting: "--slurm-srun",
                purpose: "the program that starts a step on the allocation's nodes",
            })?;
            let steps = splinter_sandbox::step::NodeSteps {
                srun,
                scancel: cluster.scancel.clone(),
                args: cluster.step_args.clone(),
                single_node_args: cluster.single_node_step_args.clone(),
                exec_prefix: cluster
                    .rank_exec_prefix
                    .clone()
                    .unwrap_or_else(|| cluster.exec_prefix.clone()),
            };
            Ok(Box::new(NodeStepLauncher::new(
                steps,
                nodes,
                settings.supervisor.as_deref(),
            )))
        }
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

    fn none() -> ClusterSettings {
        ClusterSettings::default()
    }

    fn allocation(nodes: &[&str]) -> ClusterSettings {
        ClusterSettings {
            detected: Ok(Some(crate::cluster::ClusterAllocation {
                job_id: "1".into(),
                nodes: nodes.iter().map(|n| (*n).to_string()).collect(),
                gpus_per_node: 4,
                this_node: Some(nodes[0].to_string()),
            })),
            srun: Some(PathBuf::from("/scheduler/srun")),
            ..ClusterSettings::default()
        }
    }

    #[test]
    fn the_default_settings_launch_locally() {
        let settings = JobSettings::default();
        assert_eq!(launcher(&settings, &none()).unwrap().name(), "local");
        assert_eq!(
            launcher(&settings, &allocation(&["n1"])).unwrap().name(),
            "local",
            "a one-node allocation stays on this machine"
        );
    }

    #[test]
    fn inside_an_allocation_that_spans_nodes_jobs_run_as_one_step_unless_told_otherwise() {
        let settings = JobSettings::default();
        let two = allocation(&["n1", "n2"]);
        assert_eq!(launcher(&settings, &two).unwrap().name(), "slurm-step");
        let local = JobSettings {
            launcher: LauncherKind::Local,
            ..JobSettings::default()
        };
        assert_eq!(launcher(&local, &two).unwrap().name(), "local");
        let step = JobSettings {
            launcher: LauncherKind::SlurmStep,
            ..JobSettings::default()
        };
        let Err(error) = launcher(&step, &none()) else {
            panic!("a step launcher without an allocation");
        };
        assert!(error.to_string().contains("no Slurm allocation"), "{error}");
        let no_srun = ClusterSettings {
            srun: None,
            ..allocation(&["n1", "n2"])
        };
        let Err(error) = launcher(&settings, &no_srun) else {
            panic!("a step launcher without srun");
        };
        assert!(error.to_string().contains("--slurm-srun"), "{error}");
    }

    #[test]
    fn a_supervisor_keeps_the_launcher_local() {
        let settings = JobSettings {
            supervisor: Some(PathBuf::from("/splinter")),
            ..JobSettings::default()
        };
        assert_eq!(launcher(&settings, &none()).unwrap().name(), "local");
    }

    #[test]
    fn a_slurm_configuration_builds_the_slurm_launcher() {
        assert_eq!(
            launcher(&slurm_settings(), &none()).unwrap().name(),
            "slurm"
        );
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
            let Err(error) = launcher(&settings, &none()) else {
                panic!("{setting} was not refused");
            };
            let LauncherError::MissingSetting { setting: named, .. } = &error else {
                panic!("{error}");
            };
            assert_eq!(*named, setting);
            assert!(
                error.to_string().contains(setting),
                "the refusal names {setting}: {error}"
            );
        }
    }
}
