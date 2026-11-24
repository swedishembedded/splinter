// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning agents with a stable, documented
// command line usable by people and by supervising models. If your team
// needs expertise in agent interfaces or CLI design, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The `splinter` command line: argument parsing and output only. Every
//! command is a call into `splinter-pipelines`, configured once from the
//! environment.
//!
//! Exit status: 0 on success; 1 when the work failed or did not get as far
//! as asked (an `absorb` whose candidate was not released, a `release` the
//! gate blocked, a run that failed) - a
//! script must never read either as progress; 2 when the command was
//! refused before anything ran (a usage error, an unknown id, a remote
//! model without the opt-in); 3 when a sentence was not clear enough to
//! act on and was asked back.

mod absorb_output;
mod absorb_session;
mod arms_output;
mod ask_session;
mod baseline_report;
mod build_info;
mod cli;
mod compare_output;
mod compare_session;
mod config_session;
mod consolidate_output;
mod continual_output;
mod conversation_dataset;
mod corrections_session;
mod curriculum_output;
mod dataset_output;
mod decision_fixtures_dataset;
mod diagnosis_output;
mod doctor_output;
mod env_output;
mod episodes_output;
mod eval_session;
mod exam_output;
mod exam_session;
mod experiment_output;
mod factory_session;
mod goal_session;
mod harbor_output;
mod hub_session;
mod improve_session;
mod jobs_session;
mod lineage_output;
mod objects_output;
mod output;
mod panic_report;
mod release_output;
mod session;
mod stage_output;
mod state_output;
mod tasks_output;
mod tasks_session;
mod train_output;

use std::process::ExitCode;

use clap::{CommandFactory, Parser};
use splinter_sdk::store::StateRoot;
use splinter_sdk::Config;

use cli::{Cli, Command};
use session::Session;

/// The global flags that say where things are on this machine and which
/// models play which part: each replaces the default `Config::from_env` found.
fn apply_machine_flags(config: &mut Config, global: &cli::Global) {
    if let Some(file) = &global.config {
        config.factory.config.clone_from(file);
    }
    if let Some(state) = &global.state {
        config.state_root = StateRoot::new(state);
    }
    if let Some(owner) = &global.job_owner {
        config.jobs.owner.clone_from(owner);
    }
    if let Some(root) = &global.job_attempt_root {
        config.jobs.attempt_root = Some(root.clone());
    }
    if let Some(kind) = global.job_launcher {
        config.jobs.launcher = kind;
    }
    if let Some(binary) = &global.slurm_sbatch {
        config.jobs.slurm.sbatch = Some(binary.clone());
    }
    if let Some(binary) = &global.slurm_squeue {
        config.jobs.slurm.squeue = Some(binary.clone());
    }
    if let Some(binary) = &global.slurm_scancel {
        config.jobs.slurm.scancel = Some(binary.clone());
    }
    config.jobs.slurm.args.clone_from(&global.slurm_arg);
    // The binary supervises each worker it launches, so the worker's exit is
    // recorded where a later process reads it.
    config.jobs.supervisor = std::env::current_exe().ok();
    if let Some(dir) = &global.models_dir {
        config.model_store.clone_from(dir);
        config.policy_base = dir.join(splinter_sdk::config::DEFAULT_POLICY_MODEL);
    }
    if let Some(base) = &global.policy_base {
        config.policy_base = splinter_sdk::model_ref::checkpoint_path(config, base);
    }
    config.assistant_model.clone_from(&global.assistant_model);
    config.judge_model.clone_from(&global.judge_model);
    config.front_door_model.clone_from(&global.front_door_model);
    config.bf16_base = global.bf16_base;
    if let Some(n) = global.min_calibration_controls {
        config.min_calibration_controls = usize::try_from(n).unwrap_or(usize::MAX);
    }
    config.allow_remote = global.allow_remote;
    config.workspace.docker.clone_from(&global.docker);
    config.workspace.image.clone_from(&global.workspace_image);
    config
        .workspace
        .scratch_root
        .clone_from(&global.workspace_scratch);
    config.workspace.cargo_home.clone_from(&global.cargo_home);
}

/// Puts the command line's brain settings into `config`: the server it is
/// given (`--brain-endpoint`, with what configures it) or the one it starts.
/// A flag that configures endpoints is refused without the endpoints it
/// configures.
fn apply_brain_flags(config: &mut Config, global: &cli::Global) -> Result<(), String> {
    global.device.apply_to(config).map_err(|e| e.to_string())?;
    // The program that cancels a batch job also stops a step on a node.
    if let Some(scancel) = &global.slurm_scancel {
        config.cluster.scancel = Some(scancel.clone());
    }
    config.brain_max_batch = global.attempt_concurrency;
    let Some(brain) = config.brain.take() else {
        let stray = [
            (
                global.brain_adapter_manifest.is_some(),
                "--brain-adapter-manifest",
            ),
            (global.brain_adapter_dir.is_some(), "--brain-adapter-dir"),
            (
                global.brain_artifact_store.is_some(),
                "--brain-artifact-store",
            ),
        ];
        return match stray.iter().find(|(given, _)| *given) {
            Some((_, flag)) => Err(format!(
                "{flag} configures --brain-endpoint, which is not given"
            )),
            None => Ok(()),
        };
    };
    let mut brain = brain;
    if let Some(manifest) = &global.brain_adapter_manifest {
        brain = brain.with_adapter_manifest(manifest.clone());
    }
    if let Some(directory) = &global.brain_adapter_dir {
        brain = brain.with_adapter_directory(directory.clone());
    }
    if let Some(width) = global.attempt_concurrency {
        brain = brain.with_attempt_concurrency(width as usize);
    }
    if let Some(dir) = &global.brain_artifact_store {
        let store = splinter_sdk::model::served_train::ArtifactStore::open(dir)
            .map_err(|e| e.to_string())?;
        brain = brain.with_artifact_store(store);
    }
    config.brain = Some(brain);
    Ok(())
}

/// Ctrl-C and SIGTERM cancel the runs in progress and their brain jobs, or,
/// with `detach`, leave the jobs running; each says what it did on stderr.
fn handle_interrupts(detach: bool) {
    use splinter_sdk::interrupt::{self, Interrupted, Interrupts};
    let interrupts = if detach {
        Interrupts::Detach
    } else {
        Interrupts::Cancel
    };
    let installed = interrupt::handle_interrupts(interrupts, |what| {
        eprintln!(
            "{}",
            match what {
                Interrupted::Cancelling => {
                    "interrupted: cancelling the run and its brain jobs (interrupt again to exit \
                     at once and leave them running)"
                }
                Interrupted::Detaching => {
                    "interrupted: exiting and leaving the brain jobs running; the same command \
                     continues them"
                }
                Interrupted::Exiting => "interrupted",
            }
        );
    });
    if let Err(e) = installed {
        eprintln!("Ctrl-C will not cancel brain jobs: the signal handlers failed to install: {e}");
    }
}

fn main() -> ExitCode {
    // A worker launched through the local job launcher starts this binary as
    // its supervisor, which runs the worker and records its exit.
    if let Some(code) = splinter_sdk::sandbox::job::supervise(std::env::args_os().skip(1)) {
        return ExitCode::from(u8::try_from(code).unwrap_or(1));
    }
    let cli = Cli::parse();
    if cli.sentence.is_some() && cli.command.is_some() {
        Cli::command()
            .error(
                clap::error::ErrorKind::ArgumentConflict,
                "a sentence and a command cannot be given together",
            )
            .exit();
    }
    if let Some(Command::CliReference(args)) = &cli.command {
        return cli::reference::run(args).into();
    }
    let mut config = Config::from_env();
    apply_machine_flags(&mut config, &cli.global);
    if let Some(Command::Config(command)) = &cli.command {
        return config_session::run(&config.factory.config, cli.global.json, command).into();
    }
    config.policy_context_tokens = cli.global.policy_context_tokens;
    if let Some(width) = cli.global.remote_concurrency {
        config.remote_concurrency = width as usize;
    }
    if let Err(e) = apply_brain_flags(&mut config, &cli.global) {
        output::error(cli.global.json, &splinter_sdk::Error::Refused(e));
        return session::Exit::Refused.into();
    }
    config.served_temperature = cli.global.temperature;
    config.served_top_p = cli.global.top_p;
    config.served_top_k = cli.global.top_k;
    config.served_seed = cli.global.sampling_seed;
    config.thinking |= cli.global.thinking;
    config.served_thinking_budget = cli.global.thinking_budget;
    if let Some(prompt) = cli.global.coding_prompt {
        config.coding_prompt = prompt;
    }
    if let Some(seconds) = cli.global.solve_deadline {
        config.solve_deadline = std::time::Duration::from_secs(seconds);
    }
    if let Some(bytes) = cli.global.test_output_cap {
        config.workspace.output_cap_bytes = bytes;
    }
    config.workspace.allow_unconfined = cli.global.allow_unconfined_workspace;
    config.workspace.allow_network = cli.global.allow_workspace_network;
    config.workspace.max_processes = cli.global.workspace_max_processes;
    config.workspace.env = cli.global.workspace_env.iter().cloned().collect();
    config.workspace.memory_bytes = cli.global.workspace_memory_gib.map(|gib| gib << 30);
    config.workspace.tree_memory_bytes = cli.global.workspace_tree_memory_gib.map(|gib| gib << 30);
    config
        .workspace
        .runtime_dirs
        .clone_from(&cli.global.runtime_dir);
    config.build = build_info::current(cli.global.gpu_contexts);
    handle_interrupts(cli.global.detach_on_interrupt);
    let session = match Session::new(config, cli.global.clone()) {
        Ok(session) => session,
        Err(e) => {
            // A configuration the command cannot run with is a refusal: it
            // stopped before any work began.
            let exit = if e.is_refusal() {
                session::Exit::Refused
            } else {
                session::Exit::Failed
            };
            output::error(cli.global.json, &e);
            return exit.into();
        }
    };
    // A panic that no layer below contained (a device fault, a trainer's
    // invariant) must end in a report and a failing exit status, never in a
    // bare backtrace a script reads as no output at all.
    let json = cli.global.json;
    let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        match (cli.command, cli.sentence) {
            (Some(command), _) => session.command(command),
            (None, Some(sentence)) => session.sentence(&sentence),
            (None, None) => session.repl(),
        }
    }));
    match ran {
        Ok(exit) => exit.into(),
        Err(panic) => {
            let message = panic
                .downcast_ref::<&str>()
                .map(|m| (*m).to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "a panic with no message".into());
            panic_report::panic_report(json, &message);
            session::Exit::Failed.into()
        }
    }
}
