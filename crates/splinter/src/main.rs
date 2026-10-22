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
//! as asked (a `learn` that stopped before training or whose candidate was
//! not released, a `release` the gate blocked, a run that failed) - a
//! script must never read either as progress; 2 when the command was
//! refused before anything ran (a usage error, an unknown id, a remote
//! model without the opt-in); 3 when a sentence was not clear enough to
//! act on and was asked back.

mod absorb_output;
mod absorb_session;
mod acquire_output;
mod baseline_report;
mod build_info;
mod cli;
mod compare_output;
mod compare_session;
mod consolidate_output;
mod continual_output;
mod curriculum_output;
mod doctor_output;
mod exam_output;
mod exam_session;
mod experiment_output;
mod harness_output;
mod harness_session;
mod improve_session;
mod learn_output;
mod learn_request;
mod learn_session;
mod lineage_output;
mod output;
mod practice_output;
mod practice_request;
mod release_output;
mod session;
mod state_output;
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
    if let Some(state) = &global.state {
        config.state_root = StateRoot::new(state);
    }
    if let Some(dir) = &global.models_dir {
        config.model_store.clone_from(dir);
        config.policy_base = dir.join(splinter_sdk::config::DEFAULT_POLICY_MODEL);
    }
    if let Some(base) = &global.policy_base {
        config.policy_base = splinter_sdk::model_ref::checkpoint_path(config, base);
    }
    if let Some(binary) = &global.brain_bin {
        config.brain_binary = Some(binary.clone());
    }
    config.assistant_model.clone_from(&global.assistant_model);
    config.judge_model.clone_from(&global.judge_model);
    config.front_door_model.clone_from(&global.front_door_model);
    config.default_budget.clone_from(&global.default_budget);
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

fn main() -> ExitCode {
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
    if let Err(e) = cli.global.device.apply() {
        output::error(
            cli.global.json,
            &splinter_sdk::Error::Refused(e.to_string()),
        );
        return session::Exit::Failed.into();
    }
    let mut config = Config::from_env();
    apply_machine_flags(&mut config, &cli.global);
    config.policy_context_tokens = cli.global.policy_context_tokens;
    if let Some(width) = cli.global.remote_concurrency {
        config.remote_concurrency = width as usize;
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
    let session = match Session::new(config, cli.global.clone()) {
        Ok(session) => session,
        Err(e) => {
            output::error(cli.global.json, &e);
            return session::Exit::Failed.into();
        }
    };
    let exit = match (cli.command, cli.sentence) {
        (Some(command), _) => session.command(command),
        (None, Some(sentence)) => session.sentence(&sentence),
        (None, None) => session.repl(),
    };
    exit.into()
}
