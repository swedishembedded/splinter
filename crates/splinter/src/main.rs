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
mod build_info;
mod cli;
mod compare_output;
mod consolidate_output;
mod continual_output;
mod curriculum_output;
mod doctor_output;
mod exam_output;
mod exam_session;
mod tasks_session;
mod experiment_output;
mod learn_output;
mod learn_request;
mod lineage_output;
mod output;
mod practice_output;
mod practice_request;
mod release_output;
mod session;
mod state_output;
mod train_output;

use std::process::ExitCode;

use clap::{CommandFactory, Parser};
use splinter_sdk::store::StateRoot;
use splinter_sdk::Config;

use cli::Cli;
use session::Session;

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
    if let Err(e) = cli.global.device.apply() {
        output::error(
            cli.global.json,
            &splinter_sdk::Error::Refused(e.to_string()),
        );
        return session::Exit::Failed.into();
    }
    let mut config = Config::from_env();
    if let Some(state) = &cli.global.state {
        config.state_root = StateRoot::new(state);
    }
    config.policy_context_tokens = cli.global.policy_context_tokens;
    config.build = build_info::current(config.build.gpu_contexts);
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
