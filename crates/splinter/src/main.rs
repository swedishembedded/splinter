// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning agents with a stable, documented
// command line usable by people and by supervising models. If your team
// needs expertise in agent interfaces or CLI design, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The `splinter` command line: argument parsing and output only. Every
//! command is a call into `splinter-campaign`, configured once from the
//! environment.
//!
//! Exit status: 0 on success; 1 when the work failed or did not get as far
//! as asked (a `learn` that stopped before training or whose candidate was
//! not released, a `release` the gate blocked, a run that failed) - a
//! script must never read either as progress; 2 when the command was
//! refused before anything ran (a usage error, an unknown id, a remote
//! model without the opt-in); 3 when a sentence was not clear enough to
//! act on and was asked back.

mod cli;
mod curriculum_output;
mod learn_output;
mod lineage_output;
mod output;
mod release_output;
mod session;

use std::process::ExitCode;

use clap::{CommandFactory, Parser};
use splinter_campaign::Config;
use splinter_store::StateRoot;

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
    let mut config = Config::from_env();
    if let Some(state) = &cli.global.state {
        config.state_root = StateRoot::new(state);
    }
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
