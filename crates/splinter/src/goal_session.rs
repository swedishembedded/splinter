// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `goal admit`: what may enter a selection of learning material.

use splinter_sdk::Error;

use crate::cli::GoalCommand;
use crate::output::emit;

/// Runs a `goal` command; `false` when documents were refused.
pub fn run(json: bool, command: GoalCommand) -> Result<bool, Error> {
    match command {
        GoalCommand::Admit {
            open,
            live,
            consent,
            scrub,
            out,
        } => {
            let admission =
                splinter_sdk::goal::intake::admit_paths(&open, &live, consent, scrub, &out)?;
            emit(json, &admission);
            Ok(admission.refused.is_empty())
        }
    }
}
