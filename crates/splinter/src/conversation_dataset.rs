// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The datasets made from files rather than from experience sets:
//! `dataset conversations` (graded conversations with a counterpart, as a
//! policy-gradient dataset) and `dataset decision-fixtures`.

use std::path::Path;

use splinter_sdk::conversation_policy_gradient::store;
use splinter_sdk::data::AdvantageKind;
use splinter_sdk::{Context, Error};

use crate::cli::DatasetCommand;
use crate::output::emit;

/// Reads `input` and stores the dataset its admitted conversations make.
pub fn run(
    ctx: &Context,
    json: bool,
    input: &Path,
    system_prompt: &str,
    advantage: Option<AdvantageKind>,
) -> Result<(), Error> {
    let lines = std::fs::read_to_string(input)
        .map_err(|e| Error::Refused(format!("{}: {e}", input.display())))?;
    emit(
        json,
        &store(ctx, system_prompt, &lines, advantage.unwrap_or_default())?,
    );
    Ok(())
}

/// Runs the dataset command `command`, one of the two this module serves.
pub fn dispatch(ctx: &Context, json: bool, command: DatasetCommand) -> Result<(), Error> {
    match command {
        DatasetCommand::Conversations(args) => {
            run(ctx, json, &args.input, &args.system_prompt, args.advantage)
        }
        DatasetCommand::DecisionFixtures(args) => crate::decision_fixtures_dataset::run(
            json,
            &args.runs,
            &args.out,
            splinter_sdk::decision_fixtures::SplitPlan {
                seed: splinter_sdk::seed::resolve("split-seed", args.split_seed),
                validation_fraction: args.validation_fraction,
                test_fraction: args.test_fraction,
                holdout_suites: args.holdout_suites,
            },
        ),
        other => Err(Error::Refused(format!(
            "{other:?} is not a dataset made from a file"
        ))),
    }
}
