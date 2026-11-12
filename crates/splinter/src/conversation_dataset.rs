// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `dataset conversations`: graded conversations with a counterpart, as a
//! policy-gradient dataset.

use std::path::Path;

use splinter_sdk::conversation_policy_gradient::store;
use splinter_sdk::data::AdvantageKind;
use splinter_sdk::{Context, Error};

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
