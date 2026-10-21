// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `experiences compare`: two experience sets of the same tasks, recorded as
//! a run.

use serde_json::json;
use splinter_sdk::compare::compare;
use splinter_sdk::experiences::resolve_set;
use splinter_sdk::measure::strata::StratumCutoffs;
use splinter_sdk::runs::record;
use splinter_sdk::seed::Seed;
use splinter_sdk::{Context, Error};

use crate::output::emit;

/// Compares the sets `candidate` and `baseline`, by id or unique prefix; the
/// paired tasks are also counted by difficulty stratum when `strata` is given.
pub fn compare_sets(
    ctx: &Context,
    json: bool,
    candidate: &str,
    baseline: &str,
    interval_seed: Option<Seed>,
    strata: Option<StratumCutoffs>,
) -> Result<(), Error> {
    let (candidate, baseline) = (resolve_set(ctx, candidate)?, resolve_set(ctx, baseline)?);
    let arguments = json!({
        "candidate": candidate,
        "baseline": baseline,
        "interval_seed": interval_seed,
        "strata": strata,
    });
    let compared = record(ctx, "experiences compare", &arguments, |run| {
        compare(
            ctx,
            &candidate,
            &baseline,
            run.seed("interval-seed", interval_seed)?,
            strata,
        )
    })?;
    emit(json, &compared);
    Ok(())
}
