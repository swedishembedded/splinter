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

/// `experiences compare-arms`: the trials of every `--arm NAME=DIR` and
/// `--trials FILE`, each arm against `--reference`, recorded as a run.
pub fn compare_arms(
    ctx: &Context,
    json: bool,
    args: crate::cli::CompareArmsArgs,
) -> Result<(), Error> {
    use splinter_sdk::arm_compare::{
        compare_arms, read_harbor_job, read_trials_jsonl, CompareRequest,
    };
    let mut trials = Vec::new();
    for arm in &args.arms {
        let (name, dir) = arm
            .split_once('=')
            .ok_or_else(|| Error::Refused(format!("--arm {arm}: write it as NAME=DIR")))?;
        trials.extend(read_harbor_job(name, std::path::Path::new(dir))?);
    }
    for file in &args.trials {
        trials.extend(read_trials_jsonl(file)?);
    }
    let families = match &args.families {
        None => Default::default(),
        Some(path) => {
            let text = std::fs::read_to_string(path)
                .map_err(|e| Error::Refused(format!("{}: {e}", path.display())))?;
            serde_json::from_str(&text).map_err(|e| {
                Error::Refused(format!(
                    "{}: not a JSON object of task to family: {e}",
                    path.display()
                ))
            })?
        }
    };
    let arguments = json!({
        "arms": args.arms,
        "trials": args.trials,
        "reference": args.reference,
        "families": args.families,
        "k": args.k,
        "resamples": args.resamples,
        "level": args.level,
        "interval_seed": args.interval_seed,
    });
    let compared = record(ctx, "experiences compare-arms", &arguments, |run| {
        let request = CompareRequest {
            reference: args.reference.clone(),
            k: args.k.clone(),
            resamples: args.resamples,
            level: args.level,
            seed: run.seed("interval-seed", args.interval_seed)?,
            families,
        };
        compare_arms(&trials, &request)
    })?;
    emit(json, &compared);
    Ok(())
}
