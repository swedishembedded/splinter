// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements automated improvement of agent harnesses
// that adopts a change only on verified, regression-free evidence, for its
// clients. If your team needs expertise in measuring whether a harness
// change really helped, you can procure our services by sending an email to
// info@swedishembedded.com.

//! `solve --harness` and `harness optimise`: tasks solved by a sven harness
//! document, and the search over candidate documents, as recorded runs.

use std::path::Path;

use serde_json::json;
use splinter_sdk::agent::sven::search::Strategy;
use splinter_sdk::harness_opt::gate::PromotionRule;
use splinter_sdk::harness_opt::space::MutationSpace;
use splinter_sdk::harness_opt::{optimise, OptimiseRequest};
use splinter_sdk::runs::record;
use splinter_sdk::solving::{harness_protocol, solve_tasks, SamplingChoice, SolveRequest};
use splinter_sdk::tasks::resolve_set as resolve_task_set;
use splinter_sdk::vocabulary::digest::Digest;
use splinter_sdk::{Context, Error};

use crate::cli::{HarnessCommand, SolveArgs, StrategyArg};
use crate::output::emit;
use crate::session::Exit;

/// The text of the harness document at `path`.
fn read(path: &Path) -> Result<String, Error> {
    std::fs::read_to_string(path)
        .map_err(|e| Error::Refused(format!("cannot read {}: {e}", path.display())))
}

/// `solve TASKSET --harness FILE`: every task solved by the document.
pub fn solve(ctx: &Context, json: bool, args: &SolveArgs) -> Result<Exit, Error> {
    let Some(path) = args.harness.as_deref() else {
        return Err(Error::Refused("solve --harness needs a document".into()));
    };
    let set = resolve_task_set(ctx, &args.task_set)?;
    let document = read(path)?;
    let arguments = json!({
        "task_set": set,
        "solver": args.solver,
        "harness": Digest::of(document.as_bytes()),
    });
    let solved = record(ctx, "solve", &arguments, |run| {
        solve_tasks(
            ctx,
            &SolveRequest {
                task_set: &set,
                solver: &args.solver,
                attempts: 1,
                sampling: SamplingChoice::Own,
                teacher: false,
                system: None,
                work: harness_protocol(),
                harness: Some(&document),
                deadline: None,
                cancel: run.cancel_token(),
            },
        )
    })?;
    emit(json, &solved);
    Ok(Exit::Ok)
}

/// `harness ...`.
pub fn harness(ctx: &Context, json: bool, command: HarnessCommand) -> Result<Exit, Error> {
    let HarnessCommand::Optimise(args) = command;
    let set = resolve_task_set(ctx, &args.task_set)?;
    let baseline = read(&args.baseline)?;
    let space_text = read(&args.space)?;
    let space = MutationSpace::parse(&space_text).map_err(|e| Error::Refused(e.to_string()))?;
    let strategy = match args.strategy {
        StrategyArg::BestOfN => Strategy::BestOfN,
        StrategyArg::Annealing => Strategy::SimulatedAnnealing {
            initial_temperature: args.anneal_temperature,
            cooling: args.anneal_cooling,
        },
    };
    let arguments = json!({
        "task_set": set,
        "solver": args.solver,
        "baseline": Digest::of(baseline.as_bytes()),
        "space": Digest::of(space_text.as_bytes()),
        "strategy": format!("{:?}", args.strategy),
        "evaluations": args.evaluations,
        "attempts": args.attempts,
        "alpha": args.alpha,
        "min_paired": args.min_paired,
        "search_seed": args.search_seed,
        "interval_seed": args.interval_seed,
    });
    let optimised = record(ctx, "harness optimise", &arguments, |run| {
        optimise(
            ctx,
            &OptimiseRequest {
                task_set: &set,
                solver: &args.solver,
                baseline: &baseline,
                space: space.clone(),
                strategy,
                evaluations: args.evaluations,
                attempts: args.attempts,
                rule: PromotionRule {
                    alpha: args.alpha,
                    min_paired: args.min_paired,
                },
                search_seed: args.search_seed,
                interval_seed: run.seed("interval-seed", args.interval_seed)?,
                cancel: run.cancel_token(),
            },
        )
    })?;
    emit(json, &optimised);
    Ok(Exit::Ok)
}
