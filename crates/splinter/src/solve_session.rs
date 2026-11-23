// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements recorded, resumable evaluation of coding
// agents on task sets for its clients. If your team needs expertise in
// measuring what an agent solves and how often, you can procure our services
// by sending an email to info@swedishembedded.com.

//! `solve`: a task set solved once by a solver, or (`--frontier`) attempted k
//! times to find which tasks sit at the edge of what it can do, as recorded
//! runs; `solve --harness` solves them with a sven harness document.

use std::path::Path;

use serde_json::json;
use splinter_sdk::agent::work::WorkProtocol;
use splinter_sdk::curriculum::frontier::{measure, MeasureRequest};
use splinter_sdk::experiences::resolve_set;
use splinter_sdk::runs::record;
use splinter_sdk::solving::{
    harness_protocol, solve_set, solve_tasks, SamplingChoice, SolveRequest,
};
use splinter_sdk::tasks::resolve_set as resolve_task_set;
use splinter_sdk::vocabulary::digest::Digest;
use splinter_sdk::{Context, Error};

use crate::cli::SolveArgs;
use crate::output::emit;
use crate::session::Exit;

/// Runs `solve` as `args` asks, and emits what it measured or solved.
pub fn solve(ctx: &Context, json: bool, args: &SolveArgs) -> Result<Exit, Error> {
    if let Some(path) = args.harness.as_deref() {
        return solve_with_harness(ctx, json, args, path);
    }
    let set = resolve_task_set(ctx, &args.task_set)?;
    if args.frontier {
        let pass_at_k = args.pass_at_k.pass_at_k();
        let arguments = json!({
            "task_set": set,
            "solver": args.solver,
            "teacher": args.teacher,
            "frontier": true,
            "pass_at_k": pass_at_k,
        });
        let measured = record(ctx, "solve", &arguments, |run| {
            measure(
                ctx,
                &MeasureRequest {
                    task_set: &set,
                    solver: &args.solver,
                    teacher: args.teacher.as_ref(),
                    pass_at_k,
                    deadline: None,
                    cancel: run.cancel_token(),
                },
            )
        })?;
        emit(json, &measured);
        return Ok(Exit::Ok);
    }
    let resume = args
        .resume
        .as_deref()
        .map(|id| resolve_set(ctx, id))
        .transpose()?;
    let arguments = json!({ "task_set": set, "solver": args.solver, "resume": resume });
    let work = WorkProtocol {
        compaction: !args.no_compaction,
        ..harness_protocol()
    };
    let solved = record(ctx, "solve", &arguments, |run| {
        solve_set(
            ctx,
            &set,
            &args.solver,
            None,
            &run.cancel_token(),
            resume.as_ref(),
            work,
        )
    })?;
    emit(json, &solved);
    Ok(Exit::Ok)
}

/// `solve TASKSET --harness FILE`: every task solved by the document.
fn solve_with_harness(
    ctx: &Context,
    json: bool,
    args: &SolveArgs,
    path: &Path,
) -> Result<Exit, Error> {
    let set = resolve_task_set(ctx, &args.task_set)?;
    let document = std::fs::read_to_string(path)
        .map_err(|e| Error::Refused(format!("cannot read {}: {e}", path.display())))?;
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
                resume: None,
            },
        )
    })?;
    emit(json, &solved);
    Ok(Exit::Ok)
}
