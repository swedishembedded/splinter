// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that turn the failures of an earlier
// arm into verified training material, for its clients. If your team needs
// expertise in on-policy agent data then you can procure our services by
// sending an email to info@swedishembedded.com.

//! `splinter corrections`: continue the failed or stuck attempts of an
//! experience set with a teacher, from the state the student actually
//! reached, and record the continuations the task's own hidden checks verify
//! as an experience set a dataset can be built from.

use std::time::Instant;

use serde_json::json;
use splinter_sdk::agent::work::WorkProtocol;
use splinter_sdk::curriculum::corrections::{correct, CorrectionsRequest};
use splinter_sdk::experiences::resolve_set;
use splinter_sdk::runs::record;
use splinter_sdk::tasks::resolve_set as resolve_task_set;
use splinter_sdk::vocabulary::prompt::CodingPrompt;
use splinter_sdk::{Context, Error};

use crate::cli::CorrectionsArgs;
use crate::output::emit;
use crate::session::Exit;

/// Runs the command and prints its report.
///
/// # Errors
///
/// As the run is refused or fails.
pub(crate) fn corrections_command(
    ctx: &Context,
    json: bool,
    args: CorrectionsArgs,
) -> Result<Exit, Error> {
    let attempts = resolve_set(ctx, &args.attempts)?;
    let tasks = resolve_task_set(ctx, &args.tasks)?;
    let arguments = json!({
        "attempts": attempts,
        "tasks": tasks,
        "teacher": args.teacher,
        "max": args.max(),
    });
    let deadline = args.budget.map(|budget| Instant::now() + budget);
    let corrected = record(ctx, "corrections", &arguments, |run| {
        correct(
            ctx,
            &CorrectionsRequest {
                task_set: &tasks,
                attempts: &attempts,
                teacher: &args.teacher,
                max: args.max(),
                work: WorkProtocol {
                    follow_ups: 0,
                    max_tool_calls: args.max_tool_calls,
                    max_repeated_results: None,
                    prompt: CodingPrompt::default(),
                    // The continuation is recorded, so it keeps its transcript.
                    compaction: false,
                },
                deadline,
                cancel: run.cancel_token(),
            },
        )
    })?;
    emit(json, &corrected);
    Ok(Exit::Ok)
}
