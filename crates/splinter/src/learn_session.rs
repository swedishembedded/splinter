// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for teaching models new subjects
// for its clients. If your team needs expertise in acquiring knowledge from
// documentation then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The `learn` command: a one-shot learn run, or a sourced acquisition.

use splinter_sdk::learn::{learn, Learned};
use splinter_sdk::{Context, Error, Splinter};

use crate::cli::LearnArgs;
use crate::learn_request::{acquire_request, learn_request, release_asked};
use crate::output::emit;
use crate::session::Exit;

/// Run `learn`; with `--source` it is an acquisition run, which fails
/// the exit code when the assessment does not accept.
pub fn learn_command(
    splinter: &Splinter,
    ctx: &Context,
    json: bool,
    args: LearnArgs,
) -> Result<Exit, Error> {
    if !args.source.is_empty() {
        let acquired = splinter.acquire(&acquire_request(args))?;
        emit(json, &acquired);
        return Ok(if acquired.report.accepted {
            Exit::Ok
        } else {
            Exit::Failed
        });
    }
    let release_asked = release_asked(&args);
    let request = learn_request(args);
    let learned = learn(ctx, &request, splinter.trainer())?;
    emit(json, &learned);
    let finished = match &learned {
        Learned::Planned(_) => true,
        Learned::Ran(run) => run.report.finished(release_asked),
    };
    Ok(if finished { Exit::Ok } else { Exit::Failed })
}
