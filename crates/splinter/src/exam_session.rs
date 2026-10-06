// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `exam-set create` and the powered `exam`, as recorded runs.

use serde_json::json;
use splinter_sdk::exam_set::{build, ExamBuilt, ExamSet, NewExam};
use splinter_sdk::powered::{run, PoweredExam, PoweredRequest, DEFAULT_RESAMPLES};
use splinter_sdk::runs::{record, Recorded};
use splinter_sdk::sources::SourceTarget;
use splinter_sdk::vocabulary::model_ref::ModelRef;
use splinter_sdk::{Context, Error};

use crate::cli::{ExamArgs, ExamSetArgs};

/// Captures the sources, reserves the exam's families and freezes the exam.
pub fn create(ctx: &Context, args: &ExamSetArgs) -> Result<Recorded<ExamBuilt>, Error> {
    let targets = args
        .sources
        .iter()
        .map(|s| SourceTarget::from_learn_arg(s))
        .collect::<Result<Vec<_>, _>>()?;
    let generator = args
        .generator
        .clone()
        .unwrap_or_else(ModelRef::policy_default);
    let arguments = json!({
        "sources": args.sources,
        "families": args.families,
        "tasks": args.tasks,
        "kinds": args.kinds,
        "seed": args.seed,
        "not_trained_by": args.not_trained_by,
    });
    record(ctx, "exam-set", &arguments, |run| {
        build(
            ctx,
            &NewExam {
                sources: &targets,
                families: args.families,
                seed: args.seed,
                max_tasks: args.tasks,
                kinds: &args.kinds,
                generator: &generator,
                goal: args.goal.as_deref(),
                author: args.persona.as_deref(),
                not_trained_by: &args.not_trained_by,
                cancel: run.cancel_token(),
            },
        )
    })
}

/// Puts the candidate to the frozen exam `args` names.
pub fn powered(ctx: &Context, args: &ExamArgs) -> Result<Recorded<PoweredExam>, Error> {
    let named = args.exam_set.clone().unwrap_or_default();
    let exam = ExamSet::load(ctx, &named)?;
    let arguments = json!({
        "candidate": args.candidate,
        "exam_set": exam.id,
        "judge": args.judge,
        "prompt": args.prompt,
        "resamples": args.resamples,
        "voice": !args.no_voice,
    });
    record(ctx, "exam", &arguments, |cancelled| {
        let cancel = cancelled.cancel_token();
        match run(
            ctx,
            &PoweredRequest {
                exam: &exam,
                candidate: &args.candidate,
                base: None,
                judge: args.judge.as_ref(),
                goal: args.prompt.as_deref(),
                resamples: args.resamples.map_or(DEFAULT_RESAMPLES, |n| n as usize),
                voice: !args.no_voice,
                cancel: &cancel,
            },
        ) {
            Ok(powered) => Ok(PoweredExam::Ran(Box::new(powered))),
            Err(e) if e.is_refusal() => Ok(PoweredExam::NotRun(e.to_string())),
            Err(e) => Err(e),
        }
    })
}
