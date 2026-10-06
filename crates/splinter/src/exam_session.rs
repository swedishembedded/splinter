// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `exam-set create` and the powered `exam`, as recorded runs.

use serde_json::json;
use splinter_sdk::exam_set::{build, ExamBuilt, ExamSet, NewExam};
use splinter_sdk::powered::plan::{simulate, Plan, Planned};
use splinter_sdk::powered::{run, PoweredExam, PoweredRequest, DEFAULT_RESAMPLES};
use splinter_sdk::runs::{record, Recorded};
use splinter_sdk::sources::SourceTarget;
use splinter_sdk::vocabulary::model_ref::ModelRef;
use splinter_sdk::{Context, Error};

use crate::cli::{ExamArgs, ExamSetArgs, PowerArgs};

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
        "tasks_per_family": args.tasks_per_family,
        "dev_families": args.dev_families,
        "dev_tasks_per_family": args.dev_tasks_per_family,
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
                tasks_per_family: args.tasks_per_family,
                dev_families: args.dev_families,
                dev_tasks_per_family: args.dev_tasks_per_family,
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
        "pilot_families": args.pilot_families,
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
                pilot_families: args.pilot_families,
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

/// What `exam-set power` found: the plan and what it came to.
#[derive(serde::Serialize)]
pub struct PlannedExam {
    /// The assumptions, the discordance and clustering taken from a pilot's
    /// report when one was named.
    pub plan: Plan,
    /// What simulating the planned test gave.
    pub planned: Planned,
}

/// Simulates the planned paired test.
pub fn power(args: &PowerArgs) -> Result<PlannedExam, Error> {
    let (discordance, icc) = match &args.from_report {
        None => (args.discordance, args.icc),
        Some(file) => {
            let text = std::fs::read_to_string(file)
                .map_err(|e| Error::Refused(format!("{}: {e}", file.display())))?;
            let report: serde_json::Value = serde_json::from_str(&text)
                .map_err(|e| Error::Refused(format!("{}: {e}", file.display())))?;
            // The report is the run's record or the bare report.
            let powered = report["outputs"]["ran"]
                .as_object()
                .map(|_| &report["outputs"]["ran"])
                .or_else(|| report["ran"].as_object().map(|_| &report["ran"]))
                .unwrap_or(&report);
            let power = &powered["comparisons"][0]["power"];
            let read = |name: &str| {
                power[name].as_f64().ok_or_else(|| {
                    Error::Refused(format!(
                        "{} holds no primary comparison's {name}",
                        file.display()
                    ))
                })
            };
            (read("discordance")?, read("icc")?)
        }
    };
    let plan = Plan {
        families: args.families,
        tasks_per_family: args.tasks_per_family,
        effect: args.effect,
        discordance,
        icc,
        replicates: args.replicates,
        resamples: 500,
        seed: 0,
    };
    Ok(PlannedExam {
        plan,
        planned: simulate(&plan),
    })
}
