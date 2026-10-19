// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the choice between retraining, a better
// harness, more search or no change at all, on measured cost-adjusted gain,
// for its clients. If your team needs expertise in deciding how an agent
// should be improved before spending a training budget, you can procure our
// services by sending an email to info@swedishembedded.com.

//! `improve` and `generation`: choosing how to improve an agent, and
//! generations of a coding policy.

use std::fmt::Write as _;

use serde::Serialize;
use splinter_sdk::generations::{
    run_generation, series, GenerationRequest, Limits, Outcome, ReplayChoice, SeriesReport,
};
use splinter_sdk::improve::{
    fixed_comparison, improve, parse_experiment, FixedComparison, ImprovementReport, SolveArms,
};
use splinter_sdk::measure::intervention::Decision;
use splinter_sdk::runs::Recorded;
use splinter_sdk::tasks::resolve_set;
use splinter_sdk::{Context, Error};

use crate::cli::{GenerationCommand, ImproveArgs, ReplayArg};
use crate::output::{emit, Report};
use crate::practice_request::practice_request;
use crate::session::Exit;

/// What `improve` reports over one or more experiments.
#[derive(Serialize)]
pub struct ImproveOutput {
    /// Each experiment, in the order given.
    pub experiments: Vec<Recorded<ImprovementReport>>,
    /// The choices against every fixed strategy, when asked for.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fixed: Option<FixedComparison>,
}

impl Report for ImproveOutput {
    fn human(&self) -> String {
        let mut out = String::new();
        for run in &self.experiments {
            let report = &run.report;
            let _ = writeln!(out, "{}  (run {})", report.target_capability, run.run);
            for score in &report.verdict.scores {
                let _ = writeln!(
                    out,
                    "  {:<28} gain {} bound {} cost {:.0}s net {}{}",
                    score.label,
                    score.mean_gain.map_or("-".into(), |g| format!("{g:+.3}")),
                    score.lower_bound.map_or("-".into(), |g| format!("{g:+.3}")),
                    score.cost,
                    score.net_value.map_or("-".into(), |n| format!("{n:.0}")),
                    score
                        .ineligible
                        .as_ref()
                        .map_or(String::new(), |why| format!("  ({why:?})")),
                );
            }
            for arm in report.arms.iter().filter(|a| a.measured.is_none()) {
                let _ = writeln!(
                    out,
                    "  {:<28} {}",
                    arm.label,
                    arm.note.as_deref().unwrap_or("")
                );
            }
            match &report.verdict.decision {
                Decision::Apply { label, net_value } => {
                    let _ = writeln!(out, "  choose {label} (net {net_value:.0}s)");
                }
                Decision::NoChange { reason } => {
                    let _ = writeln!(out, "  change nothing: {reason}");
                }
            }
        }
        if let Some(fixed) = &self.fixed {
            let _ = writeln!(out, "choices {:.0}s in all", fixed.selected_net);
            for (label, strategy) in &fixed.strategies {
                let _ = writeln!(
                    out,
                    "  always {label}: {:.0}s over {} experiment(s) (the choices there: {:.0}s)",
                    strategy.net, strategy.experiments, strategy.selected_net_on_these
                );
            }
            let _ = writeln!(
                out,
                "{}",
                if fixed.beats_every_fixed_strategy {
                    "the choices are worth at least every fixed strategy"
                } else {
                    "some fixed strategy was worth more than the choices"
                }
            );
        }
        out
    }
}

/// Runs `improve` over the files named.
pub fn improve_command(ctx: &Context, json: bool, args: ImproveArgs) -> Result<Exit, Error> {
    let mut experiments = Vec::new();
    for file in &args.experiments {
        let text = std::fs::read_to_string(file).map_err(|e| {
            Error::Refused(format!(
                "cannot read the experiment {}: {e}",
                file.display()
            ))
        })?;
        let parsed = parse_experiment(&text)?;
        let tasks = resolve_set(ctx, parsed.tasks())?;
        let experiment = parsed.into_experiment(ctx, tasks)?;
        let mut arms = SolveArms::new(ctx, splinter_sdk::agent::CancelToken::new());
        experiments.push(improve(ctx, &experiment, &mut arms)?);
    }
    let fixed = (args.compare_fixed || experiments.len() > 1)
        .then(|| fixed_comparison(&experiments.iter().map(|e| &e.report).collect::<Vec<_>>()));
    emit(json, &ImproveOutput { experiments, fixed });
    Ok(Exit::Ok)
}

impl Report for SeriesReport {
    fn human(&self) -> String {
        let mut out = format!(
            "{}: {} generation(s), {} accepted\n",
            self.experiment,
            self.ledger.len(),
            self.accepted
        );
        for entry in &self.ledger {
            let how = match &entry.outcome {
                Outcome::Accepted { release } => format!("accepted as {release}"),
                Outcome::Rejected { reasons } => format!("rejected: {}", reasons.join("; ")),
                Outcome::Stopped { reason } => format!("stopped: {reason}"),
            };
            let _ = writeln!(out, "  generation {}  {how}", entry.generation);
        }
        match (&self.matrix, &self.matrix_unavailable) {
            (Some(matrix), _) => {
                let _ = writeln!(
                    out,
                    "matrix of {} generation(s) over {} task(s)",
                    matrix.matrix.generations(),
                    matrix.matrix.tasks().len()
                );
            }
            (None, Some(why)) => {
                let _ = writeln!(out, "no matrix: {why}");
            }
            (None, None) => {}
        }
        out
    }
}

impl Report for splinter_sdk::generations::GenerationReport {
    fn human(&self) -> String {
        let record = &self.record;
        let how = match &record.outcome {
            Outcome::Accepted { release } => format!("accepted as {release}"),
            Outcome::Rejected { reasons } => format!("rejected: {}", reasons.join("; ")),
            Outcome::Stopped { reason } => format!("stopped: {reason}"),
        };
        format!(
            "generation {} of {}: {how}\n",
            record.generation, record.experiment
        )
    }
}

/// How a generation carries the earlier experience, given the `--replay`
/// policy named and whether it rebuilds from the base.
fn replay_choice(replay: Option<ReplayArg>, rebuild: bool) -> ReplayChoice {
    if rebuild {
        return ReplayChoice::RebuildFromBase;
    }
    match replay {
        None | Some(ReplayArg::Uniform) => ReplayChoice::Uniform,
        Some(ReplayArg::None) => ReplayChoice::None,
        Some(ReplayArg::Balanced) => ReplayChoice::CapabilityBalanced,
        Some(ReplayArg::FailureTargeted) => ReplayChoice::FailureTargeted,
    }
}

/// Runs a `generation` command.
pub fn generation_command(
    splinter: &splinter_sdk::Splinter,
    ctx: &Context,
    json: bool,
    command: GenerationCommand,
) -> Result<Exit, Error> {
    match command {
        GenerationCommand::Run(args) => {
            let mut practice = args.practice;
            let replay = replay_choice(practice.replay.take(), practice.rebuild);
            practice.rebuild = false;
            let defaults = Limits::default();
            let request = GenerationRequest {
                experiment: args.experiment,
                practice: practice_request(practice)?,
                replay,
                limits: Limits {
                    max_family_drop: args.max_family_drop.unwrap_or(defaults.max_family_drop),
                    max_mean_drop: args.max_mean_drop.unwrap_or(defaults.max_mean_drop),
                    min_new_gain: args.min_new_gain.unwrap_or(defaults.min_new_gain),
                },
            };
            let ran = run_generation(
                ctx,
                &request,
                splinter.trainer(),
                &splinter_sdk::agent::CancelToken::new(),
            )?;
            emit(json, &ran);
            Ok(
                if matches!(ran.report.record.outcome, Outcome::Accepted { .. }) {
                    Exit::Ok
                } else {
                    Exit::Failed
                },
            )
        }
        GenerationCommand::Matrix { experiment, alias } => {
            let shown = series(
                ctx,
                &experiment,
                &alias,
                splinter_sdk::measure::gate::AgenticConfig::default(),
                &splinter_sdk::agent::CancelToken::new(),
            )?;
            emit(json, &shown);
            Ok(Exit::Ok)
        }
    }
}
