// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model releases promoted only on measured
// improvement, for its clients. If your team needs expertise in model
// evaluation and release gating, you can procure our services by sending an
// email to info@swedishembedded.com.

//! `release`, `eval`, `experiment` and `lineage`.

use std::path::PathBuf;

use clap::{Args, Subcommand};
use splinter_sdk::eval::SuiteChoice;
use splinter_sdk::lineage::Direction;
use splinter_sdk::seed::Seed;
use splinter_sdk::vocabulary::model_ref::{ModelRef, POLICY_DEFAULT};

use super::{model_ref, suite, GateArgs};

/// `release`.
#[derive(Debug, Args)]
#[command(args_conflicts_with_subcommands = true)]
pub struct ReleaseArgs {
    /// The candidate, by id or unique prefix.
    #[arg(value_name = "CANDIDATE-ID")]
    pub candidate: Option<String>,
    /// The alias it replaces the champion of.
    #[arg(long, default_value = POLICY_DEFAULT, value_name = "NAME")]
    pub alias: String,
    /// The calibrated model that judges the tasks a judge decides.
    #[arg(long, value_parser = model_ref, value_name = "REF")]
    pub judge: Option<ModelRef>,
    /// Release for distribution and commercial use: refused unless the terms
    /// of everything the candidate was made from allow both. Without it the
    /// release is restricted: it records the terms and stays local.
    #[arg(long)]
    pub unrestricted: bool,
    /// The seed the gate's bootstrap intervals are resampled with (drawn and
    /// printed when not given; the same outcomes and seed give the same
    /// intervals).
    #[arg(long, value_name = "N")]
    pub interval_seed: Option<Seed>,
    /// A task set of canaries (`tasks canaries`) the candidate attempts beside
    /// the exam: a pass on any one refuses it, whatever else it passed.
    #[arg(long, value_name = "TASKSET-ID")]
    pub canaries: Option<String>,
    /// The gate's statistical settings.
    #[command(flatten)]
    pub gate: GateArgs,
    /// `list`.
    #[command(subcommand)]
    pub command: Option<ReleaseCommand>,
}

/// `release ...`.
#[derive(Debug, Subcommand)]
pub enum ReleaseCommand {
    /// List the releases and the aliases pointing at them.
    List,
}

/// `eval`.
#[derive(Debug, Args)]
#[command(args_conflicts_with_subcommands = true)]
pub struct EvalArgs {
    /// A candidate id (or prefix) or a model reference; none shows the
    /// anchor suite.
    #[arg(value_name = "REF")]
    pub model: Option<String>,
    /// held-out, retention, anchor, or an anchor-format file of tasks.
    #[arg(long, default_value = "held-out", value_parser = suite, value_name = "SUITE")]
    pub suite: SuiteChoice,
    /// With --suite anchor: make these files' tasks (repeat the flag for
    /// several) the anchor suite's next version first. JSON Lines:
    /// {"instruction", "reference", "kind"?}.
    #[arg(long, value_name = "FILE")]
    pub freeze: Vec<PathBuf>,
    /// The calibrated model that judges the tasks a judge decides.
    #[arg(long, value_parser = model_ref, value_name = "REF")]
    pub judge: Option<ModelRef>,
    /// With a factory run directory as REF: the evaluation plan to evaluate
    /// the run's candidates under, reusing them (a fork of the run into
    /// --out with the recipe's plan replaced).
    #[arg(long, value_name = "PLAN", requires = "out")]
    pub plan: Option<PathBuf>,
    /// With --plan: the new run directory.
    #[arg(long, value_name = "DIR")]
    pub out: Option<PathBuf>,
    /// `matrix`.
    #[command(subcommand)]
    pub command: Option<EvalCommand>,
}

/// `experiment ...`.
#[derive(Debug, Subcommand)]
pub enum ExperimentCommand {
    /// Validate a plan against the catalog of families and show its schedule:
    /// families disjoint by domain, no sealed family scheduled for training or
    /// retention, retention only on families an earlier generation trained.
    /// A plan that is not valid is refused with every violation. A valid plan
    /// records its sealed final exam in the state root, so a later run that is
    /// not given the plan cannot read those families either.
    Check {
        /// The plan file.
        #[arg(value_name = "PLAN")]
        plan: PathBuf,
        /// The catalog of families (default: catalog.toml beside the plan).
        #[arg(long, value_name = "FILE")]
        catalog: Option<PathBuf>,
    },
}

/// `eval ...`.
#[derive(Debug, Subcommand)]
pub enum EvalCommand {
    /// The continual-learning results matrix of a lineage: what each gate
    /// measured of each release on each earlier release's suite, with the
    /// metrics read from it. A cell no gate measured is shown absent.
    Matrix {
        /// The alias whose releases are the generations.
        #[arg(long, default_value = POLICY_DEFAULT, value_name = "NAME")]
        alias: String,
    },
}

/// `lineage`.
#[derive(Debug, Args)]
pub struct LineageArgs {
    /// Any artifact's id - a source, task, experience, dataset, candidate,
    /// release, answer, ... - or a unique prefix of it.
    #[arg(value_name = "ID")]
    pub id: String,
    /// Only where it came from.
    #[arg(long, conflicts_with_all = ["down", "both"])]
    pub up: bool,
    /// Only what came from it.
    #[arg(long, conflicts_with = "both")]
    pub down: bool,
    /// Both ways (the default).
    #[arg(long)]
    pub both: bool,
    /// The most edges walked from it (default: all the way).
    #[arg(long, value_name = "N")]
    pub depth: Option<usize>,
}

impl LineageArgs {
    /// The direction the flags ask for.
    pub fn direction(&self) -> Direction {
        match (self.up, self.down) {
            (true, _) => Direction::Up,
            (_, true) => Direction::Down,
            _ => Direction::Both,
        }
    }
}
