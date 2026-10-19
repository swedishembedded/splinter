// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements automated improvement of agent harnesses
// that adopts a change only on verified, regression-free evidence, for its
// clients. If your team needs expertise in measuring whether a harness
// change really helped, you can procure our services by sending an email to
// info@swedishembedded.com.

//! `harness ...`.

use std::path::PathBuf;

use clap::{Args, Subcommand, ValueEnum};
use splinter_sdk::seed::Seed;
use splinter_sdk::vocabulary::model_ref::ModelRef;

use super::model_ref;

/// `harness ...`.
#[derive(Debug, Subcommand)]
pub enum HarnessCommand {
    /// Search a space of variants of a baseline harness for one that
    /// verifiably beats it: each candidate is run on the task set through
    /// sven, its attempts verified by the tasks' own verifiers and compared
    /// with the baseline's (a family counted once). A candidate is promoted
    /// only when no family got worse and the sign test over families finds it
    /// better; the promotion moves the `harness/champion` pointer.
    Optimise(Box<OptimiseArgs>),
}

/// How candidates are searched.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum StrategyArg {
    /// Every candidate is one mutation of the baseline.
    BestOfN,
    /// A walk that follows the share of tasks solved, sometimes downhill.
    Annealing,
}

/// Who proposes candidates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum ProposerArg {
    /// One mutation of the space per candidate.
    Mutation,
    /// A model proposes edits.
    Model,
}

/// `harness optimise`.
#[derive(Debug, Args)]
pub struct OptimiseArgs {
    /// The frozen task set every harness is run on, by id or unique prefix.
    #[arg(value_name = "TASKSET-ID")]
    pub task_set: String,
    /// The model every harness runs on.
    #[arg(long, value_parser = model_ref, default_value_t = ModelRef::policy_default(), value_name = "REF")]
    pub solver: ModelRef,
    /// The baseline sven harness document (YAML).
    #[arg(long, required = true, value_name = "FILE")]
    pub baseline: PathBuf,
    /// The mutation space: a YAML list of knobs, each a `path` into the
    /// document and the `choices` it may take. Required by the mutation
    /// proposer; with a model proposer it is the fallback for a proposal the
    /// model cannot make compile.
    #[arg(long, value_name = "FILE")]
    pub space: Option<PathBuf>,
    /// Who proposes candidates: a mechanical mutation of the space, or a model
    /// asked for edits that sven's compiler must accept.
    #[arg(long, value_enum, default_value = "mutation")]
    pub proposer: ProposerArg,
    /// The proposing model (with `--proposer model`).
    #[arg(long, value_parser = model_ref, value_name = "REF", required_if_eq("proposer", "model"))]
    pub proposer_model: Option<ModelRef>,
    /// Seconds one model proposal may take, corrections included.
    #[arg(long, default_value_t = 300, value_name = "SECS")]
    pub proposer_deadline: u64,
    /// Corrections a model reply that does not compile is sent back for.
    #[arg(long, default_value_t = 3, value_name = "N")]
    pub proposer_repairs: u32,
    /// Candidates evaluated at once (best-of-N only): each is a whole arm, so
    /// the model server sees that many arms' attempts together.
    #[arg(long, default_value_t = 1, value_name = "N")]
    pub parallel: usize,
    /// How candidates are searched.
    #[arg(long, value_enum, default_value = "annealing")]
    pub strategy: StrategyArg,
    /// Harnesses to evaluate, the baseline's replicate (the first) included.
    #[arg(long, required = true, value_name = "N")]
    pub evaluations: u64,
    /// Solves per task and harness.
    #[arg(long, default_value_t = 1, value_name = "K")]
    pub attempts: usize,
    /// A candidate's sign test must have a p-value below this to be promoted.
    #[arg(long, default_value_t = 0.05, value_name = "P")]
    pub alpha: f64,
    /// The fewest tasks with a verdict in both arms for a promotion.
    #[arg(long, default_value_t = 5, value_name = "N")]
    pub min_paired: usize,
    /// Seeds the search's draws; the same seed reproduces the candidates
    /// drawn.
    #[arg(long, required = true, value_name = "N")]
    pub search_seed: u64,
    /// The seed the comparisons' intervals are resampled with (drawn and
    /// printed when not given).
    #[arg(long, value_name = "N")]
    pub interval_seed: Option<Seed>,
    /// Annealing: the starting temperature, in shares of tasks solved.
    #[arg(long, default_value_t = 0.05, value_name = "T")]
    pub anneal_temperature: f64,
    /// Annealing: the factor the temperature is multiplied by each step.
    #[arg(long, default_value_t = 0.97, value_name = "F")]
    pub anneal_cooling: f64,
}
