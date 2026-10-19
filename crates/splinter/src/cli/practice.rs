// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that learn a codebase from its own
// tests, for its clients. If your team needs expertise in training
// code-editing agents on executable tasks drawn from a repository, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The grammar of `practice`.

use std::path::PathBuf;
use std::time::Duration;

use clap::Args;
use splinter_sdk::canaries::CANARY_COUNT;
use splinter_sdk::curriculum::repository_teacher::RepositoryTeacher;
use splinter_sdk::practice::{
    DEFAULT_FOLLOW_UPS, DEFAULT_MAX_REPEATED_RESULTS, DEFAULT_MAX_TOOL_CALLS,
};
use splinter_sdk::seed::Seed;
use splinter_sdk::train::DEFAULT_LORA_RANK;

use super::{budget, GateArgs, MonitoringArgs, OptimiserArgs, PassAtKArgs};
use splinter_sdk::data::ReplayPolicy;

/// `practice`.
#[derive(Debug, Args)]
pub struct PracticeArgs {
    /// The git repository to learn from: a work tree or a bare repository.
    #[arg(value_name = "REPO")]
    pub repository: PathBuf,
    /// The revision of it to learn from: anything git resolves to a commit.
    /// The tracked files of that commit are the snapshot every task is
    /// pinned to; uncommitted work is not in it.
    #[arg(long, default_value = "HEAD", value_name = "REV")]
    pub rev: String,
    /// Snapshot the committed files of a revision that is checked out with
    /// uncommitted changes, instead of refusing it.
    #[arg(long)]
    pub allow_dirty: bool,
    /// A file, relative to the repository, that tasks may change; repeat for
    /// more. Without any, every source file of the project the rules
    /// recognise (a Python project with a tests directory, a Cargo project).
    #[arg(long = "file", value_name = "PATH")]
    pub files: Vec<String>,
    /// The most tasks one run admits.
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(usize))]
    pub max_tasks: Option<usize>,
    /// The most faults seeded in one file.
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(usize))]
    pub mutants_per_file: Option<usize>,
    /// How many families (a family is the file a fault was seeded in) to
    /// reserve for the exam before any attempt is made at any task: their
    /// tasks are set apart and never solved or trained on, in this run or a
    /// later one. 0 reserves none.
    #[arg(
        long,
        value_name = "N",
        help = "How many families (a family is the file a fault was \
        seeded in) to reserve for the exam before any attempt is made at any task: their tasks \
        are set apart and never solved or trained on, in this run or a later one; 0 reserves \
        none [default: a quarter of the families]"
    )]
    pub exam_families: Option<usize>,
    /// How many canaries of the exam the release gate is handed.
    #[arg(
        long,
        value_name = "N",
        help = format!("The most canaries (exam tasks no honest change can pass) the release gate \
        hands the candidate; a pass on one refuses it; 0 hands none [default: {CANARY_COUNT}]")
    )]
    pub canaries: Option<usize>,
    /// The seed the exam families are chosen by.
    #[arg(
        long,
        value_name = "N",
        help = "The seed the exam families are chosen by when this run reserves the first: \
        drawn and printed [default], then kept with the families, so every later run reads it \
        back and naming another is refused"
    )]
    pub exam_seed: Option<Seed>,
    /// How the policy's attempts are measured.
    #[command(flatten)]
    pub pass_at_k: PassAtKArgs,
    /// Follow-up turns an agent is sent after the checks it was shown stay
    /// red.
    #[arg(long, value_name = "N", help = format!(
        "Follow-up turns an agent is sent after the checks it was shown stay red \
         [default: {DEFAULT_FOLLOW_UPS}]"))]
    pub follow_ups: Option<u32>,
    /// Tool calls an agent may make in one attempt.
    #[arg(long, value_name = "N", help = format!(
        "Tool calls an agent may make in one attempt, follow-ups included \
         [default: {DEFAULT_MAX_TOOL_CALLS}]"))]
    pub max_tool_calls: Option<u32>,
    /// How often one tool call may return the same answer in an attempt.
    #[arg(long, value_name = "N", help = format!(
        "How often one tool call may return the same answer before the attempt is stopped as \
         going nowhere [default: {DEFAULT_MAX_REPEATED_RESULTS}]"))]
    pub max_repeated_results: Option<u32>,
    /// Wall-clock time the whole run may take, in h, m and s: 30m, 2h, 1h30m.
    #[arg(long, value_parser = budget, value_name = "DUR")]
    pub budget: Option<Duration>,
    /// Stop at the trained candidate: do not run the release gate.
    #[arg(long)]
    pub no_release: bool,
    /// Who demonstrates the tasks the policy never solves, so they are not
    /// dropped: `reference` replays each task's own reference change as an
    /// agent's actions, or a model that solves them as an agent
    /// (`remote:`/`local:`). Either is graded by the same independent
    /// verifier as the policy's attempts; it never grades itself.
    #[arg(long, value_parser = repository_teacher, value_name = "reference|REF")]
    pub teacher: Option<RepositoryTeacher>,
    /// Which of the earlier releases' records are replayed beside the new
    /// ones (default uniform): none, uniform over every record, or balanced
    /// over capabilities.
    #[arg(long, value_enum, value_name = "POLICY")]
    pub replay: Option<ReplayArg>,
    /// Train a fresh adapter on the frozen base from the records of every
    /// earlier release and the new ones, instead of continuing the champion.
    #[arg(long, conflicts_with = "replay")]
    pub rebuild: bool,
    /// The release gate's statistical settings.
    #[command(flatten)]
    pub gate: GateArgs,
    /// The step budget of the training: the most steps it may take.
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u32).range(1..))]
    pub steps: Option<u32>,
    /// LoRA rank of the adapter.
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u32).range(1..),
        help = format!("LoRA rank of the adapter [default: {DEFAULT_LORA_RANK}]"))]
    pub rank: Option<u32>,
    /// The longest training record, in tokens, the card fits: a longer
    /// record is left out of the dataset and counted, instead of stopping
    /// the training.
    #[arg(long, value_name = "TOKENS", value_parser = clap::value_parser!(u32).range(1..))]
    pub max_block: Option<u32>,
    /// The learning rate, alpha and weight decay of the training.
    #[command(flatten)]
    pub optimiser: OptimiserArgs,
    /// Records averaged into one optimizer step.
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u32).range(1..))]
    pub records_per_step: Option<u32>,
    /// The seed of the adapter's initialisation and the batch order (drawn,
    /// printed and recorded if not given).
    #[arg(long, value_name = "N")]
    pub seed: Option<u64>,
    /// How the training is watched.
    #[command(flatten)]
    pub monitoring: MonitoringArgs,
    /// Hold the frozen base at bf16, half the bytes of fp32.
    #[arg(long)]
    pub bf16_base: bool,
    /// An experiment plan (`experiment check`) this run is one generation of:
    /// the repository must be a family the plan trains in `--generation`, and
    /// a sealed family is always refused. The catalog is the catalog.toml
    /// beside the plan.
    #[arg(long, value_name = "FILE", requires = "generation")]
    pub plan: Option<PathBuf>,
    /// The generation of the plan, counted from 1.
    #[arg(long, value_name = "N", requires = "plan",
        value_parser = clap::value_parser!(u32).range(1..))]
    pub generation: Option<u32>,
    /// The command that runs the project's whole test suite in the root of
    /// its tree, as a program and its arguments (everything after this
    /// flag, so it comes last). Without it the rules give one: `python3 -m
    /// unittest` for a Python project, `cargo test --offline` for a Cargo
    /// project.
    #[arg(
        long,
        num_args = 1..,
        allow_hyphen_values = true,
        value_name = "PROGRAM ARG..."
    )]
    pub test_command: Vec<String>,
}

/// A repository teacher, for clap: `reference` or a model reference.
pub fn repository_teacher(text: &str) -> Result<RepositoryTeacher, String> {
    text.parse()
        .map_err(|e: splinter_sdk::vocabulary::model_ref::RefError| e.to_string())
}

/// Which earlier records a `practice` replays.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum ReplayArg {
    /// Nothing.
    None,
    /// A seeded share of every earlier record.
    Uniform,
    /// The same number of records of each capability.
    Balanced,
    /// Records of the capabilities the champion fails most, in proportion
    /// (`generation run` only: it measures the failures first).
    FailureTargeted,
}

impl ReplayArg {
    /// The policy it names, for a `practice`, which cannot measure the
    /// champion's failures.
    pub fn policy(self) -> Result<ReplayPolicy, splinter_sdk::Error> {
        match self {
            Self::None => Ok(ReplayPolicy::None),
            Self::Uniform => Ok(ReplayPolicy::Uniform),
            Self::Balanced => Ok(ReplayPolicy::CapabilityBalanced),
            Self::FailureTargeted => Err(splinter_sdk::Error::Refused(
                "failure-targeted replay weighs the champion's failures, which `generation run` \
                 measures first: use it there"
                    .into(),
            )),
        }
    }
}
