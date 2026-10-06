// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The grammar of `exam` and `exam-set`.

use clap::{Args, Subcommand};
use splinter_sdk::exam_set::DEFAULT_EXAM_TASKS;
use splinter_sdk::powered::DEFAULT_RESAMPLES;
use splinter_sdk::reserve::DEFAULT_EXAM_FAMILIES;
use splinter_sdk::vocabulary::model_ref::ModelRef;

use super::{model_ref, RetrieveArgs};

/// `exam`.
#[derive(Debug, Args)]
pub struct ExamArgs {
    /// The candidate to examine: its id, or a unique prefix of it.
    #[arg(value_name = "CANDIDATE")]
    pub candidate: String,
    /// The model that judges (default: the judge role's model).
    #[arg(long, value_parser = model_ref, value_name = "REF")]
    pub judge: Option<ModelRef>,
    /// Also ask the base under this goal, as a prompt-only baseline the
    /// candidate is compared with (`learn` uses its goal).
    #[arg(long, value_name = "GOAL")]
    pub prompt: Option<String>,
    /// Also ask the candidate with retrieved passages shown before each
    /// task, and report whether retrieval found the passage the task was
    /// written from.
    #[command(flatten)]
    pub retrieval: RetrieveArgs,
    /// Put the candidate to this frozen exam (its id, or its manifest file)
    /// instead: four arms - the base, the base told who the candidate is to
    /// be, the candidate, and the candidate under that prompt - compared per
    /// task with the family as the unit of evidence, every verdict kept.
    #[arg(long, value_name = "EXAM")]
    pub exam_set: Option<String>,
    /// With --exam-set: answers per task per arm, the first greedy and the
    /// rest sampled.
    #[arg(long, value_name = "N", requires = "exam_set",
        value_parser = clap::value_parser!(u32).range(1..),
        help = format!("With --exam-set: answers per task per arm, the first greedy and the rest \
                        sampled [default: {DEFAULT_RESAMPLES}]"))]
    pub resamples: Option<u32>,
    /// With --exam-set: leave out the voice score, which loads each arm on
    /// the device to give the writer's own text a likelihood.
    #[arg(long, requires = "exam_set")]
    pub no_voice: bool,
}

/// `exam-set`.
#[derive(Debug, Subcommand)]
pub enum ExamSetCommand {
    /// Capture the sources, reserve families of them, write the exam from
    /// that text alone and freeze it.
    Create(ExamSetArgs),
    /// Show a frozen exam.
    Show {
        /// Its id, or its manifest file.
        #[arg(value_name = "EXAM")]
        id: String,
    },
}

/// `exam-set create`.
#[derive(Debug, Args)]
pub struct ExamSetArgs {
    /// Files, directories, cmd:<command> (quoted as one argument), or stored
    /// source ids.
    #[arg(required = true, value_name = "SOURCE")]
    pub sources: Vec<String>,
    /// How many families to reserve.
    #[arg(long, value_name = "N", default_value_t = DEFAULT_EXAM_FAMILIES)]
    pub families: usize,
    /// The most tasks the exam holds, spread over the families.
    #[arg(long, value_name = "N", default_value_t = DEFAULT_EXAM_TASKS)]
    pub tasks: usize,
    /// Task kinds, comma-separated.
    #[arg(long, value_delimiter = ',', value_name = "K,..",
        default_values_t = ["converse".to_string(), "advise".to_string(), "explain".to_string()])]
    pub kinds: Vec<String>,
    /// The model that writes the tasks (default: the policy).
    #[arg(long, value_parser = model_ref, value_name = "REF")]
    pub generator: Option<ModelRef>,
    /// What the learner is after; steers the tasks.
    #[arg(long, value_name = "TEXT")]
    pub goal: Option<String>,
    /// Who wrote the sources, when they are one person's.
    #[arg(long, value_name = "NAME")]
    pub persona: Option<String>,
    /// Varies which families are reserved.
    #[arg(long, value_name = "N", default_value_t = 0)]
    pub seed: u64,
    /// A candidate (repeat the flag for several) whose training, and that of
    /// the releases it continues, no reserved family may overlap: the exam is
    /// for measuring it.
    #[arg(long, value_name = "CANDIDATE")]
    pub not_trained_by: Vec<String>,
}
