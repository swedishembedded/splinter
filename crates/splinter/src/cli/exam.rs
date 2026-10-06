// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The grammar of `exam` and `exam-set`.

use std::path::PathBuf;

use clap::{Args, Subcommand};
use splinter_sdk::exam_set::{DEFAULT_DEV_TASKS_PER_FAMILY, DEFAULT_TASKS_PER_FAMILY};
use splinter_sdk::powered::DEFAULT_RESAMPLES;
use splinter_sdk::reserve::{DEFAULT_DEV_FAMILIES, DEFAULT_EXAM_FAMILIES};
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
    /// With --exam-set: put only this many of the exam's families to the
    /// candidate, as a pilot that estimates the discordance and clustering.
    #[arg(long, value_name = "N", requires = "exam_set")]
    pub pilot_families: Option<usize>,
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
    /// Simulate the planned paired test to size an exam before it is paid
    /// for: its power at an assumed effect, and how often it calls a
    /// difference that is not there.
    Power(PowerArgs),
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
    /// Tasks of each family of the final test.
    #[arg(long, value_name = "N", default_value_t = DEFAULT_TASKS_PER_FAMILY)]
    pub tasks_per_family: usize,
    /// How many more families to reserve for the dev suite.
    #[arg(long, value_name = "N", default_value_t = DEFAULT_DEV_FAMILIES)]
    pub dev_families: usize,
    /// Tasks of each dev family.
    #[arg(long, value_name = "N", default_value_t = DEFAULT_DEV_TASKS_PER_FAMILY)]
    pub dev_tasks_per_family: usize,
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

/// `exam-set power`.
#[derive(Debug, Args)]
pub struct PowerArgs {
    /// Families of the exam.
    #[arg(long, value_name = "N", default_value_t = DEFAULT_EXAM_FAMILIES)]
    pub families: usize,
    /// Tasks of each.
    #[arg(long, value_name = "N", default_value_t = DEFAULT_TASKS_PER_FAMILY)]
    pub tasks_per_family: usize,
    /// The lead of the first arm in the share of tasks right (0.10 is ten
    /// points).
    #[arg(long, value_name = "SHARE", default_value_t = 0.10)]
    pub effect: f64,
    /// The share of tasks only one of the arms gets right.
    #[arg(
        long,
        value_name = "SHARE",
        default_value_t = 0.30,
        conflicts_with = "from_report"
    )]
    pub discordance: f64,
    /// The intraclass correlation of the per-task difference in a family.
    #[arg(
        long,
        value_name = "RHO",
        default_value_t = 0.10,
        conflicts_with = "from_report"
    )]
    pub icc: f64,
    /// Take the discordance and the intraclass correlation of the primary
    /// comparison from the report of a pilot (`exam --json` output), not from
    /// an assumption.
    #[arg(long, value_name = "FILE")]
    pub from_report: Option<PathBuf>,
    /// Simulated exams.
    #[arg(long, value_name = "N", default_value_t = 400)]
    pub replicates: usize,
}

/// `exam-report`: analyses done after an exam from what its report kept.
#[derive(Debug, Subcommand)]
pub enum ExamReportCommand {
    /// Write a blind, stratified, shuffled sample of the exam's answers for a
    /// person to label right or wrong, and the key that says which arm gave
    /// each and what the judge said.
    LabelsExport {
        /// The exam report (`exam --exam-set ... --json` output).
        #[arg(value_name = "REPORT")]
        report: PathBuf,
        /// Where the items to label are written (JSON Lines).
        #[arg(long, value_name = "FILE")]
        out: PathBuf,
        /// Where the key is written (JSON).
        #[arg(long, value_name = "FILE")]
        key: PathBuf,
        /// Items to sample.
        #[arg(long, value_name = "N", default_value_t = 400)]
        n: usize,
        /// Varies the sample.
        #[arg(long, value_name = "N", default_value_t = 0)]
        seed: u64,
    },
    /// What a person's labels say of the judge: agreement, kappa, false
    /// right and false wrong rates, length bias.
    LabelsImport {
        /// The items, each with its `label` filled in `right` or `wrong`.
        #[arg(value_name = "LABELS")]
        labels: PathBuf,
        /// The key `labels-export` wrote.
        #[arg(long, value_name = "FILE")]
        key: PathBuf,
    },
    /// How much of each arm's answers is the text the candidate was trained to
    /// produce again: overlap of 8, 13 and 20 word runs and the longest run.
    Memorisation {
        /// The exam report.
        #[arg(value_name = "REPORT")]
        report: PathBuf,
        /// The candidate the report is of.
        #[arg(long, value_name = "CANDIDATE")]
        candidate: String,
    },
}
