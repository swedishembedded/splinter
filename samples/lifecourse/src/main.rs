// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements long-horizon risk prediction from cohort and
// survey data for its clients. If your team needs expertise in building,
// validating and proving time-to-event models on real health records, you
// can procure our services by sending an email to info@swedishembedded.com.

//! Can a model read one examination of a person and say, calibrated, how
//! likely they are to die - and of what - within the next five to fifteen
//! years, better than the conventional risk factors can?
//!
//! The data are the ten continuous NHANES cycles (1999-2018) with the
//! public-use linked mortality file (through 2019). Subcommands, in the
//! order they are meant to be used:
//!
//! ```text
//! build    NHANES files and the mortality linkage as timeline-v1 subjects
//! freeze   partition once, and pin the data, the partition and the criteria
//! cv       train and score one arm on every cross-validation fold
//! final    train one arm on everything but the locked test, and score it there once
//! compare  two arms on the same folds: the corrected resampled t-test per metric
//! report   the pre-registered criteria against the locked-test results
//! amend    pin an amendment to the criteria beside them (never an edit)
//! ensemble the locked test scored by several seeds' models together (secondary)
//! temporal train on the earlier cycles, score the later ones (secondary; never the locked test)
//! intake   the harmonisation admission rules, and a model's proposals, against the hand mapping
//! ```

mod build;
mod commands;
mod concepts;
mod diet;
mod ensemble;
mod experiment;
mod intake;
mod intervals;
mod metrics;
mod nhanes;
mod report;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

use crate::experiment::Arm;

#[derive(Parser)]
#[command(about = "Long-horizon mortality prediction from one NHANES examination")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Build timeline-v1 subjects from the NHANES files and the mortality linkage.
    Build {
        /// Directory holding `<cycle start year>/*.xpt`.
        #[arg(long)]
        nhanes: PathBuf,
        /// Directory holding `NHANES_<y>_<y+1>_MORT_2019_PUBLIC.dat`.
        #[arg(long)]
        mortality: PathBuf,
        /// Output directory.
        #[arg(long)]
        data: PathBuf,
    },
    /// Partition the built subjects once and pin data, partition and criteria.
    Freeze {
        /// The build's output directory.
        #[arg(long)]
        data: PathBuf,
    },
    /// Train and score one arm on cross-validation folds.
    Cv {
        /// The build's output directory.
        #[arg(long)]
        data: PathBuf,
        /// Which arm.
        #[arg(long, value_enum)]
        arm: Arm,
        /// Seed of the run.
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Only these repeats (default: all).
        #[arg(long)]
        repeat: Vec<usize>,
        /// Only these folds (default: all).
        #[arg(long)]
        fold: Vec<usize>,
        /// Train on outcomes shuffled across the training subjects: the
        /// leakage check, which must score no better than the age-sex arm.
        #[arg(long)]
        permute: bool,
    },
    /// Train one arm on all but the locked test and score it on the locked test.
    Final {
        /// The build's output directory.
        #[arg(long)]
        data: PathBuf,
        /// Which arm.
        #[arg(long, value_enum)]
        arm: Arm,
        /// Seed of the run.
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Why the locked test is scored again for an arm and seed it already
        /// scored (it is otherwise refused).
        #[arg(long)]
        reason: Option<String>,
    },
    /// Compare two arms on the folds both ran on.
    Compare {
        /// The build's output directory.
        #[arg(long)]
        data: PathBuf,
        /// The candidate.
        #[arg(long, value_enum)]
        a: Arm,
        /// The reference.
        #[arg(long, value_enum)]
        b: Arm,
    },
    /// The pre-registered criteria against the locked-test results.
    Report {
        /// The build's output directory.
        #[arg(long)]
        data: PathBuf,
    },
    /// Pin the amendments to the criteria made after the freeze, each as its
    /// own file beside the untouched criteria.
    Amend {
        /// The build's output directory.
        #[arg(long)]
        data: PathBuf,
    },
    /// Secondary analysis: the locked test scored by an ensemble of the arm's
    /// models trained with seeds 1..=members on the same subjects as `final`.
    Ensemble {
        /// The build's output directory.
        #[arg(long)]
        data: PathBuf,
        /// Which arm.
        #[arg(long, value_enum)]
        arm: Arm,
        /// How many members.
        #[arg(long, default_value_t = 5)]
        members: u64,
    },
    /// Secondary analysis of calendar shift: train on the non-locked subjects
    /// of the cycles before `split`, score those of `split` and later.
    Temporal {
        /// The build's output directory.
        #[arg(long)]
        data: PathBuf,
        /// Which arm.
        #[arg(long, value_enum)]
        arm: Arm,
        /// Seed of the run.
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// The first cycle (start year) scored rather than trained on.
        #[arg(long, default_value_t = 2009)]
        split: u16,
    },
    /// Measure the harmonisation admission rules, and a served model's
    /// mapping proposals, against the hand-written exam concepts.
    Intake {
        /// Directory holding `<cycle start year>/*.xpt` and `*.htm` codebooks.
        #[arg(long)]
        nhanes: PathBuf,
        /// Directory holding the linkage files: the cohort is who they make eligible.
        #[arg(long)]
        mortality: PathBuf,
        /// Output directory for the reports.
        #[arg(long)]
        out: PathBuf,
        /// A served model's OpenAI-compatible address; without it only the
        /// rules are measured.
        #[arg(long, requires_all = ["api_key", "model"])]
        base_url: Option<String>,
        /// The served model's key.
        #[arg(long)]
        api_key: Option<String>,
        /// The served model's name.
        #[arg(long)]
        model: Option<String>,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Build {
            nhanes,
            mortality,
            data,
        } => commands::build(&nhanes, &mortality, &data),
        Command::Freeze { data } => commands::freeze(&data),
        Command::Cv {
            data,
            arm,
            seed,
            repeat,
            fold,
            permute,
        } => commands::cv(&data, arm, seed, &repeat, &fold, permute),
        Command::Final {
            data,
            arm,
            seed,
            reason,
        } => commands::final_test(&data, arm, seed, reason.as_deref()),
        Command::Compare { data, a, b } => commands::compare(&data, a, b),
        Command::Report { data } => report::report(&data),
        Command::Amend { data } => commands::amend(&data),
        Command::Ensemble { data, arm, members } => ensemble::ensemble(&data, arm, members),
        Command::Temporal {
            data,
            arm,
            seed,
            split,
        } => commands::temporal(&data, arm, seed, split),
        Command::Intake {
            nhanes,
            mortality,
            out,
            base_url,
            api_key,
            model,
        } => {
            let served = match (base_url, api_key, model) {
                (Some(base_url), Some(api_key), Some(model)) => Some(intake::Served {
                    base_url,
                    api_key,
                    model,
                }),
                _ => None,
            };
            intake::intake(&nhanes, &mortality, &out, served.as_ref())
        }
    }
}
