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
//! compare  two arms (or an external baseline) on the same folds: the corrected resampled t-test per metric
//! external score a baseline's out-of-fold prediction files with the arms' metrics (secondary)
//! report   the pre-registered criteria against the locked-test results
//! amend    pin an amendment to the criteria beside them (never an edit)
//! ensemble the locked test scored by several seeds' models together (secondary)
//! temporal train on the earlier cycles, score the later ones (secondary; never the locked test)
//! recipe  a named variant of the horizon spec on chosen folds, averages of variants, learning-curve subsamples (secondary)
//! intake   the harmonisation admission rules, and a model's proposals, against the hand mapping
//! ```

mod build;
mod cause_metrics;
mod causes;
mod commands;
mod compare;
mod concepts;
mod conditions;
mod contributing;
mod diet;
mod ensemble;
mod experiment;
mod external;
mod intake;
mod intervals;
mod lifeexp;
mod metrics;
mod nhanes;
mod pooled;
mod prevalence;
mod recipe;
mod recipe_run;
mod report;
mod temporal;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

use crate::compare::Comparand;
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
    /// Compare two arms, or an arm and a scored baseline, on the folds both ran on.
    Compare {
        /// The build's output directory.
        #[arg(long)]
        data: PathBuf,
        /// The candidate: an arm, or `external:<baseline>`.
        #[arg(long)]
        a: Comparand,
        /// The reference: an arm, or `external:<baseline>`.
        #[arg(long)]
        b: Comparand,
        /// Only the folds of these repeats (default: all both ran on).
        #[arg(long)]
        repeat: Vec<usize>,
    },
    /// Secondary: score a baseline's out-of-fold prediction files
    /// (`baselines/<name>/r<repeat>-k<fold>.jsonl` in the data directory) with
    /// the metrics the arms are scored with, on the cross-validation folds.
    External {
        /// The build's output directory.
        #[arg(long)]
        data: PathBuf,
        /// The baseline's name: its directory under `baselines/`.
        #[arg(long)]
        baseline: String,
        /// The files cover only some subjects of each fold (a subsample): score those.
        #[arg(long)]
        subset: bool,
    },
    /// Secondary (T1): cause-specific accuracy at 5, 10 and 15 years of
    /// baselines' out-of-fold predictions, against a reference baseline.
    Causes {
        /// The build's output directory.
        #[arg(long)]
        data: PathBuf,
        /// Baselines to report (directories under `baselines/`).
        #[arg(long, required = true)]
        model: Vec<String>,
        /// The baseline the others are compared with.
        #[arg(long, default_value = "cs-cox-standard")]
        reference: String,
        /// Score the prediction files first (otherwise earlier scores are read).
        #[arg(long)]
        score: bool,
    },
    /// Secondary (T2): death with diabetes or hypertension listed on the
    /// certificate, from pooled out-of-fold predictions of one repeat.
    Flags {
        /// The build's output directory (with `causes.jsonl`).
        #[arg(long)]
        data: PathBuf,
        /// Baselines whose files carry `flag_cif` (`baselines/flags.py`).
        #[arg(long, required = true)]
        model: Vec<String>,
        /// Baselines whose all-cause incidence is used as a ranker: the control.
        #[arg(long, required = true)]
        ranker: Vec<String>,
        /// The repeat whose folds are pooled.
        #[arg(long, default_value_t = 0)]
        repeat: usize,
        /// Skip the cluster bootstrap (point estimates only).
        #[arg(long)]
        no_intervals: bool,
    },
    /// Secondary (T3): restricted mean survival time and mortality-equivalent
    /// age from baselines' out-of-fold all-cause curves.
    Lifeexp {
        /// The build's output directory.
        #[arg(long)]
        data: PathBuf,
        /// Baselines (directories under `baselines/`).
        #[arg(long, required = true)]
        model: Vec<String>,
        /// The repeat whose folds are pooled.
        #[arg(long, default_value_t = 0)]
        repeat: usize,
        /// The restriction time in years (at most 15).
        #[arg(long, default_value_t = 10.0)]
        tau: f64,
        /// Directory to write each model's equivalent ages to.
        #[arg(long)]
        aa_dir: Option<PathBuf>,
    },
    /// Secondary (T4, T5): label every subject's prevalent conditions and
    /// undiagnosed disease from the NHANES files into `conditions.jsonl`.
    Labels {
        /// Directory holding `<cycle start year>/*.xpt`.
        #[arg(long)]
        nhanes: PathBuf,
        /// The build's output directory (with `causes.jsonl`).
        #[arg(long)]
        data: PathBuf,
    },
    /// Secondary (T4): models of prevalent conditions against the age-and-sex
    /// model, per fold (`baselines/conditions.py` writes the predictions).
    Prevalence {
        /// The build's output directory (with `conditions.jsonl`).
        #[arg(long)]
        data: PathBuf,
        /// Models (directories under `conditions/`).
        #[arg(long, required = true)]
        model: Vec<String>,
        /// The model the others are compared with.
        #[arg(long, default_value = "agesex")]
        reference: String,
    },
    /// Secondary (T5): screens for undiagnosed disease at a threshold chosen
    /// on the training subjects, pooled over the folds of one repeat.
    Screen {
        /// The build's output directory (with `conditions.jsonl`).
        #[arg(long)]
        data: PathBuf,
        /// Models (directories under `conditions/`).
        #[arg(long, required = true)]
        model: Vec<String>,
        /// The screen the others must beat.
        #[arg(long, default_value = "agebmi")]
        comparator: String,
        /// The repeat whose folds are pooled.
        #[arg(long, default_value_t = 0)]
        repeat: usize,
    },
    /// Two models on the pooled out-of-fold predictions of one repeat: the
    /// paired difference in integrated Brier score and in Brier score at ten
    /// years, with a cluster-bootstrap interval (the protocol's preferred test).
    Bootstrap {
        /// The build's output directory.
        #[arg(long)]
        data: PathBuf,
        /// A baseline name or `recipe:<name>`.
        #[arg(long)]
        a: String,
        /// The reference.
        #[arg(long)]
        b: String,
        /// The repeat whose folds are pooled.
        #[arg(long, default_value_t = 0)]
        repeat: usize,
    },
    /// Calibration at ten years per cause (observed over expected and slope
    /// with cluster-bootstrap intervals, and the calibration-curve errors).
    Calibration {
        /// The build's output directory.
        #[arg(long)]
        data: PathBuf,
        /// Baselines or `recipe:<name>`.
        #[arg(long, required = true)]
        model: Vec<String>,
        /// The repeat whose folds are pooled.
        #[arg(long, default_value_t = 0)]
        repeat: usize,
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
    /// Secondary: train named variants of the horizon spec on
    /// cross-validation folds, average their predictions, and tabulate them.
    /// Everything is kept under `<data>/recipe/`, never in `runs/`.
    Recipe {
        #[command(subcommand)]
        command: RecipeCommand,
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

#[derive(Subcommand)]
enum RecipeCommand {
    /// Train a recipe with each seed on the folds asked for; keeps the
    /// predictions and scores of member `<name>-s<seed>`.
    Run {
        /// The build's output directory.
        #[arg(long)]
        data: PathBuf,
        /// The recipe's name; members are `<name>-s<seed>`.
        #[arg(long)]
        name: String,
        /// An override, `key=value` (see `src/recipe.rs`); repeat for several.
        #[arg(long = "set")]
        set: Vec<String>,
        /// Training seeds (default: 1).
        #[arg(long, default_values_t = [1u64])]
        seed: Vec<u64>,
        /// Only these repeats (default: all).
        #[arg(long)]
        repeat: Vec<usize>,
        /// Only these folds (default: all).
        #[arg(long)]
        fold: Vec<usize>,
        /// Keep the scores only, not the prediction files.
        #[arg(long)]
        no_predictions: bool,
    },
    /// Keep the mean of members' predicted curves as `<name>`, scored like any other.
    Blend {
        /// The build's output directory.
        #[arg(long)]
        data: PathBuf,
        /// The average's name.
        #[arg(long)]
        name: String,
        /// A member: a recipe's directory name, or `external:<baseline>`.
        #[arg(long = "member", required = true)]
        member: Vec<String>,
        /// Only these repeats (default: all).
        #[arg(long)]
        repeat: Vec<usize>,
        /// Only these folds (default: all).
        #[arg(long)]
        fold: Vec<usize>,
    },
    /// Write the training subjects a `train_share` subsample keeps, per fold,
    /// for a baseline outside Rust to train on.
    Subsample {
        /// The build's output directory.
        #[arg(long)]
        data: PathBuf,
        /// Fraction of each fold's training subjects.
        #[arg(long)]
        share: f64,
        /// Seed of the subsample.
        #[arg(long, default_value_t = recipe::DEFAULT_TRAIN_SEED)]
        seed: u64,
        /// Only these repeats (default: all).
        #[arg(long)]
        repeat: Vec<usize>,
        /// Only these folds (default: all).
        #[arg(long)]
        fold: Vec<usize>,
    },
    /// Score the prediction files already in `<data>/recipe/<name>/`
    /// (written by a program outside Rust) like any other recipe.
    Score {
        /// The build's output directory.
        #[arg(long)]
        data: PathBuf,
        /// The directory name under `recipe/`.
        #[arg(long)]
        name: String,
        /// What made the predictions, for the record.
        #[arg(long)]
        made_of: String,
        /// Only these repeats (default: all).
        #[arg(long)]
        repeat: Vec<usize>,
        /// Only these folds (default: all).
        #[arg(long)]
        fold: Vec<usize>,
    },
    /// Mean metrics of models over the folds they all ran on, as a table.
    Summary {
        /// The build's output directory.
        #[arg(long)]
        data: PathBuf,
        /// A model: an arm, `external:<baseline>` or `recipe:<name>`.
        #[arg(long = "model", required = true)]
        model: Vec<Comparand>,
        /// Only the folds of these repeats.
        #[arg(long)]
        repeat: Vec<usize>,
    },
}

fn recipe_command(command: RecipeCommand) -> Result<()> {
    match command {
        RecipeCommand::Run {
            data,
            name,
            set,
            seed,
            repeat,
            fold,
            no_predictions,
        } => recipe_run::run(
            &data,
            &name,
            &recipe::Recipe::parse(&set)?,
            &seed,
            &repeat,
            &fold,
            !no_predictions,
        ),
        RecipeCommand::Blend {
            data,
            name,
            member,
            repeat,
            fold,
        } => recipe_run::blend(&data, &name, &member, &repeat, &fold),
        RecipeCommand::Subsample {
            data,
            share,
            seed,
            repeat,
            fold,
        } => recipe_run::write_subsamples(&data, share, seed, &repeat, &fold),
        RecipeCommand::Score {
            data,
            name,
            made_of,
            repeat,
            fold,
        } => recipe_run::score_files(&data, &name, &made_of, &repeat, &fold),
        RecipeCommand::Summary {
            data,
            model,
            repeat,
        } => recipe_run::summary(&data, &model, &repeat),
    }
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
        Command::Compare { data, a, b, repeat } => compare::compare(&data, &a, &b, &repeat),
        Command::External {
            data,
            baseline,
            subset,
        } => external::score(&data, &baseline, subset),
        Command::Causes {
            data,
            model,
            reference,
            score,
        } => cause_metrics::run(&data, &model, &reference, score),
        Command::Flags {
            data,
            model,
            ranker,
            repeat,
            no_intervals,
        } => contributing::run(&data, &model, &ranker, repeat, !no_intervals),
        Command::Lifeexp {
            data,
            model,
            repeat,
            tau,
            aa_dir,
        } => lifeexp::run(&data, &model, repeat, tau, aa_dir.as_deref()),
        Command::Labels { nhanes, data } => conditions::write(&nhanes, &data),
        Command::Prevalence {
            data,
            model,
            reference,
        } => {
            println!(
                "{}",
                prevalence::report_prevalent(&data, &model, &reference)?
            );
            Ok(())
        }
        Command::Screen {
            data,
            model,
            comparator,
            repeat,
        } => {
            println!(
                "{}",
                prevalence::report_screen(&data, &model, &comparator, repeat)?
            );
            Ok(())
        }
        Command::Bootstrap { data, a, b, repeat } => {
            println!("{}", pooled::difference(&data, &a, &b, repeat)?);
            Ok(())
        }
        Command::Calibration {
            data,
            model,
            repeat,
        } => {
            println!("{}", pooled::calibration(&data, &model, repeat)?);
            Ok(())
        }
        Command::Report { data } => report::report(&data),
        Command::Amend { data } => commands::amend(&data),
        Command::Ensemble { data, arm, members } => ensemble::ensemble(&data, arm, members),
        Command::Temporal {
            data,
            arm,
            seed,
            split,
        } => temporal::temporal(&data, arm, seed, split),
        Command::Recipe { command } => recipe_command(command),
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
