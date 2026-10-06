// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gated, traceable releases of clinical risk
// models trained on longitudinal records, for its clients. If your team needs
// expertise in proving that a new risk model is better than the one in place
// before it ships, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Does a candidate risk model replace the one in place only when held-out
//! evidence says it should, and can the release be traced to every
//! participant record it was trained on?
//!
//! A controlled sample on a synthetic cohort whose hazards are known, written
//! on Splinter's SDK alone. Subcommands, in the order they are meant to be
//! used (see the README for the exact commands):
//!
//! ```text
//! synth    a small deterministic synthetic cohort, as a source file with a digest and declared terms
//! import   the source file as one immutable episode per participant
//! split    the episodes as the stored parts of one split, leakage-checked
//! train    a candidate (or a baseline) from the training and validation parts
//! eval     a candidate against a champion on the test part, under registered requirements
//! release  the candidate becomes the release only if the gate passes; else it is rejected and recorded
//! predict  the released file, loaded by plain brain: a patient history forecast, a new checkup appended, forecast again
//! lineage  a release back to its training run, datasets, episodes, source lines and source file digests
//! real     the same loop on a directory of timeline-v1 files, with terms from a declared policy
//! ontology the outcome definitions and the datasets listed for each
//! ```

mod lineage;
mod ontology;
mod predict;
mod real;
mod source;
mod state;
mod steps;
mod synth;

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use splinter_sdk::model::timeline::{Backbone, Mixer, NextEvents, StackConfig};
use splinter_sdk::timeline::data::SplitPlan;

use crate::ontology::{Ontology, NHANES_DATASET};
use crate::state::{Run, DEFAULT_SECRET};
use crate::steps::{EvalArgs, ImportRecord, SplitArgs, TrainArgs};

#[derive(Parser)]
#[command(
    about = "A gated, traceable release of a risk model, on a synthetic cohort with known hazards"
)]
struct Cli {
    /// The run directory: source files, Splinter's state and the labels.
    #[arg(long, global = true, default_value = "health-run")]
    dir: PathBuf,
    #[command(subcommand)]
    command: Command,
}

#[derive(Clone, Copy, ValueEnum)]
enum Plan {
    /// By participant group: a locked share is the test.
    Participants,
    /// Earlier entries train, later ones test.
    Temporal,
    /// One whole source is the test.
    Source,
}

#[derive(clap::Args)]
struct Cut {
    /// How to cut.
    #[arg(long, value_enum, default_value = "participants")]
    plan: Plan,
    /// The seed of every shuffle.
    #[arg(long, default_value_t = 3)]
    seed: u64,
    /// Share of the units held out as the test (participants plan).
    #[arg(long, default_value_t = 0.3)]
    test_share: f64,
    /// Folds the rest is cut into; one is the early-stopping part (participants plan).
    #[arg(long, default_value_t = 5)]
    folds: u32,
    /// The calendar cutoff: later entries test (temporal plan).
    #[arg(long, default_value_t = 2007.5)]
    cutoff: f64,
    /// The source held out (source plan).
    #[arg(long, default_value = "cycle-b")]
    held_out: String,
    /// The share of the training side that early stopping reads (temporal and source plans).
    #[arg(long, default_value_t = 0.2)]
    validation_share: f64,
}

impl Cut {
    fn args(&self, dataset: Option<String>) -> SplitArgs {
        SplitArgs {
            dataset,
            seed: self.seed,
            plan: match self.plan {
                Plan::Participants => SplitPlan::Participants {
                    locked_share: self.test_share,
                    folds: self.folds,
                },
                Plan::Temporal => SplitPlan::Temporal {
                    cutoff: self.cutoff,
                    validation_share: self.validation_share,
                },
                Plan::Source => SplitPlan::LeaveSourceOut {
                    source: self.held_out.clone(),
                    validation_share: self.validation_share,
                },
            },
        }
    }
}

/// What carries the visits to the prediction time.
#[derive(Clone, Copy, ValueEnum)]
enum BackboneArg {
    /// A per-channel continuous-time state.
    State,
    /// One layer of attention with rotary angles from real time.
    Attention,
}

/// How the blocks of a stack mix the visits.
#[derive(Clone, Copy, ValueEnum)]
enum MixerArg {
    /// Attention over the visits.
    Attention,
    /// A gated delta rule whose decay follows the time between visits.
    GatedDeltaNet,
    /// Gated delta rule blocks with an attention block every fourth.
    Hybrid,
}

/// The optional heads and state of brain's timeline model; each is brain's
/// own, passed through to the training configuration. With none of them the
/// model is the default one.
#[derive(clap::Args)]
struct Shape {
    /// Read the history visit by visit, carrying a state over this many visits.
    #[arg(long)]
    visits: Option<u32>,
    /// What carries the visits to the prediction time (needs --visits).
    #[arg(long, value_enum, requires = "visits", conflicts_with = "mixer")]
    backbone: Option<BackboneArg>,
    /// Carry the visits through a stack of blocks mixing as this (needs --visits).
    #[arg(long, value_enum, requires = "visits")]
    mixer: Option<MixerArg>,
    /// The number of blocks of the stack.
    #[arg(long, default_value_t = 2, requires = "mixer")]
    blocks: u32,
    /// Event codes modelled for which comes first after the prediction time,
    /// comma separated (an outcome or a history event such as `dx`).
    #[arg(long, value_delimiter = ',')]
    next_events: Option<Vec<String>>,
    /// The weight of the next-event group against the outcomes.
    #[arg(long, default_value_t = 1.0, requires = "next_events")]
    next_events_weight: f32,
    /// A forecast head on up to this many future measurements per subject.
    #[arg(long)]
    forecasts: Option<u32>,
    /// The weight of the forecast head against the outcomes.
    #[arg(long, default_value_t = 1.0, requires = "forecasts")]
    forecast_weight: f32,
}

impl Shape {
    fn backbone(&self) -> Option<Backbone> {
        match (self.mixer, self.backbone) {
            (Some(mixer), _) => Some(Backbone::Stack(StackConfig::new(
                match mixer {
                    MixerArg::Attention => Mixer::Attention,
                    MixerArg::GatedDeltaNet => Mixer::GatedDeltaNet,
                    MixerArg::Hybrid => Mixer::Hybrid,
                },
                self.blocks,
            ))),
            (None, Some(BackboneArg::State)) => Some(Backbone::State),
            (None, Some(BackboneArg::Attention)) => Some(Backbone::Attention),
            (None, None) => None,
        }
    }
}

#[derive(Subcommand)]
enum Command {
    /// Write a synthetic cohort as a source file with a digest and declared terms.
    Synth {
        /// Participants.
        #[arg(long, default_value_t = synth::DEFAULT_PARTICIPANTS)]
        n: usize,
        /// The generator's seed.
        #[arg(long, default_value_t = synth::DEFAULT_SEED)]
        seed: u64,
        /// The usage label the data is declared to have come under:
        /// redistributable, noncommercial, research_only, restricted_DUA or unknown.
        #[arg(long, default_value = "redistributable")]
        usage: String,
    },
    /// Import a source file as one immutable episode per participant.
    Import {
        /// The declaration of the source file; the run's synthetic one by default.
        #[arg(long)]
        source: Option<PathBuf>,
        /// The secret the opaque participant keys are derived with.
        #[arg(long, default_value = DEFAULT_SECRET)]
        secret: String,
    },
    /// Cut the imported episodes into the stored parts of one split.
    Split {
        /// The imported dataset to cut; the only one by default.
        #[arg(long)]
        dataset: Option<String>,
        #[command(flatten)]
        cut: Cut,
    },
    /// Train a candidate from the training and validation parts.
    Train {
        /// The label the candidate is named by.
        #[arg(long = "as")]
        label: String,
        /// Optimiser steps at most (early stopping usually ends sooner).
        #[arg(long, default_value_t = 1500)]
        steps: u32,
        /// The seed of the weights, the batches and the masks.
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// The additive proportional-hazards baseline instead of the set encoder.
        #[arg(long)]
        additive: bool,
        /// Tokens per subject, including the summary token.
        #[arg(long)]
        max_tokens: Option<u32>,
        /// Subjects per batch.
        #[arg(long, default_value_t = 256)]
        batch: u32,
        #[command(flatten)]
        shape: Shape,
        /// Horizons, in years, whose risk is calibrated on the validation
        /// units early stopping does not read, comma separated.
        #[arg(long, value_delimiter = ',', default_values_t = [5.0, 10.0])]
        calibrate_at: Vec<f64>,
        /// Train a model that is served raw, with no calibration.
        #[arg(long)]
        no_calibration: bool,
    },
    /// Score a candidate against a champion on the test part.
    Eval {
        /// The label the evaluation is named by.
        #[arg(long = "as")]
        label: String,
        /// The candidate, by label or id.
        #[arg(long)]
        candidate: String,
        /// The champion: a candidate by label or id, or release:<alias or id>.
        #[arg(long)]
        champion: String,
        /// Horizons, in years, comma separated.
        #[arg(long, value_delimiter = ',', default_values_t = [5.0, 10.0])]
        horizons: Vec<f64>,
        /// Judge calibration on the all-cause view alone.
        #[arg(long)]
        all_cause_only: bool,
        /// Bootstrap resamples of the differences.
        #[arg(long, default_value_t = 200)]
        reps: usize,
    },
    /// Release the evaluated candidate if the gate passes; else record it as rejected.
    Release {
        /// The evaluation, by label or address.
        #[arg(long)]
        evaluation: String,
        /// The alias the release replaces its champion on.
        #[arg(long, default_value = "health")]
        alias: String,
        /// Release without restriction: needs terms that allow training, commercial use and redistribution.
        #[arg(long)]
        unrestricted: bool,
    },
    /// Predict with a released file loaded by plain brain, before and after a new checkup.
    Predict {
        /// The release: an alias, an id or a unique prefix.
        #[arg(long, default_value = "health")]
        release: String,
        /// Which subject of the test part.
        #[arg(long, default_value_t = 0)]
        subject: usize,
        /// The variable measured at the new checkup.
        #[arg(long, default_value = "x1")]
        var: String,
        /// Its value.
        #[arg(long, default_value_t = 2.0)]
        value: f64,
        /// The unit the new value is stated in; the model refuses a unit other
        /// than the one it was trained on and never converts.
        #[arg(long)]
        unit: Option<String>,
        /// How long after the first prediction time, in years.
        #[arg(long, default_value_t = 2.0)]
        after: f64,
        /// Horizons the forecast is given at, in years, comma separated.
        #[arg(long, value_delimiter = ',', default_values_t = predict::HORIZONS)]
        horizons: Vec<f64>,
    },
    /// Trace a release back to its training run, datasets, episodes and source files.
    Lineage {
        /// The release: an alias, an id or a unique prefix.
        #[arg(long, default_value = "health")]
        release: String,
        /// Episodes listed per dataset.
        #[arg(long, default_value_t = 5)]
        episodes: usize,
        /// Print the lineage as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Import and split a directory of timeline-v1 files, with terms from a declared policy.
    Real {
        /// The directory holding one timelines*.jsonl file.
        #[arg(long)]
        timelines: PathBuf,
        /// A JSON policy file declaring the data's terms; with none the terms are unknown.
        #[arg(long)]
        policy: Option<PathBuf>,
        /// The dataset id the file is imported as (the ontology lists NHANES as nhanes_mortality).
        #[arg(long, default_value = NHANES_DATASET)]
        dataset: String,
        /// The outcome codes to train on, comma separated.
        #[arg(long, value_delimiter = ',')]
        codes: Option<Vec<String>>,
        #[command(flatten)]
        cut: Cut,
    },
    /// Print the outcome definitions and the datasets listed for each, or the
    /// cause group of an NHANES linked-mortality recode.
    Ontology {
        /// The `UCOD_LEADING` recode to map to its cause group.
        #[arg(long)]
        nhanes_recode: Option<u16>,
    },
}

fn print(value: &serde_json::Value) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

/// Prints a command's report and keeps it as `reports/<name>.json`.
fn emit(run: &Run, name: &str, value: &serde_json::Value) -> Result<()> {
    print(value)?;
    let path = run.report(name, value)?;
    eprintln!("report kept as {}", path.display());
    Ok(())
}

fn run(cli: Cli) -> Result<ExitCode> {
    let run = Run::new(&cli.dir);
    match cli.command {
        Command::Synth { n, seed, usage } => {
            let declaration = synth::write(&run.source_dir(), n, seed, &usage)?;
            print(&serde_json::to_value(&declaration)?)?;
        }
        Command::Import { source, secret } => {
            let path = source.unwrap_or_else(|| run.source_dir().join(synth::DECLARATION));
            let declaration = source::read(&path)?;
            let record = ImportRecord {
                dataset: declaration.dataset.clone(),
                supplies: declaration.supplies.clone(),
                subgroups: declaration.subgroups.clone(),
                usage: declaration.usage.clone(),
            };
            let report = steps::import(
                &run,
                &declaration.file_path(&path),
                &record,
                declaration.terms()?,
                &secret,
            )?;
            print(&report)?;
        }
        Command::Split { dataset, cut } => {
            let record = steps::split(&run, &cut.args(dataset))?;
            print(&serde_json::to_value(&record)?)?;
        }
        Command::Train {
            label,
            steps: n,
            seed,
            additive,
            max_tokens,
            batch,
            shape,
            calibrate_at,
            no_calibration,
        } => {
            let name = format!("train-{label}");
            let report = steps::train(
                &run,
                &TrainArgs {
                    label,
                    steps: n,
                    seed,
                    additive,
                    max_tokens,
                    batch,
                    visits: shape.visits,
                    backbone: shape.backbone(),
                    next_events: shape.next_events.clone().map(|codes| NextEvents {
                        codes,
                        weight: shape.next_events_weight,
                    }),
                    forecasts: shape.forecasts.map(|n| (n, shape.forecast_weight)),
                    calibrate_at: if no_calibration {
                        Vec::new()
                    } else {
                        calibrate_at
                    },
                },
            )?;
            emit(&run, &name, &report)?;
        }
        Command::Eval {
            label,
            candidate,
            champion,
            horizons,
            all_cause_only,
            reps,
        } => {
            let name = format!("eval-{label}");
            let report = steps::eval(
                &run,
                &EvalArgs {
                    label,
                    candidate,
                    champion,
                    horizons,
                    all_cause_only,
                    reps,
                },
            )?;
            emit(&run, &name, &report)?;
        }
        Command::Release {
            evaluation,
            alias,
            unrestricted,
        } => {
            let decision = steps::release(&run, &evaluation, &alias, unrestricted)?;
            emit(&run, &format!("release-{evaluation}"), &decision.report)?;
            if !decision.released {
                return Ok(ExitCode::from(3));
            }
        }
        Command::Predict {
            release,
            subject,
            var,
            value,
            unit,
            after,
            horizons,
        } => {
            let report = predict::predict(
                &run,
                &release,
                subject,
                &predict::Checkup {
                    var,
                    value,
                    unit,
                    after,
                },
                &horizons,
            )?;
            emit(&run, &format!("predict-{release}"), &report)?;
        }
        Command::Lineage {
            release,
            episodes,
            json,
        } => {
            let l = lineage::lineage(&run, &release, episodes)?;
            if json {
                print(&serde_json::to_value(&l)?)?;
            } else {
                print!("{}", lineage::render(&l));
            }
        }
        Command::Real {
            timelines,
            policy,
            dataset,
            codes,
            cut,
        } => {
            let report = real::real(
                &run,
                &real::RealArgs {
                    timelines,
                    policy,
                    split: cut.args(Some(dataset.clone())),
                    codes: codes.unwrap_or_else(real::nhanes_codes),
                    dataset,
                },
            )?;
            print(&report)?;
        }
        Command::Ontology {
            nhanes_recode: Some(recode),
        } => {
            let cause = ontology::nhanes_cause(recode);
            print(&serde_json::json!({
                "ucod_leading": recode,
                "cause": cause,
                "recodes_named_for_cvd_by_the_ontology": ontology::recodes_named(
                    &Ontology::bundled()?.request("death:cvd", NHANES_DATASET)?.source_definitions[NHANES_DATASET]
                ),
                "also_all_cause": ontology::all_cause([cause]),
            }))?;
        }
        Command::Ontology {
            nhanes_recode: None,
        } => {
            let o = Ontology::bundled()?;
            for code in o.codes() {
                let outcome = o
                    .outcome(code)
                    .map(|o| o.canonical_name.as_str())
                    .unwrap_or_default();
                let by = o.supported_by(code);
                println!(
                    "{code:<14} {outcome:<28} {}",
                    if by.is_empty() {
                        "unsupported locally".to_owned()
                    } else {
                        by.join(", ")
                    }
                );
            }
        }
    }
    Ok(ExitCode::SUCCESS)
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("health: {e:#}");
            ExitCode::from(2)
        }
    }
}
