// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements generations of a code-editing agent that
// keep what earlier generations learned, for its clients. If your team needs
// expertise in measuring retention across successive fine-tunes of an
// agent, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The grammar of `generation`.

use clap::{Args, Subcommand};
use splinter_sdk::generations::Limits;

use super::{voice_share, PracticeArgs};

/// `generation`.
#[derive(Debug, Subcommand)]
pub enum GenerationCommand {
    /// Run one generation of a coding policy: measure the champion on every
    /// protected task, `practice` the repository to a candidate, measure
    /// both on the protected tasks and the new exam, reject the candidate on
    /// a retention or regression limit, release it through the gate
    /// otherwise, and keep a ledger entry with everything it was decided on.
    Run(Box<GenerationRunArgs>),
    /// The ledger of an experiment and the full retention matrix of the
    /// lineage `policy:<alias>` points at: every generation, the base
    /// included, measured on every protected task.
    Matrix {
        /// The experiment whose ledger is shown.
        #[arg(long, value_name = "NAME")]
        experiment: String,
        /// The alias whose lineage is measured.
        #[arg(long, default_value = "default", value_name = "ALIAS")]
        alias: String,
    },
}

/// `generation run`.
#[derive(Debug, Args)]
pub struct GenerationRunArgs {
    /// The experiment this generation belongs to; names its ledger.
    #[arg(long, value_name = "NAME")]
    pub experiment: String,
    /// The most a protected family's pass rate may fall.
    #[arg(long, value_name = "SHARE", value_parser = voice_share,
        help = format!("The most a protected family's pass rate may fall [default: {}]",
            Limits::default().max_family_drop))]
    pub max_family_drop: Option<f64>,
    /// The most the mean over protected families may fall.
    #[arg(long, value_name = "SHARE", value_parser = voice_share,
        help = format!("The most the mean over protected families may fall [default: {}]",
            Limits::default().max_mean_drop))]
    pub max_mean_drop: Option<f64>,
    /// What the candidate must gain on the new exam: more than this.
    #[arg(long, value_name = "SHARE", value_parser = voice_share,
        help = format!("What the candidate must gain on the new exam: more than this [default: {}]",
            Limits::default().min_new_gain))]
    pub min_new_gain: Option<f64>,
    /// The repository practised, and how.
    #[command(flatten)]
    pub practice: PracticeArgs,
}
