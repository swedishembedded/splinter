// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning factories that take a model from
// agent rollouts to an evaluated, released candidate, for its clients. If
// your team needs expertise in model fine-tuning pipelines, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The learning factory's commands: a recipe of the task repository (the
//! working directory) is validated, planned and run into a run directory,
//! which every other command takes.

use std::path::PathBuf;

use clap::Args;

/// `validate` and `plan`.
#[derive(Debug, Args)]
pub struct RecipeArgs {
    /// The recipe, relative to the task repository (the working directory).
    #[arg(value_name = "RECIPE")]
    pub recipe: PathBuf,
}

/// `run`.
#[derive(Debug, Args)]
pub struct RunArgs {
    /// The recipe, relative to the task repository (the working directory).
    #[arg(value_name = "RECIPE")]
    pub recipe: PathBuf,
    /// The run directory; an existing run of the same recipe is continued.
    #[arg(long, value_name = "DIR")]
    pub out: PathBuf,
}

/// `resume`, `cancel`, `report`.
#[derive(Debug, Args)]
pub struct RunDirArgs {
    /// The run directory.
    #[arg(value_name = "RUN")]
    pub run: PathBuf,
}

/// `inspect`.
#[derive(Debug, Args)]
pub struct InspectArgs {
    /// The run directory.
    #[arg(value_name = "RUN")]
    pub run: PathBuf,
    /// Only the stages whose key starts with this (`rollouts`, `datasets/sft`).
    #[arg(long, value_name = "STAGE")]
    pub stage: Option<String>,
    /// Open the stage's trials in a viewer instead (`harbor`).
    #[arg(long, value_name = "VIEWER", requires = "stage")]
    pub viewer: Option<String>,
}

/// `fork`.
#[derive(Debug, Args)]
pub struct ForkArgs {
    /// The run whose products are reused.
    #[arg(value_name = "RUN")]
    pub run: PathBuf,
    /// The changed recipe, relative to the task repository.
    #[arg(long, value_name = "RECIPE")]
    pub recipe: PathBuf,
    /// The new run directory.
    #[arg(long, value_name = "DIR")]
    pub out: PathBuf,
}

/// `export`.
#[derive(Debug, Args)]
pub struct ExportArgs {
    /// The run directory.
    #[arg(value_name = "RUN")]
    pub run: PathBuf,
    /// The bundle to write (`.tar.zst`).
    #[arg(long, value_name = "FILE")]
    pub out: PathBuf,
}

/// `play`.
#[derive(Debug, Args)]
pub struct PlayArgs {
    /// The run directory.
    #[arg(value_name = "RUN")]
    pub run: PathBuf,
    /// The interactive attempt stage (`interactive: true` in the recipe), by
    /// its id.
    #[arg(long, value_name = "STAGE")]
    pub stage: String,
    /// The task to play, by its id in the collection.
    #[arg(long, value_name = "TASK")]
    pub task: String,
    /// Continue the conversation of this earlier session of the stage (its
    /// number); the earlier session is left as it is.
    #[arg(long = "continue", value_name = "SESSION")]
    pub continue_from: Option<u64>,
    /// Only play: do not run the recipe again afterwards (the next `resume`
    /// takes the session in).
    #[arg(long)]
    pub no_run: bool,
}

/// `generate`.
#[derive(Debug, Args)]
pub struct GenerateArgs {
    /// The collection file, relative to the task repository.
    #[arg(value_name = "COLLECTION")]
    pub collection: PathBuf,
}
