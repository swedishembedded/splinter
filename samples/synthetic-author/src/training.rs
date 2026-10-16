// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements controlled validation of learning systems
// against ground truth that cannot have been seen in pretraining, for its
// clients. If your team needs expertise in training runs whose every setting
// is recorded and repeatable, you can procure our services by sending an
// email to info@swedishembedded.com.

//! The `train` command: fine-tune an adapter on the records of a corpus and
//! record everything that made it.

use std::path::{Path, PathBuf};

use clap::Args;
use serde::{Deserialize, Serialize};
use splinter_sdk::model::train::BasePrecision;
use splinter_sdk::vocabulary::digest::Digest;

use crate::cache::Cache;
use crate::dataset::Briefings;

/// Optimizer steps when none are asked for.
pub const DEFAULT_STEPS: u32 = 200;
/// LoRA rank when none is asked for.
pub const DEFAULT_RANK: u32 = 16;
/// LoRA alpha when none is asked for.
pub const DEFAULT_ALPHA: f32 = 32.0;
/// Records averaged into one optimizer step when none is asked for.
pub const DEFAULT_RECORDS_PER_STEP: u32 = 4;

/// Train an adapter on the records.
#[derive(Args)]
pub struct TrainArgs {
    /// The corpus directory.
    #[arg(long)]
    cache: PathBuf,
    /// The checkpoint directory of the base model.
    #[arg(long)]
    base: PathBuf,
    /// Optimizer steps.
    #[arg(long, default_value_t = DEFAULT_STEPS)]
    steps: u32,
    /// LoRA rank.
    #[arg(long, default_value_t = DEFAULT_RANK)]
    rank: u32,
    /// LoRA alpha.
    #[arg(long, default_value_t = DEFAULT_ALPHA)]
    alpha: f32,
    /// Learning rate (brain's default if not given).
    #[arg(long)]
    lr: Option<f32>,
    /// Records averaged into one optimizer step.
    #[arg(long, default_value_t = DEFAULT_RECORDS_PER_STEP)]
    records_per_step: u32,
    /// Hold the base at bf16.
    #[arg(long, conflicts_with = "int8")]
    bf16: bool,
    /// Hold the base as int8 weights with a scale per 32, about a quarter of
    /// fp32.
    #[arg(long)]
    int8: bool,
    /// Seed of the adapter's initialisation and batch order; drawn and
    /// printed when absent.
    #[arg(long)]
    train_seed: Option<u64>,
    /// Which briefings the records were made with: trains on the records
    /// `dataset --briefings` wrote for that setting.
    #[arg(long, value_enum, default_value_t = Briefings::default())]
    briefings: Briefings,
}

/// What made an adapter: every setting of the run and the records it was
/// trained on.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Settings {
    /// The seed of the adapter's initialisation and batch order.
    pub seed: u64,
    /// Optimizer steps.
    pub steps: u32,
    /// LoRA rank.
    pub rank: u32,
    /// LoRA alpha.
    pub alpha: f32,
    /// The learning rate; `null` is brain's own default.
    pub learning_rate: Option<f32>,
    /// Records averaged into one optimizer step.
    pub records_per_step: u32,
    /// Whether the base was held at bf16.
    pub bf16: bool,
    /// Whether the base was held as int8 weights; absent from the records of
    /// runs made before the setting existed, which held it at fp32 or bf16.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub int8: bool,
    /// Digest of the file of records.
    pub dataset_digest: String,
    /// Which briefings the records were made with.
    pub briefings: Briefings,
}

impl Settings {
    /// How these settings hold the base while the adapter trains.
    #[must_use]
    pub fn base_precision(&self) -> BasePrecision {
        match (self.int8, self.bf16) {
            (true, _) => BasePrecision::Int8,
            (false, true) => BasePrecision::Bf16,
            (false, false) => BasePrecision::Fp32,
        }
    }
}

/// The digest `Settings` records for the file of records at `dataset`.
///
/// # Errors
/// The file cannot be read.
pub fn dataset_digest(dataset: &Path) -> std::io::Result<String> {
    Ok(Digest::of_reader(std::fs::File::open(dataset)?)?.to_string())
}

impl TrainArgs {
    /// The settings these arguments ask for, trained with `seed` on the
    /// records in `dataset`.
    ///
    /// # Errors
    /// The records cannot be read.
    pub fn settings(&self, seed: u64, dataset: &Path) -> std::io::Result<Settings> {
        Ok(Settings {
            seed,
            steps: self.steps,
            rank: self.rank,
            alpha: self.alpha,
            learning_rate: self.lr,
            records_per_step: self.records_per_step,
            bf16: self.bf16,
            int8: self.int8,
            dataset_digest: dataset_digest(dataset)?,
            briefings: self.briefings,
        })
    }
}

/// What `summary.json` holds: the settings, then what came of them.
#[derive(Serialize)]
struct Summary<'a> {
    #[serde(flatten)]
    settings: &'a Settings,
    #[serde(flatten)]
    trained: &'a Trained,
}

/// What a training run came to.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Trained {
    /// The adapter file.
    pub adapter: PathBuf,
    /// Records trained on.
    pub records: usize,
    /// Loss and token accuracy on the held-out records before training.
    pub held_out_before: serde_json::Value,
    /// Loss and token accuracy on the held-out records after training.
    pub held_out_after: serde_json::Value,
    /// Seconds the training took.
    pub seconds: u64,
}

/// One adapter to train: where the base is, which records to train on, where
/// the attempt is kept and every setting of the run.
#[derive(Clone, Copy, Debug)]
pub struct Job<'a> {
    /// The checkpoint directory of the base model.
    pub base: &'a Path,
    /// The file of records.
    pub dataset: &'a Path,
    /// The directory the attempt, its split of the records and its
    /// `summary.json` are kept in.
    pub attempt: &'a Path,
    /// Every setting of the run.
    pub settings: &'a Settings,
}

/// Trains an adapter as `job` says and writes `summary.json` beside it.
///
/// # Errors
/// The records cannot be read or split, or the training fails.
pub fn fine_tune(job: &Job<'_>) -> anyhow::Result<Trained> {
    let settings = job.settings;
    std::fs::create_dir_all(job.attempt)?;
    let started = std::time::Instant::now();
    let report = |r: &splinter_sdk::model::train::StepReport| {
        eprintln!(
            "step {}/{} loss {:.4} elapsed {:.0}s",
            r.step,
            r.steps,
            r.loss,
            started.elapsed().as_secs_f64()
        );
    };
    let split = splinter_sdk::data::holdout::split_dataset_file(job.dataset, job.attempt)?;
    let request = splinter_sdk::model::train::FineTune {
        // A sample's datasets are dialogues: one example per supervised turn.
        pack_answers: false,
        max_block: None,
        model_dir: job.base,
        train: &split.train,
        held_out: &split.held_out,
        held_out_text: None,
        monitor: None,
        eval_every: 0,
        patience: 0,
        keep_evaluations: None,
        attempt_dir: job.attempt,
        steps: settings.steps,
        rank: settings.rank,
        alpha: settings.alpha,
        replay: &[],
        replay_share: None,
        weighted_replay: &[],
        grad_accum: settings.records_per_step,
        continue_from: None,
        cancel: None,
        base: settings.base_precision(),
        learning_rate: settings.learning_rate,
        weight_decay: 0.0,
        seed: settings.seed,
        thinking: false,
        on_step: Some(splinter_sdk::model::train::StepHook(&report)),
    };
    let trained = splinter_sdk::model::train::fine_tune(&request)?;
    let trained = Trained {
        adapter: trained.adapter,
        records: trained.records,
        held_out_before: serde_json::json!({"loss": trained.base.loss, "token_accuracy": trained.base.token_accuracy}),
        held_out_after: serde_json::json!({"loss": trained.tuned.loss, "token_accuracy": trained.tuned.token_accuracy}),
        seconds: started.elapsed().as_secs(),
    };
    std::fs::write(
        job.attempt.join("summary.json"),
        serde_json::to_string_pretty(&Summary {
            settings,
            trained: &trained,
        })?,
    )?;
    Ok(trained)
}

/// Runs `train`.
///
/// # Errors
/// The records cannot be read or split, or the training fails.
pub fn run(args: &TrainArgs) -> anyhow::Result<()> {
    let seed = splinter_sdk::seed::resolve("train-seed", args.train_seed);
    let cache = Cache::open(args.cache.clone())?;
    let dataset = crate::dataset::path(&cache, args.briefings);
    let settings = args.settings(seed, &dataset)?;
    let attempt = cache
        .dir()
        .join(format!("train{}", args.briefings.suffix()));
    let trained = fine_tune(&Job {
        base: &args.base,
        dataset: &dataset,
        attempt: &attempt,
        settings: &settings,
    })?;
    println!(
        "{}",
        serde_json::to_string_pretty(&Summary {
            settings: &settings,
            trained: &trained,
        })?
    );
    Ok(())
}
