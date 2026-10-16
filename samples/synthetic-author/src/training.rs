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
use serde::Serialize;
use splinter_sdk::vocabulary::digest::Digest;

use crate::cache::Cache;
use crate::dataset::Briefings;

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
    #[arg(long, default_value_t = 200)]
    steps: u32,
    /// LoRA rank.
    #[arg(long, default_value_t = 16)]
    rank: u32,
    /// LoRA alpha.
    #[arg(long, default_value_t = 32.0)]
    alpha: f32,
    /// Learning rate (brain's default if not given).
    #[arg(long)]
    lr: Option<f32>,
    /// Records averaged into one optimizer step.
    #[arg(long, default_value_t = 4)]
    records_per_step: u32,
    /// Hold the base at bf16.
    #[arg(long)]
    bf16: bool,
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
#[derive(Clone, Debug, PartialEq, Serialize)]
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
    /// Digest of the file of records.
    pub dataset_digest: String,
    /// Which briefings the records were made with.
    pub briefings: Briefings,
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
            dataset_digest: Digest::of_reader(std::fs::File::open(dataset)?)?.to_string(),
            briefings: self.briefings,
        })
    }
}

/// What `summary.json` holds: the settings, then what came of them.
#[derive(Serialize)]
struct Summary<'a> {
    #[serde(flatten)]
    settings: &'a Settings,
    adapter: &'a Path,
    records: usize,
    held_out_before: serde_json::Value,
    held_out_after: serde_json::Value,
    seconds: u64,
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
    std::fs::create_dir_all(&attempt)?;
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
    let split = splinter_sdk::data::holdout::split_dataset_file(&dataset, &attempt)?;
    let request = splinter_sdk::model::train::FineTune {
        max_block: None,
        model_dir: &args.base,
        train: &split.train,
        held_out: &split.held_out,
        held_out_text: None,
        monitor: None,
        eval_every: 0,
        patience: 0,
        keep_evaluations: None,
        attempt_dir: &attempt,
        steps: settings.steps,
        rank: settings.rank,
        alpha: settings.alpha,
        replay: &[],
        replay_share: None,
        weighted_replay: &[],
        grad_accum: settings.records_per_step,
        continue_from: None,
        cancel: None,
        bf16_base: settings.bf16,
        learning_rate: settings.learning_rate,
        weight_decay: 0.0,
        seed: settings.seed,
        thinking: false,
        on_step: Some(splinter_sdk::model::train::StepHook(&report)),
    };
    let trained = splinter_sdk::model::train::fine_tune(&request)?;
    let summary = Summary {
        settings: &settings,
        adapter: &trained.adapter,
        records: trained.records,
        held_out_before: serde_json::json!({"loss": trained.base.loss, "token_accuracy": trained.base.token_accuracy}),
        held_out_after: serde_json::json!({"loss": trained.tuned.loss, "token_accuracy": trained.tuned.token_accuracy}),
        seconds: started.elapsed().as_secs(),
    };
    std::fs::write(
        attempt.join("summary.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
