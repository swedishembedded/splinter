// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that decide which documents a
// person or an agent should study to reach a goal, for its clients. If your
// team needs expertise in goal-directed selection of learning material you
// can procure our services by sending an email to info@swedishembedded.com.

//! `goal-train`: train an adapter on one arm's records as a job of a brain
//! training server, and report the held-out score of the base and of the tuned
//! model, taken by the same job and so under the same load.
//!
//! ```text
//! goal-train --train arm/train.jsonl --held-out held-out.jsonl --attempt-dir DIR \
//!     --train-endpoint http://127.0.0.1:8803/v1 --brain-key-file KEY --artifact-store DIR \
//!     --model-dir ~/project/shared/containers/brain-models/Qwen/Qwen3.8-27B-FP8 --steps 150 --out result.json
//! ```

use std::path::PathBuf;

use anyhow::Context as _;
use clap::Parser;
use splinter_sdk::model::served_train::{fine_tune_served, ArtifactStore};
use splinter_sdk::model::train::{BasePrecision, FineTune};
use splinter_sdk::model::BrainEndpoints;

#[derive(Parser)]
#[command(version, about)]
struct Cli {
    /// The records to train on (chat lines).
    #[arg(long)]
    train: PathBuf,
    /// The records scored before and after, never trained on.
    #[arg(long)]
    held_out: PathBuf,
    /// This job's directory.
    #[arg(long)]
    attempt_dir: PathBuf,
    /// The training server's OpenAI base URL.
    #[arg(long)]
    train_endpoint: String,
    /// The file holding the key.
    #[arg(long)]
    brain_key_file: PathBuf,
    /// The artifact store the server shares.
    #[arg(long)]
    artifact_store: PathBuf,
    /// The base checkpoint directory.
    #[arg(long)]
    model_dir: PathBuf,
    /// Optimizer steps: the compute budget, equal for every arm.
    #[arg(long, default_value_t = 150)]
    steps: u32,
    /// LoRA rank.
    #[arg(long, default_value_t = 16)]
    rank: u32,
    /// Records averaged into one step.
    #[arg(long, default_value_t = 4)]
    grad_accum: u32,
    /// Peak learning rate.
    #[arg(long, default_value_t = 1e-4)]
    lr: f32,
    /// Seed of the adapter and the batch order; drawn and printed when not given.
    #[arg(long)]
    seed: Option<u64>,
    /// Where to write the result as JSON.
    #[arg(long)]
    out: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let seed = splinter_sdk::seed::resolve("seed", cli.seed);
    let key = BrainEndpoints::read_key(&cli.brain_key_file).context("reading the key")?;
    let store = ArtifactStore::open(&cli.artifact_store).context("opening the artifact store")?;
    let brain = BrainEndpoints::new(vec![cli.train_endpoint.clone()], key)
        .context("endpoint")?
        .with_artifact_store(store);
    std::fs::create_dir_all(&cli.attempt_dir)?;
    let trained = fine_tune_served(
        &brain,
        &FineTune {
            model_dir: &cli.model_dir,
            train: &cli.train,
            held_out: &cli.held_out,
            held_out_text: None,
            monitor: None,
            eval_every: 0,
            patience: 0,
            attempt_dir: &cli.attempt_dir,
            steps: cli.steps,
            rank: cli.rank,
            alpha: 2.0 * cli.rank as f32,
            replay: &[],
            replay_share: None,
            weighted_replay: &[],
            grad_accum: cli.grad_accum,
            continue_from: None,
            cancel: None,
            base: BasePrecision::Fp8,
            learning_rate: Some(cli.lr),
            weight_decay: 0.0,
            seed,
            max_block: None,
            thinking: false,
            pack_answers: false,
            on_step: None,
            keep_evaluations: None,
        },
    )
    .context("training")?;
    let result = serde_json::json!({
        "seed": seed, "steps": cli.steps, "records": trained.records,
        "base": trained.base, "tuned": trained.tuned,
    });
    std::fs::write(&cli.out, serde_json::to_string_pretty(&result)?)?;
    println!("{result}");
    Ok(())
}
