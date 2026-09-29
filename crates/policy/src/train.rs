// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gated self-improvement loops where a model
// update is adopted only on held-out evidence. If your team needs expertise
// in evaluation-gated fine-tuning, you can procure our services by sending
// an email to info@swedishembedded.com.

//! One LoRA fine-tune of the local model on a chat dataset, scored on the
//! held-out records before and after - brain's own trainer and evaluator,
//! called as a library, not a second implementation of either.
//!
//! The dataset is validated with the parser training uses
//! (`brain::validate_chat_dataset_for`), packed by
//! `data::chat::prepare_chat_samples`, fine-tuned by `qwen3::finetune_from`
//! as a LoRA over the base checkpoint, and scored with
//! `qwen3::eval::score_chat_dt` at one weight tier for both scores. What to
//! do with the scores - promote, reject - is the caller's decision.

use std::path::{Path, PathBuf};

use data::chat::ChatSample;
use model::FitOpts;
use qwen3::finetune::{finetune_from, Mode};
use splinter_lab::promotion::holdout_split;

/// One fine-tune: what to train, on what, and where its files go.
#[derive(Clone, Debug)]
pub struct FineTune<'a> {
    /// Base checkpoint directory (the model the agent serves).
    pub model_dir: &'a Path,
    /// Chat-format JSONL dataset; the newest records are held out.
    pub dataset: &'a Path,
    /// This attempt's own directory: the adapter and full checkpoint land here.
    pub attempt_dir: &'a Path,
    /// Where the packed token dataset is written.
    pub prepared_dir: &'a Path,
    pub steps: u32,
    /// LoRA rank / alpha for the adapter.
    pub rank: u32,
    pub alpha: f32,
}

/// One held-out score: teacher-forced loss and token accuracy over the
/// supervised positions of the held-out records.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct HeldOutScore {
    pub loss: f32,
    pub token_accuracy: f64,
    pub positions: usize,
}

/// What one fine-tune produced.
#[derive(Clone, Debug)]
pub struct Trained {
    /// The adapter file, carrying its ModelCard.
    pub adapter: PathBuf,
    /// Records in the dataset.
    pub records: usize,
    /// Training row length, sized to the longest example.
    pub block: u32,
    /// The base model on the held-out records.
    pub base: HeldOutScore,
    /// The base plus the new adapter on the same records.
    pub tuned: HeldOutScore,
}

/// Checks `dataset` the way training will parse it and returns its records,
/// before a device is claimed: a dataset that fails shape, encoding or
/// supervision is refused here with the offending record named.
pub fn read_dataset(dataset: &Path) -> anyhow::Result<Vec<ChatSample>> {
    let summary = brain::validate_chat_dataset(dataset).map_err(|e| {
        anyhow::anyhow!(
            "dataset {} is not valid trainer input: {e}",
            dataset.display()
        )
    })?;
    anyhow::ensure!(
        summary.trained_messages > 0,
        "dataset {} holds no supervised turns",
        dataset.display()
    );
    Ok(ChatSample::from_jsonl(dataset)?)
}

/// The smallest training row that can hold the longest example, rounded to
/// a power of two so a slightly longer future sample does not force a
/// re-architect. `fit` refuses a block smaller than an example.
fn block_size(longest_example: usize) -> u32 {
    let mut block = 64u32;
    while (block as usize) < longest_example {
        block *= 2;
    }
    block
}

fn fit_opts(steps: u32, block: u32) -> FitOpts {
    FitOpts {
        steps,
        batch_size: 1,
        block_size: block,
        // Scaled-down defaults: a handful of steps over a handful of
        // samples wants a short schedule, or every step is still warming up
        // when the budget ends.
        lr: 3e-4,
        min_lr: 3e-5,
        warmup: steps / 5,
        decay_iters: steps,
        weight_decay: 0.1,
        grad_clip: 1.0,
        grad_accum: 1,
        eval_interval: 0,
        eval_batches: 0,
        seed: 1337,
        checkpoint_secs: 0,
        mask_before: None,
        mask_per_line: false,
        align_to_lines: false,
        // The gate is the caller's decision over the before/after held-out
        // scores; the trainer's internal early stop would save a checkpoint
        // chosen by the TRAIN loss it sees, not the gate's evidence.
        patience: 0,
    }
}

/// Fine-tunes a LoRA on `request.dataset` and scores base and tuned on the
/// held-out records.
pub fn fine_tune(request: &FineTune<'_>) -> anyhow::Result<Trained> {
    let model_dir = request.model_dir;
    // brain's loader APIs take the checkpoint FILE; the caller passes the
    // standard directory, resolved once here.
    let weights = crate::local::resolve_base(model_dir)?;
    let weights = weights.to_str().expect("model path is utf-8");
    anyhow::ensure!(
        request.dataset.exists(),
        "no dataset at {} - learn a verified run first",
        request.dataset.display()
    );
    let summary = brain::validate_chat_dataset_for(request.dataset, model_dir)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let samples = read_dataset(request.dataset)?;
    let (train_samples, val_samples) = holdout_split(&samples).ok_or_else(|| {
        anyhow::anyhow!(
            "{} holds {} record(s); evaluation needs at least 2 so one can be held out",
            request.dataset.display(),
            summary.records
        )
    })?;

    let tmpl = data::chat_template::ChatTemplate::from_model_dir(model_dir)?;
    let tok = data::qwen_tokenizer::QwenBpe::from_file(
        model_dir
            .join("tokenizer.json")
            .to_str()
            .expect("model path is utf-8"),
    )
    .map_err(|e| anyhow::anyhow!("loading tokenizer from {model_dir:?}: {e}"))?;

    // The block must hold the longest example, or training silently
    // supervises nothing for it.
    let prepared = data::chat::prepare_chat_samples(
        train_samples,
        val_samples,
        &tok,
        &tmpl,
        tok.vocab_size(),
        request.prepared_dir,
    )?;
    let block = block_size(prepared.longest_example);
    eprintln!(
        "train: {} record(s), longest example {} tokens, block {block}",
        summary.records, prepared.longest_example
    );

    // Both scores run at the same reduced weight tier: the gate compares
    // two numbers, and two numbers at different precisions do not compare.
    let dt = gpu_core::select::Dtype::F16;
    let score = |adapter: Option<&str>| {
        let s = qwen3::eval::score_chat_dt(weights, adapter, &tok, &tmpl, val_samples, block, dt);
        HeldOutScore {
            loss: s.loss,
            token_accuracy: s.token_accuracy,
            positions: s.positions,
        }
    };
    eprintln!("train: scoring base against the held-out sample");
    let base = score(None);

    let adapter = request.attempt_dir.join("adapter.safetensors");
    // finetune_from writes a full training checkpoint (adapter tensors on
    // top of the base), which the scorer cannot fold; the scoreable,
    // ModelCard-carrying adapter file comes from re-loading that checkpoint
    // with its adapters trainable and saving through qwen3::lora, the same
    // two-step shape `brain qwen3 lora-train` uses.
    eprintln!(
        "train: tuning {} step(s), lora rank {}",
        request.steps, request.rank
    );
    let full_ckpt = request.attempt_dir.join("full.safetensors");
    let full_ckpt = full_ckpt.to_str().expect("checkpoint path is utf-8");
    finetune_from(
        weights,
        request.prepared_dir,
        &fit_opts(request.steps, block),
        &Mode::Lora {
            rank: request.rank,
            alpha: request.alpha,
        },
        full_ckpt,
        false,
    )?;
    let reader = checkpoint::weightio::WeightReader::open(full_ckpt)
        .map_err(|e| anyhow::anyhow!("reopening trained checkpoint: {e}"))?;
    let cfg = qwen3::config::QwenConfig::from_json(&reader.config());
    let shard = qwen3::Shard::whole(cfg.n_layers as usize);
    let reloaded = qwen3::model::Qwen::new_shard(cfg, 1, block, &reader, true, shard);
    // The card names what produced the adapter and what it sits on, so an
    // adapter is traceable to its own training evidence.
    let base_id = format!(
        "local/{}",
        model_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("base")
    );
    let card_id = format!("{base_id}:loop:experience:latest");
    let adapter_str = adapter.to_str().expect("adapter path is utf-8");
    qwen3::lora::save_adapter(adapter_str, &reloaded, &card_id, &base_id, None)
        .map_err(|e| anyhow::anyhow!("saving LoRA adapter: {e}"))?;

    eprintln!("train: scoring the tuned adapter against the held-out sample");
    let tuned = score(Some(adapter_str));
    Ok(Trained {
        adapter,
        records: summary.records,
        block,
        base,
        tuned,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_block_grows_to_hold_the_longest_example() {
        assert_eq!(block_size(10), 64);
        assert_eq!(block_size(64), 64);
        assert_eq!(block_size(65), 128);
        assert_eq!(block_size(1000), 1024);
    }
}
