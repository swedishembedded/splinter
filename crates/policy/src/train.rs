// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gated self-improvement loops where a model
// update is adopted only on held-out evidence. If your team needs expertise
// in evaluation-gated fine-tuning, you can procure our services by sending
// an email to info@swedishembedded.com.

//! One LoRA fine-tune of the local model on a chat dataset, scored on the
//! held-out records before and after - brain's own trainer and evaluator,
//! called through its SDK, not a second implementation of either.
//!
//! Splinter decides which records are held out ([`holdout_split`]) and
//! writes the two halves as separate files; [`brain::ChatFineTune`]
//! validates both against the base's tokenizer and chat template, trains
//! the adapter on one, scores base and tuned on the other at one precision,
//! and writes the adapter and its training record. What to do with the
//! scores - promote, reject - is the caller's decision.

use std::path::{Path, PathBuf};

use splinter_lab::promotion::holdout_split;

/// The file names one attempt's split is written under, inside its
/// attempt directory.
const TRAIN_FILE: &str = "train.jsonl";
const HELD_OUT_FILE: &str = "held_out.jsonl";

/// One fine-tune: what to train, on what, and where its files go.
#[derive(Clone, Debug)]
pub struct FineTune<'a> {
    /// Base checkpoint directory (the model the agent serves).
    pub model_dir: &'a Path,
    /// Chat-format JSONL dataset; the newest records are held out.
    pub dataset: &'a Path,
    /// This attempt's own directory: the split, the packed dataset, the
    /// adapter and its training record land here.
    pub attempt_dir: &'a Path,
    /// Optimizer steps to train for; warmup is the first fifth of them and
    /// the learning rate decays over all of them.
    pub steps: u32,
    /// LoRA rank of the adapter's low-rank update matrices.
    pub rank: u32,
    /// LoRA alpha: the update is scaled by `alpha / rank` before it is added
    /// to the base weights.
    pub alpha: f32,
}

/// One held-out score: teacher-forced loss and token accuracy over the
/// supervised positions of the held-out records.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct HeldOutScore {
    /// Mean per-token cross-entropy over the supervised positions; lower is
    /// better. `None` when no position was scored.
    pub loss: Option<f32>,
    /// Fraction of supervised positions, 0.0-1.0, where the greedy argmax
    /// matched the true next token; `None` when no position was scored.
    pub token_accuracy: Option<f64>,
    /// Supervised token positions the two numbers above were computed over.
    pub positions: usize,
    /// Held-out records scored.
    pub records: usize,
    /// Held-out records skipped (too long for the scoring row, or not
    /// encodable).
    pub skipped: usize,
}

impl From<brain::HeldOutScore> for HeldOutScore {
    fn from(s: brain::HeldOutScore) -> Self {
        Self {
            loss: s.loss,
            token_accuracy: s.token_accuracy,
            positions: s.positions,
            records: s.records,
            skipped: s.skipped,
        }
    }
}

/// What one fine-tune produced.
#[derive(Clone, Debug)]
pub struct Trained {
    /// The adapter file, carrying its ModelCard.
    pub adapter: PathBuf,
    /// `sha256:<hex>` of the adapter file - the digest brain reports for it
    /// when it serves it.
    pub adapter_digest: String,
    /// brain's training record for the adapter, beside it.
    pub training_record: PathBuf,
    /// Records in the dataset.
    pub records: usize,
    /// Training row length, sized to the longest example.
    pub block: u32,
    /// The base model on the held-out records.
    pub base: HeldOutScore,
    /// The base plus the new adapter on the same records.
    pub tuned: HeldOutScore,
}

/// What a chat dataset holds, as the trainer's own parser counts it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DatasetSummary {
    /// Conversations, one per JSONL line.
    pub records: usize,
    /// Messages across every conversation.
    pub messages: usize,
    /// Messages marked for supervision (`"train": true`).
    pub trained_messages: usize,
}

impl From<brain::ChatDatasetSummary> for DatasetSummary {
    fn from(s: brain::ChatDatasetSummary) -> Self {
        Self {
            records: s.records,
            messages: s.messages,
            trained_messages: s.trained_messages,
        }
    }
}

/// Parses `dataset` with the trainer's own parser: the wire schema, and the
/// supervision boundaries it declares. The error names the offending record.
pub fn validate_dataset(dataset: &Path) -> anyhow::Result<DatasetSummary> {
    brain::validate_chat_dataset(dataset)
        .map(DatasetSummary::from)
        .map_err(|e| {
            anyhow::anyhow!(
                "dataset {} is not valid trainer input: {e}",
                dataset.display()
            )
        })
}

/// [`validate_dataset`], and encodes every record with the checkpoint in
/// `model_dir` (its tokenizer and chat template): a record can satisfy the
/// wire schema and still have no honest loss-mask boundary under the
/// template it will train with. Needs no device.
pub fn validate_dataset_for(dataset: &Path, model_dir: &Path) -> anyhow::Result<DatasetSummary> {
    brain::validate_chat_dataset_for(dataset, model_dir)
        .map(DatasetSummary::from)
        .map_err(|e| anyhow::anyhow!("{e}"))
}

/// Fine-tunes a LoRA on `request.dataset` and scores base and tuned on the
/// held-out records.
pub fn fine_tune(request: &FineTune<'_>) -> anyhow::Result<Trained> {
    anyhow::ensure!(
        request.dataset.exists(),
        "no dataset at {} - learn a verified run first",
        request.dataset.display()
    );
    let summary = validate_dataset(request.dataset)?;
    let (train, held_out) = split_dataset(request.dataset, request.attempt_dir)?;
    // brain resolves a directory only in its model store's layout; the
    // checkpoint file is resolved here, and its directory supplies the
    // tokenizer and chat template.
    let weights = crate::local::resolve_base(request.model_dir)?;
    let weights = weights
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("{} is not valid UTF-8", weights.display()))?;
    // The card names what produced the adapter and what it sits on, so an
    // adapter is traceable to its own training evidence.
    let base_id = format!(
        "local/{}",
        request
            .model_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("base")
    );
    eprintln!(
        "train: {} record(s), tuning {} step(s), lora rank {}",
        summary.records, request.steps, request.rank
    );
    let outcome = brain::ChatFineTune::from_pretrained(weights)
        .dataset(train)
        .held_out(held_out)
        .out_dir(request.attempt_dir)
        .adapter_id(format!("{base_id}:loop:experience:latest"))
        .steps(request.steps)
        .rank(request.rank)
        .alpha(request.alpha)
        .run()
        .map_err(|e| anyhow::anyhow!("fine-tuning on {}: {e}", request.dataset.display()))?;
    let incomplete = |what: &str| {
        anyhow::anyhow!(
            "fine-tune in {} reported no {what}",
            request.attempt_dir.display()
        )
    };
    // `run` has no cancel token, so it completes or fails; a completed run
    // with a held-out set carries both scores, the adapter and its record.
    anyhow::ensure!(
        outcome.status == brain::FineTuneStatus::Completed,
        "fine-tune in {} did not complete",
        request.attempt_dir.display()
    );
    Ok(Trained {
        adapter: outcome.adapter.ok_or_else(|| incomplete("adapter"))?,
        adapter_digest: outcome
            .adapter_digest
            .ok_or_else(|| incomplete("adapter digest"))?,
        training_record: outcome
            .record
            .ok_or_else(|| incomplete("training record"))?,
        records: summary.records,
        block: outcome.block,
        base: outcome
            .base_score
            .ok_or_else(|| incomplete("base score"))?
            .into(),
        tuned: outcome
            .tuned_score
            .ok_or_else(|| incomplete("tuned score"))?
            .into(),
    })
}

/// Writes `dataset`'s records into `dir` as two files - the records to
/// train on and the newest ones, held out - and returns their paths. A
/// record is a non-blank line, as the trainer's parser reads it.
fn split_dataset(dataset: &Path, dir: &Path) -> anyhow::Result<(PathBuf, PathBuf)> {
    let text = std::fs::read_to_string(dataset)
        .map_err(|e| anyhow::anyhow!("reading {}: {e}", dataset.display()))?;
    let records: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    let (train, held_out) = holdout_split(&records).ok_or_else(|| {
        anyhow::anyhow!(
            "{} holds {} record(s); evaluation needs at least 2 so one can be held out",
            dataset.display(),
            records.len()
        )
    })?;
    let write = |name: &str, lines: &[&str]| -> anyhow::Result<PathBuf> {
        let path = dir.join(name);
        let mut body = lines.join("\n");
        body.push('\n');
        std::fs::write(&path, body).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
        Ok(path)
    };
    Ok((write(TRAIN_FILE, train)?, write(HELD_OUT_FILE, held_out)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The newest records are held out in a file of their own, never
    /// trained on; a dataset too small to hold one out is refused.
    #[test]
    fn the_newest_records_are_held_out_in_their_own_file() {
        let dir = std::env::temp_dir().join(format!("policy-split-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let dataset = dir.join("pool.jsonl");
        let lines: Vec<String> = (0..12).map(|i| format!("{{\"n\":{i}}}")).collect();
        std::fs::write(&dataset, format!("{}\n\n", lines.join("\n"))).unwrap();

        let (train, held_out) = split_dataset(&dataset, &dir).unwrap();
        let read = |p: &Path| -> Vec<String> {
            std::fs::read_to_string(p)
                .unwrap()
                .lines()
                .map(str::to_string)
                .collect()
        };
        assert_eq!(read(&train), lines[..11]);
        assert_eq!(read(&held_out), lines[11..]);

        std::fs::write(&dataset, "{\"n\":0}\n").unwrap();
        let err = split_dataset(&dataset, &dir).unwrap_err().to_string();
        assert!(err.contains("at least 2"), "{err}");

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
