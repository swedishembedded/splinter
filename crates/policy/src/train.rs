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
//! the adapter on one (with any replayed datasets mixed in, and continuing
//! an existing adapter when one is named), scores base and tuned on the
//! other at one precision, and writes the adapter and its training record.
//! What to do with the scores is the caller's decision.

use std::path::{Path, PathBuf};

use splinter_lab::holdout::holdout_split;

use crate::error::PolicyError;

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
    /// Chat datasets whose every record is mixed into training (never held
    /// out), so earlier experience is replayed beside the new.
    pub replay: &'a [PathBuf],
    /// An adapter to continue training instead of starting a fresh one; its
    /// own rank and alpha then apply.
    pub continue_from: Option<&'a Path>,
    /// Stops training at the next optimizer step once cancelled; a
    /// cancelled fine-tune exports no adapter and is reported as an error.
    pub cancel: Option<&'a sven_sdk::CancelToken>,
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
pub fn validate_dataset(dataset: &Path) -> Result<DatasetSummary, PolicyError> {
    brain::validate_chat_dataset(dataset)
        .map(DatasetSummary::from)
        .map_err(|e| PolicyError::Dataset {
            path: dataset.to_path_buf(),
            reason: e.to_string(),
        })
}

/// [`validate_dataset`], and encodes every record with the checkpoint in
/// `model_dir` (its tokenizer and chat template): a record can satisfy the
/// wire schema and still have no honest loss-mask boundary under the
/// template it will train with. Needs no device.
pub fn validate_dataset_for(
    dataset: &Path,
    model_dir: &Path,
) -> Result<DatasetSummary, PolicyError> {
    brain::validate_chat_dataset_for(dataset, model_dir)
        .map(DatasetSummary::from)
        .map_err(|e| PolicyError::Dataset {
            path: dataset.to_path_buf(),
            reason: e.to_string(),
        })
}

/// Fine-tunes a LoRA on `request.dataset` and scores base and tuned on the
/// held-out records.
pub fn fine_tune(request: &FineTune<'_>) -> Result<Trained, PolicyError> {
    let failed = |reason: String| PolicyError::Train {
        dir: request.attempt_dir.to_path_buf(),
        reason,
    };
    let summary = validate_dataset(request.dataset)?;
    let (train, held_out) = split_dataset(request.dataset, request.attempt_dir)?;
    // brain resolves a directory only in its model store's layout; the
    // checkpoint file is resolved here, and its directory supplies the
    // tokenizer and chat template.
    let weights = crate::local::resolve_base(request.model_dir)?;
    let weights = weights.to_str().ok_or_else(|| PolicyError::NotUtf8 {
        path: weights.clone(),
    })?;
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
    for replayed in request.replay {
        validate_dataset(replayed)?;
    }
    let mut fine_tune = brain::ChatFineTune::from_pretrained(weights)
        .dataset(train)
        .held_out(held_out)
        .out_dir(request.attempt_dir)
        .adapter_id(format!("{base_id}:splinter:candidate"))
        .steps(request.steps)
        .rank(request.rank)
        .alpha(request.alpha);
    for replayed in request.replay {
        fine_tune = fine_tune.replay(replayed);
    }
    if let Some(adapter) = request.continue_from {
        fine_tune = fine_tune.continue_from(adapter);
    }
    // brain's token stops training at a step boundary; sven's is the one
    // the caller cancels, so it is polled once per step.
    let brain_cancel = brain::CancelToken::armed();
    let outcome = fine_tune
        .run_with(&brain_cancel, |_| {
            if request
                .cancel
                .is_some_and(sven_sdk::CancelToken::is_cancelled)
            {
                brain_cancel.cancel();
            }
        })
        .map_err(|e| failed(format!("training on {}: {e}", request.dataset.display())))?;
    if brain_cancel.is_cancelled() {
        return Err(PolicyError::Cancelled {
            dir: request.attempt_dir.to_path_buf(),
        });
    }
    let incomplete = |what: &str| failed(format!("it reported no {what}"));
    // Not cancelled, so it completed or failed; a completed run with a
    // held-out set carries both scores, the adapter and its record.
    if outcome.status != brain::FineTuneStatus::Completed {
        return Err(failed("it did not complete".into()));
    }
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
fn split_dataset(dataset: &Path, dir: &Path) -> Result<(PathBuf, PathBuf), PolicyError> {
    let text = std::fs::read_to_string(dataset).map_err(|source| PolicyError::Io {
        path: dataset.to_path_buf(),
        source,
    })?;
    let records: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    let (train, held_out) = holdout_split(&records).ok_or_else(|| PolicyError::TooFewRecords {
        path: dataset.to_path_buf(),
        records: records.len(),
    })?;
    let write = |name: &str, lines: &[&str]| -> Result<PathBuf, PolicyError> {
        let path = dir.join(name);
        let mut body = lines.join("\n");
        body.push('\n');
        std::fs::write(&path, body).map_err(|source| PolicyError::Io {
            path: path.clone(),
            source,
        })?;
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
