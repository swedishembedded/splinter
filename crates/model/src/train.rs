// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gated self-improvement loops where a model
// update is adopted only on held-out evidence. If your team needs expertise
// in evaluation-gated fine-tuning, you can procure our services by sending
// an email to info@swedishembedded.com.

//! One LoRA fine-tune of the local model, scored on held-out records -
//! brain's own trainers and evaluators, called through its SDK, not a
//! second implementation of any of them. Two regimes:
//!
//! * [`fine_tune`]: supervised fine-tuning on a `generic-messages-v2` chat
//!   dataset through [`brain::ChatFineTune`], which validates both halves
//!   of the split against the base's tokenizer and chat template, trains
//!   the adapter on one (with any replayed datasets mixed in, each weighted
//!   file at its own share of the draws), and scores base and tuned on the
//!   other at one precision.
//! * [`train_preference`]: direct preference optimisation on a
//!   `generic-preference-v1` pair dataset through
//!   [`brain::PreferenceFineTune`], against a frozen reference - the model
//!   the run starts from - and scored as brain's preference score: on the
//!   held-out pairs, how much more the tuned adapter prefers each chosen
//!   answer over its rejected one than the reference does.
//!
//! In both, the caller hands over the dataset already split into the records
//! to train on and the ones held out, a named adapter is
//! continued instead of a fresh one started, and brain writes the adapter
//! and its training record into the attempt directory. What to do with the
//! scores is the caller's decision.

use std::path::{Path, PathBuf};

use splinter_core::training::{
    CurvePoint, HeldOutScore, PreferenceScore, Selection, TrainingCurve,
};

use crate::error::PolicyError;
use crate::local::{closed_think_pairs, opens_think_block};

/// The DPO temperature `beta` a preference fine-tune uses when its caller
/// names none: brain's default.
pub const DEFAULT_DPO_BETA: f32 = brain::DEFAULT_DPO_BETA;

/// One fine-tune: what to train, on what, and where its files go.
#[derive(Clone, Debug)]
pub struct FineTune<'a> {
    /// Base checkpoint directory (the model the agent serves).
    pub model_dir: &'a Path,
    /// Chat-format JSONL records to train on.
    pub train: &'a Path,
    /// Chat-format JSONL records scored before and after, never trained on.
    pub held_out: &'a Path,
    /// Chat-format JSONL records scored every `eval_every` steps as the run
    /// trains, never trained on: the curve, and what the best step is
    /// selected on. `None` monitors nothing (and `eval_every` must be 0).
    pub monitor: Option<&'a Path>,
    /// Steps between evaluations of the monitoring records; 0 never
    /// evaluates and the last step is exported.
    pub eval_every: u32,
    /// Evaluations without improvement before the run stops; 0 runs the
    /// whole budget. With `eval_every > 0` the adapter exported is the best
    /// evaluation's either way.
    pub patience: u32,
    /// This attempt's own directory: the packed dataset, the adapter and its
    /// training record land here.
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
    /// The share of training draws that come from `replay` in all; `None`
    /// mixes the plain union, in which a large replay set takes most steps.
    pub replay_share: Option<f32>,
    /// Chat datasets each mixed into training at a share of the draws of
    /// its own (never held out), beside `replay`: a base model's own answers
    /// rehearsed at a set rate, say. The shares, with `replay_share`, leave
    /// the new records the rest.
    pub weighted_replay: &'a [(PathBuf, f32)],
    /// Records averaged into one optimizer step: the effective batch size.
    pub grad_accum: u32,
    /// An adapter to continue training instead of starting a fresh one; its
    /// own rank and alpha then apply.
    pub continue_from: Option<&'a Path>,
    /// Stops training at the next optimizer step once cancelled; a
    /// cancelled fine-tune exports no adapter and is reported as an error.
    pub cancel: Option<&'a sven_sdk::CancelToken>,
    /// Hold the frozen base at bf16, half the bytes of fp32: what lets a 7B
    /// base train on one 24 GiB card. Scoring before and after runs at the
    /// same tier. brain's default, fp32, when `false`.
    pub bf16_base: bool,
    /// The peak learning rate; brain's default when `None`.
    pub learning_rate: Option<f32>,
    /// The seed of a fresh adapter's initialisation and the batch order;
    /// brain's default when `None`. Two runs that differ only here show
    /// how much of a measured difference is the draw.
    pub seed: Option<u64>,
    /// Whether the model is trained to reason before it answers. Off, it is
    /// trained for no-think mode - what Splinter asks of it by default - so a
    /// reasoning model trains on the state a no-think prompt leaves it in
    /// (see [`brain::ChatFineTune::thinking`]); a base that never reasons is
    /// unaffected.
    pub thinking: bool,
    /// Told after every optimizer step, so a long run can be watched.
    pub on_step: Option<StepHook<'a>>,
}

/// A caller's function told of each optimizer step.
#[derive(Clone, Copy)]
pub struct StepHook<'a>(pub &'a dyn Fn(&StepReport));

impl std::fmt::Debug for StepHook<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("StepHook")
    }
}

/// One completed optimizer step of a fine-tune.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepReport {
    /// Steps completed so far.
    pub step: u32,
    /// Steps the run will take.
    pub steps: u32,
    /// This step's training loss.
    pub loss: f32,
}

/// brain's held-out score as Splinter records it.
fn held_out_score(s: brain::HeldOutScore) -> HeldOutScore {
    HeldOutScore {
        loss: s.loss,
        token_accuracy: s.token_accuracy,
        positions: s.positions,
        records: s.records,
        skipped: s.skipped,
    }
}

/// brain's preference score as Splinter records it.
fn preference_score(s: brain::PreferenceScore) -> PreferenceScore {
    PreferenceScore {
        accuracy: s.accuracy,
        mean_margin: s.mean_margin,
        pairs: s.pairs,
        skipped: s.skipped,
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
    /// `sha256:<hex>` of the base checkpoint the adapter was trained on -
    /// the digest its card records, which brain checks when it serves it.
    pub base_digest: String,
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
    /// The monitoring curve, and which step the adapter is.
    pub curve: TrainingCurve,
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

/// What a preference dataset holds, as the trainer's own parser counts it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreferenceDatasetSummary {
    /// Preference pairs, one per JSONL line.
    pub pairs: usize,
    /// Prompt messages across every pair.
    pub prompt_messages: usize,
    /// Pairs carrying a tool schema.
    pub pairs_with_tools: usize,
}

impl From<brain::PreferenceDatasetSummary> for PreferenceDatasetSummary {
    fn from(s: brain::PreferenceDatasetSummary) -> Self {
        Self {
            pairs: s.pairs,
            prompt_messages: s.prompt_messages,
            pairs_with_tools: s.pairs_with_tools,
        }
    }
}

/// Parses `dataset` as `generic-preference-v1` with the trainer's own
/// parser: the wire schema, every candidate an assistant turn, and no pair
/// whose two candidates are the same turn. The error names the offending
/// record and field.
pub fn validate_preference_dataset(
    dataset: &Path,
) -> Result<PreferenceDatasetSummary, PolicyError> {
    brain::validate_preference_dataset(dataset)
        .map(PreferenceDatasetSummary::from)
        .map_err(|reason| PolicyError::Dataset {
            path: dataset.to_path_buf(),
            reason,
        })
}

/// Fine-tunes a LoRA on `request.train` and scores base and tuned on the
/// held-out records.
pub fn fine_tune(request: &FineTune<'_>) -> Result<Trained, PolicyError> {
    let failed = |reason: String| PolicyError::Train {
        dir: request.attempt_dir.to_path_buf(),
        reason,
    };
    let trained_on = validate_dataset(request.train)?;
    let held_out_summary = validate_dataset(request.held_out)?;
    let (weights, base_id) = base_weights(request.model_dir)?;
    for replayed in request.replay {
        validate_dataset(replayed)?;
    }
    for (weighted, _) in request.weighted_replay {
        validate_dataset(weighted)?;
    }
    if let Some(monitor) = request.monitor {
        validate_dataset(monitor)?;
    } else if request.eval_every > 0 {
        return Err(failed(
            "the run was asked to evaluate as it trains but given no monitoring records".into(),
        ));
    }
    let mut fine_tune = brain::ChatFineTune::from_pretrained(weights.as_str())
        .dataset(request.train)
        .held_out(request.held_out)
        .thinking(request.thinking)
        .out_dir(request.attempt_dir)
        .adapter_id(format!("{base_id}:splinter:candidate"))
        .steps(request.steps)
        .rank(request.rank)
        .alpha(request.alpha)
        .bf16_base(request.bf16_base)
        .grad_accum(request.grad_accum)
        .eval_every(request.eval_every)
        .patience(request.patience)
        // A monitored run exports its best evaluation whether or not its
        // patience ever runs out.
        .keep_best(request.eval_every > 0);
    if let Some(monitor) = request.monitor {
        fine_tune = fine_tune.monitor(monitor);
    }
    if let Some(share) = request.replay_share {
        fine_tune = fine_tune.replay_share(share);
    }
    if let Some(lr) = request.learning_rate {
        fine_tune = fine_tune.lr(lr);
    }
    if let Some(seed) = request.seed {
        fine_tune = fine_tune.seed(seed);
    }
    for replayed in request.replay {
        fine_tune = fine_tune.replay(replayed);
    }
    for (weighted, share) in request.weighted_replay {
        fine_tune = fine_tune.replay_at(weighted, *share);
    }
    if let Some(adapter) = request.continue_from {
        fine_tune = fine_tune.continue_from(adapter);
    }
    let brain_cancel = brain::CancelToken::armed();
    let outcome = fine_tune
        .run_with(
            &brain_cancel,
            poll(request.cancel, &brain_cancel, request.on_step),
        )
        .map_err(|e| failed(format!("training on {}: {e}", request.train.display())))?;
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
    let curve = curve_of(&outcome);
    Ok(Trained {
        adapter: outcome.adapter.ok_or_else(|| incomplete("adapter"))?,
        adapter_digest: outcome
            .adapter_digest
            .ok_or_else(|| incomplete("adapter digest"))?,
        base_digest: outcome
            .base_digest
            .ok_or_else(|| incomplete("base digest"))?,
        training_record: outcome
            .record
            .ok_or_else(|| incomplete("training record"))?,
        records: trained_on.records + held_out_summary.records,
        block: outcome.block,
        base: outcome
            .base_score
            .map(held_out_score)
            .ok_or_else(|| incomplete("base score"))?,
        tuned: outcome
            .tuned_score
            .map(held_out_score)
            .ok_or_else(|| incomplete("tuned score"))?,
        curve,
    })
}

/// brain's monitoring curve and selection as Splinter records them.
fn curve_of(outcome: &brain::ChatFineTuneOutcome) -> TrainingCurve {
    TrainingCurve {
        steps: outcome.steps,
        steps_completed: outcome.steps_completed,
        eval_every: outcome.eval_every,
        patience: outcome.patience,
        monitor_records: outcome.monitor_records,
        points: outcome
            .curve
            .iter()
            .map(|p| CurvePoint {
                step: p.step,
                train_loss: p.train_loss,
                monitor_loss: p.monitor_loss,
            })
            .collect(),
        selected_step: outcome.selected_step,
        selection: match outcome.selection {
            brain::Selection::LastStep => Selection::LastStep,
            brain::Selection::BestMonitorLoss => Selection::BestMonitorLoss,
        },
        stopped_early: outcome.stopped_early,
    }
}

/// One preference fine-tune: what to train, on which pairs, and where its
/// files go.
#[derive(Clone, Debug)]
pub struct PreferenceTune<'a> {
    /// Base checkpoint directory (the model the agent serves).
    pub model_dir: &'a Path,
    /// `generic-preference-v1` JSONL pairs to train on.
    pub train: &'a Path,
    /// `generic-preference-v1` JSONL pairs scored, never trained on.
    pub held_out: &'a Path,
    /// This attempt's own directory: the adapter and its training record
    /// land here.
    pub attempt_dir: &'a Path,
    /// Optimizer steps, one pair each.
    pub steps: u32,
    /// LoRA rank of a fresh adapter.
    pub rank: u32,
    /// LoRA alpha of a fresh adapter.
    pub alpha: f32,
    /// The DPO temperature scaling the reference-normalised margin
    /// ([`DEFAULT_DPO_BETA`] unless the caller chooses otherwise).
    pub beta: f32,
    /// Weight of an anchor on the chosen answer's own likelihood, added to
    /// the DPO loss; 0 is plain DPO.
    pub nll_weight: f32,
    /// Pairs summed into each optimizer step: the effective batch size.
    pub grad_accum: u32,
    /// The peak learning rate; brain's default when `None`.
    pub learning_rate: Option<f32>,
    /// An adapter to continue instead of starting a fresh one; base plus
    /// this adapter is then the frozen reference, and its own rank and
    /// alpha apply.
    pub continue_from: Option<&'a Path>,
    /// Stops training at the next optimizer step once cancelled; a
    /// cancelled fine-tune exports no adapter and is reported as an error.
    pub cancel: Option<&'a sven_sdk::CancelToken>,
}

/// What one preference fine-tune produced.
#[derive(Clone, Debug)]
pub struct TrainedPreference {
    /// The adapter file, carrying its ModelCard.
    pub adapter: PathBuf,
    /// `sha256:<hex>` of the adapter file.
    pub adapter_digest: String,
    /// `sha256:<hex>` of the base checkpoint the adapter was trained on -
    /// the digest its card records, which brain checks when it serves it.
    pub base_digest: String,
    /// brain's training record for the adapter, beside it.
    pub training_record: PathBuf,
    /// Pairs in the dataset, trained and held out together.
    pub records: usize,
    /// Training row length, sized to the longest candidate.
    pub block: u32,
    /// The DPO temperature it trained with.
    pub beta: f32,
    /// The digest of the adapter the reference carried (the one continued);
    /// `None` when the reference was the base alone.
    pub reference_adapter: Option<String>,
    /// The tuned adapter against the reference on the pairs trained on;
    /// `None` when brain did not measure it.
    pub train_score: Option<PreferenceScore>,
    /// The same on the held-out pairs; `None` when brain did not measure
    /// it.
    pub held_out_score: Option<PreferenceScore>,
}

/// Fine-tunes a LoRA by DPO on `request.train`'s pairs and scores it
/// against its reference on the held-out pairs.
pub fn train_preference(request: &PreferenceTune<'_>) -> Result<TrainedPreference, PolicyError> {
    let failed = |reason: String| PolicyError::Train {
        dir: request.attempt_dir.to_path_buf(),
        reason,
    };
    let trained_on = validate_preference_dataset(request.train)?;
    let held_out_summary = validate_preference_dataset(request.held_out)?;
    let (weights, base_id) = base_weights(request.model_dir)?;
    // As in [`fine_tune`]: a model asked from an open think block trains on
    // candidates that follow an empty closed one, rendered with it kept.
    let reasoning = opens_think_block(request.model_dir);
    let prepared = |path: &Path, name: &str| -> Result<std::path::PathBuf, PolicyError> {
        if !reasoning {
            return Ok(path.to_path_buf());
        }
        let io = |source| PolicyError::Io {
            path: path.to_path_buf(),
            source,
        };
        let text = std::fs::read_to_string(path).map_err(io)?;
        let rewritten = closed_think_pairs(&text).map_err(|e| PolicyError::Dataset {
            path: path.to_path_buf(),
            reason: e.to_string(),
        })?;
        let dest = request.attempt_dir.join(name);
        std::fs::create_dir_all(request.attempt_dir).map_err(io)?;
        std::fs::write(&dest, rewritten).map_err(io)?;
        Ok(dest)
    };
    let train = prepared(request.train, "pairs.think.jsonl")?;
    let held_out = prepared(request.held_out, "held_out_pairs.think.jsonl")?;
    let mut fine_tune = brain::PreferenceFineTune::from_pretrained(weights.as_str())
        .dataset(&train)
        .held_out(&held_out)
        .keep_reasoning(reasoning)
        .nll_weight(request.nll_weight)
        .grad_accum(request.grad_accum)
        .out_dir(request.attempt_dir)
        .adapter_id(format!("{base_id}:splinter:candidate"))
        .steps(request.steps)
        .rank(request.rank)
        .alpha(request.alpha)
        .beta(request.beta);
    if let Some(lr) = request.learning_rate {
        fine_tune = fine_tune.lr(lr);
    }
    if let Some(adapter) = request.continue_from {
        fine_tune = fine_tune.continue_from(adapter);
    }
    let brain_cancel = brain::CancelToken::armed();
    let outcome = fine_tune
        .run_with(&brain_cancel, poll(request.cancel, &brain_cancel, None))
        .map_err(|e| failed(format!("training on {}: {e}", request.train.display())))?;
    if brain_cancel.is_cancelled() {
        return Err(PolicyError::Cancelled {
            dir: request.attempt_dir.to_path_buf(),
        });
    }
    if outcome.status != brain::FineTuneStatus::Completed {
        return Err(failed("it did not complete".into()));
    }
    let incomplete = |what: &str| failed(format!("it reported no {what}"));
    Ok(TrainedPreference {
        adapter: outcome.adapter.ok_or_else(|| incomplete("adapter"))?,
        adapter_digest: outcome
            .adapter_digest
            .ok_or_else(|| incomplete("adapter digest"))?,
        base_digest: outcome
            .base_digest
            .ok_or_else(|| incomplete("base digest"))?,
        training_record: outcome
            .record
            .ok_or_else(|| incomplete("training record"))?,
        records: trained_on.pairs + held_out_summary.pairs,
        block: outcome.block,
        beta: outcome.beta,
        reference_adapter: outcome.trained_from,
        train_score: outcome.train_score.map(preference_score),
        held_out_score: outcome.held_out_score.map(preference_score),
    })
}

/// What brain trains from under `model_dir`, as the UTF-8 path it takes
/// (the directory of a Hugging Face checkpoint, else the checkpoint file),
/// and the base id an adapter's card names. brain resolves a directory only
/// in its model store's layout, so the source is resolved here; its
/// directory supplies the tokenizer and chat template. The card names what
/// produced the adapter and what it sits on, so an adapter is traceable to
/// its own training evidence.
fn base_weights(model_dir: &Path) -> Result<(String, String), PolicyError> {
    let weights = crate::local::load_source(&crate::local::resolve_base(model_dir)?);
    let weights = weights
        .to_str()
        .ok_or_else(|| PolicyError::NotUtf8 {
            path: weights.clone(),
        })?
        .to_string();
    let base_id = format!(
        "local/{}",
        model_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("base")
    );
    Ok((weights, base_id))
}

/// The progress callback that cancels `brain_cancel` once `cancel` fires:
/// brain's token stops training at a step boundary, sven's is the one the
/// caller cancels, so it is polled once per step.
fn poll<'a>(
    cancel: Option<&'a sven_sdk::CancelToken>,
    brain_cancel: &'a brain::CancelToken,
    on_step: Option<StepHook<'a>>,
) -> impl FnMut(&brain::FineTuneProgress) + 'a {
    move |progress| {
        if let Some(StepHook(report)) = on_step {
            report(&StepReport {
                step: progress.step,
                steps: progress.steps,
                loss: progress.loss,
            });
        }
        if cancel.is_some_and(sven_sdk::CancelToken::is_cancelled) {
            brain_cancel.cancel();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every optimizer step is reported to the caller, and a cancelled
    /// caller cancels brain's token at the next step.
    #[test]
    fn a_step_is_reported_and_a_cancel_is_forwarded() {
        let seen = std::cell::RefCell::new(Vec::new());
        let report = |r: &StepReport| seen.borrow_mut().push((r.step, r.steps, r.loss));
        let brain_cancel = brain::CancelToken::armed();
        let caller = sven_sdk::CancelToken::new();
        let mut step = poll(Some(&caller), &brain_cancel, Some(StepHook(&report)));
        let progress = |n| brain::FineTuneProgress {
            step: n,
            steps: 3,
            loss: 1.5,
            lr: 1e-4,
        };
        step(&progress(1));
        assert!(!brain_cancel.is_cancelled());
        caller.cancel();
        step(&progress(2));
        assert!(
            brain_cancel.is_cancelled(),
            "the caller's cancel reaches brain"
        );
        assert_eq!(*seen.borrow(), vec![(1, 3, 1.5), (2, 3, 1.5)]);
    }

    /// A Hugging Face checkpoint trains from its directory, where its
    /// `config.json` is; a brain-format file trains as itself.
    #[test]
    fn a_hugging_face_base_is_trained_from_its_directory() {
        let root = std::env::temp_dir().join(format!("policy-base-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let hf = root.join("DeepSeek-R1-Distill-Qwen-1.5B");
        let native = root.join("Qwen3-0.6B");
        std::fs::create_dir_all(&hf).unwrap();
        std::fs::create_dir_all(&native).unwrap();
        std::fs::write(hf.join("model.safetensors"), b"x").unwrap();
        std::fs::write(hf.join("config.json"), b"{}").unwrap();
        std::fs::write(native.join("model.brain.safetensors"), b"x").unwrap();
        std::fs::write(native.join("config.json"), b"{}").unwrap();

        let (weights, base_id) = base_weights(&hf).unwrap();
        assert_eq!(weights, hf.to_str().unwrap());
        assert_eq!(base_id, "local/DeepSeek-R1-Distill-Qwen-1.5B");
        let (weights, _) = base_weights(&native).unwrap();
        assert_eq!(
            weights,
            native.join("model.brain.safetensors").to_str().unwrap()
        );

        std::fs::remove_dir_all(&root).unwrap();
    }
}
