// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gated self-improvement loops where a model
// update is adopted only on held-out evidence. If your team needs expertise
// in evaluation-gated fine-tuning, you can procure our services by sending
// an email to info@swedishembedded.com.

//! What one LoRA fine-tune of the local model is told and what it produced,
//! in two regimes, each trained as a job of the brain service
//! ([`crate::served_train`]):
//!
//! * [`FineTune`]: supervised fine-tuning on a `generic-messages-v2` chat
//!   dataset, scored on held-out records before and after.
//! * [`PreferenceTune`]: direct preference optimisation on a
//!   `generic-preference-v1` pair dataset against a frozen reference (the
//!   model the run starts from), scored on held-out pairs by how much more
//!   the tuned adapter prefers each chosen answer than the reference does.
//!
//! In both, the caller hands over the dataset already split into the records
//! to train on and the ones held out, and a named adapter is continued
//! instead of a fresh one started. What to do with the scores is the
//! caller's decision.

use std::path::{Path, PathBuf};

use splinter_core::training::{HeldOutScore, PreferenceScore, TrainingCurve};

use crate::error::PolicyError;

/// The DPO temperature `beta` a preference fine-tune uses when its caller
/// names none.
pub use crate::served_train::DEFAULT_DPO_BETA;

pub use crate::precision::BasePrecision;

/// One fine-tune: what to train, on what, and where its files go.
#[derive(Clone, Debug)]
pub struct FineTune<'a> {
    /// Base checkpoint directory (the model the agent serves).
    pub model_dir: &'a Path,
    /// Chat-format JSONL records to train on.
    pub train: &'a Path,
    /// Chat-format JSONL records scored before and after, never trained on.
    pub held_out: &'a Path,
    /// The writer's own text of the held-out families, scored before and
    /// after beside them and reported apart in the training record; `None`
    /// when the data has none.
    pub held_out_text: Option<&'a Path>,
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
    /// Optimizer steps to train for; warmup is the first twentieth of them
    /// and the learning rate decays over all of them (a run that stops on
    /// its patience cools the rate down first).
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
    /// How the frozen base is held while the adapter trains. Scoring before
    /// and after runs at the same tier.
    pub base: BasePrecision,
    /// The peak learning rate; brain's default when `None`.
    pub learning_rate: Option<f32>,
    /// The AdamW weight decay on the adapter's matrices.
    pub weight_decay: f32,
    /// The seed of a fresh adapter's initialisation and the batch order.
    /// Two runs that differ only here show how much of a measured
    /// difference is the draw; the caller draws it when the user gave none.
    pub seed: u64,
    /// The longest training row in tokens; `None` sizes the row to the
    /// longest record. Set, a record longer than it is named (see
    /// [`crate::capabilities::overlong_records`]) before any device is claimed, and the run does
    /// not start.
    pub max_block: Option<u32>,
    /// Whether the model is trained to reason before it answers. Off, it is
    /// trained for no-think mode - what Splinter asks of it by default - so a
    /// reasoning model trains on the state a no-think prompt leaves it in
    /// (brain's `thinking` setting); a base that never reasons is
    /// unaffected.
    pub thinking: bool,
    /// Train a record that supervises several turns as one sequence, not
    /// one example per turn. Set for datasets built packed
    /// (`dataset build --pack`); any other dataset keeps a turn per example.
    pub pack_answers: bool,
    /// Told after every optimizer step, so a long run can be watched.
    pub on_step: Option<StepHook<'a>>,
    /// Where to keep the adapter of every evaluation, as
    /// `step-<N>.safetensors`; `None` keeps only the one exported. Needs
    /// `eval_every > 0`.
    pub keep_evaluations: Option<&'a Path>,
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
    /// The adapter of every evaluation, by the steps trained, when the run
    /// was asked to keep them.
    pub evaluations: Vec<(u32, PathBuf)>,
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
    /// The seed of a fresh adapter's initialisation and the batch order; the
    /// caller draws it when the user gave none.
    pub seed: u64,
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

/// The card id a trained candidate's adapter is recorded under: brain's
/// model-reference grammar for an adapter, `<base>:<owner>:<name>:<tag>`,
/// with the attempt directory's name (the candidate's own id) as the tag.
/// brain serves an adapter under its card id, so two candidates sharing one
/// id could never be offered together and a served id would not say which
/// candidate answered.
pub(crate) fn candidate_adapter_id(
    base_id: &str,
    attempt_dir: &Path,
) -> Result<String, PolicyError> {
    let tag = attempt_dir
        .file_name()
        .and_then(|n| n.to_str())
        .filter(|n| !n.is_empty() && !n.contains([':', '/']))
        .ok_or_else(|| PolicyError::Train {
            dir: attempt_dir.to_path_buf(),
            reason: "the attempt directory has no name to tag the adapter's card id with".into(),
        })?;
    Ok(format!("{base_id}:splinter:candidate:{tag}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `brain serve --adapter` refuses an adapter whose card id is not
    /// `<base>:<owner>:<name>:<tag>`, each segment non-empty, so a candidate
    /// recorded under any other shape could never pass the gate's serve check;
    /// and two candidates must not share an id, or a server could offer only
    /// one of them.
    #[test]
    fn a_candidates_card_id_is_an_adapter_reference_unique_to_its_attempt() {
        let a =
            candidate_adapter_id("local/Qwen3-0.6B", Path::new("/w/train/candidate-1")).unwrap();
        let b =
            candidate_adapter_id("local/Qwen3-0.6B", Path::new("/w/train/candidate-2")).unwrap();
        let parts: Vec<&str> = a.split(':').collect();
        assert_eq!(parts.len(), 4, "{a}");
        assert_eq!(parts[0], "local/Qwen3-0.6B");
        assert!(parts.iter().all(|p| !p.is_empty()), "{a}");
        assert_eq!(parts[3], "candidate-1");
        assert_ne!(a, b);
        assert!(candidate_adapter_id("local/x", Path::new("/")).is_err());
    }
}
