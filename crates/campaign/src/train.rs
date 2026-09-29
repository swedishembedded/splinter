// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gated self-improvement loops where a model
// update is adopted only on held-out evidence. If your team needs expertise
// in evaluation-gated fine-tuning, you can procure our services by sending
// an email to info@swedishembedded.com.

//! A training attempt with the gate outside the trainer's optimism: the
//! policy fine-tunes and scores, the lab's gate decides, and only a
//! promotion moves the adapter pointer serving reads. Every attempt -
//! adopted or rejected - leaves both scores in its decision record, so
//! "the model got worse" is evidence, not folklore.

use std::path::PathBuf;

use splinter_lab::promotion::{decide, Decision, Scores};
use splinter_policy::train::{fine_tune, FineTune};
use splinter_store::{write_atomic, StateRoot};

/// Training steps when a command names none: small, because the loop
/// trains on its own verified experience a few records at a time.
pub const DEFAULT_STEPS: u32 = 40;
/// LoRA rank of a new adapter when a command names none.
pub const DEFAULT_LORA_RANK: u32 = 8;
/// LoRA alpha of a new adapter when a command names none.
pub const DEFAULT_LORA_ALPHA: f32 = 16.0;

/// Options for one training attempt.
#[derive(Clone, Debug)]
pub struct TrainOptions {
    /// Base checkpoint to fine-tune against (the model the agent serves).
    pub model_dir: PathBuf,
    /// Dataset file to train on; defaults to the accumulated pool.
    pub dataset: Option<PathBuf>,
    /// Training steps (small by default: the loop trains on its own
    /// verified experience, a few samples at a time).
    pub steps: u32,
    /// LoRA rank / alpha for the adapter.
    pub rank: u32,
    pub alpha: f32,
}

/// One training attempt, end to end. Returns the decision and the attempt's
/// directory, which holds the decision record and the adapter.
pub fn run(root: &StateRoot, options: &TrainOptions) -> anyhow::Result<(Decision, PathBuf)> {
    let dataset = options
        .dataset
        .clone()
        .unwrap_or_else(|| root.experience_pool());
    let out = attempt_dir(root)?;
    let trained = fine_tune(&FineTune {
        model_dir: &options.model_dir,
        dataset: &dataset,
        attempt_dir: &out,
        prepared_dir: &prepared_dir(root)?,
        steps: options.steps,
        rank: options.rank,
        alpha: options.alpha,
    })?;

    let scores = Scores {
        base_loss: trained.base.loss,
        tuned_loss: trained.tuned.loss,
    };
    // The standing champion's scores, from the pointer: beats-base is not
    // enough when a better adapter already serves. Only a champion scored
    // on the same pool and split bounds a candidate; `decide` checks that by
    // base loss.
    let champion = read_champion_scores(&root.adapter_pointer());
    let decision = decide(&scores, champion.as_ref());

    let record = serde_json::json!({
        "dataset": dataset,
        "records": trained.records,
        "block_size": trained.block,
        "steps": options.steps,
        "rank": options.rank,
        "base": trained.base,
        "tuned": trained.tuned,
        "decision": match decision { Decision::Promoted => "promoted", Decision::Rejected => "rejected" },
        "adapter": trained.adapter,
    });
    let decision_record = out.join("decision.json");
    write_atomic(&decision_record, &serde_json::to_string_pretty(&record)?)?;
    apply_decision(
        &decision,
        &root.adapter_pointer(),
        &trained.adapter,
        &options.model_dir,
        &scores,
        &decision_record,
    )?;
    Ok((decision, out))
}

/// Applies the gate's verdict to durable state. Promotion repoints the
/// adapter pointer at the new adapter, so serving follows the promotion.
/// Rejection changes nothing: the pointer names the previous known-good
/// adapter, and a failed training attempt must not take it out of service -
/// "no pointer" means this attempt wrote none, never that the standing
/// promotion was torn down.
fn apply_decision(
    decision: &Decision,
    pointer: &std::path::Path,
    adapter: &std::path::Path,
    model_dir: &std::path::Path,
    scores: &Scores,
    decision_record: &std::path::Path,
) -> anyhow::Result<()> {
    if *decision != Decision::Promoted {
        return Ok(());
    }
    let record = serde_json::json!({
        "adapter": adapter,
        "model_dir": model_dir,
        "scores": { "base_loss": scores.base_loss, "tuned_loss": scores.tuned_loss },
        "decision_record": decision_record,
    });
    write_atomic(pointer, &serde_json::to_string_pretty(&record)?)
        .map_err(|e| anyhow::anyhow!("{}: {e}", pointer.display()))
}

/// One training attempt's own output directory, created exactly once.
fn attempt_dir(root: &StateRoot) -> anyhow::Result<PathBuf> {
    let dir = root
        .train()
        .join(splinter_store::new_id_with_prefix("train"));
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

fn prepared_dir(root: &StateRoot) -> anyhow::Result<PathBuf> {
    let dir = root.train().join("prepared");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// The champion's recorded gate scores, when a pointer is in place. A
/// missing or malformed pointer is no champion - the base-only rule
/// applies - never an error: a rejected first attempt leaves no pointer,
/// and training must still work from there.
fn read_champion_scores(pointer: &std::path::Path) -> Option<Scores> {
    let text = std::fs::read_to_string(pointer).ok()?;
    let value: serde_json::Value = serde_json::from_str(&text).ok()?;
    Some(Scores {
        base_loss: value.get("scores")?.get("base_loss")?.as_f64()? as f32,
        tuned_loss: value.get("scores")?.get("tuned_loss")?.as_f64()? as f32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rejected_attempt_leaves_the_promoted_adapter_in_service() {
        let root = StateRoot::new(
            std::env::temp_dir().join(format!("splinter-train-gate-{}", std::process::id())),
        );
        let pointer = root.adapter_pointer();
        std::fs::create_dir_all(pointer.parent().unwrap()).unwrap();
        let standing = serde_json::json!({ "adapter": "/previous/good-adapter.safetensors" });
        std::fs::write(&pointer, serde_json::to_string_pretty(&standing).unwrap()).unwrap();

        // A rejection writes nothing and tears nothing down: the standing
        // promotion stays servable.
        apply_decision(
            &Decision::Rejected,
            &pointer,
            std::path::Path::new("/this/attempt/adapter.safetensors"),
            std::path::Path::new("/base"),
            &Scores {
                base_loss: 1.0,
                tuned_loss: 1.2,
            },
            std::path::Path::new("/this/attempt/decision.json"),
        )
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(&pointer).unwrap(),
            serde_json::to_string_pretty(&standing).unwrap(),
            "a failed training must not remove the known-good promotion"
        );

        // A promotion repoints the pointer at the new adapter.
        let adapter = std::path::Path::new("/this/attempt/adapter.safetensors");
        apply_decision(
            &Decision::Promoted,
            &pointer,
            adapter,
            std::path::Path::new("/base"),
            &Scores {
                base_loss: 1.2,
                tuned_loss: 1.0,
            },
            std::path::Path::new("/this/attempt/decision.json"),
        )
        .unwrap();
        let promoted: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&pointer).unwrap()).unwrap();
        assert_eq!(promoted["adapter"], adapter.display().to_string());
        assert_eq!(promoted["scores"]["tuned_loss"], 1.0);

        let _ = std::fs::remove_dir_all(root.path());
    }
}
