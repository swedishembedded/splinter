// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that turn a verifier's answers to
// a model's own mistakes into supervised training data, for its clients. If
// your team needs expertise in self-taught fine-tuning against an oracle you
// can procure our services by sending an email to info@swedishembedded.com.

//! The supervised recipe of a lineage round and the datasets it trains on.
//!
//! A round's records are the programs the policy proposed, each answered by
//! the interpreter: the request is what the policy was asked and the target is
//! the interpreter's output in one fenced block, the shape the neutral
//! protocol reads. Programs the policy mispredicted are the new information
//! and form the training set; the ones it predicted right are mixed in at a
//! stated share of the draws, and so are earlier accepted rounds' records.

use std::io::Write as _;
use std::path::Path;

use anyhow::{Context as _, Result};
use clap::Args;
use serde_json::json;
use splinter_sdk::model::precision::BasePrecision;

/// The settings of a round's training.
#[derive(Args, Clone, Debug)]
pub struct Recipe {
    /// Optimiser steps per round.
    #[arg(long)]
    pub steps: u32,
    /// Peak learning rate.
    #[arg(long)]
    pub lr: f32,
    /// Adapter rank.
    #[arg(long)]
    pub rank: u32,
    /// Records averaged into one optimiser step.
    #[arg(long)]
    pub records_per_step: u32,
    /// Share of the draws that replay earlier accepted rounds' records.
    #[arg(long)]
    pub replay_share: f32,
    /// Share of the draws that are programs the policy already predicted
    /// right; the mispredicted ones take the rest of the draws.
    #[arg(long)]
    pub correct_share: f32,
    /// Weight decay on the adapter.
    #[arg(long, default_value_t = 0.0)]
    pub weight_decay: f32,
    /// The longest training row, in tokens.
    #[arg(long, default_value_t = 1024)]
    pub max_block: u32,
    /// Precision of the frozen base during training: `bf16` or `fp32`.
    #[arg(long, default_value = "bf16")]
    pub base_precision: String,
    /// Dev programs scored before and after training as brain's own held-out
    /// measure (never trained on).
    #[arg(long, default_value_t = 64)]
    pub held_out_records: usize,
}

impl Recipe {
    /// The precision of the frozen base.
    ///
    /// # Errors
    /// The setting is not `bf16` or `fp32`.
    pub fn precision(&self) -> Result<BasePrecision> {
        match self.base_precision.as_str() {
            "bf16" => Ok(BasePrecision::Bf16),
            "fp32" => Ok(BasePrecision::Fp32),
            other => anyhow::bail!("{other:?} is not a base precision: bf16 or fp32"),
        }
    }

    /// Checks the shares leave the mispredicted programs a share of the draws.
    ///
    /// # Errors
    /// A share is outside `[0, 1)` or the two leave nothing for new records.
    pub fn validate(&self) -> Result<()> {
        for (name, share) in [
            ("replay-share", self.replay_share),
            ("correct-share", self.correct_share),
        ] {
            anyhow::ensure!(
                (0.0..1.0).contains(&share),
                "--{name} {share} is outside [0, 1)"
            );
        }
        anyhow::ensure!(
            self.replay_share + self.correct_share < 1.0,
            "the replay and correct shares leave no draws for the mispredicted programs"
        );
        anyhow::ensure!(
            self.steps > 0 && self.records_per_step > 0,
            "no training steps"
        );
        Ok(())
    }
}

/// One training example: what the policy was asked and what the interpreter
/// printed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Example {
    /// The task id, to trace a record back to its program.
    pub id: String,
    /// What the policy was asked.
    pub request: String,
    /// What the interpreter printed.
    pub expected: String,
}

/// The assistant's side of an example: the output in one fenced block.
#[must_use]
pub fn target(expected: &str) -> String {
    format!("```\n{}\n```", expected.trim())
}

/// Writes `examples` as a chat dataset (one conversation per line, only the
/// assistant's turn supervised), whole or not at all.
///
/// # Errors
/// The file cannot be written.
pub fn write_chat(path: &Path, examples: &[Example]) -> Result<()> {
    let partial = path.with_extension("partial");
    let mut file = std::fs::File::create(&partial)
        .with_context(|| format!("creating {}", partial.display()))?;
    for example in examples {
        let line = json!({
            "messages": [
                {"role": "user", "content": example.request, "train": false},
                {"role": "assistant", "content": target(&example.expected), "train": true},
            ],
            "metadata": {"task": example.id},
        });
        writeln!(file, "{line}")?;
    }
    file.sync_all()?;
    std::fs::rename(&partial, path).with_context(|| format!("placing {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recipe(replay: f32, correct: f32) -> Recipe {
        Recipe {
            steps: 10,
            lr: 1e-5,
            rank: 16,
            records_per_step: 8,
            replay_share: replay,
            correct_share: correct,
            weight_decay: 0.0,
            max_block: 1024,
            base_precision: "bf16".into(),
            held_out_records: 8,
        }
    }

    #[test]
    fn the_target_is_the_interpreters_output_in_one_fenced_block() {
        assert_eq!(target("12\n"), "```\n12\n```");
        assert!(crate::neutral::answers(&target(" 1 2 "), "1 2"));
    }

    #[test]
    fn shares_that_leave_nothing_for_mispredicted_programs_are_refused() {
        assert!(recipe(0.25, 0.25).validate().is_ok());
        assert!(recipe(0.6, 0.4).validate().is_err());
        assert!(recipe(1.0, 0.0).validate().is_err());
        assert!(recipe(-0.1, 0.0).validate().is_err());
    }

    #[test]
    fn a_dataset_is_written_whole_with_only_the_answer_supervised() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("train.jsonl");
        write_chat(
            &path,
            &[Example {
                id: "p".into(),
                request: "```lua\nprint(1)\n```\nWhat?".into(),
                expected: "1".into(),
            }],
        )
        .unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let line: serde_json::Value = serde_json::from_str(text.trim()).unwrap();
        assert_eq!(line["messages"][0]["train"], false);
        assert_eq!(line["messages"][1]["train"], true);
        assert_eq!(line["messages"][1]["content"], "```\n1\n```");
        assert!(!path.with_extension("partial").exists());
    }
}
