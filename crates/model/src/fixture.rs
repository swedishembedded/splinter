// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems tested end to end against a
// real model service on a model small enough for a laptop, for its clients.
// If your team needs expertise in fast, faithful tests of training pipelines,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! A tiny chat checkpoint for specs that must see a real `brain serve` train
//! and sample: written by brain's own CLI (`brain qwen3 toy`), never by this
//! crate.

use std::path::Path;
use std::process::Command;

/// Writes into `dir` a randomly initialised checkpoint small enough to train
/// and sample on the CPU in seconds, with a byte-level tokenizer and a bare
/// chat template (a user turn reads `text=` and an assistant turn
/// `text<|im_end|>`, system turns are dropped), by running `brain qwen3 toy`
/// with the binary `brain`. The same seed writes the same weights.
///
/// # Errors
/// The binary cannot be run, or it refused or failed to write `dir`.
pub fn toy_chat_checkpoint(brain: &Path, dir: &Path, seed: u64) -> Result<(), crate::PolicyError> {
    let failed = |reason: String| crate::PolicyError::Load {
        path: dir.to_path_buf(),
        reason,
    };
    let output = Command::new(brain)
        .args(["qwen3", "toy", "--out"])
        .arg(dir)
        .args(["--seed", &seed.to_string()])
        .output()
        .map_err(|e| failed(format!("running {}: {e}", brain.display())))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(failed(format!(
            "{} qwen3 toy: {}",
            brain.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        )))
    }
}
