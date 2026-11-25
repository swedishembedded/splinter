// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems tested end to end against a
// real model service on a model small enough for a laptop, for its clients.
// If your team needs expertise in fast, faithful tests of training pipelines,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! A tiny chat checkpoint for specs that must see a real `brain serve` train
//! and sample: written by brain's qwen3 sample (`sample-models-qwen3 toy`),
//! never by this crate.

use std::path::Path;
use std::process::Command;

/// The binary brain builds its qwen3 sample as; a brain build places it beside
/// `brain` (`target/release`).
pub const QWEN3_SAMPLE: &str = "sample-models-qwen3";

/// Writes into `dir` a randomly initialised checkpoint small enough to train
/// and sample on the CPU in seconds, with a byte-level tokenizer and a bare
/// chat template (a user turn reads `text=` and an assistant turn
/// `text<|im_end|>`, system turns are dropped), by running the toy action of
/// brain's qwen3 sample, the binary `qwen3` ([`QWEN3_SAMPLE`]). The same seed
/// writes the same weights. `dir` is `<models-dir>/<vendor>/<name>`: the
/// checkpoint is written in the model store's layout, with a card naming it
/// `<vendor>/<name>`, so a `brain serve` over that store serves it by name.
///
/// # Errors
/// The binary cannot be run, or it refused or failed to write `dir`.
pub fn toy_chat_checkpoint(qwen3: &Path, dir: &Path, seed: u64) -> Result<(), crate::PolicyError> {
    let failed = |reason: String| crate::PolicyError::Load {
        path: dir.to_path_buf(),
        reason,
    };
    let output = Command::new(qwen3)
        .args(["toy", "--out"])
        .arg(dir)
        .args(["--seed", &seed.to_string(), "--store_layout", "true"])
        .output()
        .map_err(|e| failed(format!("running {}: {e}", qwen3.display())))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(failed(format!(
            "{} toy: {}",
            qwen3.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        )))
    }
}
