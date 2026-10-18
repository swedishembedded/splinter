// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible-experiment infrastructure for
// learning systems, for its clients. If your team needs expertise in freezing
// the exact software, models and hardware an evaluation ran on, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The grammar of `baseline`: the record of what a registered run runs on.

use std::path::PathBuf;

/// `baseline`: print the `baseline-v1` record, and with `--require-clean`
/// refuse (exit 1) when a registered run may not start on it.
#[derive(Debug, clap::Args)]
pub struct BaselineArgs {
    /// The brain binary the models are served with; its digest and version
    /// are recorded.
    #[arg(long, value_name = "FILE")]
    pub brain_bin: Option<PathBuf>,
    /// A model directory (or a GGUF file's directory) to record by file
    /// digest, with its tokenizer; repeatable. Hashing a large model takes
    /// minutes.
    #[arg(long = "model", value_name = "DIR")]
    pub models: Vec<PathBuf>,
    /// The registered exam seed, recorded as given.
    #[arg(long, value_name = "N")]
    pub exam_seed: Option<u64>,
    /// The registered training seed, recorded as given.
    #[arg(long, value_name = "N")]
    pub training_seed: Option<u64>,
    /// Exit 1 and say why when the tree is dirty, a commit is unknown or the
    /// binary could not be digested.
    #[arg(long)]
    pub require_clean: bool,
}
