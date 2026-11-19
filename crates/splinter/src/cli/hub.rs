// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements distribution of curated training data,
// adapters and evaluation verdicts as immutable, verifiable versions, for its
// clients. If your team needs expertise in sharing learning artifacts between
// machines without losing their provenance, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The grammar of `publish` and `experiences ingest-hf`: bundles in a private
//! Hugging Face dataset repository.

use std::path::PathBuf;

use clap::Args;
use splinter_sdk::hub::DEFAULT_ENDPOINT;

/// Where the Hub is and how to authenticate to it.
#[derive(Debug, Args)]
pub struct HubArgs {
    /// The Hub to talk to (a local emulator's `http://127.0.0.1:PORT` in a
    /// test).
    #[arg(long = "hub-endpoint", value_name = "URL", default_value = DEFAULT_ENDPOINT)]
    pub endpoint: String,
    /// A file holding the access token (the `hf` CLI keeps one at
    /// `~/.cache/huggingface/token`); without it, calls are anonymous.
    #[arg(long = "token-file", value_name = "FILE")]
    pub token_file: Option<PathBuf>,
}

/// `publish`: a bundle of experience sets, datasets, candidates and signed
/// verdicts as one tagged revision of a private dataset repository.
#[derive(Debug, Args)]
pub struct PublishArgs {
    /// The dataset repository, `owner/name`; created private when missing,
    /// and refused unless the Hub states it private.
    #[arg(long, value_name = "OWNER/NAME")]
    pub repo: String,
    /// An experience set, by id or unique prefix (repeatable).
    #[arg(long = "set", value_name = "SET")]
    pub sets: Vec<String>,
    /// A dataset, by id or unique prefix (repeatable).
    #[arg(long = "dataset", value_name = "DATASET")]
    pub datasets: Vec<String>,
    /// A trained candidate, by id or unique prefix (repeatable); its adapter,
    /// training record and the datasets it was trained on come along.
    #[arg(long = "candidate", value_name = "CANDIDATE")]
    pub candidates: Vec<String>,
    /// A signed aggregate verdict of the examiner (repeatable).
    #[arg(long = "verdict", value_name = "FILE", requires = "examiner_key")]
    pub verdicts: Vec<PathBuf>,
    /// The examiner's public key the verdicts verify under.
    #[arg(long = "examiner-key", value_name = "FILE")]
    pub examiner_key: Option<PathBuf>,
    /// The Hub.
    #[command(flatten)]
    pub hub: HubArgs,
}

/// `experiences ingest-hf`: a published bundle read at a pinned revision.
#[derive(Debug, Args)]
pub struct IngestHfArgs {
    /// The bundle, `OWNER/NAME@REVISION`: a commit, or a branch or tag, which
    /// is resolved to the commit it names and recorded as that commit.
    #[arg(value_name = "REPO@REVISION")]
    pub source: String,
    /// The Hub.
    #[command(flatten)]
    pub hub: HubArgs,
}
