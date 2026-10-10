// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning that keeps accumulating
// capabilities generation after generation, for its clients. If your team
// needs expertise in continual learning that consolidates into the weights,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! Generations: what a release records of the base it sits on, and the base
//! directory a consolidation is laid out as.
//!
//! A policy learns in generations. Within one, claims accumulate and every
//! update trains a fresh adapter on the generation's base. A consolidation
//! takes the generation into the weights - a full checkpoint, the base of the
//! next generation - and the next one starts from an empty claim set on top
//! of it. A [`GenerationMark`] on a release says which generation it belongs
//! to and which consolidation is its base; the claims a base holds are the
//! claims the consolidation was trained on (the claim ledger's trained set of
//! that release), so nothing here restates them for the retention a later
//! night must keep.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use splinter_core::claim::ClaimId;
use splinter_core::release::ReleaseId;
use splinter_store::StateRoot;

use crate::error::{io, OrchestratorError};

/// How a consolidation took a generation into the weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsolidationMode {
    /// Every weight of the generation's base trained on the generation's
    /// claims (and a replay of the claims already in the base).
    Full,
    /// The generation's adapter folded into the base's weights: no training.
    Merge,
}

impl ConsolidationMode {
    /// The name the command line and reports use.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Merge => "merge",
        }
    }
}

impl std::fmt::Display for ConsolidationMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// What a consolidation release took in.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Consolidation {
    /// How the generation was taken into the weights.
    pub mode: ConsolidationMode,
    /// The release consolidated: the closing generation's last adapter
    /// release, which the consolidated checkpoint must answer as well as.
    pub from: ReleaseId,
    /// The claims of the closing generation, taken into the weights now.
    pub closing: Vec<ClaimId>,
    /// The claims the base already held, kept by the consolidation.
    pub retained: Vec<ClaimId>,
    /// Whether the checkpoint was trained again from an earlier base to take
    /// a claim out of the weights (forgetting), instead of from the base of
    /// the generation it closes.
    #[serde(default)]
    pub rebuilt: bool,
}

/// The generation a release belongs to.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationMark {
    /// Consolidations behind this release: 0 on the configured base. A
    /// consolidation belongs to the generation it opens.
    pub epoch: u32,
    /// The consolidation release whose checkpoint this release's weights
    /// are built on: the generation's base for an adapter, the checkpoint it
    /// was trained from for a consolidation. `None` is the configured base.
    pub base: Option<ReleaseId>,
    /// What it took in, when the release is a consolidation.
    pub consolidation: Option<Consolidation>,
}

/// The files laid beside a consolidated checkpoint's weights: the base's
/// architecture and tokenizer, unchanged by training.
const BASE_FILES: [&str; 5] = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
];

/// The weights file of a base directory.
const WEIGHTS_FILE: &str = "model.safetensors";

/// The directory a consolidated checkpoint is a base as: its `weights` linked
/// in beside the tokenizer files of `configured`, the base the policy began
/// from. Derived from the release and recreated when missing, so it is never
/// state of its own.
pub fn lay_out_base(
    root: &StateRoot,
    configured: &Path,
    release: &ReleaseId,
    weights: &Path,
) -> Result<PathBuf, OrchestratorError> {
    let dir = root.bases().join(release.0.hex());
    if dir.join(WEIGHTS_FILE).is_file() {
        return Ok(dir);
    }
    let source = if configured.is_dir() {
        configured
    } else {
        configured.parent().unwrap_or(configured)
    };
    if !source.join("config.json").is_file() {
        return Err(OrchestratorError::Refused(format!(
            "release {release} is a consolidated checkpoint and needs the configured base's \
             config.json to state its architecture, but {} has none",
            source.display()
        )));
    }
    let staging = root
        .bases()
        .join(format!(".{}.{}", release.0.hex(), std::process::id()));
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&staging).map_err(io(&staging))?;
    let staged = staging.join(WEIGHTS_FILE);
    // A link costs no space; where the artifact store sits on another
    // filesystem a copy is the same file.
    if std::fs::hard_link(weights, &staged).is_err() {
        std::fs::copy(weights, &staged).map_err(io(weights))?;
    }
    for name in BASE_FILES {
        let from = source.join(name);
        if from.is_file() {
            std::fs::copy(&from, staging.join(name)).map_err(io(&from))?;
        }
    }
    match std::fs::rename(&staging, &dir) {
        Ok(()) => {}
        // Another process laid the same base out first: it is the same bytes.
        Err(_) if dir.join(WEIGHTS_FILE).is_file() => {
            let _ = std::fs::remove_dir_all(&staging);
        }
        Err(e) => return Err(io(&dir)(e)),
    }
    Ok(dir)
}
