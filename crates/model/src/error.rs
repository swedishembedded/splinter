// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Why the policy could not load, reach or train a model.

use std::path::PathBuf;

/// Why a policy operation failed. Each names what failed and on which
/// input; brain's and sven's own errors are carried as their text.
#[derive(Debug, thiserror::Error)]
pub enum PolicyError {
    /// No checkpoint file where one was named.
    #[error("no checkpoint file found under {path}")]
    NoCheckpoint {
        /// The path named.
        path: PathBuf,
    },
    /// brain has no trainer for the objective, and the caller did not ask
    /// for an export-only file.
    #[error(
        "brain cannot train objective {objective:?}; write it with export_only to get \
         Splinter's export format instead"
    )]
    ObjectiveNotTrainable {
        /// The objective.
        objective: splinter_data::Objective,
    },
    /// The embedding model could not embed.
    #[error("embedding: {reason}")]
    Embedding {
        /// brain's error.
        reason: String,
    },
    /// A path brain must be given as UTF-8 is not.
    #[error("{path} is not valid UTF-8")]
    NotUtf8 {
        /// The path.
        path: PathBuf,
    },
    /// brain could not load the weights.
    #[error("loading {path}: {reason}")]
    Load {
        /// The checkpoint.
        path: PathBuf,
        /// brain's error.
        reason: String,
    },
    /// A resident base refused to switch to a model's adapter.
    #[error("{} on {path}: {reason}", adapter.as_ref().map_or_else(|| "detaching the adapter".to_string(), |a| format!("attaching {}", a.display())))]
    Adapter {
        /// The base checkpoint.
        path: PathBuf,
        /// The adapter asked for; `None` for the plain base.
        adapter: Option<PathBuf>,
        /// brain's error.
        reason: String,
    },
    /// A generation on a resident base failed.
    #[error("generating on {path}{}: {reason}", adapter.as_ref().map_or_else(String::new, |a| format!(" with {}", a.display())))]
    Generate {
        /// The base checkpoint.
        path: PathBuf,
        /// The adapter attached; `None` for the plain base.
        adapter: Option<PathBuf>,
        /// brain's error.
        reason: String,
    },
    /// A remote model could not be configured or reached.
    #[error("remote model {spec}: {reason}")]
    Remote {
        /// The model, `provider/name`.
        spec: String,
        /// Why.
        reason: String,
    },
    /// A dataset is not valid trainer input.
    #[error("dataset {path} is not valid trainer input: {reason}")]
    Dataset {
        /// The dataset file.
        path: PathBuf,
        /// brain's parser's error.
        reason: String,
    },
    /// A fine-tune failed, or completed without what it must report.
    #[error("fine-tune in {dir}: {reason}")]
    Train {
        /// The fine-tune's directory.
        dir: PathBuf,
        /// What went wrong.
        reason: String,
    },
    /// A fine-tune was cancelled; it exported no adapter.
    #[error("fine-tune in {dir} was cancelled; it exported no adapter")]
    Cancelled {
        /// The fine-tune's directory.
        dir: PathBuf,
    },
    /// A file operation failed.
    #[error("{path}: {source}")]
    Io {
        /// The file.
        path: PathBuf,
        /// The underlying error.
        source: std::io::Error,
    },
}
