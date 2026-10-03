// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Why a command could not do what it was asked: the lower crates' errors,
//! named, and the controller's own refusals.

use std::path::PathBuf;

use splinter_agent::repair::RepairError;
use splinter_agent::replay::ReplayError;
use splinter_agent::solve::SolveError;
use splinter_core::experience::ExperienceError;
use splinter_knowledge::capture::CaptureError;
use splinter_knowledge::tasks::GenerateError;
use splinter_lab::verifiers::VerifyError;
use splinter_record::experiences::StoreError;
use splinter_sandbox::SandboxError;
use splinter_views::ViewError;

use crate::model_ref::RefError;

/// Why a command failed.
#[derive(Debug, thiserror::Error)]
pub enum CampaignError {
    /// A model reference is malformed, or needs the network opt-in.
    #[error(transparent)]
    Ref(#[from] RefError),
    /// An argument is refused before anything runs.
    #[error("{0}")]
    Refused(String),
    /// No stored object of this kind has this id (or id prefix).
    #[error("no {what} {id} in the store")]
    NotFound {
        /// What was looked for.
        what: &'static str,
        /// The id or prefix given.
        id: String,
    },
    /// An id prefix matches more than one stored object.
    #[error("{id} matches {matches} {what}s; give more of the id")]
    AmbiguousId {
        /// What was looked for.
        what: &'static str,
        /// The prefix given.
        id: String,
        /// How many stored objects it matches.
        matches: usize,
    },
    /// An id prefix names artifacts in more than one store, or more than
    /// one artifact in one.
    #[error("{id} names {} artifacts: {}; give more of the id", candidates.len(), candidates.join(", "))]
    AmbiguousArtifact {
        /// The prefix given.
        id: String,
        /// Each artifact it names, as `<kind> <id>`.
        candidates: Vec<String>,
    },
    /// A model could not be loaded or reached.
    #[error("model {model} could not be loaded: {detail}")]
    Model {
        /// The model's reference.
        model: String,
        /// Why.
        detail: String,
    },
    /// A training run failed.
    #[error("training failed: {0}")]
    Train(String),
    /// A typed model call failed.
    #[error("the {method} call failed: {source}")]
    Call {
        /// The method.
        method: &'static str,
        /// What failed.
        source: sven_sdk::CallError,
    },
    /// The run was cancelled.
    #[error("cancelled")]
    Cancelled,
    /// A store failed.
    #[error(transparent)]
    Store(#[from] StoreError),
    /// A dataset could not be projected, written or read.
    #[error(transparent)]
    View(#[from] ViewError),
    /// A source could not be captured.
    #[error(transparent)]
    Capture(#[from] CaptureError),
    /// Tasks could not be generated.
    #[error(transparent)]
    Generate(#[from] GenerateError),
    /// A task could not be solved.
    #[error(transparent)]
    Solve(#[from] SolveError),
    /// A verifier could not run.
    #[error(transparent)]
    Verify(#[from] VerifyError),
    /// A critique or retry could not run.
    #[error(transparent)]
    Repair(#[from] RepairError),
    /// A sandbox could not resolve a runtime or run code.
    #[error(transparent)]
    Sandbox(#[from] SandboxError),
    /// An experience could not be replayed.
    #[error(transparent)]
    Replay(#[from] ReplayError),
    /// A task or experience is not valid.
    #[error(transparent)]
    Experience(#[from] ExperienceError),
    /// A file operation failed.
    #[error("{path}: {source}")]
    Io {
        /// The file or directory.
        path: PathBuf,
        /// The underlying error.
        source: std::io::Error,
    },
    /// The async runtime could not be started.
    #[error("cannot start the async runtime: {0}")]
    Runtime(std::io::Error),
    /// A record could not be serialized or parsed.
    #[error("{what}: {source}")]
    Json {
        /// What was being read or written.
        what: String,
        /// The serializer's error.
        source: serde_json::Error,
    },
}

impl CampaignError {
    /// Whether the command was refused before it did anything - a usage
    /// error rather than a failure of the work.
    #[must_use]
    pub fn is_refusal(&self) -> bool {
        matches!(
            self,
            Self::Ref(_)
                | Self::Refused(_)
                | Self::NotFound { .. }
                | Self::AmbiguousId { .. }
                | Self::AmbiguousArtifact { .. }
        )
    }
}

/// Wraps an I/O error on `path`.
pub(crate) fn io(path: &std::path::Path) -> impl FnOnce(std::io::Error) -> CampaignError + '_ {
    move |source| CampaignError::Io {
        path: path.to_path_buf(),
        source,
    }
}
