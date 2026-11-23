// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning pipelines that never redo work
// whose inputs did not change, for its clients. If your team needs expertise
// in content-addressed build and experiment systems, you can procure our
// services by sending an email to info@swedishembedded.com.

//! What makes a stage's product reusable: its identity.
//!
//! A stage's identity is the digest of everything its output depends on: the
//! schema of the stage, its resolved parameters, the digests of the products
//! it consumes, the implementation that computes it and its explicit seed.
//! Two stages with one identity produce interchangeable products, so a
//! completed one is reused; any change makes a new identity and leaves the
//! old product for whichever run still wants it.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::digest::{digest_of, Digest};

/// The schema of a [`StageIdentity`].
pub const STAGE_SCHEMA: &str = "splinter.stage/1";

/// The kinds of stage a recipe is planned into.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum StageKind {
    /// Load and pin the task collection.
    Tasks,
    /// One rollout pass: agents attempt tasks.
    Rollout,
    /// Project trials into a training dataset.
    Dataset,
    /// Train one candidate.
    Training,
    /// Evaluate candidates against a baseline.
    Eval,
    /// Promote a candidate.
    Release,
}

impl StageKind {
    /// The directory a stage of this kind is viewed under in a run.
    #[must_use]
    pub fn directory(self) -> &'static str {
        match self {
            Self::Tasks => "tasks",
            Self::Rollout => "rollouts",
            Self::Dataset => "datasets",
            Self::Training => "training",
            Self::Eval => "eval",
            Self::Release => "release",
        }
    }
}

/// Everything a stage's output depends on.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct StageIdentity {
    /// Always [`STAGE_SCHEMA`].
    pub schema: String,
    /// What the stage does.
    pub kind: StageKind,
    /// The stage's name in its recipe (`local`, `sft`, `lora-r16`).
    pub name: String,
    /// The resolved parameters, after defaults and file contents (as their
    /// digests) are substituted.
    pub parameters: Value,
    /// The digests of the products the stage consumes, by role.
    pub inputs: Vec<(String, Digest)>,
    /// The implementation that computes it: Splinter's version plus any
    /// worker or trainer digest the stage runs.
    pub implementation: String,
    /// The explicit seed, when the stage draws randomness.
    pub seed: Option<u64>,
}

impl StageIdentity {
    /// A stage identity; `inputs` is sorted so their order in a recipe does
    /// not change it.
    #[must_use]
    pub fn new(
        kind: StageKind,
        name: impl Into<String>,
        parameters: Value,
        mut inputs: Vec<(String, Digest)>,
        implementation: impl Into<String>,
        seed: Option<u64>,
    ) -> Self {
        inputs.sort();
        Self {
            schema: STAGE_SCHEMA.into(),
            kind,
            name: name.into(),
            parameters,
            inputs,
            implementation: implementation.into(),
            seed,
        }
    }

    /// The stage's identity digest.
    #[must_use]
    pub fn digest(&self) -> Digest {
        digest_of(self)
    }
}
