// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements versioned data contracts between Splinter and
// the trainers it drives, for its clients. If your team needs expertise in
// cross-system schema compatibility or checkpoint lineage, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The contract a trainer publishes when it commits a checkpoint.
//!
//! A training job writes its durable output as a small JSON manifest beside the
//! files it names. The manifest is versioned by its own shape: both sides
//! declare every field, refuse unknown ones, and a change to either side is a
//! change to the pair. The trainer owns the bytes; Splinter reads them, checks
//! each file against the length and checksum the manifest states, and keeps the
//! verified bytes as its own artifacts.
//!
//! The plan travels as the JSON text it was submitted with, so its identity is
//! the exact string a run was launched under and not a re-encoding of it.

use serde::{Deserialize, Serialize};

use crate::digest::Digest;

/// The plan a distributed trainer runs under, as its coordinator was told it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JobPlan {
    /// The name the training run is known by.
    pub job: String,
    /// How many ranks the run has.
    pub world_size: usize,
    /// How the ranks are split across tensor parallelism.
    pub tp: u32,
    /// How the ranks are split across pipeline parallelism.
    pub pp: u32,
    /// How the ranks are split across data parallelism.
    pub dp: u32,
    /// The coordinator every rank registers with.
    pub coordinator: String,
    /// How long a rank waits for the others to register, in milliseconds.
    pub handshake_timeout_ms: u64,
    /// How long one collective operation may take, in milliseconds.
    pub op_timeout_ms: u64,
}

impl JobPlan {
    /// Parses a plan, refusing unknown fields.
    pub fn parse(text: &str) -> Result<JobPlan, CheckpointManifestError> {
        serde_json::from_str(text).map_err(|source| CheckpointManifestError::Json {
            reason: source.to_string(),
        })
    }
}

/// What a trainer committed at one step.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointManifest {
    /// The name of the training run that wrote the commit.
    pub job: String,
    /// The plan the run was launched under, as the exact JSON text it was
    /// submitted with.
    pub plan: String,
    /// The training step the checkpoint belongs to.
    pub step: u32,
    /// How many ranks wrote it.
    pub world_size: usize,
    /// Every file the commit holds.
    pub files: Vec<CheckpointFile>,
}

impl CheckpointManifest {
    /// Parses a manifest, refusing unknown fields, malformed paths and
    /// checksums that are not `sha256:<64 lowercase hex>`.
    pub fn parse(bytes: &[u8]) -> Result<CheckpointManifest, CheckpointManifestError> {
        serde_json::from_slice(bytes).map_err(|source| CheckpointManifestError::Json {
            reason: source.to_string(),
        })
    }
}

/// One file of a commit.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointFile {
    /// What the file is for, such as `state` or `weights`.
    pub role: String,
    /// Where it lies beneath the commit directory.
    pub path: ManifestPath,
    /// Its length in bytes.
    pub len: u64,
    /// Its SHA-256, as the trainer computed it.
    pub sha256: Sha256Sum,
}

/// A path a manifest names: relative, and never leading out of the commit
/// directory.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ManifestPath(String);

impl ManifestPath {
    /// The path as the manifest spelled it, with `/` separators.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for ManifestPath {
    type Error = CheckpointManifestError;

    fn try_from(value: String) -> Result<ManifestPath, CheckpointManifestError> {
        let refused = || CheckpointManifestError::Path {
            value: value.clone(),
        };
        if value.is_empty() || value.starts_with('/') {
            return Err(refused());
        }
        if value.split('/').any(|part| part == "..") {
            return Err(refused());
        }
        Ok(ManifestPath(value))
    }
}

impl From<ManifestPath> for String {
    fn from(path: ManifestPath) -> String {
        path.0
    }
}

/// A SHA-256 checksum in the form an external trainer reports it:
/// `sha256:<64 lowercase hex>`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Sha256Sum(Digest);

impl Sha256Sum {
    /// The checksum as a digest, for comparing with the bytes it names.
    #[must_use]
    pub fn digest(&self) -> &Digest {
        &self.0
    }
}

impl TryFrom<String> for Sha256Sum {
    type Error = CheckpointManifestError;

    fn try_from(value: String) -> Result<Sha256Sum, CheckpointManifestError> {
        match Digest::parse(&value) {
            Ok(digest) if value.starts_with("sha256:") => Ok(Sha256Sum(digest)),
            _ => Err(CheckpointManifestError::Sha256 { value }),
        }
    }
}

impl From<Sha256Sum> for String {
    fn from(sum: Sha256Sum) -> String {
        sum.0.into()
    }
}

/// Why a manifest is not part of the contract.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum CheckpointManifestError {
    /// The bytes are not a manifest of this shape.
    #[error("a checkpoint manifest is not valid JSON: {reason}")]
    Json {
        /// Why it did not parse.
        reason: String,
    },
    /// A file path is absolute, empty or leads out of the commit directory.
    #[error("{value:?} is not a relative path a checkpoint manifest may name")]
    Path {
        /// The path as it was written.
        value: String,
    },
    /// A checksum is not `sha256:<64 lowercase hex>`.
    #[error("{value:?} is not a sha256:<64 lowercase hex> checksum")]
    Sha256 {
        /// The checksum as it was written.
        value: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest(sha256: &str, path: &str) -> String {
        format!(
            r#"{{"job":"run","plan":"{{}}","step":1,"world_size":1,"files":[{{"role":"weights","path":"{path}","len":1,"sha256":"{sha256}"}}]}}"#
        )
    }

    const SHA: &str = "sha256:1111111111111111111111111111111111111111111111111111111111111111";

    #[test]
    fn a_manifest_parses_and_keeps_the_plan_text_verbatim() {
        let plan = r#"{"job":"run","world_size":1}"#;
        let text = format!(r#"{{"job":"run","plan":{plan:?},"step":7,"world_size":1,"files":[]}}"#);
        let parsed = CheckpointManifest::parse(text.as_bytes()).unwrap();
        assert_eq!(parsed.step, 7);
        assert_eq!(parsed.plan, plan);
        assert_eq!(parsed.files.len(), 0);
    }

    #[test]
    fn unknown_fields_are_refused() {
        let text = r#"{"job":"run","plan":"{}","step":1,"world_size":1,"files":[],"extra":true}"#;
        assert!(CheckpointManifest::parse(text.as_bytes()).is_err());
    }

    #[test]
    fn paths_out_of_the_commit_directory_are_refused() {
        for path in ["../weights", "/weights", "state/../../weights", ""] {
            assert!(
                CheckpointManifest::parse(manifest(SHA, path).as_bytes()).is_err(),
                "{path:?}"
            );
        }
        let parsed = CheckpointManifest::parse(manifest(SHA, "state/weights").as_bytes()).unwrap();
        assert_eq!(parsed.files[0].path.as_str(), "state/weights");
    }

    #[test]
    fn only_a_lowercase_sha256_is_a_checksum() {
        for bad in [
            "blake3:1111111111111111111111111111111111111111111111111111111111111111",
            "sha256:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            "sha256:1111",
        ] {
            assert!(
                CheckpointManifest::parse(manifest(bad, "weights").as_bytes()).is_err(),
                "{bad:?}"
            );
        }
        let parsed = CheckpointManifest::parse(manifest(SHA, "weights").as_bytes()).unwrap();
        assert_eq!(parsed.files[0].sha256.digest().as_str(), SHA);
    }
}
