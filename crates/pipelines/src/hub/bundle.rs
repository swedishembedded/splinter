// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The bundle's manifest: what a published revision holds, every file of it
//! by SHA-256, and where it came from.
//!
//! The bundle's identity is its [`Contents`]: the content addresses of the
//! experience sets, experiences, datasets, candidates and verdicts it
//! carries, never the bytes of a file that only packs them (an archive of
//! the same experiences made twice need not be the same bytes). Publishing
//! the same contents again is therefore recognised however the files came
//! out, and the tag a publish leaves is named by that identity.

use serde::{Deserialize, Serialize};
use splinter_core::dataset::DatasetId;
use splinter_core::digest::{canonical_json, Digest};
use splinter_core::experience::ExperienceId;
use splinter_orchestrator::error::OrchestratorError;
use splinter_store::experiences::SetId;

/// The schema a bundle's manifest declares.
pub const SCHEMA: &str = "splinter.bundle/1";

/// The manifest's path in the repository.
pub const MANIFEST: &str = "manifest.json";

/// The archive of the experience database holding the bundle's sets and
/// experiences.
pub const EXPERIENCES: &str = "experiences.tar.zst";

/// The examiner's public key the bundle's verdicts verify under.
pub const EXAMINER_KEY: &str = "verdicts/examiner.pub";

/// The prefix of the tag a publish leaves on its commit.
pub const TAG_PREFIX: &str = "splinter-";

/// A bundle's manifest.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BundleManifest {
    /// [`SCHEMA`].
    pub schema: String,
    /// The bundle's identity: the SHA-256 of the canonical form of
    /// `contents`.
    pub content: Digest,
    /// What it carries.
    pub contents: Contents,
    /// Every file of it but this manifest, in path order.
    pub files: Vec<BundleFile>,
    /// Where it came from.
    pub provenance: Provenance,
}

/// What a bundle carries, by content address.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Contents {
    /// The experience sets, in the order asked for.
    pub sets: Vec<SetEntry>,
    /// The experiences its datasets were built from, in id order (held in
    /// the archive whether or not a set names them).
    pub experiences: Vec<ExperienceId>,
    /// The datasets, in id order.
    pub datasets: Vec<DatasetEntry>,
    /// The trained candidates, in id order.
    pub candidates: Vec<CandidateEntry>,
    /// The examiner's signed aggregate verdicts, in digest order.
    pub verdicts: Vec<VerdictEntry>,
}

impl Contents {
    /// The bundle identity of these contents.
    ///
    /// # Errors
    /// They cannot be written as JSON.
    pub fn identity(&self) -> Result<Digest, OrchestratorError> {
        let bytes = canonical_json(self).map_err(|source| OrchestratorError::Json {
            what: "the bundle's contents".into(),
            source,
        })?;
        Ok(Digest::sha256_of(&bytes))
    }

    /// The name of the tag a publish of these contents leaves.
    ///
    /// # Errors
    /// As [`Contents::identity`].
    pub fn tag(&self) -> Result<String, OrchestratorError> {
        Ok(tag_of(&self.identity()?))
    }

    /// Whether the bundle carries nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sets.is_empty()
            && self.experiences.is_empty()
            && self.datasets.is_empty()
            && self.candidates.is_empty()
            && self.verdicts.is_empty()
    }
}

/// The tag of the bundle whose identity is `content`.
#[must_use]
pub fn tag_of(content: &Digest) -> String {
    format!("{TAG_PREFIX}{}", &content.hex()[..20])
}

/// An experience set of a bundle.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SetEntry {
    /// Its id.
    pub id: SetId,
    /// Its name.
    pub name: String,
    /// How many experiences it holds.
    pub members: usize,
}

/// A dataset of a bundle.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DatasetEntry {
    /// Its id.
    pub id: DatasetId,
    /// The view that projected it.
    pub view: String,
    /// Its records.
    pub records: usize,
}

/// A trained candidate of a bundle.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CandidateEntry {
    /// Its id.
    pub candidate: String,
    /// Its adapter's SHA-256, as brain reports it.
    pub adapter_digest: String,
    /// The content address of its record (with the training record brain
    /// wrote).
    pub record: Digest,
}

/// A signed aggregate verdict of a bundle.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VerdictEntry {
    /// The SHA-256 of the document (hex).
    pub sha256: String,
    /// The schema it declares.
    pub schema: String,
    /// The id of the examiner's key it is signed with.
    pub key_id: String,
}

/// A file of a bundle.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BundleFile {
    /// Its path in the repository.
    pub path: String,
    /// The SHA-256 of its bytes (hex).
    pub sha256: String,
    /// Its size in bytes.
    pub bytes: u64,
    /// What it is.
    #[serde(flatten)]
    pub kind: FileKind,
}

/// What a file of a bundle is.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FileKind {
    /// The archive of the experience database ([`EXPERIENCES`]).
    Experiences,
    /// A dataset's records.
    DatasetRecords {
        /// The dataset.
        dataset: DatasetId,
    },
    /// A dataset's manifest, in its canonical form.
    DatasetManifest {
        /// The dataset.
        dataset: DatasetId,
    },
    /// A candidate's record.
    Candidate {
        /// The candidate.
        candidate: String,
    },
    /// A file a candidate names by content address (an adapter, a
    /// rehearsal), kept as the source root kept it.
    Artifact {
        /// Its content address.
        digest: Digest,
        /// What it is for.
        role: String,
        /// Who made it.
        producer: String,
        /// The extension its file carries.
        extension: String,
        /// Whether its SHA-256 is recorded beside its address.
        sha256_recorded: bool,
    },
    /// A signed verdict of the examiner.
    Verdict,
    /// The examiner's public key ([`EXAMINER_KEY`]).
    ExaminerKey,
}

/// Where a bundle came from.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    /// The runs that produced what it carries: each run whose outputs name
    /// one of its sets, datasets or candidates.
    pub source_runs: Vec<SourceRun>,
    /// The commits of the programs that made it (each run's and each
    /// candidate trainer's splinter, sven and brain), and of the publisher.
    pub commits: Vec<String>,
    /// The tokenizers the records' sampled token ids are of.
    pub tokenizer_digests: Vec<String>,
    /// The bases the candidates were trained on, as their cards state them.
    pub base_digests: Vec<String>,
}

/// A run a bundle's contents came from.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceRun {
    /// Its id.
    pub run: String,
    /// The command it ran.
    pub command: String,
}

/// The repository path of a dataset's records.
#[must_use]
pub fn dataset_records_path(id: &DatasetId) -> String {
    format!("datasets/{}.jsonl", id.0.hex())
}

/// The repository path of a dataset's manifest.
#[must_use]
pub fn dataset_manifest_path(id: &DatasetId) -> String {
    format!("datasets/{}.manifest.json", id.0.hex())
}

/// The repository path of a candidate's record.
#[must_use]
pub fn candidate_path(candidate: &str) -> String {
    format!("candidates/{candidate}.json")
}

/// The repository path of an artifact.
#[must_use]
pub fn artifact_path(digest: &Digest, extension: &str) -> String {
    format!("artifacts/{}{extension}", digest.hex())
}

/// The repository path of a verdict.
#[must_use]
pub fn verdict_path(sha256: &str) -> String {
    format!("verdicts/{sha256}.json")
}
