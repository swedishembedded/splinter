// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning experiments whose every result
// names the exact code, data and hardware that produced it, for its clients.
// If your team needs expertise in attributable, reproducible machine-learning
// runs, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The revision manifest of a run that trains or evaluates: which code it
//! was built from (Splinter's, sven's and brain's commits), which
//! training-record contract this build writes, what the brain server that
//! served it says it is (its build, backend, hardware and record contract),
//! the model and tokenizer digests it reported, the digests of the stage's
//! datasets and task collections and the stage's resolved configuration.
//!
//! Two refusals come with it. A run pins its revisions when it starts
//! ([`RunRevisions`]); resumed by a build of other commits it refuses,
//! naming each that differs, unless its recipe allows mixed revisions. And a
//! brain server that reports another training-record contract than the one
//! this build was pinned against ([`RECORD_CONTRACT_DIGEST`]) is refused by
//! name ([`ServedRevision::check_contract`]): the records Splinter writes are
//! checked by that contract. A server that reports none (a brain build older
//! than the report) is recorded as unknown, `null`, and refused only where
//! the run requires the contract.
//!
//! Every fact a server or build did not establish is absent, never filled
//! in.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::build::BuildInfo;
use crate::digest::Digest;

/// The schema of a stage's revision manifest.
pub const REVISIONS_SCHEMA: &str = "splinter.revisions/1";
/// The schema of a run's pinned revisions.
pub const RUN_REVISIONS_SCHEMA: &str = "splinter.run-revisions/1";
/// The file a stage's revision manifest is kept in, in its product, and a
/// run's pinned revisions, in its directory.
pub const REVISIONS_FILE: &str = "revisions.json";

/// The training-record contract the records this build writes keep.
pub const RECORD_CONTRACT: &str = "training-records-v2";
/// Its version number.
pub const RECORD_CONTRACT_VERSION: u32 = 2;
/// The digest of the contract's example set as this build keeps a copy of
/// it (brain's, unchanged): what a brain server must report as its own
/// (`server.training_records.digest` of `GET /v1/capabilities`). It is
/// `sha256:<hex>` of one line `<path>\0<sha256 hex of the file>\n` per file
/// of the set but its `README.md`, sorted by path, as brain computes it; a
/// test holds it to the copy.
pub const RECORD_CONTRACT_DIGEST: &str =
    "sha256:4fb13e2f610fb2d3d4ede4731c5da509d2ccfae1071143c796015e824a4bc20b";

/// A training-record contract: its name, version and the digest of its
/// example set.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecordContract {
    /// Its name (`training-records-v2`).
    pub contract: String,
    /// Its version number.
    pub version: u32,
    /// `sha256:<hex>` of its example set (the digest of its files).
    pub digest: String,
}

impl RecordContract {
    /// The contract this build is pinned against.
    #[must_use]
    pub fn pinned() -> Self {
        Self {
            contract: RECORD_CONTRACT.to_owned(),
            version: RECORD_CONTRACT_VERSION,
            digest: RECORD_CONTRACT_DIGEST.to_owned(),
        }
    }
}

/// A brain build as its server reports it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineBuild {
    /// Its version.
    pub version: String,
    /// The commit it was built from, when the build knew it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,
    /// Whether its tree had uncommitted changes, when the build knew it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dirty: Option<bool>,
}

/// What one brain server says it is: its endpoint, its build, the backend it
/// computes on, the training-record contract its trainer checks and its
/// devices (`GET /v1/devices`, as reported).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServedRevision {
    /// The endpoint's URL.
    pub endpoint: String,
    /// Its build; `null` for a server that does not report it.
    pub engine: Option<EngineBuild>,
    /// Its compute backend.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    /// The contract its trainer checks records against; `null` (written
    /// out, never left out) for a server that does not report it: unknown,
    /// never guessed.
    pub training_records: Option<RecordContract>,
    /// Its device report: the host and every device it schedules on.
    pub hardware: Value,
}

impl ServedRevision {
    /// Refuse a server whose trainer checks another training-record contract
    /// than the one this build writes, and, when the contract is `required`,
    /// one that does not say which it checks. A server that says none is
    /// otherwise accepted: its contract is unknown.
    ///
    /// # Errors
    /// Why, naming both digests (or the endpoint that reports none).
    pub fn check_contract(&self, required: bool) -> Result<(), String> {
        let pinned = RecordContract::pinned();
        match &self.training_records {
            Some(served) if *served == pinned => Ok(()),
            Some(served) => Err(format!(
                "the brain server at {} checks training records against {} version {} ({}), and this build of Splinter writes {} version {} ({}); build Splinter against that brain (its contract fixtures copied), or serve the brain it was pinned against",
                self.endpoint, served.contract, served.version, served.digest, pinned.contract, pinned.version, pinned.digest
            )),
            None if !required => Ok(()),
            None => Err(format!(
                "the brain server at {} reports no training-record contract, the recipe requires it (`contract: required`), and this build of Splinter writes {} ({}); serve a brain that reports it",
                self.endpoint, pinned.contract, pinned.digest
            )),
        }
    }
}

/// The run-level identity of a policy version a brain training run
/// published, as the server reports it beside the serving-local version: the
/// training run, the job of it (`epoch`), the optimiser update the weights
/// hold, the base checkpoint and the adapter file.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyLineage {
    /// The training run (its resume key).
    pub run: String,
    /// The job of the run that took the update.
    pub epoch: u64,
    /// The optimiser updates the weights hold, over the whole run.
    pub update: u64,
    /// `sha256:<hex>` of the base checkpoint.
    pub base_digest: String,
    /// `sha256:<hex>` of the adapter file.
    pub adapter_digest: String,
}

/// A model and tokenizer a stage used, by the digests its server reported
/// for them; each absent when the server did not report it.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelDigests {
    /// `sha256:<hex>` of the base checkpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_digest: Option<String>,
    /// `sha256:<hex>` of its tokenizer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer_digest: Option<String>,
}

/// The revision manifest of a stage that trains or evaluates. See the module
/// documentation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageRevisions {
    /// Always [`REVISIONS_SCHEMA`].
    pub schema: String,
    /// The build that ran the stage: Splinter's, sven's and brain's commits,
    /// whether the tree was dirty, the executable's digest.
    pub build: BuildInfo,
    /// The training-record contract this build writes.
    pub training_records: RecordContract,
    /// What every brain server the stage reached says it is; empty when the
    /// stage reached none.
    pub served: Vec<ServedRevision>,
    /// The models and tokenizers the servers reported for the stage, where
    /// they reported them.
    pub models: Vec<ModelDigests>,
    /// The stage's datasets, task collections and other inputs, by input
    /// role, as the digests of their products.
    pub inputs: Vec<InputDigest>,
    /// The stage's resolved configuration: its parameters as checked (every
    /// default written in, files as their digests), its seed and its
    /// implementation.
    pub configuration: Value,
}

/// One input of a stage, by role, as its product's digest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputDigest {
    /// The role it is consumed under.
    pub input: String,
    /// Its product's digest.
    pub digest: Digest,
}

/// What a run pins when it starts and holds every resume of it to.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunRevisions {
    /// Always [`RUN_REVISIONS_SCHEMA`].
    pub schema: String,
    /// The commits the build that started it was made from.
    pub revisions: crate::build::Revisions,
    /// Whether its tree was dirty.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dirty: Option<bool>,
    /// The SHA-256 of the executable that started it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub binary_digest: Option<Digest>,
    /// The training-record contract it writes.
    pub training_records: RecordContract,
}

impl RunRevisions {
    /// What a run started by `build` pins.
    #[must_use]
    pub fn of(build: &BuildInfo) -> Self {
        Self {
            schema: RUN_REVISIONS_SCHEMA.to_owned(),
            revisions: build.revisions.clone(),
            dirty: build.dirty,
            binary_digest: build.binary_digest.clone(),
            training_records: RecordContract::pinned(),
        }
    }

    /// Every way `now` differs from these, one line each, by name: a commit
    /// of a repository, the dirty flag or the record contract. The same
    /// executable (by digest) is the same code. Otherwise a commit that was
    /// unknown or is unknown now differs, and so does a dirty tree: neither
    /// says which code ran.
    #[must_use]
    pub fn changes(&self, now: &Self) -> Vec<String> {
        if self.binary_digest.is_some() && self.binary_digest == now.binary_digest {
            return Vec::new();
        }
        let mut changes = Vec::new();
        let shown = |rev: &Option<String>| rev.clone().unwrap_or_else(|| "unknown".into());
        for (name, was, is) in [
            (
                "splinter",
                &self.revisions.splinter,
                &now.revisions.splinter,
            ),
            ("sven", &self.revisions.sven, &now.revisions.sven),
            ("brain", &self.revisions.brain, &now.revisions.brain),
        ] {
            if was != is || was.is_none() {
                changes.push(format!(
                    "{name} commit: the run started at {}, this build is {}",
                    shown(was),
                    shown(is)
                ));
            }
        }
        if self.dirty != now.dirty || self.dirty == Some(true) {
            let flag = |d: Option<bool>| match d {
                Some(true) => "dirty",
                Some(false) => "clean",
                None => "unknown",
            };
            changes.push(format!(
                "splinter tree: the run started {}, this build is {}",
                flag(self.dirty),
                flag(now.dirty)
            ));
        }
        if self.training_records != now.training_records {
            changes.push(format!(
                "training-record contract: the run started on {}, this build writes {}",
                self.training_records.digest, now.training_records.digest
            ));
        }
        changes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::Revisions;
    use std::path::Path;

    /// The digest of a contract example set in `dir`, as brain computes it for
    /// its own: `sha256:<hex>` of one line `<path>\0<sha256 hex of the file>\n`
    /// per file under `dir` but its `README.md`, the path relative to `dir`
    /// with `/` separators, sorted by path.
    ///
    fn record_contract_digest(dir: &Path) -> std::io::Result<String> {
        fn walk(root: &Path, dir: &Path, out: &mut Vec<(String, String)>) -> std::io::Result<()> {
            for entry in std::fs::read_dir(dir)? {
                let path = entry?.path();
                if path.is_dir() {
                    walk(root, &path, out)?;
                    continue;
                }
                let relative = path.strip_prefix(root).map_err(std::io::Error::other)?;
                let name = relative
                    .components()
                    .map(|c| c.as_os_str().to_string_lossy().into_owned())
                    .collect::<Vec<_>>()
                    .join("/");
                if name != "README.md" {
                    let digest = Digest::sha256_of_reader(std::fs::File::open(&path)?)?;
                    out.push((name, digest.hex().to_owned()));
                }
            }
            Ok(())
        }
        let mut files = Vec::new();
        walk(dir, dir, &mut files)?;
        files.sort();
        let listing: String = files
            .iter()
            .map(|(name, hex)| format!("{name}\0{hex}\n"))
            .collect();
        Ok(Digest::sha256_of(listing.as_bytes()).as_str().to_owned())
    }

    fn build(splinter: &str, dirty: bool) -> BuildInfo {
        BuildInfo {
            revisions: Revisions {
                splinter: Some(splinter.into()),
                sven: Some("s1".into()),
                brain: Some("b1".into()),
            },
            dirty: Some(dirty),
            ..BuildInfo::default()
        }
    }

    /// The pinned digest is the digest of this build's copy of brain's
    /// example set: a copy that changes without the pin (or the reverse)
    /// fails here, before a server is refused for it.
    #[test]
    fn the_pinned_record_contract_is_the_digest_of_this_builds_copy() {
        let copy = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../data/tests/fixtures/training-records/v2");
        assert_eq!(
            record_contract_digest(&copy).unwrap(),
            RECORD_CONTRACT_DIGEST
        );
        let manifest: Value =
            serde_json::from_slice(&std::fs::read(copy.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(manifest["contract"], RECORD_CONTRACT);
    }

    #[test]
    fn a_run_names_every_revision_a_resume_would_change() {
        let started = RunRevisions::of(&build("aaa", false));
        assert!(started
            .changes(&RunRevisions::of(&build("aaa", false)))
            .is_empty());
        let moved = started.changes(&RunRevisions::of(&build("bbb", false)));
        assert_eq!(moved.len(), 1, "{moved:?}");
        assert!(
            moved[0].contains("splinter commit")
                && moved[0].contains("aaa")
                && moved[0].contains("bbb")
        );
        // A dirty tree is never the same code twice, and an unknown commit is not a known one.
        assert!(!started
            .changes(&RunRevisions::of(&build("aaa", true)))
            .is_empty());
        let dirty = RunRevisions::of(&build("aaa", true));
        assert!(!dirty.changes(&dirty).is_empty());
        let unknown = RunRevisions::of(&BuildInfo::default());
        assert!(unknown
            .changes(&unknown)
            .iter()
            .any(|c| c.contains("unknown")));
        // The same executable is the same code, whatever its tree was.
        let exe = BuildInfo {
            binary_digest: Some(Digest::sha256_of(b"exe")),
            ..build("aaa", true)
        };
        assert!(RunRevisions::of(&exe)
            .changes(&RunRevisions::of(&exe))
            .is_empty());
        let rebuilt = BuildInfo {
            binary_digest: Some(Digest::sha256_of(b"other")),
            ..build("aaa", true)
        };
        assert!(!RunRevisions::of(&exe)
            .changes(&RunRevisions::of(&rebuilt))
            .is_empty());
    }

    #[test]
    fn a_server_with_another_record_contract_is_refused_and_one_with_none_only_when_required() {
        let served = |training_records| ServedRevision {
            endpoint: "http://brain:8080".into(),
            engine: None,
            backend: Some("cpu".into()),
            training_records,
            hardware: Value::Null,
        };
        for required in [false, true] {
            assert_eq!(
                served(Some(RecordContract::pinned())).check_contract(required),
                Ok(())
            );
            let other = RecordContract {
                digest: format!("sha256:{}", "0".repeat(64)),
                ..RecordContract::pinned()
            };
            let refused = served(Some(other)).check_contract(required).unwrap_err();
            assert!(
                refused.contains(RECORD_CONTRACT_DIGEST)
                    && refused.contains(&"0".repeat(64))
                    && refused.contains("http://brain:8080"),
                "{refused}"
            );
        }
        assert_eq!(
            served(None).check_contract(false),
            Ok(()),
            "unknown is not a mismatch"
        );
        let refused = served(None).check_contract(true).unwrap_err();
        assert!(
            refused.contains("http://brain:8080") && refused.contains("contract: required"),
            "{refused}"
        );
        // Unknown is written out, never left out.
        let written = serde_json::to_value(served(None)).unwrap();
        assert_eq!(
            (
                written["training_records"].clone(),
                written["engine"].clone()
            ),
            (Value::Null, Value::Null)
        );
    }
}
