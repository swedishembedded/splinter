// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements versioned stores of validated training
// environments with their admission history, for its clients. If your team
// needs expertise in managing generated environments across a campaign, you
// can procure our services by sending an email to info@swedishembedded.com.

//! Where environments live on disk.
//!
//! ```text
//! <root>/
//!   <environment-id>/
//!     admission.json        the AdmissionReport: gates, attacks, band, status
//!     instances.json        every stored instance of the environment: one
//!                           archive, files shared between instances (the
//!                           tool, the verifier) kept once by content address
//!   compilations/<id>.json  what each objective was compiled to
//! ```
//!
//! A generated collection is one archive, never a directory per task: a
//! campaign draws thousands of instances and a file per instance file would
//! exhaust the inode quota of a shared disk. A task folder exists only while
//! something runs it: [`EnvironmentStore::stage`] writes one into a scratch
//! directory and removes it when the guard is dropped.
//!
//! Instances are archived only for an environment that may still enter a
//! campaign (admitted, or waiting for its band): a rejected or quarantined
//! environment keeps its report, with the attack that beat it, and nothing a
//! campaign could pick up by mistake. The same environment, split and seed
//! always has the same content, so a differing archived instance is a bug to
//! report, not to overwrite.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use splinter_core::digest::Digest;
use splinter_core::verifiability::Split;

use super::admission::{AdmissionReport, Status};
use super::artifact::{ArtifactError, Instance};
use super::band::BandMeasurement;
use super::folder::{Folder, FolderError};

const REPORT_FILE: &str = "admission.json";
const ARCHIVE_FILE: &str = "instances.json";
const COMPILATIONS_DIR: &str = "compilations";

/// Why the store could not read or write.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    /// The id is not an environment id (it could name a path elsewhere).
    #[error("{0:?} is not an environment id (env- and 16 hex digits)")]
    BadId(String),
    /// No such environment.
    #[error("no environment {0} in the store")]
    NotFound(String),
    /// The disk refused.
    #[error("{path}: {source}")]
    Io {
        /// What was being touched.
        path: String,
        /// The error.
        source: std::io::Error,
    },
    /// A report that does not parse.
    #[error("{path}: {source}")]
    Json {
        /// The file.
        path: String,
        /// The error.
        source: serde_json::Error,
    },
    /// A task folder that cannot be written or read.
    #[error(transparent)]
    Folder(#[from] FolderError),
    /// A task folder that is not an artifact.
    #[error(transparent)]
    Artifact(#[from] ArtifactError),
    /// An instance already archived differs from the one to store.
    #[error("{0} is already archived with different content")]
    Differs(String),
    /// An archive that does not hold together.
    #[error("the archive is corrupt: {0}")]
    Corrupt(String),
}

fn io(path: &Path) -> impl FnOnce(std::io::Error) -> StoreError + '_ {
    move |source| StoreError::Io {
        path: path.display().to_string(),
        source,
    }
}

fn valid_id(id: &str) -> bool {
    id.strip_prefix("env-")
        .is_some_and(|hex| hex.len() == 16 && hex.bytes().all(|b| b.is_ascii_hexdigit()))
}

/// One archived instance: its files by content address.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Archived {
    split: Split,
    seed: u64,
    version: String,
    /// `(path, executable, blob)` for every file of the folder.
    files: Vec<(String, bool, String)>,
}

/// An environment's instances in one file.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
struct Archive {
    /// File contents as hex, by content address.
    blobs: BTreeMap<String, String>,
    instances: Vec<Archived>,
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut out, b| {
            let _ = write!(out, "{b:02x}");
            out
        })
}

fn unhex(text: &str) -> Option<Vec<u8>> {
    if !text.len().is_multiple_of(2) || !text.is_ascii() {
        return None;
    }
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).ok())
        .collect()
}

impl Archive {
    fn add(&mut self, instance: &Instance) -> Result<(), String> {
        let meta = instance.meta();
        let version = instance.version().to_string();
        if let Some(existing) = self
            .instances
            .iter()
            .find(|a| a.split == meta.split && a.seed == meta.seed)
        {
            return if existing.version == version {
                Ok(())
            } else {
                Err(format!("{}-{}", meta.split, meta.seed))
            };
        }
        let mut files = Vec::new();
        for (path, entry) in instance.folder().entries() {
            let address = Digest::of(&entry.bytes).to_string();
            self.blobs
                .entry(address.clone())
                .or_insert_with(|| hex(&entry.bytes));
            files.push((path.to_string(), entry.executable, address));
        }
        self.instances.push(Archived {
            split: meta.split,
            seed: meta.seed,
            version,
            files,
        });
        Ok(())
    }

    fn instance(&self, archived: &Archived) -> Result<Instance, StoreError> {
        let mut folder = Folder::new();
        for (path, executable, address) in &archived.files {
            let bytes = self
                .blobs
                .get(address)
                .and_then(|h| unhex(h))
                .ok_or_else(|| StoreError::Corrupt(format!("blob {address} of {path}")))?;
            folder.put(path, bytes, *executable)?;
        }
        let instance = Instance::from_folder(folder)?;
        if instance.version().to_string() != archived.version {
            return Err(StoreError::Corrupt(format!(
                "{}-{} does not match its recorded version",
                archived.split, archived.seed
            )));
        }
        Ok(instance)
    }
}

/// A task folder staged for a run; removed when dropped.
pub struct Staged {
    _scratch: tempfile::TempDir,
    dir: PathBuf,
}

impl Staged {
    /// The task folder.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.dir
    }
}

/// A directory of environments.
#[derive(Clone, Debug)]
pub struct EnvironmentStore {
    root: PathBuf,
}

impl EnvironmentStore {
    /// The store at `root` (created on the first write).
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The directory the store lives in.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn dir(&self, id: &str) -> Result<PathBuf, StoreError> {
        if valid_id(id) {
            Ok(self.root.join(id))
        } else {
            Err(StoreError::BadId(id.to_string()))
        }
    }

    fn write_json(path: &Path, value: &impl Serialize) -> Result<(), StoreError> {
        let text = serde_json::to_string_pretty(value).map_err(|source| StoreError::Json {
            path: path.display().to_string(),
            source,
        })?;
        let temporary = path.with_extension("json.tmp");
        std::fs::write(&temporary, text + "\n").map_err(io(&temporary))?;
        std::fs::rename(&temporary, path).map_err(io(path))
    }

    /// Stores `report`, and `instances` when the environment may still enter
    /// a campaign.
    pub fn save(&self, report: &AdmissionReport, instances: &[Instance]) -> Result<(), StoreError> {
        let dir = self.dir(&report.environment)?;
        std::fs::create_dir_all(&dir).map_err(io(&dir))?;
        if matches!(report.status, Status::Admitted | Status::PendingBand) {
            let mut archive = self.archive(&report.environment)?;
            for instance in instances {
                archive.add(instance).map_err(StoreError::Differs)?;
            }
            Self::write_json(&dir.join(ARCHIVE_FILE), &archive)?;
        }
        Self::write_json(&dir.join(REPORT_FILE), report)
    }

    /// The admission report of `id`.
    pub fn report(&self, id: &str) -> Result<AdmissionReport, StoreError> {
        let path = self.dir(id)?.join(REPORT_FILE);
        let text = std::fs::read_to_string(&path).map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => StoreError::NotFound(id.to_string()),
            _ => io(&path)(e),
        })?;
        serde_json::from_str(&text).map_err(|source| StoreError::Json {
            path: path.display().to_string(),
            source,
        })
    }

    /// Every report, by environment id.
    pub fn list(&self) -> Result<Vec<AdmissionReport>, StoreError> {
        let Ok(entries) = std::fs::read_dir(&self.root) else {
            return Ok(Vec::new());
        };
        let mut ids: Vec<String> = entries
            .filter_map(Result::ok)
            .filter_map(|e| e.file_name().into_string().ok())
            .filter(|name| valid_id(name))
            .collect();
        ids.sort();
        ids.iter().map(|id| self.report(id)).collect()
    }

    /// Records a base-policy measurement for `id` and re-judges it: an
    /// environment waiting for its band is admitted or rejected, and one
    /// already judged is judged again against the newer measurement.
    pub fn record_band(
        &self,
        id: &str,
        measurement: &BandMeasurement,
    ) -> Result<AdmissionReport, StoreError> {
        let report = self.report(id)?.with_band(Some(measurement));
        Self::write_json(&self.dir(id)?.join(REPORT_FILE), &report)?;
        Ok(report)
    }

    /// The content versions of every instance ever drawn, for the novelty
    /// gate of the next environment.
    pub fn known_versions(&self) -> Result<BTreeSet<Digest>, StoreError> {
        Ok(self
            .list()?
            .iter()
            .flat_map(|r| r.instances.iter())
            .filter_map(|i| Digest::parse(&i.version).ok())
            .collect())
    }

    fn archive(&self, id: &str) -> Result<Archive, StoreError> {
        let path = self.dir(id)?.join(ARCHIVE_FILE);
        match std::fs::read_to_string(&path) {
            Ok(text) => serde_json::from_str(&text).map_err(|source| StoreError::Json {
                path: path.display().to_string(),
                source,
            }),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Archive::default()),
            Err(e) => Err(io(&path)(e)),
        }
    }

    /// The archived instances of `id` in `split`, by seed.
    pub fn instances(&self, id: &str, split: Split) -> Result<Vec<Instance>, StoreError> {
        let archive = self.archive(id)?;
        let mut chosen: Vec<&Archived> = archive
            .instances
            .iter()
            .filter(|a| a.split == split)
            .collect();
        chosen.sort_by_key(|a| a.seed);
        chosen.into_iter().map(|a| archive.instance(a)).collect()
    }

    /// Writes `instance`'s task folder into a new scratch directory under
    /// `parent` and returns the guard that removes it. This is the only way
    /// a task folder comes to exist on disk; a run holds the guard for as
    /// long as it needs the folder.
    pub fn stage(parent: &Path, instance: &Instance) -> Result<Staged, StoreError> {
        std::fs::create_dir_all(parent).map_err(io(parent))?;
        let scratch = tempfile::Builder::new()
            .prefix("task-")
            .tempdir_in(parent)
            .map_err(io(parent))?;
        let dir = scratch.path().join("task");
        instance.folder().write_to(&dir)?;
        Ok(Staged {
            _scratch: scratch,
            dir,
        })
    }

    /// Writes a document under `compilations/`.
    pub fn record_compilation(
        &self,
        name: &str,
        document: &impl Serialize,
    ) -> Result<PathBuf, StoreError> {
        let dir = self.root.join(COMPILATIONS_DIR);
        std::fs::create_dir_all(&dir).map_err(io(&dir))?;
        let path = dir.join(format!("{name}.json"));
        Self::write_json(&path, document)?;
        Ok(path)
    }
}
