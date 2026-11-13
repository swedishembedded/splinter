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
//!     train-<seed>/         one task folder per instance (Harbor layout)
//!     dev-<seed>/
//!   compilations/<id>.json  what each objective was compiled to
//! ```
//!
//! Instances are written only for an environment that may still enter a
//! campaign (admitted, or waiting for its band): a rejected or quarantined
//! environment keeps its report, with the attack that beat it, and nothing a
//! campaign could pick up by mistake. An instance directory is never written
//! over another: the same environment, split and seed always has the same
//! content, so a differing file is a bug to report, not to overwrite.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde::Serialize;
use splinter_core::digest::Digest;
use splinter_core::verifiability::Split;

use super::admission::{AdmissionReport, Status};
use super::artifact::{ArtifactError, Instance};
use super::band::BandMeasurement;
use super::folder::{Folder, FolderError};

const REPORT_FILE: &str = "admission.json";
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
    /// An instance already stored differs from the one to store.
    #[error("{0} is already stored with different content")]
    Differs(String),
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
            for instance in instances {
                let meta = instance.meta();
                let target = dir.join(format!("{}-{}", meta.split, meta.seed));
                if target.exists() {
                    let existing = Folder::read_from(&target)?;
                    if existing.digest() != instance.version() {
                        return Err(StoreError::Differs(target.display().to_string()));
                    }
                } else {
                    instance.folder().write_to(&target)?;
                }
            }
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

    /// The stored instances of `id` in `split`, as task folders on disk.
    pub fn instance_dirs(&self, id: &str, split: Split) -> Result<Vec<PathBuf>, StoreError> {
        let dir = self.dir(id)?;
        let prefix = format!("{split}-");
        let mut found: Vec<PathBuf> = std::fs::read_dir(&dir)
            .map_err(io(&dir))?
            .filter_map(Result::ok)
            .filter(|e| e.file_name().to_string_lossy().starts_with(&prefix))
            .map(|e| e.path())
            .collect();
        found.sort();
        Ok(found)
    }

    /// Reads a stored instance back.
    pub fn load_instance(&self, dir: &Path) -> Result<Instance, StoreError> {
        Ok(Instance::from_folder(Folder::read_from(dir)?)?)
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
