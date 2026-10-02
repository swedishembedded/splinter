// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements versioned, content-addressed training
// datasets curated from verified agent experience, for its clients. If your
// team needs expertise in dataset curation for fine-tuning, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Datasets, named by their manifest.
//!
//! A dataset is two things: the records file brain trains on, a real file kept
//! as an artifact at a stable path, and its manifest, a document in the
//! experience database. A [`DatasetId`] is the digest of the manifest's
//! canonical form, and the manifest names the records file by its digest in
//! turn, so one id pins both. The file is stored first and the manifest makes
//! the dataset official in one commit; storing the same projection again finds
//! it already there. [`DatasetStore::get`] checks both digests on every read.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use splinter_record::artifacts::{ArtifactSpec, ArtifactState, ArtifactStore};
use splinter_record::digest::Digest;
use splinter_record::experiences::StoreError;
use splinter_record::workspace::Workspace;
use splinter_record::StateRoot;

use crate::dataset::{manifest_path, write_dataset, Manifest, WriteOptions};
use crate::{Projection, ViewError};

const MANIFEST: &str = "dataset_manifest";

/// The file name a dataset's records are written under.
const DATASET_FILE: &str = "dataset.jsonl";

/// A stored dataset's id: the digest of its manifest's canonical form.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DatasetId(pub Digest);

impl std::fmt::Display for DatasetId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// A dataset as stored and verified.
#[derive(Clone, Debug, PartialEq)]
pub struct StoredDataset {
    /// Its id.
    pub id: DatasetId,
    /// The records file, a real file a tool can open.
    pub path: PathBuf,
    /// Its manifest.
    pub manifest: Manifest,
}

/// The datasets of one state root.
#[derive(Clone, Debug)]
pub struct DatasetStore {
    workspace: Workspace,
    artifacts: ArtifactStore,
    work: PathBuf,
}

impl DatasetStore {
    /// The store over `workspace`, keeping files under `root`.
    #[must_use]
    pub fn new(workspace: &Workspace, root: &StateRoot) -> Self {
        Self {
            workspace: workspace.clone(),
            artifacts: ArtifactStore::new(workspace, root),
            work: root.work().join("datasets"),
        }
    }

    fn spec() -> ArtifactSpec {
        ArtifactSpec::new("dataset", "splinter-views").with_extension(".jsonl")
    }

    /// Writes `projection` as [`write_dataset`] does and stores it under its
    /// id; the refusals are [`write_dataset`]'s. A dataset already stored
    /// under the same id is kept and returned.
    pub fn put(
        &self,
        projection: &Projection,
        options: WriteOptions,
    ) -> Result<StoredDataset, ViewError> {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let scratch = self.work.join(format!(
            "{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&scratch);
        fs::create_dir_all(&scratch).map_err(io(&scratch))?;
        let stored = self.put_from(&scratch, projection, options);
        // A scratch directory is only ever a working place.
        let _ = fs::remove_dir_all(&scratch);
        stored
    }

    fn put_from(
        &self,
        scratch: &Path,
        projection: &Projection,
        options: WriteOptions,
    ) -> Result<StoredDataset, ViewError> {
        let file = scratch.join(DATASET_FILE);
        let written = write_dataset(&file, projection, options)?;
        // The records file first, then the manifest that makes the dataset
        // official: a crash between them leaves an orphan file and nothing
        // that names it.
        let artifact = self.artifacts.put_file(&file, &Self::spec())?;
        let manifest_file = manifest_path(&file);
        let manifest_bytes = fs::read(&manifest_file).map_err(io(&manifest_file))?;
        let manifest: Manifest = serde_json::from_slice(&manifest_bytes)?;
        let stored = self.workspace.put_document(MANIFEST, &manifest)?;
        if stored != written.manifest || artifact.digest != written.digest {
            return Err(ViewError::Store(StoreError::Altered {
                what: "a dataset as it was stored".into(),
                expected: written.manifest,
                found: stored,
            }));
        }
        self.get(&DatasetId(stored))
    }

    /// The dataset stored under `id`, verified: the manifest hashes to `id`,
    /// and the records file to the digest the manifest names.
    pub fn get(&self, id: &DatasetId) -> Result<StoredDataset, ViewError> {
        let manifest: Manifest = self
            .workspace
            .get_document(MANIFEST, &id.0)?
            .ok_or_else(|| ViewError::UnknownDataset(id.clone()))?;
        let path = self.artifacts.path(&manifest.dataset)?;
        match self.artifacts.check(&manifest.dataset, true)? {
            ArtifactState::Sound => {}
            ArtifactState::Missing => {
                return Err(StoreError::MissingArtifact {
                    digest: manifest.dataset.clone(),
                    role: "dataset".into(),
                    path,
                }
                .into())
            }
            ArtifactState::SizeChanged { .. } => {
                return Err(StoreError::Altered {
                    what: format!("dataset {id}"),
                    expected: manifest.dataset.clone(),
                    found: manifest.dataset.clone(),
                }
                .into())
            }
            ArtifactState::Corrupt { found } => {
                return Err(StoreError::Altered {
                    what: format!("dataset {id}"),
                    expected: manifest.dataset.clone(),
                    found,
                }
                .into())
            }
        }
        Ok(StoredDataset {
            id: id.clone(),
            path,
            manifest,
        })
    }

    /// Every stored dataset's id, in id order (without verifying them).
    pub fn list(&self) -> Result<Vec<DatasetId>, ViewError> {
        Ok(self
            .workspace
            .document_ids(MANIFEST)?
            .into_iter()
            .map(DatasetId)
            .collect())
    }
}

/// Wraps an I/O error on `path`.
fn io(path: &Path) -> impl FnOnce(std::io::Error) -> ViewError + '_ {
    move |source| ViewError::Io {
        path: path.to_path_buf(),
        source,
    }
}
