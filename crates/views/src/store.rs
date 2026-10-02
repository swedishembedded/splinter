// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements versioned, content-addressed training
// datasets curated from verified agent experience, for its clients. If your
// team needs expertise in dataset curation for fine-tuning, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Datasets under the state root, named by their manifest.
//!
//! ```text
//! <root>/datasets/<hex>/
//!   dataset.jsonl                  the records, as [`write_dataset`] wrote them
//!   dataset.jsonl.manifest.json    its manifest; <hex> is this file's digest
//! ```
//!
//! A [`DatasetId`] is the digest of the manifest's bytes, and the manifest
//! names the dataset file by its digest in turn, so one id pins both. A
//! dataset is written into a pending directory beside the others and moved
//! into place whole once its manifest exists; storing the same projection
//! again finds it already there. [`DatasetStore::get`] checks both digests
//! on every read.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use splinter_record::digest::Digest;
use splinter_record::experiences::StoreError;
use splinter_record::StateRoot;

use crate::dataset::{manifest_path, write_dataset, Manifest, WriteOptions};
use crate::{Projection, ViewError};

/// The dataset file's name inside its directory.
pub const DATASET_FILE: &str = "dataset.jsonl";

/// A stored dataset's id: the digest of its manifest's bytes.
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
    /// The dataset file.
    pub path: PathBuf,
    /// Its manifest.
    pub manifest: Manifest,
}

/// The datasets under one state root.
#[derive(Clone, Debug)]
pub struct DatasetStore {
    dir: PathBuf,
}

impl DatasetStore {
    /// The store under `root`. Nothing is created until something is
    /// written.
    #[must_use]
    pub fn open(root: &StateRoot) -> Self {
        Self {
            dir: root.datasets(),
        }
    }

    fn dataset_dir(&self, id: &DatasetId) -> PathBuf {
        self.dir.join(id.0.hex())
    }

    /// Writes `projection` as [`write_dataset`] does and stores it under
    /// its id; the refusals are [`write_dataset`]'s. A dataset already
    /// stored under the same id is kept and returned.
    pub fn put(
        &self,
        projection: &Projection,
        options: WriteOptions,
    ) -> Result<StoredDataset, ViewError> {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let pending = self.dir.join(format!(
            ".pending-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&pending);
        fs::create_dir_all(&pending).map_err(io(&pending))?;
        let written = write_dataset(&pending.join(DATASET_FILE), projection, options);
        let written = match written {
            Ok(written) => written,
            Err(e) => {
                // The refusal is the error worth reporting; a leftover
                // pending directory is harmless and named for this process.
                let _ = fs::remove_dir_all(&pending);
                return Err(e);
            }
        };
        let id = DatasetId(written.manifest);
        let target = self.dataset_dir(&id);
        if target.is_dir() {
            fs::remove_dir_all(&pending).map_err(io(&pending))?;
        } else if let Err(e) = fs::rename(&pending, &target) {
            // A writer racing on the same dataset got there first.
            if !target.is_dir() {
                return Err(io(&target)(e));
            }
            fs::remove_dir_all(&pending).map_err(io(&pending))?;
        }
        self.get(&id)
    }

    /// The dataset stored under `id`, verified: the manifest's bytes hash
    /// to `id`, and the dataset's bytes to the digest the manifest names.
    pub fn get(&self, id: &DatasetId) -> Result<StoredDataset, ViewError> {
        let path = self.dataset_dir(id).join(DATASET_FILE);
        let manifest_file = manifest_path(&path);
        if !manifest_file.is_file() {
            return Err(ViewError::UnknownDataset(id.clone()));
        }
        let manifest_bytes = fs::read(&manifest_file).map_err(io(&manifest_file))?;
        verify(&manifest_file, &id.0, &manifest_bytes)?;
        let manifest: Manifest = serde_json::from_slice(&manifest_bytes)?;
        let dataset_bytes = fs::read(&path).map_err(io(&path))?;
        verify(&path, &manifest.dataset, &dataset_bytes)?;
        Ok(StoredDataset {
            id: id.clone(),
            path,
            manifest,
        })
    }

    /// Every stored dataset's id, in id order (without verifying them).
    pub fn list(&self) -> Result<Vec<DatasetId>, ViewError> {
        let entries = match fs::read_dir(&self.dir) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(io(&self.dir)(e)),
        };
        let mut ids = Vec::new();
        for entry in entries {
            let name = entry.map_err(io(&self.dir))?.file_name();
            // Anything but a `<64 hex>` directory (a pending write) is not
            // a stored dataset.
            if let Some(Ok(digest)) = name.to_str().map(Digest::from_content_hex) {
                ids.push(DatasetId(digest));
            }
        }
        ids.sort();
        Ok(ids)
    }
}

/// Refuses `bytes`, read from `path`, unless they hash to `expected`.
fn verify(path: &Path, expected: &Digest, bytes: &[u8]) -> Result<(), ViewError> {
    let found = Digest::of(bytes);
    if found != *expected {
        return Err(ViewError::Store(StoreError::Corrupt {
            path: path.to_path_buf(),
            expected: expected.clone(),
            found,
        }));
    }
    Ok(())
}

/// Wraps an I/O error on `path`.
fn io(path: &Path) -> impl FnOnce(std::io::Error) -> ViewError + '_ {
    move |source| ViewError::Io {
        path: path.to_path_buf(),
        source,
    }
}
