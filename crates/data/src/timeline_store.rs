// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements versioned, content-addressed datasets of
// longitudinal records with their splits, for its clients. If your team
// needs expertise in reproducible cohort datasets for risk models, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Timeline datasets, named by their manifest, and the splits that cut them.
//!
//! The same shape as the chat [`DatasetStore`](crate::DatasetStore): the
//! records file is kept as an artifact at a stable path and the
//! [`TimelineManifest`] is a document, a dataset's id is the digest of the
//! manifest's canonical form, and the manifest names the records file by its
//! digest, so one id pins both. A [`DataSplit`] is a document too, kept by its
//! own address, so a model trained on one part and scored on another can show
//! that the two parts were cut from one split.

use std::path::PathBuf;

use splinter_core::dataset::DatasetId;
use splinter_core::digest::Digest;
use splinter_store::artifacts::{ArtifactSpec, ArtifactState, ArtifactStore};
use splinter_store::experiences::StoreError;
use splinter_store::workspace::Workspace;
use splinter_store::StateRoot;

use crate::dataset::{manifest_path, Dataset};
use crate::split::DataSplit;
use crate::timeline_dataset::TimelineManifest;
use crate::ViewError;

const MANIFEST: &str = "timeline_manifest";
const SPLIT: &str = "data_split";

/// A timeline dataset as stored and verified.
#[derive(Clone, Debug, PartialEq)]
pub struct StoredTimeline {
    /// Its id.
    pub id: DatasetId,
    /// The records file, a real file a trainer can open.
    pub path: PathBuf,
    /// Its manifest.
    pub manifest: TimelineManifest,
}

/// The timeline datasets and splits of one state root.
#[derive(Clone, Debug)]
pub struct TimelineStore {
    workspace: Workspace,
    artifacts: ArtifactStore,
}

impl TimelineStore {
    /// The store over `workspace`, keeping files under `root`.
    #[must_use]
    pub fn new(workspace: &Workspace, root: &StateRoot) -> Self {
        Self {
            workspace: workspace.clone(),
            artifacts: ArtifactStore::new(workspace, root),
        }
    }

    /// Stores a dataset [`write_timeline_dataset`](crate::timeline_dataset::write_timeline_dataset)
    /// or [`write_timeline_splits`](crate::timeline_dataset::write_timeline_splits)
    /// wrote, with the manifest beside its file. Storing it again finds it
    /// already there.
    pub fn put(&self, written: &Dataset) -> Result<StoredTimeline, ViewError> {
        let io = |path: &std::path::Path| {
            let path = path.to_path_buf();
            move |source| ViewError::Io { path, source }
        };
        // The records file first, then the manifest that makes the dataset
        // official: a crash between them leaves an orphan file and nothing
        // that names it.
        let artifact = self.artifacts.put_file(
            &written.path,
            &ArtifactSpec::new("dataset", "splinter-timeline").with_extension(".jsonl"),
        )?;
        let manifest_file = manifest_path(&written.path);
        let bytes = std::fs::read(&manifest_file).map_err(io(&manifest_file))?;
        let manifest: TimelineManifest = serde_json::from_slice(&bytes)?;
        let stored = self.workspace.put_document(MANIFEST, &manifest)?;
        if stored != written.manifest || artifact.digest != written.digest {
            return Err(ViewError::Store(StoreError::Altered {
                what: "a timeline dataset as it was stored".into(),
                expected: written.manifest.clone(),
                found: stored,
            }));
        }
        self.get(&DatasetId(stored))
    }

    /// The dataset stored under `id`, verified: the manifest hashes to `id`
    /// and the records file to the digest the manifest names.
    pub fn get(&self, id: &DatasetId) -> Result<StoredTimeline, ViewError> {
        let manifest: TimelineManifest = self
            .workspace
            .get_document(MANIFEST, &id.0)?
            .ok_or_else(|| ViewError::UnknownDataset(id.clone()))?;
        let path = self.artifacts.path(&manifest.dataset)?;
        let altered = |found: Digest| {
            ViewError::Store(StoreError::Altered {
                what: format!("timeline dataset {id}"),
                expected: manifest.dataset.clone(),
                found,
            })
        };
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
            ArtifactState::SizeChanged { .. } => return Err(altered(manifest.dataset.clone())),
            ArtifactState::Corrupt { found } => return Err(altered(found)),
        }
        Ok(StoredTimeline {
            id: id.clone(),
            path,
            manifest,
        })
    }

    /// Every stored timeline dataset's id, in id order (not verified).
    pub fn list(&self) -> Result<Vec<DatasetId>, ViewError> {
        Ok(self
            .workspace
            .document_ids(MANIFEST)?
            .into_iter()
            .map(DatasetId)
            .collect())
    }

    /// Keeps `split` under its own address and returns it.
    pub fn put_split(&self, split: &DataSplit) -> Result<Digest, ViewError> {
        Ok(self.workspace.put_document(SPLIT, split)?)
    }

    /// The split kept under `address`, checked against it.
    pub fn split(&self, address: &Digest) -> Result<DataSplit, ViewError> {
        self.workspace.get_document(SPLIT, address)?.ok_or_else(|| {
            ViewError::Store(StoreError::Rejected {
                what: "data split",
                reason: format!("no split {address} is stored"),
            })
        })
    }
}
