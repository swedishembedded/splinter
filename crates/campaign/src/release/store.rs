// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements immutable, content-addressed model
// releases with full lineage, for its clients. If your team needs
// expertise in model release management, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Releases, and the aliases that point at them.
//!
//! A release is a manifest in the experience database and the adapter file it
//! names, which is the artifact its candidate was already kept as: nothing is
//! copied, and one file is never two. A [`ReleaseId`] is the digest of the
//! manifest's canonical form, and the manifest names the adapter by its digest,
//! so one id pins both. The manifest and the release's place in the lineage are
//! made official in one commit; one that already exists is refused.
//!
//! An alias is a pointer, so its history is kept: moving one is a
//! compare-and-set on its version ([`ReleaseStore::move_alias`]): it moves only
//! from the release its caller measured against, so two releases decided
//! against the same champion cannot both land, whichever processes decide
//! them, and there is no lock file to go stale. Rolling back is moving to the
//! older release.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use splinter_core::digest::Digest;
use splinter_store::artifacts::ArtifactStore;
use splinter_store::experiences::StoreError;
use splinter_store::workspace::Workspace;
use splinter_store::StateRoot;
use splinter_views::DatasetId;

use crate::error::CampaignError;
use crate::model_ref::is_alias_name;
use crate::release::gate::GateReport;
use crate::train::{ReplaySample, TrainingSummary};

const RELEASE: &str = "release";
const ALIAS_PREFIX: &str = "alias-";
/// The `format` every manifest carries.
pub const RELEASE_FORMAT: &str = "splinter-release-v2";

/// The longest lineage walked: a guard against a corrupt store, far past
/// any real history.
const MAX_LINEAGE: usize = 10_000;

/// A release's id: the digest of its manifest's bytes.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ReleaseId(pub Digest);

impl std::fmt::Display for ReleaseId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// What a release is and why it was released.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReleaseManifest {
    /// Always [`RELEASE_FORMAT`].
    pub format: String,
    /// The base model, as brain's model store names it.
    pub base_model: String,
    /// The digest of the base checkpoint file the adapter sits on.
    pub base_digest: Digest,
    /// The digest of the adapter file, as brain reports it.
    pub adapter_digest: Digest,
    /// The artifact the adapter is kept as.
    pub adapter_artifact: Digest,
    /// The release it was trained from; `None` for the first.
    pub parent: Option<ReleaseId>,
    /// The candidate it was.
    pub candidate: String,
    /// The new datasets it was trained on; their held-out records are its
    /// suite.
    pub datasets: Vec<DatasetId>,
    /// The earlier records replayed beside them.
    pub replay: Option<ReplaySample>,
    /// How it was trained.
    pub training: TrainingSummary,
    /// The gate it passed, with every number.
    pub gate: GateReport,
    /// When it was released, from the injected clock.
    pub created_at: String,
}

/// A release as stored and verified.
#[derive(Clone, Debug, PartialEq)]
pub struct StoredRelease {
    /// Its id.
    pub id: ReleaseId,
    /// Its adapter file, a real file brain loads from this path.
    pub adapter: PathBuf,
    /// Its manifest.
    pub manifest: ReleaseManifest,
}

/// The releases of one state root.
#[derive(Clone, Debug)]
pub struct ReleaseStore {
    workspace: Workspace,
    artifacts: ArtifactStore,
}

impl ReleaseStore {
    /// The store over `workspace`, with its files under `root`.
    #[must_use]
    pub fn new(workspace: &Workspace, root: &StateRoot) -> Self {
        Self {
            workspace: workspace.clone(),
            artifacts: ArtifactStore::new(workspace, root),
        }
    }

    /// Stores `manifest`, whose adapter must already be kept as an artifact
    /// whose SHA-256 is the manifest's adapter digest, and records where the
    /// release came from, in one commit. Refused when the release already
    /// exists: a release is written once.
    pub fn put(&self, manifest: &ReleaseManifest) -> Result<StoredRelease, CampaignError> {
        let artifact = self.artifacts.get(&manifest.adapter_artifact)?;
        if artifact.sha256.as_ref() != Some(&manifest.adapter_digest) {
            return Err(CampaignError::Refused(format!(
                "the adapter {} is kept with SHA-256 {}, not the {} the manifest names",
                artifact.digest,
                artifact
                    .sha256
                    .as_ref()
                    .map_or_else(|| "none".to_string(), ToString::to_string),
                manifest.adapter_digest
            )));
        }
        let id = ReleaseId(self.workspace.record_release(
            manifest,
            "release",
            &manifest.candidate,
            manifest.parent.as_ref().map(|p| &p.0),
        )?);
        self.get(&id)
    }

    /// The release `id`, verified: the manifest hashes to `id`, and its
    /// adapter is there at the size that was kept.
    pub fn get(&self, id: &ReleaseId) -> Result<StoredRelease, CampaignError> {
        let manifest: ReleaseManifest =
            self.workspace
                .get_document(RELEASE, &id.0)?
                .ok_or_else(|| CampaignError::NotFound {
                    what: "release",
                    id: id.to_string(),
                })?;
        let adapter = self.artifacts.path(&manifest.adapter_artifact)?;
        Ok(StoredRelease {
            id: id.clone(),
            adapter,
            manifest,
        })
    }

    /// Every stored release's id, in id order.
    pub fn list(&self) -> Result<Vec<ReleaseId>, CampaignError> {
        Ok(self
            .workspace
            .document_ids(RELEASE)?
            .into_iter()
            .map(ReleaseId)
            .collect())
    }

    /// The release made from `candidate`, if one was made: a candidate is
    /// released at most once.
    pub fn of_candidate(&self, candidate: &str) -> Result<Option<StoredRelease>, CampaignError> {
        for id in self.list()? {
            let release = self.get(&id)?;
            if release.manifest.candidate == candidate {
                return Ok(Some(release));
            }
        }
        Ok(None)
    }

    /// The release `alias` points at; `None` when it points nowhere yet.
    pub fn alias(&self, alias: &str) -> Result<Option<ReleaseId>, CampaignError> {
        let name = self.pointer(alias)?;
        self.workspace.refresh()?;
        match self.workspace.pointer(&name)? {
            None => Ok(None),
            Some((_, value)) => Digest::parse(&value)
                .map(|digest| Some(ReleaseId(digest)))
                .map_err(|e| CampaignError::Refused(format!("alias {alias} is corrupt: {e}"))),
        }
    }

    /// Every alias and the release it points at.
    pub fn aliases(&self) -> Result<BTreeMap<String, ReleaseId>, CampaignError> {
        self.workspace.refresh()?;
        let mut aliases = BTreeMap::new();
        for (name, value) in self.workspace.pointers(ALIAS_PREFIX)? {
            let Some(alias) = name.strip_prefix(ALIAS_PREFIX).filter(|a| is_alias_name(a)) else {
                continue;
            };
            let digest = Digest::parse(&value)
                .map_err(|e| CampaignError::Refused(format!("alias {alias} is corrupt: {e}")))?;
            aliases.insert(alias.to_string(), ReleaseId(digest));
        }
        Ok(aliases)
    }

    /// Points `alias` at `to`, provided it still points at `from` (`None`:
    /// nowhere yet); refused otherwise, naming where it points now. `to` must
    /// be a stored release. `at` stamps the move.
    pub fn move_alias(
        &self,
        alias: &str,
        from: Option<&ReleaseId>,
        to: &ReleaseId,
        at: &str,
    ) -> Result<(), CampaignError> {
        self.get(to)?;
        let pointer = self.pointer(alias)?;
        match self
            .workspace
            .move_pointer(&pointer, from.map(|id| id.0.as_str()), to.0.as_str(), at)
        {
            Ok(_) => Ok(()),
            Err(StoreError::PointerConflict {
                found, expected, ..
            }) => Err(CampaignError::Refused(format!(
                "alias {alias} moved to {} while this was decided against {}; decide again",
                found.unwrap_or_else(|| "nothing".into()),
                expected.unwrap_or_else(|| "nothing".into()),
            ))),
            Err(other) => Err(other.into()),
        }
    }

    /// Where an alias has pointed, oldest first: the release and when it was
    /// moved there.
    pub fn alias_history(&self, alias: &str) -> Result<Vec<(ReleaseId, String)>, CampaignError> {
        let pointer = self.pointer(alias)?;
        self.workspace.refresh()?;
        self.workspace
            .pointer_history(&pointer)?
            .into_iter()
            .map(|m| {
                Digest::parse(&m.value)
                    .map(|d| (ReleaseId(d), m.at))
                    .map_err(|e| CampaignError::Refused(format!("alias {alias} is corrupt: {e}")))
            })
            .collect()
    }

    /// `id` and the releases it descends from, newest first.
    pub fn lineage(&self, id: &ReleaseId) -> Result<Vec<StoredRelease>, CampaignError> {
        let mut lineage: Vec<StoredRelease> = Vec::new();
        let mut next = Some(id.clone());
        while let Some(id) = next {
            if lineage.len() >= MAX_LINEAGE || lineage.iter().any(|r| r.id == id) {
                return Err(CampaignError::Refused(format!(
                    "the lineage of release {id} does not end; the release store is corrupt"
                )));
            }
            let release = self.get(&id)?;
            next = release.manifest.parent.clone();
            lineage.push(release);
        }
        Ok(lineage)
    }

    fn pointer(&self, alias: &str) -> Result<String, CampaignError> {
        if !is_alias_name(alias) {
            return Err(CampaignError::Refused(format!(
                "{alias:?} is not an alias name: a lowercase letter, then up to 63 lowercase \
                 letters, digits, - or _"
            )));
        }
        Ok(format!("{ALIAS_PREFIX}{alias}"))
    }
}
