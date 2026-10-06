// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements immutable, content-addressed model
// releases with full lineage, for its clients. If your team needs
// expertise in model release management, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The release manifest and the formats it has been stored in.
//!
//! A manifest is content-addressed: its id is the digest of the canonical
//! form of the document as stored, and a read checks it. A reader therefore
//! decodes each stored format as that format and only then converts it, so
//! the check still runs over the bytes that were written.

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize};
use splinter_core::dataset::DatasetId;
use splinter_core::digest::Digest;
use splinter_core::release::ReleaseId;
use splinter_core::terms::{Distribution, Terms};
use splinter_core::training::{ReplaySample, TrainingSummary};
use splinter_eval::gate::GateReport;

/// The `format` every manifest is written with.
pub const RELEASE_FORMAT: &str = "splinter-release-v3";

/// The format of manifests written before usage terms were recorded. They
/// are still read: what they never recorded is unknown, so such a release is
/// restricted and its terms fail closed.
pub const MANIFEST_V2: &str = "splinter-release-v2";

/// What a release is and why it was released.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReleaseManifest {
    /// [`RELEASE_FORMAT`] for a release written now; [`MANIFEST_V2`] for one
    /// read from an older manifest.
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
    /// The terms of everything it was made from, combined over its datasets
    /// and the release it continues; unknown when none were stated.
    pub terms: Terms,
    /// How widely it was released. `Unrestricted` only when `terms` allow
    /// it; `Restricted` records that the release stays where it was made.
    pub distribution: Distribution,
    /// When it was released, from the injected clock.
    pub created_at: String,
}

/// A manifest as it was written before terms were recorded: exactly the
/// fields of [`MANIFEST_V2`], so that it re-encodes to the bytes its id
/// addresses.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct ManifestV2 {
    format: String,
    base_model: String,
    base_digest: Digest,
    adapter_digest: Digest,
    adapter_artifact: Digest,
    parent: Option<ReleaseId>,
    candidate: String,
    datasets: Vec<DatasetId>,
    replay: Option<ReplaySample>,
    training: TrainingSummary,
    gate: GateReport,
    created_at: String,
}

/// A stored manifest in whichever format it was written.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub(crate) enum ManifestDocument {
    V2(ManifestV2),
    Current(ReleaseManifest),
}

impl<'de> Deserialize<'de> for ManifestDocument {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(deserializer)?;
        let format = value
            .get("format")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned);
        match format.as_deref() {
            Some(MANIFEST_V2) => serde_json::from_value(value)
                .map(Self::V2)
                .map_err(D::Error::custom),
            Some(RELEASE_FORMAT) => serde_json::from_value(value)
                .map(Self::Current)
                .map_err(D::Error::custom),
            other => Err(D::Error::custom(format!(
                "release format {other:?} is neither {RELEASE_FORMAT} nor {MANIFEST_V2}"
            ))),
        }
    }
}

impl ManifestDocument {
    /// The manifest, converted to the current shape. A [`MANIFEST_V2`]
    /// release keeps its format string, so it stays recognisable as read
    /// from an older manifest, and its terms are unknown.
    pub(crate) fn into_manifest(self) -> ReleaseManifest {
        match self {
            Self::Current(manifest) => manifest,
            Self::V2(old) => ReleaseManifest {
                format: old.format,
                base_model: old.base_model,
                base_digest: old.base_digest,
                adapter_digest: old.adapter_digest,
                adapter_artifact: old.adapter_artifact,
                parent: old.parent,
                candidate: old.candidate,
                datasets: old.datasets,
                replay: old.replay,
                training: old.training,
                gate: old.gate,
                terms: Terms::unknown("unstated: released before terms were recorded"),
                distribution: Distribution::Restricted,
                created_at: old.created_at,
            },
        }
    }
}
