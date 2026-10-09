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
//!
//! A release is of one [`ReleasedArtifact`]: a LoRA adapter on a named base,
//! or a full checkpoint whose format brain owns. Splinter keeps either as an
//! immutable content-addressed file and records its digest; it never reads
//! the file.

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize};
use splinter_core::dataset::DatasetId;
use splinter_core::digest::Digest;
use splinter_core::release::ReleaseId;
use splinter_core::terms::{Distribution, Terms};
use splinter_core::training::{RehearsalSample, ReplaySample, TrainingSummary};
use splinter_eval::claim_gate::ClaimGate;
use splinter_eval::gate::GateReport;
use splinter_eval::metric_gate::Evidence;
use splinter_eval::predictive_gate::PredictiveReport;

/// The `format` every manifest is written with.
pub const RELEASE_FORMAT: &str = "splinter-release-v4";

/// The format of manifests that recorded terms but assumed every release an
/// adapter. Read, never written.
pub const MANIFEST_V3: &str = "splinter-release-v3";

/// The format of manifests written before usage terms were recorded. They
/// are still read: what they never recorded is unknown, so such a release is
/// restricted and its terms fail closed.
pub const MANIFEST_V2: &str = "splinter-release-v2";

/// What kind of file a release is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    /// A LoRA adapter, which only means something on its base.
    Adapter,
    /// A whole model checkpoint, in a format brain owns.
    FullCheckpoint,
}

/// The one file a release is, as the manifest names it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ReleasedArtifact {
    /// A LoRA adapter on a base.
    Adapter {
        /// The base model, as brain's model store names it.
        base_model: String,
        /// The digest of the base checkpoint file the adapter sits on.
        base_digest: Digest,
        /// The digest of the adapter file, as brain reports it.
        adapter_digest: Digest,
        /// The artifact the adapter is kept as.
        adapter_artifact: Digest,
    },
    /// A full checkpoint: no base, no adapter.
    FullCheckpoint {
        /// The model architecture, as brain names it.
        architecture: String,
        /// The digest of the checkpoint file, as brain reports it.
        checkpoint_digest: Digest,
        /// The artifact the checkpoint is kept as.
        checkpoint_artifact: Digest,
    },
}

impl ReleasedArtifact {
    /// Which kind of file it is.
    #[must_use]
    pub fn kind(&self) -> ArtifactKind {
        match self {
            Self::Adapter { .. } => ArtifactKind::Adapter,
            Self::FullCheckpoint { .. } => ArtifactKind::FullCheckpoint,
        }
    }

    /// The SHA-256 of the file as its producer reported it, which the store
    /// holds the file to.
    #[must_use]
    pub fn content_digest(&self) -> &Digest {
        match self {
            Self::Adapter { adapter_digest, .. } => adapter_digest,
            Self::FullCheckpoint {
                checkpoint_digest, ..
            } => checkpoint_digest,
        }
    }

    /// The artifact the file is kept as.
    #[must_use]
    pub fn stored(&self) -> &Digest {
        match self {
            Self::Adapter {
                adapter_artifact, ..
            } => adapter_artifact,
            Self::FullCheckpoint {
                checkpoint_artifact,
                ..
            } => checkpoint_artifact,
        }
    }
}

/// The gate a release passed, with every number.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ReleaseGate {
    /// The four-check gate over graded answers.
    Llm {
        /// The report, boxed so the variants stay small.
        report: Box<GateReport>,
    },
    /// The gate over the facts a person taught, decided on counts.
    Claims {
        /// The report, boxed so the variants stay small.
        report: Box<ClaimGate>,
    },
    /// The five-check gate over a predictive model's metrics on paired
    /// held-out units.
    Predictive {
        /// The report, boxed so the variants stay small.
        report: Box<PredictiveReport>,
    },
}

impl ReleaseGate {
    /// The graded-answer gate's report, when that is the gate the release
    /// passed.
    #[must_use]
    pub fn llm(&self) -> Option<&GateReport> {
        match self {
            Self::Llm { report } => Some(report.as_ref()),
            Self::Predictive { .. } | Self::Claims { .. } => None,
        }
    }

    /// The claim gate's report, when that is the gate the release passed.
    #[must_use]
    pub fn claims(&self) -> Option<&ClaimGate> {
        match self {
            Self::Claims { report } => Some(report.as_ref()),
            Self::Llm { .. } | Self::Predictive { .. } => None,
        }
    }

    /// The predictive gate's report, when that is the gate the release
    /// passed.
    #[must_use]
    pub fn predictive(&self) -> Option<&PredictiveReport> {
        match self {
            Self::Predictive { report } => Some(report.as_ref()),
            Self::Llm { .. } | Self::Claims { .. } => None,
        }
    }
}

/// What a release was made and judged from, by digest, beyond the datasets
/// it names: enough to rebuild the claim that it was measured fairly.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    /// The digests of the datasets' record files it was trained on.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dataset_snapshots: Vec<Digest>,
    /// The digest of the training configuration it was trained under.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training_config: Option<Digest>,
    /// The digests of the held-out evaluation splits it was judged on.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evaluation_splits: Vec<Digest>,
    /// The digest of its calibration artifact, when it has one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration: Option<Digest>,
    /// The commit of brain that trained and serves it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub brain_commit: Option<String>,
    /// The commit of splinter that decided it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub splinter_commit: Option<String>,
}

impl Provenance {
    /// Whether nothing is recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self == &Self::default()
    }
}

/// What a release is and why it was released.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReleaseManifest {
    /// [`RELEASE_FORMAT`] for a release written now; [`MANIFEST_V3`] or
    /// [`MANIFEST_V2`] for one read from an older manifest.
    pub format: String,
    /// The file it is.
    pub artifact: ReleasedArtifact,
    /// The release it was trained from; `None` for the first.
    pub parent: Option<ReleaseId>,
    /// The candidate it was.
    pub candidate: String,
    /// The new datasets it was trained on; their held-out records are its
    /// suite.
    pub datasets: Vec<DatasetId>,
    /// The earlier records replayed beside them.
    pub replay: Option<ReplaySample>,
    /// The base's own answers rehearsed beside them; `None` when none
    /// were, and for a release recorded before rehearsal was.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rehearsal: Option<RehearsalSample>,
    /// How it was trained.
    pub training: TrainingSummary,
    /// The gate it passed, with every number.
    pub gate: ReleaseGate,
    /// The metrics it was released on, by name with their intervals, when the
    /// gate was handed some; `None` for a gate that grades answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metrics: Option<Evidence>,
    /// The terms of everything it was made from, combined over its datasets
    /// and the release it continues; unknown when none were stated.
    pub terms: Terms,
    /// How widely it was released. `Unrestricted` only when `terms` allow
    /// it; `Restricted` records that the release stays where it was made.
    pub distribution: Distribution,
    /// What else it was made and judged from.
    #[serde(default, skip_serializing_if = "Provenance::is_empty")]
    pub provenance: Provenance,
    /// When it was released, from the injected clock.
    pub created_at: String,
}

/// A manifest of [`MANIFEST_V3`]: terms recorded, every release an adapter.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct ManifestV3 {
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
    terms: Terms,
    distribution: Distribution,
    created_at: String,
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
    V3(ManifestV3),
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
            Some(MANIFEST_V3) => serde_json::from_value(value)
                .map(Self::V3)
                .map_err(D::Error::custom),
            Some(RELEASE_FORMAT) => serde_json::from_value(value)
                .map(Self::Current)
                .map_err(D::Error::custom),
            other => Err(D::Error::custom(format!(
                "release format {other:?} is none of {RELEASE_FORMAT}, {MANIFEST_V3}, {MANIFEST_V2}"
            ))),
        }
    }
}

impl ManifestV3 {
    fn into_manifest(self) -> ReleaseManifest {
        ReleaseManifest {
            format: self.format,
            artifact: ReleasedArtifact::Adapter {
                base_model: self.base_model,
                base_digest: self.base_digest,
                adapter_digest: self.adapter_digest,
                adapter_artifact: self.adapter_artifact,
            },
            parent: self.parent,
            candidate: self.candidate,
            datasets: self.datasets,
            replay: self.replay,
            rehearsal: None,
            training: self.training,
            gate: ReleaseGate::Llm {
                report: Box::new(self.gate),
            },
            metrics: None,
            terms: self.terms,
            distribution: self.distribution,
            provenance: Provenance::default(),
            created_at: self.created_at,
        }
    }
}

impl ManifestDocument {
    /// The manifest, converted to the current shape. An older release keeps
    /// its format string, so it stays recognisable as read from an older
    /// manifest; one from before terms were recorded has unknown terms and
    /// is restricted.
    pub(crate) fn into_manifest(self) -> ReleaseManifest {
        match self {
            Self::Current(manifest) => manifest,
            Self::V3(old) => old.into_manifest(),
            Self::V2(old) => ManifestV3 {
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
            }
            .into_manifest(),
        }
    }
}
