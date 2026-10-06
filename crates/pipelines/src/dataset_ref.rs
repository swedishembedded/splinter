// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements traceable training data for learned models,
// from a release back to the records it was trained on, for its clients. If
// your team needs expertise in dataset lineage for audited models, you can
// procure our services by sending an email to info@swedishembedded.com.

//! A stored dataset of either family, as a release names it.
//!
//! A release is trained on datasets named by id. A chat dataset (messages and
//! preference pairs, projected from experience) and a timeline dataset
//! (participant-safe `timeline-v1` records, projected from longitudinal
//! episodes) are kept by different stores, but a release needs the same three
//! things of both: the id, the terms the records came under and the digest of
//! the records file. Their lineage differs: a chat dataset reaches the
//! experiences it was projected from, a timeline dataset the episodes.

use splinter_core::dataset::DatasetId;
use splinter_core::digest::Digest;
use splinter_core::terms::Terms;
use splinter_data::timeline_store::StoredTimeline;
use splinter_data::StoredDataset;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_orchestrator::ids;
use splinter_orchestrator::runs::to_json;
use splinter_store::longitudinal::LongitudinalStore;

use crate::datasets::{record_dataset_lineage, resolve_dataset};

/// A stored dataset of either family.
#[derive(Clone, Debug, PartialEq)]
pub enum AnyDataset {
    /// Chat or preference records projected from experience.
    Chat(StoredDataset),
    /// Timeline records projected from longitudinal episodes.
    Timeline(StoredTimeline),
}

impl AnyDataset {
    /// The dataset's id.
    #[must_use]
    pub fn id(&self) -> &DatasetId {
        match self {
            Self::Chat(d) => &d.id,
            Self::Timeline(d) => &d.id,
        }
    }

    /// The digest of the records file.
    #[must_use]
    pub fn snapshot(&self) -> &Digest {
        match self {
            Self::Chat(d) => &d.manifest.dataset,
            Self::Timeline(d) => &d.manifest.dataset,
        }
    }

    /// The terms its records came under, when any source stated them.
    #[must_use]
    pub fn terms(&self) -> Option<&Terms> {
        match self {
            Self::Chat(d) => d.manifest.terms.as_ref(),
            Self::Timeline(d) => d.manifest.terms.as_ref(),
        }
    }
}

/// The stored dataset `id` (or a unique prefix of it) names, whichever store
/// holds it, verified.
pub fn resolve_any(ctx: &Context, id: &str) -> Result<AnyDataset, OrchestratorError> {
    let timeline = ctx.timeline_datasets().list()?;
    let chat = ctx.datasets().list()?;
    let all = timeline.iter().chain(&chat).map(|d| d.0.clone());
    let found = DatasetId(ids::resolve("dataset", id, all)?);
    if timeline.contains(&found) {
        Ok(AnyDataset::Timeline(ctx.timeline_datasets().get(&found)?))
    } else {
        resolve_dataset(ctx, &found.to_string()).map(AnyDataset::Chat)
    }
}

/// Records where `dataset` came from, if that was not recorded yet: the
/// experiences of a chat dataset, the episodes of a timeline one.
pub fn record_lineage(ctx: &Context, dataset: &AnyDataset) -> Result<(), OrchestratorError> {
    match dataset {
        AnyDataset::Chat(d) => record_dataset_lineage(ctx, d),
        AnyDataset::Timeline(d) => {
            LongitudinalStore::new(ctx.workspace()).record_dataset(
                &d.id.0,
                to_json("timeline dataset manifest", &d.manifest)?,
                d.manifest.records as u64,
                &d.manifest.episodes,
            )?;
            Ok(())
        }
    }
}
