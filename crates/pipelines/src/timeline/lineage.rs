// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements audit trails for released risk models, from
// the release back to the raw source file of every participant it was trained
// on, for its clients. If your team needs expertise in model provenance for
// regulated prediction, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The lineage of a released timeline model, read from what the stores record:
//! the release, the candidate it was, the datasets it was trained on, the
//! episodes those were projected from, the line of each source file every
//! episode was imported from, and the digest of that file.
//!
//! Nothing is stored for it: the release manifest names the candidate and the
//! datasets, the candidate's record names its configuration, split and
//! commits, each dataset's manifest names its episodes and source files, and
//! each episode's own provenance names the file line. The participant keys in
//! it are the opaque ones; no raw identifier exists to show.

use std::collections::{BTreeMap, HashSet};

use serde::Serialize;
use splinter_core::dataset::DatasetId;
use splinter_core::digest::Digest;
use splinter_core::release::ReleaseId;
use splinter_core::terms::{Distribution, Terms};
use splinter_data::split::Part;
use splinter_data::timeline_dataset::SourceFile;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_store::longitudinal::LongitudinalStore;

use super::records::TimelineCandidate;
use super::release::resolve_release;
use super::train::load_timeline_candidate;

/// The most episodes listed per dataset unless the caller asks for more.
pub const DEFAULT_EPISODE_LIMIT: usize = 5;

/// One episode and where its record came from.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EpisodeSource {
    /// The episode's content address.
    pub episode: Digest,
    /// The opaque participant key.
    pub participant: String,
    /// The imported dataset.
    pub dataset: String,
    /// The digest of the raw source file.
    pub file: Digest,
    /// The line of that file the record is on.
    pub line: u64,
}

/// A dataset a release was trained on.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DatasetTrace {
    /// The dataset's id.
    pub id: DatasetId,
    /// Which part of its split it is.
    pub part: Option<Part>,
    /// The digest of its records file.
    pub snapshot: Digest,
    /// Records in it.
    pub records: usize,
    /// Episodes it was projected from.
    pub episodes: usize,
    /// The first of those, with their source lines.
    pub listed: Vec<EpisodeSource>,
    /// The source files the episodes came from, with how many of the dataset's
    /// episodes each holds.
    pub files: Vec<(SourceFile, usize)>,
}

/// The release, the run that made it and everything it was made from.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TimelineLineage {
    /// The release.
    pub release: ReleaseId,
    /// The release it continues.
    pub parent: Option<ReleaseId>,
    /// The candidate it was.
    pub candidate: String,
    /// The digest of its checkpoint file.
    pub checkpoint: Digest,
    /// The digest of its training configuration.
    pub config: Digest,
    /// Its seed.
    pub seed: u64,
    /// The address of the split it was cut from.
    pub split: Digest,
    /// The digests of the evaluation splits it was judged on.
    pub evaluation_splits: Vec<Digest>,
    /// The commit of brain that trained it.
    pub brain_commit: Option<String>,
    /// The commit of Splinter that trained it.
    pub splinter_commit: Option<String>,
    /// How widely it may be handed on.
    pub distribution: Distribution,
    /// The terms of everything it was made from.
    pub terms: Terms,
    /// The datasets it was trained on.
    pub datasets: Vec<DatasetTrace>,
}

fn trace(
    ctx: &Context,
    id: &DatasetId,
    episodes: &LongitudinalStore,
    limit: usize,
) -> Result<DatasetTrace, OrchestratorError> {
    let stored = ctx.timeline_datasets().get(id)?;
    let wanted: HashSet<&Digest> = stored.manifest.episodes.iter().collect();
    let mut found: BTreeMap<Digest, EpisodeSource> = BTreeMap::new();
    episodes.for_each_history(|address, history| {
        if wanted.contains(address) {
            found.insert(
                address.clone(),
                EpisodeSource {
                    episode: address.clone(),
                    participant: history.participant.to_string(),
                    dataset: history.provenance.dataset.clone(),
                    file: history.provenance.file.clone(),
                    line: history.provenance.line,
                },
            );
        }
        Ok(())
    })?;
    if found.len() != wanted.len() {
        return Err(OrchestratorError::Refused(format!(
            "dataset {id} names {} episodes but the store holds {} of them",
            wanted.len(),
            found.len()
        )));
    }
    let mut per_file: BTreeMap<(String, Digest), usize> = BTreeMap::new();
    for e in found.values() {
        *per_file
            .entry((e.dataset.clone(), e.file.clone()))
            .or_default() += 1;
    }
    Ok(DatasetTrace {
        id: id.clone(),
        part: stored.manifest.part,
        snapshot: stored.manifest.dataset.clone(),
        records: stored.manifest.records,
        episodes: wanted.len(),
        listed: stored
            .manifest
            .episodes
            .iter()
            .take(limit)
            .filter_map(|a| found.get(a).cloned())
            .collect(),
        files: per_file
            .into_iter()
            .map(|((dataset, file), n)| (SourceFile { dataset, file }, n))
            .collect(),
    })
}

/// The lineage of `release` (an id, a unique prefix or an alias), listing at
/// most `limit` episodes per dataset.
pub fn timeline_lineage(
    ctx: &Context,
    release: &str,
    limit: usize,
) -> Result<TimelineLineage, OrchestratorError> {
    let store = ctx.releases();
    let id = resolve_release(ctx, release)?;
    let stored = store.get(&id)?;
    let (_, record): (String, TimelineCandidate) =
        load_timeline_candidate(ctx, &stored.manifest.candidate).map_err(|e| {
            OrchestratorError::Refused(format!(
                "release {id} was not made from a timeline candidate this store holds: {e}"
            ))
        })?;
    let episodes = LongitudinalStore::new(ctx.workspace());
    let datasets = stored
        .manifest
        .datasets
        .iter()
        .map(|d| trace(ctx, d, &episodes, limit))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(TimelineLineage {
        release: id,
        parent: stored.manifest.parent.clone(),
        candidate: stored.manifest.candidate.clone(),
        checkpoint: record.checkpoint_sha256.clone(),
        config: record.config_digest.clone(),
        seed: record.seed,
        split: record.split.clone(),
        evaluation_splits: stored.manifest.provenance.evaluation_splits.clone(),
        brain_commit: record.brain_commit.clone(),
        splinter_commit: record.splinter_commit.clone(),
        distribution: stored.manifest.distribution,
        terms: stored.manifest.terms.clone(),
        datasets,
    })
}
