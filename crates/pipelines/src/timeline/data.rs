// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements leakage-free preparation of longitudinal
// cohort data for risk models, from a source file to stored, split datasets
// whose every record traces to its file line, for its clients. If your team
// needs expertise in reproducible cohort splits, you can procure our services
// by sending an email to info@swedishembedded.com.

//! The data stages of the timeline pipeline: a record file becomes immutable
//! episodes (`import`), and the episodes become the stored parts of one split
//! (`split`).
//!
//! The import keeps one episode per participant under opaque keys, with the
//! terms its caller states; importing the same file again adds nothing. The
//! split is a pure function of the episodes, the plan and the seed: it cuts
//! by participant group, by a calendar cutoff or by holding one source out,
//! projects each part at the prediction point (a temporal split projects
//! the earlier parts with the cutoff as their administrative end, so nothing
//! of the future is in them), runs the leakage gates over the records it is
//! about to write, and stores the parts and the split under their content
//! addresses. Every participant is accounted for: each is in exactly one part
//! or is counted as left out, with the reason.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::Serialize;
use splinter_core::dataset::DatasetId;
use splinter_core::digest::{canonical_json, Digest};
use splinter_core::longitudinal::ParticipantKeying;
use splinter_core::terms::Terms;
use splinter_data::partition::{partition, PartitionSpec};
use splinter_data::split::{leave_one_source_out, temporal_split, DataSplit, Part};
use splinter_data::timeline_dataset::{
    members, project_split, write_timeline_splits, ProjectionSpec,
};
use splinter_model::BrainDatasetCheck;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_store::longitudinal::{ImportReport, ImportSpec, LongitudinalStore};

use super::train::Scratch;

/// One `import` command.
#[derive(Clone, Debug)]
pub struct ImportRequest {
    /// The record file: one `timeline-v1` line per participant, with
    /// interventions.
    pub file: PathBuf,
    /// The dataset's id: part of every participant key and every item's
    /// provenance.
    pub dataset: String,
    /// The terms the data came under, as its holder states them.
    pub terms: Terms,
    /// The secret the opaque keys are derived with; losing it means a later
    /// import cannot reproduce earlier keys.
    pub secret: Vec<u8>,
}

/// Imports the file as episodes; importing it again adds nothing.
pub fn import_records(
    ctx: &Context,
    request: &ImportRequest,
) -> Result<ImportReport, OrchestratorError> {
    Ok(LongitudinalStore::new(ctx.workspace()).import_jsonl(
        &request.file,
        &ImportSpec {
            dataset: request.dataset.clone(),
            terms: request.terms.clone(),
            keying: ParticipantKeying::from_secret(&request.secret),
        },
    )?)
}

/// How the episodes are cut into parts.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "plan", rename_all = "snake_case")]
pub enum SplitPlan {
    /// By participant group: a locked share is the test, the rest trains and
    /// one fold of `folds` is the early-stopping part.
    Participants {
        /// Share of the units held out as the test.
        locked_share: f64,
        /// Folds the rest is cut into; one is the validation part.
        folds: u32,
    },
    /// Units whose entry is at or before `cutoff` on the calendar train, later
    /// ones test.
    Temporal {
        /// The calendar cutoff.
        cutoff: f64,
        /// Share of the earlier units that validate.
        validation_share: f64,
    },
    /// One whole source is the test.
    LeaveSourceOut {
        /// The held-out source.
        source: String,
        /// Share of the others that validate.
        validation_share: f64,
    },
}

/// One `split` command.
#[derive(Clone, Debug, Serialize)]
pub struct SplitRequest {
    /// How to cut.
    pub plan: SplitPlan,
    /// The seed of every shuffle.
    pub seed: u64,
    /// Where inputs end and outcomes begin.
    pub projection: ProjectionSpec,
}

/// What `split_timeline` stored.
#[derive(Clone, Debug, Serialize)]
pub struct SplitReport {
    /// The split's address.
    pub split: Digest,
    /// The stored training part.
    pub train: DatasetId,
    /// The stored validation part.
    pub validation: DatasetId,
    /// The stored test part.
    pub test: DatasetId,
    /// Participants in the imported dataset.
    pub participants: usize,
    /// Records in each part.
    pub records: BTreeMap<Part, usize>,
    /// Participants the split left out, by reason.
    pub excluded: BTreeMap<String, usize>,
    /// Participants the split assigned to a part whose projection left them out
    /// (no observation window after the prediction point). Every participant is
    /// in one record, or in `excluded`, or here.
    pub unprojected: usize,
}

fn refuse(what: &str, e: impl std::fmt::Display) -> OrchestratorError {
    OrchestratorError::Refused(format!("{what}: {e}"))
}

/// Cuts the imported episodes by `request.plan`, projects and writes each part
/// through the leakage gates and stores the parts and the split.
pub fn split_timeline(
    ctx: &Context,
    request: &SplitRequest,
) -> Result<SplitReport, OrchestratorError> {
    let episodes = LongitudinalStore::new(ctx.workspace());
    let all = members(&episodes, |_| String::new())?;
    if all.is_empty() {
        return Err(OrchestratorError::Refused(
            "no episode is imported: import a record file first".into(),
        ));
    }
    let basis = canonical_json(&episodes.addresses()?)
        .map(|bytes| Digest::of(&bytes))
        .map_err(|e| refuse("the imported episodes have no address", e))?;
    let split: DataSplit = match &request.plan {
        SplitPlan::Participants {
            locked_share,
            folds,
        } => {
            let spec = PartitionSpec {
                locked_share: *locked_share,
                repeats: 1,
                folds: *folds,
                seed: request.seed,
            };
            let units: Vec<_> = all.iter().map(|m| m.unit.clone()).collect();
            let cut = partition(&units, basis, &spec).map_err(|e| refuse("partition", e))?;
            DataSplit::from_partition(&cut, &units, 0, 0).map_err(|e| refuse("split", e))?
        }
        SplitPlan::Temporal {
            cutoff,
            validation_share,
        } => temporal_split(&all, *cutoff, *validation_share, request.seed, &basis)
            .map_err(|e| refuse("temporal split", e))?,
        SplitPlan::LeaveSourceOut {
            source,
            validation_share,
        } => leave_one_source_out(&all, source, *validation_share, request.seed, &basis)
            .map_err(|e| refuse("leave-one-source-out split", e))?,
    };
    split
        .verify_members(&all)
        .map_err(|e| refuse("the split leaks", e))?;
    let parts = project_split(&episodes, &split, &request.projection)?;
    let scratch = Scratch::new(ctx, "split")?;
    let written = write_timeline_splits(&scratch.0, &split, &parts, &BrainDatasetCheck)?;
    let store = ctx.timeline_datasets();
    let address = store.put_split(&split)?;
    let mut stored = BTreeMap::new();
    for (part, dataset) in &written.datasets {
        stored.insert(*part, store.put(dataset)?.id);
    }
    let take = |part: Part| {
        stored.get(&part).cloned().ok_or_else(|| {
            OrchestratorError::Refused(format!(
                "the {} part has no record: the split left nothing to {} on",
                part.name(),
                part.name()
            ))
        })
    };
    let unprojected = parts
        .iter()
        .map(|(part, p)| split.ids(*part).len().saturating_sub(p.records.len()))
        .sum();
    Ok(SplitReport {
        split: address,
        train: take(Part::Train)?,
        validation: take(Part::Validation)?,
        test: take(Part::Test)?,
        participants: all.len(),
        records: parts.iter().map(|(p, d)| (*p, d.records.len())).collect(),
        excluded: split.excluded.clone(),
        unprojected,
    })
}
