// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible training of risk models on
// participant-safe longitudinal data, with the checkpoint, the split and the
// source files tied together by digest, for its clients. If your team needs
// expertise in training audited time-to-event models, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The train stage for `timeline-v1` datasets.
//!
//! A candidate is trained from two stored parts of one split: the training
//! part, which the model (its vocabulary, its value normalisation and its
//! weights) is fitted on, and the validation part, which only early stopping
//! reads. The test part is never opened. The stage refuses datasets that are
//! not parts of one split, a training part that shares a group with the
//! validation part, and terms that do not permit training; it certifies,
//! with the split's own ledger, that what was fitted consumed training units
//! only. The training algorithm is brain's: this stage hands brain the
//! subjects and the configuration ([`splinter_model::timeline`]).
//!
//! The trained directory is packed deterministically into one file
//! ([`splinter_store::bundle`]) and the stage proves the round trip before it
//! keeps anything: the file is unpacked, loaded by plain brain and must
//! predict what the model as trained predicted. The file is kept as an
//! immutable content-addressed artifact and its record - configuration,
//! split, fit certificate, episodes, source files, seed, commits, the probe a
//! release is later held to - as an immutable document. A candidate is never
//! written over: training again makes another record, or finds the same one.

use std::path::{Path, PathBuf};

use serde::Serialize;
use splinter_core::terms::Use;
use splinter_data::partition::Unit;
use splinter_data::split::{verify_disjoint, FitLedger, Part};
use splinter_data::timeline_store::StoredTimeline;
use splinter_model::timeline::probe::{max_abs_difference, probe_values};
use splinter_model::timeline::{
    brain_revision, read_jsonl, train_timeline, Subject, TimelineError, TimelineModel,
    TimelineTraining,
};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_store::artifacts::ArtifactSpec;
use splinter_store::bundle::{pack_directory, unpack_directory};

use super::records::{
    Probe, TimelineCandidate, TrainingOutcome, CANDIDATE_CLASS, CANDIDATE_FORMAT,
};

/// How many early-stopping subjects the probe a release is held to is taken on.
pub const DEFAULT_PROBE_SUBJECTS: usize = 16;
/// The most the packed file may differ from the model as trained, on the
/// probe: the round trip is exact up to float printing, never up to a model.
pub const ROUND_TRIP_TOLERANCE: f64 = 1e-6;

/// One `train_timeline` command.
#[derive(Clone, Debug, Serialize)]
pub struct TimelineTrainRequest {
    /// The stored training part, by id or unique prefix.
    pub train: String,
    /// The stored validation part early stopping reads, by id or prefix.
    pub held_out: String,
    /// What to train; knots are derived from the training outcome times when
    /// it names none.
    pub config: TimelineTraining,
    /// Early-stopping subjects the probe is taken on.
    pub probe_subjects: usize,
    /// The commit of Splinter that is training, as the caller knows it.
    pub splinter_commit: Option<String>,
}

impl TimelineTrainRequest {
    /// A request to train `config` on the stored parts `train` and `held_out`.
    #[must_use]
    pub fn new(train: &str, held_out: &str, config: TimelineTraining) -> Self {
        Self {
            train: train.into(),
            held_out: held_out.into(),
            config,
            probe_subjects: DEFAULT_PROBE_SUBJECTS,
            splinter_commit: None,
        }
    }
}

/// What `train_timeline_candidate` reports.
#[derive(Clone, Debug, Serialize)]
pub struct TimelineTrained {
    /// The candidate's id.
    pub id: String,
    /// Its record.
    pub candidate: TimelineCandidate,
    /// The packed checkpoint, a real file in the artifact store.
    pub checkpoint: PathBuf,
}

/// Maps a timeline failure onto the command error it is.
pub(crate) fn lift(e: TimelineError) -> OrchestratorError {
    match e {
        TimelineError::Request(why) => OrchestratorError::Refused(why),
        other => OrchestratorError::Train(other.to_string()),
    }
}

fn units(subjects: &[Subject]) -> Vec<Unit> {
    subjects
        .iter()
        .map(|s| Unit {
            id: s.subject_id.clone(),
            group: s.group_id.clone().unwrap_or_else(|| s.subject_id.clone()),
            stratum: String::new(),
        })
        .collect()
}

fn part_of(ctx: &Context, given: &str, part: Part) -> Result<StoredTimeline, OrchestratorError> {
    let store = ctx.timeline_datasets();
    let id = splinter_core::dataset::DatasetId(splinter_orchestrator::ids::resolve(
        "timeline dataset",
        given,
        store.list()?.into_iter().map(|d| d.0),
    )?);
    let stored = store.get(&id)?;
    if stored.manifest.part != Some(part) {
        return Err(OrchestratorError::Refused(format!(
            "dataset {id} is the {:?} part of its split, not the {part:?} part this was asked to read",
            stored.manifest.part
        )));
    }
    Ok(stored)
}

/// A scratch directory under the state root's work area, removed on drop.
pub(crate) struct Scratch(pub(crate) PathBuf);

impl Scratch {
    pub(crate) fn new(ctx: &Context, what: &str) -> Result<Self, OrchestratorError> {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let dir = ctx.root().work().join("timeline").join(format!(
            "{what}-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).map_err(|source| OrchestratorError::Io {
            path: dir.clone(),
            source,
        })?;
        Ok(Self(dir))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The horizons the probe is taken at: a quarter, a half and all of the
/// model's last knot.
fn probe_horizons(config: &TimelineTraining) -> Vec<f64> {
    let last = config
        .knots
        .as_ref()
        .and_then(|k| k.last().copied())
        .map_or(1.0, f64::from);
    vec![last / 4.0, last / 2.0, last]
}

/// Loads the model packed in `bundle` the way a consumer does: unpacked into
/// a new directory under `scratch`, then loaded by plain brain.
pub fn load_bundle(
    bundle: &Path,
    scratch: &Path,
    name: &str,
) -> Result<TimelineModel, OrchestratorError> {
    let dir = scratch.join(name);
    unpack_directory(bundle, &dir)?;
    TimelineModel::load(&dir).map_err(|e| lift(e.into()))
}

/// Trains a timeline candidate from stored parts and keeps it. Nothing is
/// kept when any check fails.
pub fn train_timeline_candidate(
    ctx: &Context,
    request: &TimelineTrainRequest,
) -> Result<TimelineTrained, OrchestratorError> {
    let train_part = part_of(ctx, &request.train, Part::Train)?;
    let held_part = part_of(ctx, &request.held_out, Part::Validation)?;
    let (Some(split_a), Some(split_b)) = (&train_part.manifest.split, &held_part.manifest.split)
    else {
        return Err(OrchestratorError::Refused(
            "a part that names no split cannot be trained from: the split is what certifies the fit".into(),
        ));
    };
    if split_a != split_b {
        return Err(OrchestratorError::Refused(format!(
            "the training part is of split {split_a} but the validation part is of split {split_b}"
        )));
    }
    let split = ctx.timeline_datasets().split(split_a)?;
    let terms = splinter_core::terms::combine_stated([
        train_part.manifest.terms.as_ref(),
        held_part.manifest.terms.as_ref(),
    ]);
    let terms_ok = terms
        .as_ref()
        .ok_or_else(|| {
            "no terms were recorded for the data: declare them when it is imported".to_string()
        })
        .and_then(|t| t.permits(Use::Training));
    if let Err(why) = terms_ok {
        return Err(OrchestratorError::Refused(format!(
            "training is refused: {why}"
        )));
    }
    let train = read_jsonl(&train_part.path).map_err(|e| lift(e.into()))?;
    let held = read_jsonl(&held_part.path).map_err(|e| lift(e.into()))?;
    for (name, subjects, part) in [
        ("training", &train, Part::Train),
        ("validation", &held, Part::Validation),
    ] {
        let assigned: std::collections::HashSet<&str> =
            split.ids(part).iter().map(String::as_str).collect();
        let strays = subjects
            .iter()
            .filter(|s| !assigned.contains(s.subject_id.as_str()))
            .count();
        if strays > 0 {
            return Err(OrchestratorError::Refused(format!(
                "{strays} subject(s) of the {name} dataset are not in the {part:?} part of split {split_a}"
            )));
        }
    }
    let (train_units, held_units) = (units(&train), units(&held));
    verify_disjoint(&[(Part::Train, &train_units), (Part::Validation, &held_units)])
        .map_err(|e| OrchestratorError::Refused(format!("the parts share a group: {e}")))?;

    // The ledger names every unit brain's fit reads: exactly the slice it is
    // handed as training data, and nothing else is.
    let mut ledger = FitLedger::new();
    for s in &train {
        ledger.consume(&s.subject_id);
    }
    let fit = split
        .certify_fit(&ledger)
        .map_err(|e| OrchestratorError::Refused(format!("the fit is not certified: {e}")))?;

    let trained = train_timeline(&train, &held, &request.config).map_err(lift)?;
    let scratch = Scratch::new(ctx, "train")?;
    let saved = scratch.0.join("saved");
    trained.model.save(&saved).map_err(|e| lift(e.into()))?;
    let bundle = scratch.0.join("checkpoint.bundle");
    let packed = pack_directory(&saved, &bundle)?;

    let probe_subjects: Vec<Subject> = held
        .iter()
        .take(request.probe_subjects.max(1))
        .cloned()
        .collect();
    let horizons = probe_horizons(&trained.config);
    let codes = &trained.config.codes;
    let as_trained =
        probe_values(&trained.model, &probe_subjects, codes, &horizons).map_err(lift)?;
    let reloaded = load_bundle(&bundle, &scratch.0, "round-trip")?;
    let from_file = probe_values(&reloaded, &probe_subjects, codes, &horizons).map_err(lift)?;
    let round_trip = max_abs_difference(&as_trained, &from_file);
    if round_trip.is_nan() || round_trip > ROUND_TRIP_TOLERANCE {
        return Err(OrchestratorError::Train(format!(
            "the packed checkpoint predicts differently from the model as trained (largest \
             difference {round_trip}, tolerance {ROUND_TRIP_TOLERANCE}); nothing was kept"
        )));
    }

    let kept = ctx.artifacts().put_file(
        &bundle,
        &ArtifactSpec::new("checkpoint", "splinter-timeline")
            .with_extension(".bundle")
            .with_sha256(),
    )?;
    let candidate = TimelineCandidate {
        format: CANDIDATE_FORMAT.into(),
        config_digest: trained.config.digest().map_err(lift)?,
        seed: trained.config.seed,
        config: trained.config.clone(),
        train: train_part.id.clone(),
        held_out: held_part.id.clone(),
        split: split_a.clone(),
        fit,
        episodes: train_part.manifest.episodes.clone(),
        sources: train_part.manifest.sources.clone(),
        terms,
        checkpoint: kept.digest.clone(),
        checkpoint_sha256: packed.sha256,
        files: packed.files,
        outcome: TrainingOutcome {
            steps: trained.report.steps,
            initial_loss: trained.report.initial_loss,
            final_loss: trained.report.final_loss,
            held_out_event_nll: trained.report.held_out_event_nll,
            parameters: trained.report.parameters,
            truncated_tokens: trained.report.truncated_tokens,
        },
        probe: Probe {
            subjects: probe_subjects
                .iter()
                .map(|s| s.subject_id.clone())
                .collect(),
            horizons,
            values: as_trained,
        },
        round_trip_max_abs_diff: round_trip,
        brain_commit: brain_revision().map(str::to_owned),
        splinter_commit: request.splinter_commit.clone(),
    };
    let id = candidate.id().map_err(|e| {
        OrchestratorError::Train(format!("the candidate has no canonical form: {e}"))
    })?;
    ctx.workspace().put_document(CANDIDATE_CLASS, &candidate)?;
    Ok(TimelineTrained {
        id,
        checkpoint: ctx.artifacts().path(&kept.digest)?,
        candidate,
    })
}

/// The candidate `id` (or a unique prefix of it) names.
pub fn load_timeline_candidate(
    ctx: &Context,
    id: &str,
) -> Result<(String, TimelineCandidate), OrchestratorError> {
    let wanted = id.strip_prefix("tl-").unwrap_or(id);
    let ids = ctx.workspace().document_ids(CANDIDATE_CLASS)?;
    let mut found = ids.iter().filter(|d| d.hex().starts_with(wanted));
    let digest = found.next().ok_or_else(|| OrchestratorError::NotFound {
        what: "timeline candidate",
        id: id.to_string(),
    })?;
    let more = found.count();
    if more > 0 || wanted.len() < 4 {
        return Err(OrchestratorError::AmbiguousId {
            what: "timeline candidate",
            id: id.to_string(),
            matches: more + 1,
        });
    }
    let candidate: TimelineCandidate = ctx
        .workspace()
        .get_document(CANDIDATE_CLASS, digest)?
        .ok_or_else(|| OrchestratorError::NotFound {
            what: "timeline candidate",
            id: id.to_string(),
        })?;
    let id = candidate.id().map_err(|e| {
        OrchestratorError::Train(format!("the candidate has no canonical form: {e}"))
    })?;
    Ok((id, candidate))
}
