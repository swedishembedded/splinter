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
//! and calibration read. The test part is never opened.
//!
//! When the request asks for a calibration, the validation part is divided by
//! participant group (never dividing a group) into the units early stopping
//! reads and the units the calibration is fitted on: brain calibrates on
//! subjects the model was neither trained nor early-stopped on, so the two
//! uses cannot share a unit. The calibration is written beside the weights
//! (`calibration.json`), packed with them, and its digest recorded. The stage refuses datasets that are
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
use splinter_core::digest::{canonical_json, Digest};
use splinter_core::terms::Use;
use splinter_data::holdout::{split_by_rule, Membership, SplitRule};
use splinter_data::partition::Unit;
use splinter_data::split::{verify_disjoint, FitLedger, Part};
use splinter_data::timeline_store::StoredTimeline;
use splinter_model::timeline::probe::{
    max_abs_difference, max_abs_difference_optional, probe_calibrated, probe_values,
};
use splinter_model::timeline::{
    brain_revision, calibrate_timeline, read_jsonl, train_timeline, CalibrationPlan, Subject,
    TimelineError, TimelineModel, TimelineTraining,
};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_store::artifacts::ArtifactSpec;
use splinter_store::bundle::{pack_directory, unpack_directory};

use super::records::{
    CalibrationRecord, Probe, TimelineCandidate, TrainingOutcome, CANDIDATE_CLASS, CANDIDATE_FORMAT,
};

/// The file brain writes a calibration to, beside the weights.
pub const CALIBRATION_FILE: &str = "calibration.json";
/// The file brain records the training support in, beside the weights.
pub const SUPPORT_FILE: &str = "support.json";
/// The share of the validation part's groups the calibration is fitted on;
/// early stopping reads the rest.
pub const CALIBRATION_SHARE: f64 = 0.5;

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
    /// Calibrate the trained model's risks on a share of the validation part
    /// ([`CALIBRATION_SHARE`]) early stopping then does not read; `None`
    /// trains a model that is served, and judged, raw.
    pub calibration: Option<CalibrationPlan>,
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
            calibration: None,
            splinter_commit: None,
        }
    }

    /// The same request, also calibrating the risk by each of `horizons`.
    #[must_use]
    pub fn calibrated_at(mut self, horizons: Vec<f64>) -> Self {
        self.calibration = Some(CalibrationPlan {
            horizons,
            min_events: None,
        });
        self
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
/// model's last knot, and every horizon the calibration is fitted at (a
/// calibrated risk exists only at exactly those).
fn probe_horizons(config: &TimelineTraining, calibration: Option<&CalibrationPlan>) -> Vec<f64> {
    let last = config
        .knots
        .as_ref()
        .and_then(|k| k.last().copied())
        .map_or(1.0, f64::from);
    let mut horizons = vec![last / 4.0, last / 2.0, last];
    horizons.extend(
        calibration
            .into_iter()
            .flat_map(|c| c.horizons.iter().copied()),
    );
    horizons.sort_by(f64::total_cmp);
    horizons.dedup();
    horizons
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

/// The validation part divided into the units early stopping reads and the
/// units a calibration is fitted on, by participant group: a group is never
/// divided, nothing is dropped, and the same units always divide the same way.
pub fn divide_validation(
    held: &[Subject],
) -> Result<(Vec<Subject>, Vec<Subject>), OrchestratorError> {
    let (early, calibration) = split_by_rule(
        held,
        |s| Membership {
            group: s.group_id.clone(),
            examinable: true,
            side: None,
        },
        &SplitRule::monitor(CALIBRATION_SHARE),
    )
    .ok_or_else(|| {
        OrchestratorError::Refused(format!(
            "{} validation unit(s) cannot be divided into early-stopping and calibration units",
            held.len()
        ))
    })?;
    // `split_by_rule` takes the share out as its second half.
    Ok((
        early.into_iter().cloned().collect(),
        calibration.into_iter().cloned().collect(),
    ))
}

/// The units a calibration was fitted on, as one address.
fn units_digest(units: &[Subject]) -> Result<Digest, OrchestratorError> {
    let mut ids: Vec<&str> = units.iter().map(|s| s.subject_id.as_str()).collect();
    ids.sort_unstable();
    let bytes = canonical_json(&ids).map_err(|e| {
        OrchestratorError::Train(format!("the calibration units have no canonical form: {e}"))
    })?;
    Ok(Digest::of(&bytes))
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

    let (early, calibration_units) = match &request.calibration {
        Some(_) => divide_validation(&held)?,
        None => (held.clone(), Vec::new()),
    };
    let mut trained = train_timeline(&train, &early, &request.config).map_err(lift)?;
    let calibration_outcome = request
        .calibration
        .as_ref()
        .map(|plan| calibrate_timeline(&mut trained.model, &calibration_units, plan).map_err(lift))
        .transpose()?;
    let scratch = Scratch::new(ctx, "train")?;
    let saved = scratch.0.join("saved");
    trained.model.save(&saved).map_err(|e| lift(e.into()))?;
    let bundle = scratch.0.join("checkpoint.bundle");
    let packed = pack_directory(&saved, &bundle)?;
    let calibration = calibration_outcome
        .map(|outcome| -> Result<CalibrationRecord, OrchestratorError> {
            let file = saved.join(CALIBRATION_FILE);
            let bytes = std::fs::read(&file).map_err(|source| OrchestratorError::Io {
                path: file.clone(),
                source,
            })?;
            Ok(CalibrationRecord {
                digest: Digest::sha256_of(&bytes),
                outcome,
                early_stopping_units: early.len(),
                units_digest: units_digest(&calibration_units)?,
            })
        })
        .transpose()?;
    for needed in [SUPPORT_FILE]
        .into_iter()
        .chain(calibration.as_ref().map(|_| CALIBRATION_FILE))
    {
        if !packed.files.iter().any(|f| f == needed) {
            return Err(OrchestratorError::Train(format!(
                "the packed checkpoint holds {:?} but not {needed}: a model is shipped with \
                 everything brain saved for it; nothing was kept",
                packed.files
            )));
        }
    }

    let probe_subjects: Vec<Subject> = early
        .iter()
        .take(request.probe_subjects.max(1))
        .cloned()
        .collect();
    let horizons = probe_horizons(&trained.config, request.calibration.as_ref());
    let codes = &trained.config.codes;
    let as_trained =
        probe_values(&trained.model, &probe_subjects, codes, &horizons).map_err(lift)?;
    let reloaded = load_bundle(&bundle, &scratch.0, "round-trip")?;
    let from_file = probe_values(&reloaded, &probe_subjects, codes, &horizons).map_err(lift)?;
    let calibrated_as_trained =
        probe_calibrated(&trained.model, &probe_subjects, codes, &horizons).map_err(lift)?;
    let calibrated_from_file =
        probe_calibrated(&reloaded, &probe_subjects, codes, &horizons).map_err(lift)?;
    let raw_diff = max_abs_difference(&as_trained, &from_file);
    let calibrated_diff =
        max_abs_difference_optional(&calibrated_as_trained, &calibrated_from_file);
    // A mismatch in shape (a calibrated risk one side lacks) is NaN, which
    // `f64::max` would silently drop, so each is judged before they are joined.
    let round_trip = if raw_diff.is_nan() || calibrated_diff.is_nan() {
        f64::NAN
    } else {
        raw_diff.max(calibrated_diff)
    };
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
            calibrated: calibrated_as_trained,
        },
        calibration,
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

#[cfg(test)]
mod tests {
    use super::*;
    use splinter_model::timeline::synthetic;

    fn grouped(n: usize) -> Vec<Subject> {
        let mut subjects = synthetic::population(n, 4).0;
        for (i, s) in subjects.iter_mut().enumerate() {
            s.group_id = Some(format!("household-{}", i / 3));
        }
        subjects
    }

    #[test]
    fn validation_units_divide_by_group_without_loss_or_overlap() {
        let held = grouped(60);
        let (early, calibration) = divide_validation(&held).unwrap();
        assert_eq!(early.len() + calibration.len(), held.len());
        assert!(!early.is_empty() && !calibration.is_empty());
        let groups = |part: &[Subject]| -> std::collections::HashSet<String> {
            part.iter().filter_map(|s| s.group_id.clone()).collect()
        };
        assert!(groups(&early).is_disjoint(&groups(&calibration)));
        let ids = |part: &[Subject]| -> std::collections::HashSet<String> {
            part.iter().map(|s| s.subject_id.clone()).collect()
        };
        assert!(ids(&early).is_disjoint(&ids(&calibration)));
        let (again_early, again_calibration) = divide_validation(&held).unwrap();
        assert_eq!(ids(&early), ids(&again_early));
        assert_eq!(ids(&calibration), ids(&again_calibration));
        assert!(
            calibration.len() * 2 <= held.len(),
            "early stopping keeps at least half"
        );
    }

    #[test]
    fn too_few_validation_units_are_refused_by_name() {
        let why = divide_validation(&grouped(1)).unwrap_err().to_string();
        assert!(why.contains("1 validation unit"), "{why}");
    }
}
