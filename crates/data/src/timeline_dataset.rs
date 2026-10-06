// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for turning longitudinal records
// into participant-safe training datasets with leakage gates, for its
// clients. If your team needs expertise in survival and risk-model data
// pipelines, you can procure our services by sending an email to
// info@swedishembedded.com.

//! `timeline-v1` datasets projected from longitudinal episodes.
//!
//! [`project`] rebuilds one record per episode as it stood at a prediction
//! point ([`ProjectionSpec`]):
//!
//! - inputs are what was known then: observations at or before it, events
//!   strictly before it, and interventions at or before it (as the variable
//!   `intervention:<code>` whose level is `randomised` or `observational`, so
//!   the two are never one flag);
//! - outcomes are events strictly after it that fall inside their observation
//!   window; windows are clipped to open no earlier than the prediction point
//!   (a window that opened later stays delayed, so left truncation is
//!   kept) and to close by the administrative end, so right censoring is
//!   kept; an episode with no window left is excluded and counted;
//! - nothing later than the prediction point is an input, and a later
//!   measurement is not carried as a forecast target;
//! - the subject id is the opaque participant key and the group id the opaque
//!   group key: no raw identifier exists to project;
//! - an absent measurement is absent.
//!
//! The projection is a pure function of the episodes and the spec. It reads
//! the episodes' own event, stream and clock structure through the store; the
//! experience database's world-model windows (`train::world`) anchor on
//! regular raw streams, which irregular visits are not, so the cut is made
//! here on the episode clock instead.
//!
//! [`write_timeline_dataset`] writes a projection as `timeline-v1` with a
//! manifest naming the format, the digest, the spec, the counts, the episodes
//! and source files the records came from, and the terms combined over their
//! sources. [`write_timeline_splits`] writes the parts of a split and runs
//! [`verify_disjoint`] over the records it is about to write first.

use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use serde::{Deserialize, Serialize};
use splinter_core::digest::{canonical_json, Digest};
use splinter_core::longitudinal::{AtRisk, Event, History, Observation, Value};
use splinter_core::terms::Terms;
use splinter_store::longitudinal::LongitudinalStore;

use crate::dataset::{place_dataset, Dataset, DatasetCheck, Format};
use crate::partition::Unit;
use crate::split::{verify_disjoint, DataSplit, LeakageError, Member, Part};
use crate::ViewError;

/// Where the inputs end and the outcomes begin.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "at", rename_all = "snake_case")]
pub enum PredictionPoint {
    /// The record's own entry time.
    AtEntry,
    /// This time on the participant's clock.
    SubjectTime {
        /// The time.
        t: f64,
    },
    /// This time on the calendar, read on each participant's clock through
    /// the clock mapping the episode records.
    CalendarTime {
        /// The calendar time.
        t: f64,
    },
}

/// How a projection cuts each episode.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProjectionSpec {
    /// Where inputs end and outcomes begin.
    pub prediction: PredictionPoint,
    /// A calendar time after which nothing exists for this projection: later
    /// items are dropped and windows are censored there. A model trained
    /// with the cutoff of a temporal split sets it to that cutoff.
    pub administrative_end: Option<f64>,
}

impl ProjectionSpec {
    /// The record's own entry time, with nothing hidden after it.
    #[must_use]
    pub fn at_entry() -> Self {
        Self {
            prediction: PredictionPoint::AtEntry,
            administrative_end: None,
        }
    }
}

/// Why an episode yields no record.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimelineExclusion {
    /// No observation window is open after the prediction point.
    NotAtRiskAfterPrediction,
}

/// An item a projection does not carry, by reason.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Dropped {
    /// A measurement after the prediction point.
    ObservationAfterPrediction,
    /// An intervention after the prediction point.
    InterventionAfterPrediction,
    /// An event after the prediction point outside its observation window.
    EventOutsideWindow,
    /// Anything after the administrative end.
    AfterAdministrativeEnd,
}

/// One line of a `timeline-v1` dataset.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TimelineRecord {
    /// The participant's opaque key.
    pub subject_id: String,
    /// The group's opaque key.
    pub group_id: String,
    /// Sampling weight.
    pub weight: f64,
    /// Where the record came from within its dataset.
    pub source: String,
    /// The prediction time.
    pub entry: f64,
    /// The calendar time at `entry`.
    pub calendar_at_entry: f64,
    /// Inputs: what was known at or before `entry`.
    pub observations: Vec<Observation>,
    /// History before `entry`, outcomes after it.
    pub events: Vec<Event>,
    /// Observation windows for the outcomes.
    pub at_risk: Vec<AtRisk>,
}

impl TimelineRecord {
    /// The record as a unit for splitting: its participant, its group.
    #[must_use]
    pub fn unit(&self, stratum: &str) -> Unit {
        Unit {
            id: self.subject_id.clone(),
            group: self.group_id.clone(),
            stratum: stratum.to_owned(),
        }
    }
}

/// A source file a record came from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SourceFile {
    /// The source dataset's id.
    pub dataset: String,
    /// The digest of the file.
    pub file: Digest,
}

/// A record and where it came from.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectedRecord {
    /// The address of the episode it was projected from.
    pub episode: Digest,
    /// The record.
    pub record: TimelineRecord,
    /// The index of its source file in [`TimelineProjection::sources`].
    pub source: usize,
}

/// The records projected from a set of episodes.
#[derive(Clone, Debug, PartialEq)]
pub struct TimelineProjection {
    /// How they were cut.
    pub spec: ProjectionSpec,
    /// The records, in episode address order.
    pub records: Vec<ProjectedRecord>,
    /// Episodes left out, by reason.
    pub excluded: BTreeMap<TimelineExclusion, usize>,
    /// Items not carried, by reason.
    pub dropped: BTreeMap<Dropped, usize>,
    /// The source files of the records, with the terms each came under.
    pub sources: Vec<(SourceFile, Terms)>,
}

const INTERVENTION_PREFIX: &str = "intervention:";

fn by_time(a: &Observation, b: &Observation) -> std::cmp::Ordering {
    a.t.total_cmp(&b.t).then_with(|| a.var.cmp(&b.var))
}

/// The record `history` projects to under `spec`, or why it has none. Items
/// not carried are counted in `dropped`.
pub fn project_history(
    history: &History,
    spec: &ProjectionSpec,
    dropped: &mut BTreeMap<Dropped, usize>,
) -> Result<TimelineRecord, TimelineExclusion> {
    let offset = history.calendar_at_entry - history.entry;
    let cut = match spec.prediction {
        PredictionPoint::AtEntry => history.entry,
        PredictionPoint::SubjectTime { t } => t,
        PredictionPoint::CalendarTime { t } => t - offset,
    };
    let end = spec.administrative_end.map(|calendar| calendar - offset);
    let mut count = |why: Dropped| *dropped.entry(why).or_default() += 1;
    let beyond_end = |t: f64| end.is_some_and(|e| t > e);

    let mut observations = Vec::new();
    for o in &history.observations {
        if beyond_end(o.t) {
            count(Dropped::AfterAdministrativeEnd);
        } else if o.t <= cut {
            observations.push(o.clone());
        } else {
            count(Dropped::ObservationAfterPrediction);
        }
    }
    for i in &history.interventions {
        if beyond_end(i.t) {
            count(Dropped::AfterAdministrativeEnd);
        } else if i.t <= cut {
            observations.push(Observation {
                t: i.t,
                var: format!("{INTERVENTION_PREFIX}{}", i.code),
                value: Value::Category(i.assignment.label().to_owned()),
            });
        } else {
            count(Dropped::InterventionAfterPrediction);
        }
    }
    observations.sort_by(by_time);

    let windows: Vec<AtRisk> = history
        .at_risk
        .iter()
        .filter_map(|w| {
            let to = end.map_or(w.to, |e| w.to.min(e));
            let from = w.from.max(cut);
            (to > from).then(|| AtRisk {
                code: w.code.clone(),
                from,
                to,
            })
        })
        .collect();
    if windows.is_empty() {
        return Err(TimelineExclusion::NotAtRiskAfterPrediction);
    }
    let window_of = |code: &str| {
        windows
            .iter()
            .find(|w| w.code == code)
            .or_else(|| windows.iter().find(|w| w.code == "*"))
    };

    let mut events = Vec::new();
    for e in &history.events {
        if e.t < cut {
            events.push(e.clone());
        } else if e.t > cut {
            if beyond_end(e.t) {
                count(Dropped::AfterAdministrativeEnd);
            } else if window_of(&e.code).is_some_and(|w| w.from <= e.t && e.t <= w.to) {
                events.push(e.clone());
            } else {
                count(Dropped::EventOutsideWindow);
            }
        }
    }
    events.sort_by(|a, b| a.t.total_cmp(&b.t).then_with(|| a.code.cmp(&b.code)));

    Ok(TimelineRecord {
        subject_id: history.participant.to_string(),
        group_id: history.group.to_string(),
        weight: history.weight,
        source: history.source.clone(),
        entry: cut,
        calendar_at_entry: cut + offset,
        observations,
        events,
        at_risk: windows,
    })
}

/// Projects every episode of `store` under `spec`.
pub fn project(
    store: &LongitudinalStore,
    spec: &ProjectionSpec,
) -> Result<TimelineProjection, ViewError> {
    let mut projection = TimelineProjection {
        spec: *spec,
        records: Vec::new(),
        excluded: BTreeMap::new(),
        dropped: BTreeMap::new(),
        sources: Vec::new(),
    };
    store.for_each_history(|address, history| {
        match project_history(&history, spec, &mut projection.dropped) {
            Ok(record) => {
                let file = SourceFile {
                    dataset: history.provenance.dataset.clone(),
                    file: history.provenance.file.clone(),
                };
                let source = match projection.sources.iter().position(|(f, _)| *f == file) {
                    Some(at) => at,
                    None => {
                        projection.sources.push((file, history.terms.clone()));
                        projection.sources.len() - 1
                    }
                };
                projection.records.push(ProjectedRecord {
                    episode: address.clone(),
                    record,
                    source,
                });
            }
            Err(why) => *projection.excluded.entry(why).or_default() += 1,
        }
        Ok(())
    })?;
    Ok(projection)
}

/// Every episode of `store` as a [`Member`] for the temporal and external
/// splits: its participant, group, source and entry on the calendar, with
/// `stratum` naming the stratum partitions balance.
pub fn members(
    store: &LongitudinalStore,
    stratum: impl Fn(&History) -> String,
) -> Result<Vec<Member>, ViewError> {
    members_of(store, None, stratum)
}

/// The episodes of the one imported dataset `dataset` (every episode when
/// `None`) as [`Member`]s: a store may hold several cohorts, and a split cuts
/// one of them.
pub fn members_of(
    store: &LongitudinalStore,
    dataset: Option<&str>,
    stratum: impl Fn(&History) -> String,
) -> Result<Vec<Member>, ViewError> {
    let mut found = Vec::new();
    store.for_each_history(|_, history| {
        if dataset.is_some_and(|d| history.provenance.dataset != d) {
            return Ok(());
        }
        found.push(Member {
            unit: Unit {
                id: history.participant.to_string(),
                group: history.group.to_string(),
                stratum: stratum(&history),
            },
            source: history.source.clone(),
            at: history.calendar_at_entry,
        });
        Ok(())
    })?;
    Ok(found)
}

impl TimelineProjection {
    /// The records whose subject is one of `ids`, with the sources they
    /// came from.
    #[must_use]
    pub fn select(&self, ids: &HashSet<&str>) -> TimelineProjection {
        let records: Vec<ProjectedRecord> = self
            .records
            .iter()
            .filter(|r| ids.contains(r.record.subject_id.as_str()))
            .cloned()
            .collect();
        let used: HashSet<usize> = records.iter().map(|r| r.source).collect();
        let mut remap = BTreeMap::new();
        let mut sources = Vec::new();
        for (at, source) in self.sources.iter().enumerate() {
            if used.contains(&at) {
                remap.insert(at, sources.len());
                sources.push(source.clone());
            }
        }
        TimelineProjection {
            spec: self.spec,
            records: records
                .into_iter()
                .map(|mut r| {
                    r.source = remap[&r.source];
                    r
                })
                .collect(),
            excluded: self.excluded.clone(),
            dropped: self.dropped.clone(),
            sources,
        }
    }
}

/// The projection of each part of `split`: for a temporal split, training
/// and validation records are cut with the cutoff as administrative end (no
/// training unit's outcomes or inputs reach past it) and test records are
/// not; for any other split every part is cut by `spec`.
pub fn project_split(
    store: &LongitudinalStore,
    split: &DataSplit,
    spec: &ProjectionSpec,
) -> Result<BTreeMap<Part, TimelineProjection>, ViewError> {
    let cut = |end: Option<f64>| ProjectionSpec {
        administrative_end: end.or(spec.administrative_end),
        ..*spec
    };
    let (early, late) = match split.kind {
        crate::split::SplitKind::Temporal { cutoff } => {
            (project(store, &cut(Some(cutoff)))?, project(store, spec)?)
        }
        _ => {
            let all = project(store, spec)?;
            (all.clone(), all)
        }
    };
    Ok(Part::ALL
        .into_iter()
        .map(|part| {
            let ids: HashSet<&str> = split.ids(part).iter().map(String::as_str).collect();
            let from = if part == Part::Test { &late } else { &early };
            (part, from.select(&ids))
        })
        .collect())
}

/// What a `timeline-v1` dataset holds and where it came from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TimelineManifest {
    /// Always `timeline-v1`.
    pub format: Format,
    /// The digest of the dataset's bytes.
    pub dataset: Digest,
    /// The part of a split this is, if it is one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub part: Option<Part>,
    /// The address of the split it is a part of.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub split: Option<Digest>,
    /// How the records were cut.
    pub projection: ProjectionSpec,
    /// Records written.
    pub records: usize,
    /// Episodes the whole projection left out, by reason.
    pub excluded: BTreeMap<TimelineExclusion, usize>,
    /// Items the whole projection did not carry, by reason.
    pub dropped: BTreeMap<Dropped, usize>,
    /// The episodes the records were projected from, in file order.
    pub episodes: Vec<Digest>,
    /// The source files the records came from.
    pub sources: Vec<SourceFile>,
    /// The terms combined over the sources (the most restrictive of each
    /// axis, a source stating none counting as unknown); a consumer treats
    /// them as the terms of every record.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terms: Option<Terms>,
}

fn manifest(
    projection: &TimelineProjection,
    dataset: &Digest,
    part: Option<Part>,
    split: Option<&Digest>,
) -> TimelineManifest {
    let terms: Vec<Terms> = projection.sources.iter().map(|(_, t)| t.clone()).collect();
    TimelineManifest {
        format: Format::TimelineV1,
        dataset: dataset.clone(),
        part,
        split: split.cloned(),
        projection: projection.spec,
        records: projection.records.len(),
        excluded: projection.excluded.clone(),
        dropped: projection.dropped.clone(),
        episodes: projection
            .records
            .iter()
            .map(|r| r.episode.clone())
            .collect(),
        sources: projection.sources.iter().map(|(f, _)| f.clone()).collect(),
        terms: Terms::combine(&terms),
    }
}

/// Writes `projection` to `path` as `timeline-v1`, and its
/// [`TimelineManifest`] beside it. Refuses an empty projection and a file
/// `check` refuses; in every case nothing is left at `path`.
pub fn write_timeline_dataset(
    path: &Path,
    projection: &TimelineProjection,
    check: &dyn DatasetCheck,
) -> Result<Dataset, ViewError> {
    write_part(path, projection, check, None, None)
}

fn write_part(
    path: &Path,
    projection: &TimelineProjection,
    check: &dyn DatasetCheck,
    part: Option<Part>,
    split: Option<&Digest>,
) -> Result<Dataset, ViewError> {
    if projection.records.is_empty() {
        return Err(ViewError::Empty);
    }
    let mut text = String::new();
    for r in &projection.records {
        text.push_str(&serde_json::to_string(&r.record)?);
        text.push('\n');
    }
    let (digest, manifest) = place_dataset(
        path,
        &text,
        Format::TimelineV1,
        projection.records.len(),
        check,
        |digest| Ok(canonical_json(&manifest(projection, digest, part, split))?),
    )?;
    Ok(Dataset {
        path: path.to_path_buf(),
        format: Format::TimelineV1,
        digest,
        records: projection.records.len(),
        trained_messages: None,
        manifest,
    })
}

/// The files of a written split.
#[derive(Clone, Debug, PartialEq)]
pub struct WrittenSplit {
    /// The split's content address.
    pub split: Digest,
    /// Where the split's own manifest is.
    pub manifest: std::path::PathBuf,
    /// The dataset of each part that has records.
    pub datasets: BTreeMap<Part, Dataset>,
}

/// Writes each part of `split` into `dir` as `<part>.jsonl` with its
/// manifest, and the split's own manifest as `split.json`. Before writing
/// anything it runs [`verify_disjoint`] over the groups of the records it is
/// about to write, and refuses a record in a part its split does not assign
/// it to: a leak fails the write with counts and leaves no file behind.
pub fn write_timeline_splits(
    dir: &Path,
    split: &DataSplit,
    parts: &BTreeMap<Part, TimelineProjection>,
    check: &dyn DatasetCheck,
) -> Result<WrittenSplit, ViewError> {
    let units: Vec<(Part, Vec<Unit>)> = parts
        .iter()
        .map(|(part, p)| (*part, p.records.iter().map(|r| r.record.unit("")).collect()))
        .collect();
    let borrowed: Vec<(Part, &[Unit])> = units.iter().map(|(p, u)| (*p, u.as_slice())).collect();
    verify_disjoint(&borrowed)?;
    let misplaced = parts
        .iter()
        .map(|(part, p)| {
            let assigned: HashSet<&str> = split.ids(*part).iter().map(String::as_str).collect();
            p.records
                .iter()
                .filter(|r| !assigned.contains(r.record.subject_id.as_str()))
                .count()
        })
        .sum::<usize>();
    if misplaced > 0 {
        return Err(LeakageError::NotInPart { units: misplaced }.into());
    }
    let address = split.digest()?;
    let mut datasets = BTreeMap::new();
    for (part, projection) in parts {
        if projection.records.is_empty() {
            continue;
        }
        let path = dir.join(format!("{}.jsonl", part.name()));
        datasets.insert(
            *part,
            write_part(&path, projection, check, Some(*part), Some(&address))?,
        );
    }
    let manifest = dir.join("split.json");
    std::fs::write(&manifest, split.canonical()?).map_err(|source| ViewError::Io {
        path: manifest.clone(),
        source,
    })?;
    Ok(WrittenSplit {
        split: address,
        manifest,
        datasets,
    })
}
