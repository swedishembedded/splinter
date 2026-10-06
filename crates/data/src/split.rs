// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for leakage-free evaluation of
// models trained on longitudinal data, for its clients. If your team needs
// expertise in train/test hygiene, temporal validation or external
// validation designs, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Splits of units into training, validation and test parts, and the gates
//! that catch what leaks between them.
//!
//! A *unit* is one record; a *group* is what must never be divided (a person
//! seen twice, a household, a site). A [`DataSplit`] is a pure function of
//! its units, the dataset digest and a seed, has a content address, and is
//! verified before it is returned:
//!
//! - [`DataSplit::from_partition`]: participant-exclusive, from a
//!   [`Partition`](crate::partition::Partition) (the locked test is `Test`,
//!   one fold is `Validation`).
//! - [`temporal_split`]: units whose time is at or before a cutoff train
//!   (a seeded share of them validate), later units test. A group with
//!   members on both sides is left out and counted, because keeping it would
//!   put one group in two parts.
//! - [`leave_one_source_out`]: one whole source is the test; no other unit
//!   shares a group with it.
//!
//! The gates fail with counts, never with the identities involved:
//! [`verify_disjoint`] (a group in more than one part), the checks a split of
//! each kind runs on its members, and [`DataSplit::certify_fit`], which
//! refuses statistics (a normalisation, a vocabulary) that consumed any unit
//! outside the training part.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use serde::{Deserialize, Serialize};
use splinter_core::digest::{canonical_json, Digest};
use thiserror::Error;

use crate::partition::{partition, Partition, PartitionError, PartitionSpec, Unit};

/// A part of a split.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Part {
    /// What a model is fitted on.
    Train,
    /// What model choices are made on.
    Validation,
    /// What a claim is measured on.
    Test,
}

impl Part {
    /// Every part, training first.
    pub const ALL: [Part; 3] = [Part::Train, Part::Validation, Part::Test];

    /// The part's name, as file names and manifests spell it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Part::Train => "train",
            Part::Validation => "validation",
            Part::Test => "test",
        }
    }
}

/// A unit with what the temporal and external splits need to know of it.
#[derive(Clone, Debug, PartialEq)]
pub struct Member {
    /// The unit: id, group and stratum.
    pub unit: Unit,
    /// The source the unit came from.
    pub source: String,
    /// When the unit's prediction time falls on the dataset's shared calendar.
    pub at: f64,
}

/// How a split was made.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SplitKind {
    /// No group in two parts, from one fold of a partition.
    ParticipantExclusive {
        /// The cross-validation repeat.
        repeat: usize,
        /// The fold held out as validation.
        fold: usize,
    },
    /// Training at or before `cutoff`, testing after it.
    Temporal {
        /// The calendar time of the cutoff.
        cutoff: f64,
    },
    /// One source held out whole.
    LeaveOneSourceOut {
        /// The held-out source.
        held_out: String,
    },
}

/// A split: which units are in which part, and how that was decided.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DataSplit {
    /// How the split was made.
    pub kind: SplitKind,
    /// The seed of every shuffle in it.
    pub seed: u64,
    /// The dataset the units came from.
    pub dataset: Digest,
    /// The unit ids of each part, sorted; every part is present.
    pub parts: BTreeMap<Part, Vec<String>>,
    /// Units left out of every part, by reason.
    pub excluded: BTreeMap<String, usize>,
}

/// Why a gate refused. Each carries counts only: a count is enough to act
/// on and names nobody.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum LeakageError {
    /// A group (or a unit) is in more than one part.
    #[error("{groups} group(s) and {units} unit(s) appear in more than one of train, validation and test")]
    SharedGroups {
        /// Groups in more than one part.
        groups: usize,
        /// Unit ids in more than one part.
        units: usize,
    },
    /// A part names units the dataset does not have.
    #[error("{units} unit(s) of the split are not in the dataset")]
    UnknownUnits {
        /// How many.
        units: usize,
    },
    /// A unit of the held-out source is in training or validation, or a
    /// unit of another source is in the test.
    #[error("the held-out source is not held out whole: {units} unit(s) are on the wrong side")]
    SourceNotHeldOut {
        /// How many units are on the wrong side.
        units: usize,
    },
    /// A training or validation unit is later than the cutoff, or a test
    /// unit is not.
    #[error("{units} unit(s) are on the wrong side of the temporal cutoff")]
    WrongSideOfCutoff {
        /// How many units are on the wrong side.
        units: usize,
    },
    /// Records are written to a part their split does not assign them to.
    #[error("{units} record(s) are written to a part their split does not assign them to")]
    NotInPart {
        /// How many.
        units: usize,
    },
    /// Statistics consumed units outside the training part.
    #[error("statistics were fitted on {units} unit(s) outside the training part ({held_out} of them in validation or test)")]
    FittedOutsideTraining {
        /// Units consumed that are not training units.
        units: usize,
        /// Of those, units that are in validation or test.
        held_out: usize,
    },
    /// Statistics that consumed nothing.
    #[error("the statistics consumed no units, so nothing shows what they were fitted on")]
    FittedOnNothing,
}

/// Why a split could not be made.
#[derive(Debug, Error, PartialEq)]
pub enum SplitError {
    /// The partition underneath was refused.
    #[error(transparent)]
    Partition(#[from] PartitionError),
    /// The split it produced fails a gate.
    #[error(transparent)]
    Leakage(#[from] LeakageError),
    /// The request cannot be met.
    #[error("invalid split: {0}")]
    Spec(String),
}

/// Fails, with counts, when a group (or a unit id) appears in more than one
/// of `parts`. Passing means a model trained on one part has seen nothing of
/// any group it is scored on in another.
pub fn verify_disjoint(parts: &[(Part, &[Unit])]) -> Result<(), LeakageError> {
    let mut group_part: HashMap<&str, Part> = HashMap::new();
    let mut unit_part: HashMap<&str, Part> = HashMap::new();
    let (mut groups, mut units) = (HashSet::new(), HashSet::new());
    for (part, members) in parts {
        for u in *members {
            match group_part.get(u.group.as_str()) {
                Some(held) if held != part => {
                    groups.insert(u.group.as_str());
                }
                Some(_) => {}
                None => {
                    group_part.insert(&u.group, *part);
                }
            }
            match unit_part.get(u.id.as_str()) {
                Some(held) if held != part => {
                    units.insert(u.id.as_str());
                }
                Some(_) => {}
                None => {
                    unit_part.insert(&u.id, *part);
                }
            }
        }
    }
    if groups.is_empty() && units.is_empty() {
        Ok(())
    } else {
        Err(LeakageError::SharedGroups {
            groups: groups.len(),
            units: units.len(),
        })
    }
}

/// What statistics were fitted on, as the fitting code reports it: every
/// unit it read, once each.
#[derive(Clone, Debug, Default)]
pub struct FitLedger {
    units: BTreeSet<String>,
}

impl FitLedger {
    /// A ledger with nothing consumed.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records that a fit read the unit `id`.
    pub fn consume(&mut self, id: &str) {
        self.units.insert(id.to_owned());
    }

    /// How many distinct units were consumed.
    #[must_use]
    pub fn len(&self) -> usize {
        self.units.len()
    }

    /// Whether nothing was consumed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.units.is_empty()
    }
}

/// Proof that statistics were fitted on training units only: kept beside
/// the fitted values.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FitCertificate {
    /// The split the statistics were checked against.
    pub split: Digest,
    /// How many units were consumed.
    pub units: usize,
    /// The address of the consumed unit ids.
    pub fitted_on: Digest,
}

impl DataSplit {
    /// The ids in `part`.
    #[must_use]
    pub fn ids(&self, part: Part) -> &[String] {
        self.parts.get(&part).map_or(&[], Vec::as_slice)
    }

    /// The canonical JSON the split's address is computed over.
    pub fn canonical(&self) -> Result<Vec<u8>, serde_json::Error> {
        canonical_json(self)
    }

    /// The split's content address.
    pub fn digest(&self) -> Result<Digest, serde_json::Error> {
        Ok(Digest::of(&self.canonical()?))
    }

    fn part_of(&self) -> HashMap<&str, Part> {
        Part::ALL
            .iter()
            .flat_map(|p| self.ids(*p).iter().map(move |id| (id.as_str(), *p)))
            .collect()
    }

    /// The gate over the units' groups: every id of every part is a unit of
    /// `units`, and no group is in two parts.
    pub fn verify(&self, units: &[Unit]) -> Result<(), LeakageError> {
        let by_id: HashMap<&str, &Unit> = units.iter().map(|u| (u.id.as_str(), u)).collect();
        let mut unknown = 0;
        let mut resolved: Vec<(Part, Vec<Unit>)> = Vec::new();
        for part in Part::ALL {
            let mut members = Vec::new();
            for id in self.ids(part) {
                match by_id.get(id.as_str()) {
                    Some(u) => members.push((*u).clone()),
                    None => unknown += 1,
                }
            }
            resolved.push((part, members));
        }
        if unknown > 0 {
            return Err(LeakageError::UnknownUnits { units: unknown });
        }
        let borrowed: Vec<(Part, &[Unit])> =
            resolved.iter().map(|(p, u)| (*p, u.as_slice())).collect();
        verify_disjoint(&borrowed)
    }

    /// The gate over members: [`DataSplit::verify`], and the check its kind
    /// adds (a held-out source is held out whole; no training unit is after
    /// a temporal cutoff and no test unit is at or before it).
    pub fn verify_members(&self, members: &[Member]) -> Result<(), LeakageError> {
        let units: Vec<Unit> = members.iter().map(|m| m.unit.clone()).collect();
        self.verify(&units)?;
        let part_of = self.part_of();
        let wrong = |is_wrong: &dyn Fn(Part, &Member) -> bool| {
            members
                .iter()
                .filter(|m| {
                    part_of
                        .get(m.unit.id.as_str())
                        .is_some_and(|part| is_wrong(*part, m))
                })
                .count()
        };
        match &self.kind {
            SplitKind::ParticipantExclusive { .. } => {}
            SplitKind::LeaveOneSourceOut { held_out } => {
                let units = wrong(&|part, m| (part == Part::Test) != (&m.source == held_out));
                if units > 0 {
                    return Err(LeakageError::SourceNotHeldOut { units });
                }
            }
            SplitKind::Temporal { cutoff } => {
                let units = wrong(&|part, m| (part == Part::Test) != (m.at > *cutoff));
                if units > 0 {
                    return Err(LeakageError::WrongSideOfCutoff { units });
                }
            }
        }
        Ok(())
    }

    /// Certifies that the statistics `ledger` describes were fitted on
    /// training units only. Fails with the counts of consumed units outside
    /// the training part, or when nothing was consumed.
    pub fn certify_fit(&self, ledger: &FitLedger) -> Result<FitCertificate, LeakageError> {
        if ledger.is_empty() {
            return Err(LeakageError::FittedOnNothing);
        }
        let training: HashSet<&str> = self.ids(Part::Train).iter().map(String::as_str).collect();
        let outside: Vec<&String> = ledger
            .units
            .iter()
            .filter(|id| !training.contains(id.as_str()))
            .collect();
        if !outside.is_empty() {
            let part_of = self.part_of();
            let held_out = outside
                .iter()
                .filter(|id| part_of.contains_key(id.as_str()))
                .count();
            return Err(LeakageError::FittedOutsideTraining {
                units: outside.len(),
                held_out,
            });
        }
        let ids: Vec<&String> = ledger.units.iter().collect();
        let fitted_on = serde_json::to_vec(&ids)
            .map(|bytes| Digest::of(&bytes))
            .unwrap_or_else(|_| Digest::of(b""));
        Ok(FitCertificate {
            split: self.digest().unwrap_or_else(|_| Digest::of(b"")),
            units: ledger.len(),
            fitted_on,
        })
    }

    /// The participant-exclusive split of one fold of `partition`: its
    /// locked test is `Test`, fold `fold` of repeat `repeat` is `Validation`
    /// and the other folds are `Train`.
    pub fn from_partition(
        partition: &Partition,
        units: &[Unit],
        repeat: usize,
        fold: usize,
    ) -> Result<DataSplit, SplitError> {
        if repeat >= partition.repeats.len() || fold >= partition.spec.folds as usize {
            return Err(SplitError::Spec(format!(
                "repeat {repeat} fold {fold} is outside the {} repeats of {} folds",
                partition.repeats.len(),
                partition.spec.folds
            )));
        }
        let (train, validation) = partition.fold(repeat, fold);
        let split = DataSplit {
            kind: SplitKind::ParticipantExclusive { repeat, fold },
            seed: partition.spec.seed,
            dataset: partition.dataset.clone(),
            parts: sorted_parts(
                train.into_iter().map(String::from).collect(),
                validation.into_iter().map(String::from).collect(),
                partition.locked.clone(),
            ),
            excluded: BTreeMap::new(),
        };
        split.verify(units)?;
        Ok(split)
    }
}

fn sorted_parts(
    mut train: Vec<String>,
    mut validation: Vec<String>,
    mut test: Vec<String>,
) -> BTreeMap<Part, Vec<String>> {
    for ids in [&mut train, &mut validation, &mut test] {
        ids.sort();
    }
    BTreeMap::from([
        (Part::Train, train),
        (Part::Validation, validation),
        (Part::Test, test),
    ])
}

/// Divides `pool` into training and validation by group, `validation_share`
/// of each stratum to validation, as a pure function of the seed.
fn carve_validation(
    pool: &[&Member],
    dataset: &Digest,
    validation_share: f64,
    seed: u64,
) -> Result<(Vec<String>, Vec<String>), SplitError> {
    let units: Vec<Unit> = pool.iter().map(|m| m.unit.clone()).collect();
    let spec = PartitionSpec {
        locked_share: validation_share,
        repeats: 1,
        folds: 2,
        seed,
    };
    let carved = partition(&units, dataset.clone(), &spec)?;
    let validation: HashSet<&str> = carved.locked.iter().map(String::as_str).collect();
    let train = units
        .iter()
        .filter(|u| !validation.contains(u.id.as_str()))
        .map(|u| u.id.clone())
        .collect();
    Ok((train, carved.locked))
}

/// The groups with members on both sides of `in_a`, and how many members
/// those groups have in all.
fn straddling<'a>(
    members: &'a [Member],
    in_a: &dyn Fn(&Member) -> bool,
) -> (HashSet<&'a str>, usize) {
    let mut sides: HashMap<&str, (bool, bool)> = HashMap::new();
    for m in members {
        let entry = sides.entry(m.unit.group.as_str()).or_default();
        if in_a(m) {
            entry.0 = true;
        } else {
            entry.1 = true;
        }
    }
    let groups: HashSet<&str> = sides
        .into_iter()
        .filter(|(_, (a, b))| *a && *b)
        .map(|(g, _)| g)
        .collect();
    let units = members
        .iter()
        .filter(|m| groups.contains(m.unit.group.as_str()))
        .count();
    (groups, units)
}

/// `split`, once it passes the gates for its members.
fn verified(split: DataSplit, members: &[Member]) -> Result<DataSplit, SplitError> {
    split.verify_members(members)?;
    Ok(split)
}

fn check_share(validation_share: f64) -> Result<(), SplitError> {
    if (0.0..1.0).contains(&validation_share) {
        Ok(())
    } else {
        Err(SplitError::Spec(format!(
            "the validation share must be in [0, 1), got {validation_share}"
        )))
    }
}

/// The temporal split: members at or before `cutoff` train (`validation_share`
/// of them, by group, validate) and later members test. A group with members
/// on both sides of the cutoff is left out and counted as
/// `straddles_cutoff`. Refused when either side is empty.
pub fn temporal_split(
    members: &[Member],
    cutoff: f64,
    validation_share: f64,
    seed: u64,
    dataset: &Digest,
) -> Result<DataSplit, SplitError> {
    check_share(validation_share)?;
    if !cutoff.is_finite() {
        return Err(SplitError::Spec(format!(
            "the cutoff {cutoff} is not finite"
        )));
    }
    let (groups, left_out) = straddling(members, &|m| m.at <= cutoff);
    let kept: Vec<&Member> = members
        .iter()
        .filter(|m| !groups.contains(m.unit.group.as_str()))
        .collect();
    let (early, late): (Vec<&Member>, Vec<&Member>) =
        kept.into_iter().partition(|m| m.at <= cutoff);
    if early.is_empty() || late.is_empty() {
        return Err(SplitError::Spec(format!(
            "the cutoff leaves {} unit(s) before it and {} after",
            early.len(),
            late.len()
        )));
    }
    let (train, validation) = carve_validation(&early, dataset, validation_share, seed)?;
    let test = late.iter().map(|m| m.unit.id.clone()).collect();
    let excluded = BTreeMap::from([("straddles_cutoff".to_owned(), left_out)])
        .into_iter()
        .filter(|(_, n)| *n > 0)
        .collect();
    verified(
        DataSplit {
            kind: SplitKind::Temporal { cutoff },
            seed,
            dataset: dataset.clone(),
            parts: sorted_parts(train, validation, test),
            excluded,
        },
        members,
    )
}

/// The external split: every member of `held_out` is the test, the others
/// train (`validation_share` of them, by group, validate). A group with
/// members in the held-out source and another is left out and counted as
/// `spans_sources`. Refused when the source has no member or nothing is
/// left to train on.
pub fn leave_one_source_out(
    members: &[Member],
    held_out: &str,
    validation_share: f64,
    seed: u64,
    dataset: &Digest,
) -> Result<DataSplit, SplitError> {
    check_share(validation_share)?;
    if !members.iter().any(|m| m.source == held_out) {
        return Err(SplitError::Spec(format!(
            "no unit comes from the source {held_out:?}"
        )));
    }
    let (groups, left_out) = straddling(members, &|m| m.source == held_out);
    let kept: Vec<&Member> = members
        .iter()
        .filter(|m| !groups.contains(m.unit.group.as_str()))
        .collect();
    let (test, rest): (Vec<&Member>, Vec<&Member>) =
        kept.into_iter().partition(|m| m.source == held_out);
    if test.is_empty() || rest.is_empty() {
        return Err(SplitError::Spec(format!(
            "holding out {held_out:?} leaves {} test and {} training unit(s)",
            test.len(),
            rest.len()
        )));
    }
    let (train, validation) = carve_validation(&rest, dataset, validation_share, seed)?;
    let excluded = BTreeMap::from([("spans_sources".to_owned(), left_out)])
        .into_iter()
        .filter(|(_, n)| *n > 0)
        .collect();
    verified(
        DataSplit {
            kind: SplitKind::LeaveOneSourceOut {
                held_out: held_out.to_owned(),
            },
            seed,
            dataset: dataset.clone(),
            parts: sorted_parts(
                train,
                validation,
                test.iter().map(|m| m.unit.id.clone()).collect(),
            ),
            excluded,
        },
        members,
    )
}
