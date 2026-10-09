// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The rule that decides which records of a training set are held out for
//! scoring, and the split of a dataset file by it.
//!
//! A tenth of the samples (at least one) is held out, the newest, because a
//! single record's 20-odd token positions cannot carry a verdict on a set of
//! hundreds: its noise would masquerade as improvement or regression. One
//! held-out sample is the minimum honest evaluation; scoring on the training
//! set would read training loss as generalisation.
//!
//! What is held out is also what a release is decided on, by a paired sign
//! test in which records about one family of source text count once. A tenth
//! of a set generated letter by letter is a handful of records about two or
//! three letters: two or three units of evidence, on which no test can reach
//! significance. So the held-out set also runs to [`MIN_HELD_OUT_UNITS`]
//! units - groups, or ungrouped samples - where the data has them, taking
//! whole groups from the newest end as long as it stays within
//! [`MAX_HELD_OUT_SHARE`] of the samples.
//!
//! The samples the split chooses from are the records that can be examined:
//! those projected from a task or an experience, whose question a gate or an
//! exam can put to a model. A record of the writer's own text alone (the
//! `voice` and `cpt` views) has no question to ask and is not a unit; it
//! follows its family, held out with it and trained on otherwise, so
//! nothing trained on is a print of a held-out letter and the families held
//! out are the same however much of the writer's text is beside them. A set
//! with nothing to examine holds out its records as units, as before.
//!
//! The writer's text of a held-out family is written to a file of its own
//! ([`Split::held_out_text`]): it is not what the policy is asked as, so it
//! is not in the examinable held-out file a gate scores, but it is the one
//! measure of the writer's text the run never trained on, and no record of
//! the set goes nowhere: every record is in exactly one of the files.
//!
//! The records to train on are split once more, by the same family rule
//! ([`monitor_split_file`]): a share of their families is the monitoring
//! set a run scores as it trains and selects its best step on. It is carved
//! from the training families, never from the held-out ones, because the
//! gate and the exam decide on the held-out families, and a step chosen for
//! its loss on them would make that decision on evidence the choice had
//! already seen. The price is a tenth of the training records. The
//! monitoring set spans at least [`MIN_MONITOR_FAMILIES`] families where the
//! data has them, within twice the share, because a loss over one family
//! measures that family; and the writer's text of the monitoring families
//! goes to its own file ([`MonitorSplit::monitor_text`]), where a run that
//! trains mostly on the writer's text can watch what it is learning.

use std::path::{Path, PathBuf};

use crate::partition::Unit;
use crate::split::{verify_disjoint, Part};
use crate::{Side, ViewError};

/// How one sample stands to the split: the group it is held out or trained
/// on with, and whether it is a unit the split chooses from.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Membership {
    /// The group of overlapping source text; `None` is its own.
    pub group: Option<String>,
    /// Whether the sample has a question a model can be examined on. One
    /// that has not follows its group.
    pub examinable: bool,
    /// The side its producer fixed for it, when it did: a sample that must
    /// be trained on (a fact the run exists to teach) or must be held out
    /// (a wording of it no record trains). The rule never moves it, and
    /// does not count it among the units it chooses from.
    pub side: Option<Side>,
}

/// The fewest samples [`holdout_split_grouped`] splits: one to train on, one to
/// hold out.
pub const MIN_SAMPLES: usize = 2;

/// The fewest units of evidence - groups, or ungrouped samples - a held-out
/// set runs to when the samples allow: enough for a one-sided paired sign
/// test at the gate's level to be significant with one unit lost.
pub const MIN_HELD_OUT_UNITS: usize = 8;

/// The share of the samples the held-out set may grow to for the sake of
/// [`MIN_HELD_OUT_UNITS`], as the divisor of the sample count: a quarter.
pub const MAX_HELD_OUT_SHARE: usize = 4;

/// The fewest families a monitoring set spans when the training families
/// allow: a loss over one family is a measure of that family, and a step
/// selected on it is selected for it.
pub const MIN_MONITOR_FAMILIES: usize = 3;

/// `samples` split into the ones trained on and the ones held out, a group
/// never divided: the samples `group` puts under one name (two prints of one
/// letter, the tasks written from them) are all held out or all trained on,
/// so nothing trained on is a near-copy of what is held out. A sample with no
/// group is its own. Groups are taken from the end, whole, until a tenth of
/// the samples (at least one) are held out and [`MIN_HELD_OUT_UNITS`] groups
/// with them; a group that would take the held out past half the samples is
/// skipped, and one taken only for the units past
/// a [`MAX_HELD_OUT_SHARE`]th of them. When no group fits - the samples are
/// all from one text - the newest tenth is held out, whole groups or not.
/// `None` when there are fewer than [`MIN_SAMPLES`]. Both halves keep the
/// samples' order.
pub fn holdout_split_grouped<T>(
    samples: &[T],
    group: impl Fn(&T) -> Option<String>,
) -> Option<(Vec<&T>, Vec<&T>)> {
    holdout_split_by(samples, |sample| Membership {
        group: group(sample),
        examinable: true,
        side: None,
    })
}

/// How much of a set a split takes out, in units of the examinable samples.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SplitRule {
    /// The share of the units wanted out (at least one unit).
    pub share: f64,
    /// Whole groups are taken past the share until this many are out, as
    /// long as that stays within `most`.
    pub min_units: usize,
    /// The share of the units the set taken out may grow to for the sake of
    /// `min_units`.
    pub most: f64,
}

impl SplitRule {
    /// The held-out rule: a tenth, grown to [`MIN_HELD_OUT_UNITS`] units
    /// within a [`MAX_HELD_OUT_SHARE`]th, for the paired test the gate runs.
    pub const HELD_OUT: SplitRule = SplitRule {
        share: 0.1,
        min_units: MIN_HELD_OUT_UNITS,
        most: 1.0 / MAX_HELD_OUT_SHARE as f64,
    };

    /// The monitoring rule: `share` of the units, whole groups, grown to
    /// [`MIN_MONITOR_FAMILIES`] families within twice the share.
    #[must_use]
    pub fn monitor(share: f64) -> SplitRule {
        SplitRule {
            share,
            min_units: MIN_MONITOR_FAMILIES,
            most: (2.0 * share).min(1.0),
        }
    }
}

/// [`holdout_split_grouped`] over the samples `membership` says can be
/// examined, the rest following their group: held out with it, else trained
/// on (see the module documentation). When fewer than [`MIN_SAMPLES`]
/// samples can be examined, every sample is a unit.
pub fn holdout_split_by<T>(
    samples: &[T],
    membership: impl Fn(&T) -> Membership,
) -> Option<(Vec<&T>, Vec<&T>)> {
    split_by_rule(samples, membership, &SplitRule::HELD_OUT)
}

/// [`holdout_split_by`] taking out what `rule` says instead of the
/// held-out rule. A sample whose [`Membership::side`] is fixed stays on that
/// side and is not one of the units the rule chooses from.
pub fn split_by_rule<'a, T>(
    samples: &'a [T],
    membership: impl Fn(&T) -> Membership,
    rule: &SplitRule,
) -> Option<(Vec<&'a T>, Vec<&'a T>)> {
    if samples.len() < MIN_SAMPLES {
        return None;
    }
    let memberships: Vec<Membership> = samples.iter().map(membership).collect();
    let held = held_out_flags(&memberships, rule);
    let (out, kept): (Vec<_>, Vec<_>) = samples.iter().zip(held).partition(|(_, held)| *held);
    Some((
        kept.into_iter().map(|(s, _)| s).collect(),
        out.into_iter().map(|(s, _)| s).collect(),
    ))
}

/// Which of the samples `memberships` describe are held out: the ones whose
/// side is fixed as held out, and of the others those `rule` chooses.
fn held_out_flags(memberships: &[Membership], rule: &SplitRule) -> Vec<bool> {
    let free: Vec<usize> = (0..memberships.len())
        .filter(|&n| memberships[n].side.is_none())
        .collect();
    let mut held: Vec<bool> = memberships
        .iter()
        .map(|m| m.side == Some(Side::HeldOut))
        .collect();
    if free.len() < MIN_SAMPLES {
        return held;
    }
    let free_memberships: Vec<Membership> = free.iter().map(|&n| memberships[n].clone()).collect();
    for (n, chosen) in free
        .into_iter()
        .zip(choose_held_out(free_memberships, rule))
    {
        held[n] = chosen;
    }
    held
}

/// Which of `memberships` (at least [`MIN_SAMPLES`], none with a side fixed)
/// `rule` takes out.
fn choose_held_out(mut memberships: Vec<Membership>, rule: &SplitRule) -> Vec<bool> {
    if memberships.iter().filter(|m| m.examinable).count() < MIN_SAMPLES {
        for m in &mut memberships {
            m.examinable = true;
        }
    }
    let keys: Vec<String> = memberships
        .iter()
        .enumerate()
        .map(|(n, m)| {
            m.group
                .clone()
                .unwrap_or_else(|| format!("\u{0}sample-{n}"))
        })
        .collect();
    // The units and the counts are the examinable samples'.
    let units: Vec<usize> = (0..memberships.len())
        .filter(|&n| memberships[n].examinable)
        .collect();
    let mut sizes: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for &n in &units {
        *sizes.entry(keys[n].as_str()).or_default() += 1;
    }
    // Floors, as a tenth and a quarter of the count always were.
    let of_units = |share: f64| (units.len() as f64 * share).floor() as usize;
    let wanted = of_units(rule.share).max(1);
    let half = units.len() / 2;
    let most = of_units(rule.most);
    let mut held: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let mut count = 0;
    for &n in units.iter().rev() {
        if count >= wanted && held.len() >= rule.min_units {
            break;
        }
        let key = keys[n].as_str();
        if held.contains(key) {
            continue;
        }
        let size = sizes[key];
        let for_the_tenth = count < wanted && count + size <= half;
        let for_the_units = held.len() < rule.min_units && count + size <= most;
        if for_the_tenth || for_the_units {
            held.insert(key);
            count += size;
        }
    }
    if held.is_empty() {
        // Every group is too large to hold out whole - the samples are all
        // from one text - so the newest tenth of the units is held out, as
        // it is when there are no groups.
        let newest: std::collections::HashSet<usize> =
            units[units.len() - wanted..].iter().copied().collect();
        return (0..memberships.len())
            .map(|n| newest.contains(&n))
            .collect();
    }
    keys.iter().map(|key| held.contains(key.as_str())).collect()
}

/// How a dataset record stands to the split, from its metadata: the group
/// it names in `metadata.group`, whether it was projected from a task
/// or an experience (`metadata.task`, `metadata.experiences`), which is
/// what makes it examinable, and the side it fixes in `metadata.split`. A record that is not JSON is its own group and
/// examinable.
fn record_membership<S: AsRef<str>>(record: &S) -> Membership {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(record.as_ref()) else {
        return Membership {
            group: None,
            examinable: true,
            side: None,
        };
    };
    let metadata = &value["metadata"];
    Membership {
        group: metadata["group"].as_str().map(str::to_string),
        examinable: metadata["task"].is_string()
            || metadata["experiences"]
                .as_array()
                .is_some_and(|experiences| !experiences.is_empty()),
        side: serde_json::from_value(metadata["split"].clone()).ok(),
    }
}

/// Dataset `records` (one JSON object per line) split as
/// [`holdout_split_by`] splits them, by the group each names in its
/// `metadata.group` and whether it came from a task or an experience. The
/// one rule training and the release gate share, so the gate scores what
/// training held out.
pub fn holdout_split_records<S: AsRef<str>>(records: &[S]) -> Option<(Vec<&S>, Vec<&S>)> {
    holdout_split_by(records, record_membership)
}

/// The parts of a split dataset file: every record is in exactly one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Split {
    /// The records to train on.
    pub train: PathBuf,
    /// The held-out records a score is measured on: the examinable ones.
    pub held_out: PathBuf,
    /// The records of the held-out families with no question to ask - the
    /// writer's text - never trained on; `None` when there are none.
    pub held_out_text: Option<PathBuf>,
}

/// The file name the records to train on are written under.
const TRAIN_FILE: &str = "train.jsonl";
/// The file name the held-out records are written under.
const HELD_OUT_FILE: &str = "held_out.jsonl";
/// The file name the records a monitored run fits are written under.
const FIT_FILE: &str = "fit.jsonl";
/// The file name the monitoring records are written under.
const MONITOR_FILE: &str = "monitor.jsonl";
/// The suffix of the file the writer's text of the taken-out families is
/// written under, after the taken-out file's stem: `held_out_text.jsonl`.
const TEXT_SUFFIX: &str = "_text";

/// Writes `dataset`'s records into `dir` as two files - the records to train
/// on and the ones held out that can be scored - split by
/// [`holdout_split_records`]. A record is a non-blank line, as the trainers'
/// parsers read it. Refused when the dataset has fewer than [`MIN_SAMPLES`]
/// records.
pub fn split_dataset_file(dataset: &Path, dir: &Path) -> Result<Split, ViewError> {
    let (train, held_out, held_out_text) = split_file(
        dataset,
        dir,
        &SplitRule::HELD_OUT,
        TRAIN_FILE,
        HELD_OUT_FILE,
    )?;
    Ok(Split {
        train,
        held_out,
        held_out_text,
    })
}

/// The parts of a split of the records to train on: every record is in
/// exactly one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MonitorSplit {
    /// The records the run fits.
    pub fit: PathBuf,
    /// The records it scores as it trains: the examinable ones of the
    /// monitoring families.
    pub monitor: PathBuf,
    /// The records of the monitoring families with no question to ask - the
    /// writer's text - scored as the run trains, never fitted; `None` when
    /// there are none.
    pub monitor_text: Option<PathBuf>,
}

/// Writes the records of `train` (a [`Split::train`] file) into `dir` as two
/// files - the records to fit and the monitoring records - split by the
/// family rule with [`SplitRule::monitor`] of `share`. Refused when there
/// are fewer than [`MIN_SAMPLES`] records.
pub fn monitor_split_file(train: &Path, dir: &Path, share: f64) -> Result<MonitorSplit, ViewError> {
    monitor_split_named(train, dir, share, (FIT_FILE, MONITOR_FILE))
}

/// [`monitor_split_file`] writing its two files under `names` - `(fit,
/// monitor)` - so a second set split into the same directory keeps its
/// files apart from the first's.
pub fn monitor_split_named(
    train: &Path,
    dir: &Path,
    share: f64,
    names: (&str, &str),
) -> Result<MonitorSplit, ViewError> {
    let (fit, monitor, monitor_text) =
        split_file(train, dir, &SplitRule::monitor(share), names.0, names.1)?;
    Ok(MonitorSplit {
        fit,
        monitor,
        monitor_text,
    })
}

/// The records of `dataset` split by `rule` into three files of `dir`: `kept`;
/// `out`, the examinable records taken out (every record, when none is
/// examinable); and `<out stem>_text.jsonl`, the taken-out records that
/// cannot be examined, when there are any.
fn split_file(
    dataset: &Path,
    dir: &Path,
    rule: &SplitRule,
    kept: &str,
    out: &str,
) -> Result<(PathBuf, PathBuf, Option<PathBuf>), ViewError> {
    let io = |path: &Path| {
        let path = path.to_path_buf();
        move |source| ViewError::Io { path, source }
    };
    let text = std::fs::read_to_string(dataset).map_err(io(dataset))?;
    let records: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    let (train, held_out) = split_by_rule(&records, record_membership, rule).ok_or_else(|| {
        ViewError::TooFewRecords {
            path: dataset.to_path_buf(),
            records: records.len(),
        }
    })?;
    // A score over the examinable records is a score of what the policy is
    // asked as, comparable whatever the writer's text beside them; when
    // none is, every record is (see [`holdout_split_by`]). The writer's text
    // of the taken-out families is scored apart, in its own file.
    let (examinable, text): (Vec<&&str>, Vec<&&str>) = held_out
        .iter()
        .partition(|record| record_membership(record).examinable);
    let (scored, text) = if examinable.is_empty() {
        (held_out, Vec::new())
    } else {
        (examinable, text)
    };
    // The rule never divides a group; the gate checks the files about to be
    // written rather than trusting the rule that made them.
    let units = |lines: &[&&str], side: &str| -> Vec<Unit> {
        lines
            .iter()
            .enumerate()
            .map(|(n, line)| Unit {
                id: format!("{side}-{n}"),
                group: record_membership(line)
                    .group
                    .unwrap_or_else(|| format!("{side}-{n}")),
                stratum: String::new(),
            })
            .collect()
    };
    verify_disjoint(&[
        (Part::Train, &units(&train, "kept")),
        (Part::Test, &units(&scored, "out")),
    ])?;
    let write = |name: &str, lines: &[&&str]| -> Result<PathBuf, ViewError> {
        let path = dir.join(name);
        let mut body = lines.iter().map(|l| **l).collect::<Vec<_>>().join("\n");
        body.push('\n');
        std::fs::write(&path, body).map_err(io(&path))?;
        Ok(path)
    };
    let text_file = if text.is_empty() {
        None
    } else {
        let stem = out.strip_suffix(".jsonl").unwrap_or(out);
        Some(write(&format!("{stem}{TEXT_SUFFIX}.jsonl"), &text)?)
    };
    Ok((write(kept, &train)?, write(out, &scored)?, text_file))
}

#[cfg(test)]
mod tests;
