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
//! The records to train on are split once more, by the same family rule
//! ([`monitor_split_file`]): a share of their families is the monitoring
//! set a run scores as it trains and selects its best step on. It is carved
//! from the training families, never from the held-out ones, because the
//! gate and the exam decide on the held-out families, and a step chosen for
//! its loss on them would make that decision on evidence the choice had
//! already seen. The price is a tenth of the training records.

use std::path::{Path, PathBuf};

use crate::ViewError;

/// How one sample stands to the split: the group it is held out or trained
/// on with, and whether it is a unit the split chooses from.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Membership {
    /// The group of overlapping source text; `None` is its own.
    pub group: Option<String>,
    /// Whether the sample has a question a model can be examined on. One
    /// that has not follows its group.
    pub examinable: bool,
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

    /// The monitoring rule: `share` of the units, whole groups, and no more
    /// - a loss curve needs positions, not units of evidence.
    #[must_use]
    pub fn monitor(share: f64) -> SplitRule {
        SplitRule {
            share,
            min_units: 1,
            most: share,
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
/// held-out rule.
pub fn split_by_rule<'a, T>(
    samples: &'a [T],
    membership: impl Fn(&T) -> Membership,
    rule: &SplitRule,
) -> Option<(Vec<&'a T>, Vec<&'a T>)> {
    if samples.len() < MIN_SAMPLES {
        return None;
    }
    let mut memberships: Vec<Membership> = samples.iter().map(membership).collect();
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
    let units: Vec<usize> = (0..samples.len())
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
        let (out, kept): (Vec<_>, Vec<_>) = samples
            .iter()
            .enumerate()
            .partition(|(n, _)| newest.contains(n));
        return Some((
            kept.into_iter().map(|(_, s)| s).collect(),
            out.into_iter().map(|(_, s)| s).collect(),
        ));
    }
    let (out, kept): (Vec<_>, Vec<_>) = samples
        .iter()
        .zip(&keys)
        .partition(|(_, key)| held.contains(key.as_str()));
    Some((
        kept.into_iter().map(|(s, _)| s).collect(),
        out.into_iter().map(|(s, _)| s).collect(),
    ))
}

/// How a dataset record stands to the split, from its metadata: the group
/// it names in `metadata.group`, and whether it was projected from a task
/// or an experience (`metadata.task`, `metadata.experiences`), which is
/// what makes it examinable. A record that is not JSON is its own group and
/// examinable.
fn record_membership<S: AsRef<str>>(record: &S) -> Membership {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(record.as_ref()) else {
        return Membership {
            group: None,
            examinable: true,
        };
    };
    let metadata = &value["metadata"];
    Membership {
        group: metadata["group"].as_str().map(str::to_string),
        examinable: metadata["task"].is_string()
            || metadata["experiences"]
                .as_array()
                .is_some_and(|experiences| !experiences.is_empty()),
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

/// The two halves of a split dataset file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Split {
    /// The records to train on.
    pub train: PathBuf,
    /// The held-out records a score is measured on: the examinable ones.
    /// The writer's text held out with their families is in neither file.
    pub held_out: PathBuf,
}

/// The file name the records to train on are written under.
const TRAIN_FILE: &str = "train.jsonl";
/// The file name the held-out records are written under.
const HELD_OUT_FILE: &str = "held_out.jsonl";
/// The file name the records a monitored run fits are written under.
const FIT_FILE: &str = "fit.jsonl";
/// The file name the monitoring records are written under.
const MONITOR_FILE: &str = "monitor.jsonl";

/// Writes `dataset`'s records into `dir` as two files - the records to train
/// on and the ones held out that can be scored - split by
/// [`holdout_split_records`]. A record is a non-blank line, as the trainers'
/// parsers read it. Refused when the dataset has fewer than [`MIN_SAMPLES`]
/// records.
pub fn split_dataset_file(dataset: &Path, dir: &Path) -> Result<Split, ViewError> {
    let (train, held_out) = split_file(
        dataset,
        dir,
        &SplitRule::HELD_OUT,
        TRAIN_FILE,
        HELD_OUT_FILE,
    )?;
    Ok(Split { train, held_out })
}

/// The two halves of a split of the records to train on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MonitorSplit {
    /// The records the run fits.
    pub fit: PathBuf,
    /// The records it scores as it trains: the examinable ones of the
    /// monitoring families. The writer's text of those families is in
    /// neither file.
    pub monitor: PathBuf,
}

/// Writes the records of `train` (a [`Split::train`] file) into `dir` as two
/// files - the records to fit and the monitoring records - split by the
/// family rule with [`SplitRule::monitor`] of `share`. Refused when there
/// are fewer than [`MIN_SAMPLES`] records.
pub fn monitor_split_file(train: &Path, dir: &Path, share: f64) -> Result<MonitorSplit, ViewError> {
    let (fit, monitor) = split_file(
        train,
        dir,
        &SplitRule::monitor(share),
        FIT_FILE,
        MONITOR_FILE,
    )?;
    Ok(MonitorSplit { fit, monitor })
}

/// The records of `dataset` split by `rule` into `dir/kept` and `dir/out`,
/// the latter the examinable records taken out (every record, when none is
/// examinable).
fn split_file(
    dataset: &Path,
    dir: &Path,
    rule: &SplitRule,
    kept: &str,
    out: &str,
) -> Result<(PathBuf, PathBuf), ViewError> {
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
    // none is, every record is (see [`holdout_split_by`]).
    let examinable: Vec<&&str> = held_out
        .iter()
        .filter(|record| record_membership(record).examinable)
        .copied()
        .collect();
    let scored = if examinable.is_empty() {
        held_out
    } else {
        examinable
    };
    let write = |name: &str, lines: &[&&str]| -> Result<PathBuf, ViewError> {
        let path = dir.join(name);
        let mut body = lines.iter().map(|l| **l).collect::<Vec<_>>().join("\n");
        body.push('\n');
        std::fs::write(&path, body).map_err(io(&path))?;
        Ok(path)
    };
    Ok((write(kept, &train)?, write(out, &scored)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A record of the writer's own text (from source text alone, no task or
    /// experience behind it) is not a unit the split chooses from: it is
    /// held out when its family is, so nothing trained on is a print of a
    /// held-out letter, and trained on otherwise. The families held out are
    /// decided by the records that can be examined, however many records of
    /// the writer's text there are and wherever they sit.
    #[test]
    fn records_from_source_text_alone_follow_the_family_the_examined_records_decide() {
        let examined = |group: &str, n: u32| {
            format!(
                r#"{{"messages":[],"metadata":{{"experiences":["blake3:{n:064x}"],"group":"{group}"}}}}"#
            )
        };
        let voice = |group: &str, n: u32| {
            format!(
                r#"{{"messages":[],"metadata":{{"sources":["blake3:{n:064x}"],"group":"{group}","view":"voice"}}}}"#
            )
        };
        // Twenty examined records over ten families, and six times as many
        // records of the writer's text: two per family, and forty of
        // families no examined record is about (letters nothing was asked
        // of). The writer's text comes last, where the newest tenth would
        // be if it counted.
        let mut records: Vec<String> = Vec::new();
        for n in 0..20u32 {
            records.push(examined(&format!("family-{}", n / 2), n));
        }
        for n in 0..20u32 {
            records.push(voice(&format!("family-{}", n / 2), 100 + n));
        }
        for n in 0..40u32 {
            records.push(voice(&format!("alone-{n}"), 200 + n));
        }
        let (train, held) = holdout_split_records(&records).unwrap();
        let examined_held: Vec<&&String> =
            held.iter().filter(|r| r.contains("experiences")).collect();
        // The split of the examined records is what it would be without the
        // writer's text: the newest families, to the units a paired test
        // needs, within a quarter of the twenty.
        assert_eq!(examined_held.len(), 4, "{examined_held:?}");
        assert!(examined_held
            .iter()
            .all(|r| r.contains("family-9") || r.contains("family-8")));
        // Every record of the writer's text of a held-out family is held out
        // with it, and no other.
        let voice_held: Vec<&&String> = held.iter().filter(|r| r.contains("voice")).collect();
        assert_eq!(voice_held.len(), 4, "{voice_held:?}");
        assert!(voice_held
            .iter()
            .all(|r| r.contains("family-9") || r.contains("family-8")));
        assert!(train
            .iter()
            .all(|r| !r.contains("family-9") && !r.contains("family-8")));
        assert_eq!(train.len() + held.len(), records.len());

        // A set of the writer's text alone has nothing to examine: every
        // record is then a unit, as before.
        let alone: Vec<String> = (0..30u32).map(|n| voice(&format!("g{n}"), n)).collect();
        let (train, held) = holdout_split_records(&alone).unwrap();
        assert_eq!((train.len(), held.len()), (23, 7));

        // Split into files, the held-out file holds what a score is measured
        // on: the examined records. The writer's text held out with them is
        // in neither file - trained on by nobody, and not what the policy is
        // asked as.
        let dir = std::env::temp_dir().join(format!("views-voice-split-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let dataset = dir.join("mixed.jsonl");
        std::fs::write(&dataset, format!("{}\n", records.join("\n"))).unwrap();
        let split = split_dataset_file(&dataset, &dir).unwrap();
        let held_file = std::fs::read_to_string(&split.held_out).unwrap();
        assert_eq!(held_file.lines().count(), 4);
        assert!(held_file.lines().all(|l| l.contains("experiences")));
        let train_file = std::fs::read_to_string(&split.train).unwrap();
        assert_eq!(train_file.lines().count(), records.len() - 8);
        assert!(!train_file.contains("family-9"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The monitoring set is a share of the training families, whole, and
    /// no more; the writer's text of a monitored family is in neither file,
    /// as with the held-out families.
    #[test]
    fn the_monitoring_set_is_a_share_of_the_training_families_taken_whole() {
        let examined = |group: &str, n: u32| {
            format!(
                r#"{{"messages":[],"metadata":{{"experiences":["blake3:{n:064x}"],"group":"{group}"}}}}"#
            )
        };
        let voice = |group: &str, n: u32| {
            format!(
                r#"{{"messages":[],"metadata":{{"sources":["blake3:{n:064x}"],"group":"{group}","view":"voice"}}}}"#
            )
        };
        // Ten families of four examined records, and two records of the
        // writer's text per family.
        let mut records: Vec<String> = Vec::new();
        for n in 0..40u32 {
            records.push(examined(&format!("family-{}", n / 4), n));
        }
        for n in 0..20u32 {
            records.push(voice(&format!("family-{}", n / 2), 100 + n));
        }
        let dir = std::env::temp_dir().join(format!("views-monitor-split-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let train = dir.join("train.jsonl");
        std::fs::write(&train, format!("{}\n", records.join("\n"))).unwrap();

        let split = monitor_split_file(&train, &dir, 0.1).unwrap();
        let monitor = std::fs::read_to_string(&split.monitor).unwrap();
        let fit = std::fs::read_to_string(&split.fit).unwrap();
        // A tenth of forty units is one family of four, the newest, and the
        // paired test's eight units are not asked for.
        assert_eq!(monitor.lines().count(), 4, "{monitor}");
        assert!(monitor
            .lines()
            .all(|l| l.contains("family-9") && l.contains("experiences")));
        assert_eq!(
            fit.lines().count(),
            records.len() - 6,
            "the family's four examined and two voice records leave"
        );
        assert!(!fit.contains("family-9"));

        let split = monitor_split_file(&train, &dir, 0.3).unwrap();
        let monitor = std::fs::read_to_string(&split.monitor).unwrap();
        assert_eq!(monitor.lines().count(), 12, "three families of four");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Records name their group in their metadata; one without is its own.
    #[test]
    fn dataset_records_are_grouped_by_their_metadata() {
        let line = |group: Option<&str>, n: u32| match group {
            Some(g) => format!(r#"{{"messages":[],"metadata":{{"group":"{g}","n":{n}}}}}"#),
            None => format!(r#"{{"messages":[],"n":{n}}}"#),
        };
        let mut records: Vec<String> = (0..16).map(|n| line(Some(&format!("g{n}")), n)).collect();
        records.insert(2, line(Some("held"), 100));
        records.insert(5, line(None, 102));
        records.push(line(Some("held"), 101));
        let (train, held) = holdout_split_records(&records).unwrap();
        // The group "held" whole, wherever its records sit, and the newest
        // groups after it up to a quarter of the records.
        assert_eq!(held.len(), 4, "{held:?}");
        assert_eq!(
            held.iter()
                .filter(|r| r.contains(r#""group":"held""#))
                .count(),
            2,
            "{held:?}"
        );
        assert!(train.iter().all(|r| !r.contains(r#""group":"held""#)));
        assert_eq!(train.len(), 15);
    }

    fn by_tag(s: &(char, u32)) -> Option<String> {
        Some(s.0.to_string())
    }

    /// A group is held out whole, wherever its samples sit, and the held-out
    /// set is the newest groups.
    #[test]
    fn a_group_is_held_out_whole_wherever_its_samples_sit() {
        // 20 samples; group 'z' is the newest and has a copy near the start.
        let mut samples: Vec<(char, u32)> = (0..18).map(|n| ('a', n)).collect();
        for (n, s) in samples.iter_mut().enumerate() {
            s.0 = char::from(b'a' + u8::try_from(n % 9).unwrap());
        }
        samples.insert(1, ('z', 100));
        samples.push(('z', 101));
        let (train, held) = holdout_split_grouped(&samples, by_tag).unwrap();
        // The newest group whole, then the next newest that fits within a
        // quarter of the samples, for the units a paired test needs.
        assert!(held.contains(&&('z', 100)) && held.contains(&&('z', 101)));
        assert_eq!(held.len(), 4, "{held:?}");
        assert_eq!(train.len(), 16);
        let split: Vec<char> = held
            .iter()
            .filter(|s| train.iter().any(|t| t.0 == s.0))
            .map(|s| s.0)
            .collect();
        assert!(split.is_empty(), "no group is on both sides: {split:?}");
    }

    /// Ungrouped samples are each a unit: the newest are held out until a
    /// tenth and the units a paired test needs are, within a quarter.
    #[test]
    fn ungrouped_samples_are_each_their_own_group() {
        let samples: Vec<u32> = (0..30).collect();
        let (train, held) = holdout_split_grouped(&samples, |_| None).unwrap();
        assert_eq!(held, [&23, &24, &25, &26, &27, &28, &29], "a quarter of 30");
        assert_eq!(train.len(), 23);
        let samples: Vec<u32> = (0..60).collect();
        let (train, held) = holdout_split_grouped(&samples, |_| None).unwrap();
        assert_eq!(
            held.len(),
            MIN_HELD_OUT_UNITS,
            "past the tenth, to the units"
        );
        assert_eq!(train.len(), 52);
    }

    /// Records generated letter by letter come in runs about one letter: a
    /// tenth of them is two or three letters, on which no paired test over
    /// families can be significant. Whole groups are held out until there
    /// are enough of them, within a quarter of the records.
    #[test]
    fn enough_groups_are_held_out_for_a_paired_test_over_them() {
        let mut samples: Vec<(u32, u32)> = Vec::new();
        for letter in 0..28u32 {
            for n in 0..(8 + letter % 2) {
                samples.push((letter, n));
            }
        }
        let (train, held) = holdout_split_grouped(&samples, |s| Some(s.0.to_string())).unwrap();
        let families: std::collections::BTreeSet<u32> = held.iter().map(|s| s.0).collect();
        assert!(
            (6..=MIN_HELD_OUT_UNITS).contains(&families.len()),
            "{} families held out",
            families.len()
        );
        assert!(
            held.len() <= samples.len() / MAX_HELD_OUT_SHARE,
            "{}",
            held.len()
        );
        assert!(held.len() >= samples.len() / 10);
        assert!(
            train.iter().all(|s| !families.contains(&s.0)),
            "a family is never split"
        );
        assert_eq!(train.len() + held.len(), samples.len());
    }

    /// A group too big to hold out is skipped for an older one, and a set
    /// that is one group holds out its newest tenth.
    #[test]
    fn a_group_that_would_take_over_half_is_skipped_and_one_group_holds_out_its_newest_tenth() {
        let mut samples: Vec<(char, u32)> = (0..40).map(|n| ('a', n)).collect();
        samples.extend((0..4).map(|n| ('b', n)));
        samples.extend((0..4).map(|n| ('c', n)));
        let (train, held) = holdout_split_grouped(&samples, by_tag).unwrap();
        assert!(held.iter().all(|s| s.0 != 'a'), "the large group trains");
        assert!(train.iter().any(|s| s.0 == 'a'));
        let one: Vec<(char, u32)> = (0..10).map(|n| ('a', n)).collect();
        let (train, held) = holdout_split_grouped(&one, by_tag).unwrap();
        assert_eq!((train.len(), held.len()), (9, 1));
        assert_eq!(held[0].1, 9, "the newest");
    }

    #[test]
    fn the_split_holds_the_newest_sample_out_and_needs_two() {
        let one = vec![0];
        assert!(
            holdout_split_grouped(&one, |_| None).is_none(),
            "one record has nothing to hold out"
        );

        let three = vec![0, 1, 2];
        let (train, val) = holdout_split_grouped(&three, |_| None).unwrap();
        assert_eq!(train.len(), 2);
        // The held-out sample is the LAST one, the newest verified
        // experience.
        assert_eq!(val, [&2]);
    }

    /// One held-out sample cannot score a large set: the holdout grows with
    /// the set (a tenth, at least one, and the units a paired test needs) so
    /// a verdict rests on more than one record's noise.
    #[test]
    fn the_holdout_grows_with_the_pool() {
        let big: Vec<u32> = (0..150).collect();
        let (train, val) = holdout_split_grouped(&big, |_| None).unwrap();
        assert_eq!(val.len(), 15, "10% of 150");
        assert_eq!(train.len(), 135);
        // Newest records are the held-out ones.
        assert_eq!(val[0], &135);

        let small: Vec<u32> = (0..9).collect();
        let (train, val) = holdout_split_grouped(&small, |_| None).unwrap();
        assert_eq!(val.len(), 2, "a quarter of nine, for the units");
        assert_eq!(train.len(), 7);
        let tiny: Vec<u32> = (0..3).collect();
        let (train, val) = holdout_split_grouped(&tiny, |_| None).unwrap();
        assert_eq!(val.len(), 1, "at least one, even where a quarter is none");
        assert_eq!(train.len(), 2);
    }

    /// The newest records are held out in a file of their own, never trained
    /// on; a dataset too small to hold one out is refused.
    #[test]
    fn the_newest_records_are_held_out_in_their_own_file() {
        let dir = std::env::temp_dir().join(format!("views-split-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let dataset = dir.join("pool.jsonl");
        let lines: Vec<String> = (0..12).map(|i| format!("{{\"n\":{i}}}")).collect();
        std::fs::write(&dataset, format!("{}\n\n", lines.join("\n"))).unwrap();

        let split = split_dataset_file(&dataset, &dir).unwrap();
        let read = |p: &Path| -> Vec<String> {
            std::fs::read_to_string(p)
                .unwrap()
                .lines()
                .map(str::to_string)
                .collect()
        };
        assert_eq!(read(&split.train), lines[..9]);
        assert_eq!(read(&split.held_out), lines[9..], "a quarter of twelve");

        std::fs::write(&dataset, "{\"n\":0}\n").unwrap();
        let err = split_dataset_file(&dataset, &dir).unwrap_err().to_string();
        assert!(err.contains("at least 2"), "{err}");

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
