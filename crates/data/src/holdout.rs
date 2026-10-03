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

use std::path::{Path, PathBuf};

use crate::ViewError;

/// The fewest samples [`holdout_split_grouped`] splits: one to train on, one to
/// hold out.
pub const MIN_SAMPLES: usize = 2;

/// `samples` split into the ones trained on and the ones held out, a group
/// never divided: the samples `group` puts under one name (two prints of one
/// letter, the tasks written from them) are all held out or all trained on,
/// so nothing trained on is a near-copy of what is held out. A sample with no
/// group is its own. Groups are taken from the end, whole, until a tenth of
/// the samples (at least one) are held out; a group that would take the held
/// out past half the samples is skipped. When no group fits - the samples are
/// all from one text - the newest tenth is held out, whole groups or not.
/// `None` when there are fewer than [`MIN_SAMPLES`]. Both halves keep the
/// samples' order.
pub fn holdout_split_grouped<T>(
    samples: &[T],
    group: impl Fn(&T) -> Option<String>,
) -> Option<(Vec<&T>, Vec<&T>)> {
    if samples.len() < MIN_SAMPLES {
        return None;
    }
    let keys: Vec<String> = samples
        .iter()
        .enumerate()
        .map(|(n, s)| group(s).unwrap_or_else(|| format!("\u{0}sample-{n}")))
        .collect();
    let mut sizes: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for key in &keys {
        *sizes.entry(key.as_str()).or_default() += 1;
    }
    let wanted = (samples.len() / 10).max(1);
    let mut held: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let mut count = 0;
    for key in keys.iter().rev() {
        if count >= wanted {
            break;
        }
        let size = sizes[key.as_str()];
        if held.contains(key.as_str()) || count + size > samples.len() / 2 {
            continue;
        }
        held.insert(key.as_str());
        count += size;
    }
    if held.is_empty() {
        // Every group is too large to hold out whole - the samples are all
        // from one text - so the newest tenth is held out, as it is when
        // there are no groups.
        let (train, out) = samples.split_at(samples.len() - wanted);
        return Some((train.iter().collect(), out.iter().collect()));
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

/// The group a dataset record names in its `metadata.group`; `None` for a
/// record that names none or is not JSON.
fn record_group<S: AsRef<str>>(record: &S) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(record.as_ref()).ok()?;
    value
        .get("metadata")?
        .get("group")?
        .as_str()
        .map(str::to_string)
}

/// Dataset `records` (one JSON object per line) split as
/// [`holdout_split_grouped`] splits them, by the group each names in its
/// `metadata.group`. The one rule training and the release gate share, so
/// the gate scores what training held out.
pub fn holdout_split_records<S: AsRef<str>>(records: &[S]) -> Option<(Vec<&S>, Vec<&S>)> {
    holdout_split_grouped(records, record_group)
}

/// The two halves of a split dataset file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Split {
    /// The records to train on.
    pub train: PathBuf,
    /// The newest records, held out for scoring.
    pub held_out: PathBuf,
}

/// The file name the records to train on are written under.
const TRAIN_FILE: &str = "train.jsonl";
/// The file name the held-out records are written under.
const HELD_OUT_FILE: &str = "held_out.jsonl";

/// Writes `dataset`'s records into `dir` as two files - the records to train
/// on and the ones held out - split by [`holdout_split_records`]. A record is
/// a non-blank line, as the trainers' parsers read it. Refused when the
/// dataset has fewer than [`MIN_SAMPLES`] records.
pub fn split_dataset_file(dataset: &Path, dir: &Path) -> Result<Split, ViewError> {
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
    let (train, held_out) =
        holdout_split_records(&records).ok_or_else(|| ViewError::TooFewRecords {
            path: dataset.to_path_buf(),
            records: records.len(),
        })?;
    let write = |name: &str, lines: &[&&str]| -> Result<PathBuf, ViewError> {
        let path = dir.join(name);
        let mut body = lines.iter().map(|l| **l).collect::<Vec<_>>().join("\n");
        body.push('\n');
        std::fs::write(&path, body).map_err(io(&path))?;
        Ok(path)
    };
    Ok(Split {
        train: write(TRAIN_FILE, &train)?,
        held_out: write(HELD_OUT_FILE, &held_out)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(held.len(), 2, "{held:?}");
        assert!(
            held.iter().all(|r| r.contains(r#""group":"held""#)),
            "{held:?}"
        );
        assert!(train.iter().all(|r| !r.contains(r#""group":"held""#)));
        assert_eq!(train.len(), 17);
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
        assert_eq!(held, [&('z', 100), &('z', 101)]);
        assert_eq!(train.len(), 18);
        assert!(
            train.iter().all(|s| s.0 != 'z'),
            "no group is on both sides"
        );
    }

    /// Ungrouped samples behave as the plain rule: the newest tenth.
    #[test]
    fn ungrouped_samples_are_each_their_own_group() {
        let samples: Vec<u32> = (0..30).collect();
        let (train, held) = holdout_split_grouped(&samples, |_| None).unwrap();
        assert_eq!(held, [&27, &28, &29]);
        assert_eq!(train.len(), 27);
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
    /// the set (10%, minimum one) so a verdict rests on more than one
    /// record's noise.
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
        assert_eq!(val.len(), 1, "at least one, even at 10% < 1");
        assert_eq!(train.len(), 8);
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
        assert_eq!(read(&split.train), lines[..11]);
        assert_eq!(read(&split.held_out), lines[11..]);

        std::fs::write(&dataset, "{\"n\":0}\n").unwrap();
        let err = split_dataset_file(&dataset, &dir).unwrap_err().to_string();
        assert!(err.contains("at least 2"), "{err}");

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
