// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The rule that decides which records of a training set are held out for
//! scoring. Pure: no model, no files.
//!
//! A tenth of the samples (at least one) is held out, the newest, because a
//! single record's 20-odd token positions cannot carry a verdict on a set of
//! hundreds: its noise would masquerade as improvement or regression. One
//! held-out sample is the minimum honest evaluation; scoring on the training
//! set would read training loss as generalisation.

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
}
