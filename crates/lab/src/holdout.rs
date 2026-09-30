// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The rule that decides which records of a training set are held out for
//! scoring. Pure: no model, no files.

/// The fewest samples [`holdout_split`] splits: one to train on, one to
/// hold out.
pub const MIN_SAMPLES: usize = 2;

/// `samples` split into the ones trained on and the ones held out, `None`
/// when there are fewer than [`MIN_SAMPLES`]. One held-out sample is the
/// minimum honest evaluation; anything less has nothing to hold out, and
/// scoring on the training set would read training loss as
/// generalisation. The newest samples - the last ones - are held out.
pub fn holdout_split<T>(samples: &[T]) -> Option<(&[T], &[T])> {
    if samples.len() < MIN_SAMPLES {
        return None;
    }
    // A tenth held out, at least one: a single record's 20-odd token
    // positions cannot carry a verdict on a set of hundreds - its noise
    // would masquerade as improvement or regression.
    let val = (samples.len() / 10).max(1);
    let (train, val) = samples.split_at(samples.len() - val);
    Some((train, val))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_split_holds_the_newest_sample_out_and_needs_two() {
        let one = vec![0];
        assert!(
            holdout_split(&one).is_none(),
            "one record has nothing to hold out"
        );

        let three = vec![0, 1, 2];
        let (train, val) = holdout_split(&three).unwrap();
        assert_eq!(train.len(), 2);
        // The held-out sample is the LAST one, the newest verified
        // experience.
        assert_eq!(val, &[2]);
    }

    /// One held-out sample cannot score a large set: the holdout grows with
    /// the set (10%, minimum one) so a verdict rests on more than one
    /// record's noise.
    #[test]
    fn the_holdout_grows_with_the_pool() {
        let big: Vec<u32> = (0..150).collect();
        let (train, val) = holdout_split(&big).unwrap();
        assert_eq!(val.len(), 15, "10% of 150");
        assert_eq!(train.len(), 135);
        // Newest records are the held-out ones.
        assert_eq!(val[0], 135);

        let small: Vec<u32> = (0..9).collect();
        let (train, val) = holdout_split(&small).unwrap();
        assert_eq!(val.len(), 1, "at least one, even at 10% < 1");
        assert_eq!(train.len(), 8);
    }
}
