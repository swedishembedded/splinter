// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements continual fine-tuning that replays earlier
// verified experience so a model keeps what it learned, for its clients. If
// your team needs expertise in continual learning, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Which earlier records are replayed beside new ones: a seeded sample of
//! a fixed size, the same every time for the same records, fraction and
//! seed.
//!
//! Each record is drawn by a hash of the seed and the record's own bytes,
//! so the choice depends on what a record is, not on where it sits; the
//! `ceil(fraction * n)` records with the lowest draws are kept, in the
//! order they were given. Mixing the sample into training is the trainer's
//! business: brain replays exactly the files it is handed.

use splinter_core::digest::Digest;

use crate::strip::draw;
use crate::Fraction;

/// The domain of the replay draw, apart from every other seeded draw.
// The name is part of the draw: renaming it would change which earlier
// records a seed replays.
const REPLAY_DOMAIN: &str = "splinter-views/replay";

/// The indices of the records of `records` replayed at `fraction` under
/// `seed`, in ascending order.
#[must_use]
pub fn replay_sample<T: AsRef<str>>(records: &[T], fraction: Fraction, seed: u64) -> Vec<usize> {
    let wanted = (fraction.get() * records.len() as f64).ceil() as usize;
    let mut drawn: Vec<(f64, usize)> = records
        .iter()
        .enumerate()
        .map(|(index, record)| {
            let subject = Digest::of(record.as_ref().as_bytes());
            (draw(REPLAY_DOMAIN, seed, &subject), index)
        })
        .collect();
    // Draws are finite, so the order is total; equal draws (equal records)
    // fall back to position.
    drawn.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut kept: Vec<usize> = drawn
        .into_iter()
        .take(wanted.min(records.len()))
        .map(|(_, i)| i)
        .collect();
    kept.sort_unstable();
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fraction(value: f64) -> Fraction {
        Fraction::new(value).unwrap_or_else(|e| panic!("{e}"))
    }

    #[test]
    fn the_sample_has_a_fixed_size_and_is_reproducible() {
        let records: Vec<String> = (0..40).map(|i| format!("{{\"n\":{i}}}")).collect();
        let a = replay_sample(&records, fraction(0.25), 7);
        assert_eq!(a.len(), 10);
        assert_eq!(a, replay_sample(&records, fraction(0.25), 7));
        assert_ne!(
            a,
            replay_sample(&records, fraction(0.25), 8),
            "the seed matters"
        );
        assert!(a.windows(2).all(|w| w[0] < w[1]), "given order: {a:?}");
        assert_eq!(
            replay_sample(&records, fraction(0.01), 7).len(),
            1,
            "rounded up"
        );
        assert!(replay_sample(&records, fraction(0.0), 7).is_empty());
        assert_eq!(replay_sample(&records, fraction(1.0), 7).len(), 40);
    }
}
