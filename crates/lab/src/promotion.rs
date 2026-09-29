// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The promotion gate over held-out scores, and the rule that decides
//! which records are held out. Pure: no model, no files - the numbers a
//! training attempt produced go in, a decision comes out.

/// The held-out verdict for one training attempt.
#[derive(Clone, Debug, PartialEq)]
pub struct Scores {
    /// Held-out loss of the base model alone: mean per-token cross-entropy
    /// over the supervised positions of the held-out records.
    pub base_loss: f32,
    /// Held-out loss of the base plus the candidate adapter, on the same
    /// records and measured the same way.
    pub tuned_loss: f32,
}

/// What the gate decided for one candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decision {
    /// The candidate beat the base, and any matched champion; it may serve.
    Promoted,
    /// The candidate did not earn adoption: no strict improvement, a
    /// non-finite score, or it lost to a matched champion.
    Rejected,
}

/// The gate. A non-finite score is a rejection, never an adoption - "the
/// evaluator failed" must not read as "the model improved". A candidate is promoted only when its held-out loss strictly
/// improves on the base AND, when a standing champion measured on the same
/// pool and split (equal base loss - same data, same holdout), on the
/// champion too. Losses from different held-out sets do not compare, so an
/// unmatched champion never bounds a candidate. A candidate that beats the
/// base but not the champion is rejected: promoting it would make serving
/// worse than what it replaces.
pub fn decide(scores: &Scores, champion: Option<&Scores>) -> Decision {
    if !scores.base_loss.is_finite() || !scores.tuned_loss.is_finite() {
        return Decision::Rejected;
    }
    if scores.tuned_loss >= scores.base_loss {
        return Decision::Rejected;
    }
    if let Some(champ) = champion {
        if champ.base_loss == scores.base_loss
            && champ.base_loss.is_finite()
            && champ.tuned_loss.is_finite()
            && scores.tuned_loss >= champ.tuned_loss
        {
            return Decision::Rejected;
        }
    }
    Decision::Promoted
}

/// One held-out sample is the minimum honest evaluation; anything less has
/// nothing to hold out, and training on the evaluation set would make the
/// gate read training loss. The newest sample is held out: the pool is
/// appended in run order, so the most recent verified experience is the
/// one that must still generalize.
pub fn holdout_split<T>(samples: &[T]) -> Option<(&[T], &[T])> {
    if samples.len() < 2 {
        return None;
    }
    // A tenth held out, at least one: a single record's 20-odd token positions
    // cannot carry a promotion verdict on a pool of hundreds - its noise
    // would masquerade as improvement or regression. The newest records are
    // held out: the pool is appended in run order, so the most recent
    // verified experience is the one that must still generalize.
    let val = (samples.len() / 10).max(1);
    let (train, val) = samples.split_at(samples.len() - val);
    Some((train, val))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn promotion_requires_a_strictly_lower_heldout_loss() {
        assert_eq!(
            decide(
                &Scores {
                    base_loss: 2.0,
                    tuned_loss: 1.9
                },
                None,
            ),
            Decision::Promoted
        );
        assert_eq!(
            decide(
                &Scores {
                    base_loss: 1.9,
                    tuned_loss: 1.9
                },
                None,
            ),
            Decision::Rejected
        );
        assert_eq!(
            decide(
                &Scores {
                    base_loss: 1.9,
                    tuned_loss: 2.0
                },
                None,
            ),
            Decision::Rejected
        );
        // A failed evaluation (NaN) is never an improvement.
        assert_eq!(
            decide(
                &Scores {
                    base_loss: f32::NAN,
                    tuned_loss: 1.0
                },
                None,
            ),
            Decision::Rejected
        );
        assert_eq!(
            decide(
                &Scores {
                    base_loss: 2.0,
                    tuned_loss: f32::NAN
                },
                None,
            ),
            Decision::Rejected
        );
    }

    /// A candidate that beats the base but not the standing champion must
    /// not displace it: the gate exists so serving never gets worse, and
    /// the champion's held-out loss is the bar when the two attempts
    /// measured on the same pool, split, and base (equal base loss).
    #[test]
    fn a_candidate_worse_than_the_champion_is_rejected() {
        // Same base loss: comparable. Champion tuned 0.165; candidate 0.9
        // beats the base (4.8) but not the champion - rejected.
        assert_eq!(
            decide(
                &Scores {
                    base_loss: 4.8,
                    tuned_loss: 0.9
                },
                Some(&Scores {
                    base_loss: 4.8,
                    tuned_loss: 0.165
                }),
            ),
            Decision::Rejected
        );
        // A candidate better than both is promoted.
        assert_eq!(
            decide(
                &Scores {
                    base_loss: 4.8,
                    tuned_loss: 0.1
                },
                Some(&Scores {
                    base_loss: 4.8,
                    tuned_loss: 0.165
                }),
            ),
            Decision::Promoted
        );
        // Champion scores from a DIFFERENT pool (base loss differs) do not
        // bound the candidate: losses measured on different held-out data
        // do not compare, so the base-only rule applies.
        assert_eq!(
            decide(
                &Scores {
                    base_loss: 4.8,
                    tuned_loss: 0.9
                },
                Some(&Scores {
                    base_loss: 3.0,
                    tuned_loss: 0.165
                }),
            ),
            Decision::Promoted
        );
        // No champion recorded: base-only rule.
        assert_eq!(
            decide(
                &Scores {
                    base_loss: 4.8,
                    tuned_loss: 0.9
                },
                None,
            ),
            Decision::Promoted
        );
    }

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

    /// One held-out sample cannot gate a large pool: 21 token positions of
    /// one record decided a promotion that the train split contradicted.
    /// The holdout grows with the pool (10%, minimum one) so a promotion
    /// verdict rests on more than one record's noise.
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
