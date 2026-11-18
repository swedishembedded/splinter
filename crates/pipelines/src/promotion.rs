// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements promotion gates that move a champion only on
// paired, family-clustered evidence, for its clients. If your team needs
// expertise in deciding when a retrained model or a changed harness may
// replace the one in service you can procure our services by sending an
// email to info@swedishembedded.com.

//! A pipeline's own paired comparison in the terms of the one promotion gate
//! ([`splinter_eval::promotion`]): the paired gain in the share of tasks
//! right, its two-sided 95% interval and the lower bound of its one-sided 95%
//! interval, both resampling whole families, so that a gain resting on a few
//! families shows as a wide interval.
//!
//! A caller without the examiner's signed verdicts decides with
//! [`splinter_eval::promotion::decide_unexamined`] on these measurements; the
//! decision record then says it is not exam-verified, and why.

use std::collections::BTreeMap;

use splinter_core::rng::Seed;
use splinter_eval::paired::PairedOutcome;
use splinter_eval::promotion::{Measurement, ALPHA};

use crate::powered::analysis::Resampled;

/// Resamples a measurement's bounds are read from.
const RESAMPLES: usize = 10_000;

/// The measurement over families given as `(sum of the paired differences,
/// paired tasks)`, a difference being the candidate's verdict less the
/// predecessor's on one task, resampled from the generator that starts at
/// `seed`. With no paired task the gain is absent; with fewer than two
/// families there is no interval and no bound.
#[must_use]
pub(crate) fn measurement(families: &[(f64, usize)], seed: Seed) -> Measurement {
    let items: usize = families.iter().map(|(_, n)| n).sum();
    let gain = (items > 0).then(|| families.iter().map(|(d, _)| d).sum::<f64>() / items as f64);
    let resampled = Resampled::of(families, RESAMPLES, seed);
    Measurement {
        items: u64::try_from(items).unwrap_or(u64::MAX),
        gain,
        interval: resampled
            .as_ref()
            .map(|r| (r.quantile(ALPHA / 2.0), r.quantile(1.0 - ALPHA / 2.0))),
        one_sided_lower: resampled.as_ref().map(|r| r.quantile(ALPHA)),
    }
}

/// The measurement over the tasks of `outcomes` both arms were graded on,
/// a task in the family its cluster names (a task of no cluster is a family
/// of its own).
#[must_use]
pub(crate) fn measurement_of(outcomes: &[PairedOutcome], seed: Seed) -> Measurement {
    let mut families: BTreeMap<&str, (f64, usize)> = BTreeMap::new();
    for outcome in outcomes {
        let Some((candidate, predecessor)) = outcome.paired() else {
            continue;
        };
        let family = outcome.cluster.as_deref().unwrap_or(outcome.item.as_str());
        let entry = families.entry(family).or_default();
        entry.0 += f64::from(candidate) - f64::from(predecessor);
        entry.1 += 1;
    }
    measurement(&families.into_values().collect::<Vec<_>>(), seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    // seed: the specs read which side of a bound the resamples fall on,
    // which one fixed stream answers the same way on every run.
    const SEED: Seed = Seed::new(3);

    fn outcome(item: &str, family: &str, candidate: bool, predecessor: bool) -> PairedOutcome {
        PairedOutcome {
            item: item.into(),
            candidate: Some(candidate),
            baseline: Some(predecessor),
            cluster: Some(family.into()),
        }
    }

    #[test]
    fn a_gain_in_every_family_has_an_interval_above_zero() {
        let outcomes: Vec<PairedOutcome> = (0..6)
            .map(|i| outcome(&format!("t{i}"), &format!("f{i}"), true, false))
            .collect();
        let m = measurement_of(&outcomes, SEED);
        assert_eq!((m.items, m.gain), (6, Some(1.0)));
        assert_eq!(m.interval, Some((1.0, 1.0)));
        assert_eq!(m.one_sided_lower, Some(1.0));
    }

    /// Unanimous families resample to one mean: two of them already give an
    /// interval of no width, above zero. The count of families is not in the
    /// interval; the sign test reported beside it shows it.
    #[test]
    fn two_unanimous_families_give_an_interval_of_no_width() {
        let outcomes = [
            outcome("a", "f1", true, false),
            outcome("b", "f2", true, false),
        ];
        let m = measurement_of(&outcomes, SEED);
        assert_eq!(
            (m.interval, m.one_sided_lower),
            (Some((1.0, 1.0)), Some(1.0))
        );
    }

    #[test]
    fn one_family_has_a_gain_but_no_interval_and_unpaired_tasks_are_left_out() {
        let mut outcomes = vec![
            outcome("a", "only", true, false),
            outcome("b", "only", true, true),
        ];
        outcomes.push(PairedOutcome {
            item: "c".into(),
            candidate: None,
            baseline: Some(true),
            cluster: Some("other".into()),
        });
        let m = measurement_of(&outcomes, SEED);
        assert_eq!((m.items, m.gain), (2, Some(0.5)));
        assert_eq!((m.interval, m.one_sided_lower), (None, None));
        assert_eq!(measurement(&[], SEED).gain, None);
    }

    #[test]
    fn the_one_sided_bound_lies_inside_the_two_sided_interval() {
        let outcomes: Vec<PairedOutcome> = (0..40)
            .map(|i| {
                outcome(
                    &format!("t{i}"),
                    &format!("f{}", i % 10),
                    i % 3 != 0,
                    i % 4 == 0,
                )
            })
            .collect();
        let m = measurement_of(&outcomes, SEED);
        let (low, high) = m.interval.unwrap();
        let lower = m.one_sided_lower.unwrap();
        assert!(low <= lower && lower <= high, "{m:?}");
        assert!(
            low < high,
            "mixed families resample to different means: {m:?}"
        );
    }
}
