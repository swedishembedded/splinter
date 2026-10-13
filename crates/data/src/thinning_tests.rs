// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: a projection thinned to a record or token limit keeps an even
//! spread of what it holds and counts the rest.

use super::*;

/// A projection thinned to a limit keeps an even spread of its records,
/// first and last included, and counts the rest.
#[test]
fn a_thinned_projection_keeps_an_even_spread_and_counts_the_rest() {
    let mut projection = Projection::new("voice", Objective::Sft, None, None);
    for n in 0..10 {
        projection.push(
            RecordBody::Text {
                text: n.to_string(),
            },
            Provenance::default(),
        );
    }
    let thinned = projection.clone().thinned_to(4);
    let kept: Vec<&str> = thinned
        .records
        .iter()
        .map(|r| match &r.body {
            RecordBody::Text { text } => text.as_str(),
            _ => unreachable!(),
        })
        .collect();
    assert_eq!(kept, ["0", "3", "6", "9"]);
    assert_eq!(thinned.count(Exclusion::OverLimit), 6);
    assert_eq!(
        projection
            .clone()
            .thinned_to(10)
            .count(Exclusion::OverLimit),
        0
    );
    assert_eq!(projection.thinned_to(1).records.len(), 1);
}

/// A token budget keeps a steady stride of the records, never over the
/// budget, from the first part of the projection to the last.
#[test]
fn a_token_budget_keeps_a_spread_of_the_records_within_it() {
    let mut projection = Projection::new("voice", Objective::Sft, None, None);
    for n in 0..20 {
        projection.push(
            RecordBody::Text {
                text: "w".repeat(10 + n % 3),
            },
            Provenance::default(),
        );
    }
    let size = |record: &Record| match &record.body {
        RecordBody::Text { text } => text.len(),
        _ => unreachable!(),
    };
    let total: usize = projection.records.iter().map(size).sum();
    let thinned = projection.clone().thinned_to_tokens(total / 2, &size);
    let used: usize = thinned.records.iter().map(size).sum();
    assert!(
        used <= total / 2 && used > total / 2 - 24,
        "{used} of {total}"
    );
    let kept_from_each_half = |half: std::ops::Range<usize>| {
        half.filter(|&n| thinned.records.iter().any(|r| r == &projection.records[n]))
            .count()
    };
    assert!(kept_from_each_half(0..10) >= 4 && kept_from_each_half(10..20) >= 4);
    assert_eq!(
        thinned.count(Exclusion::OverLimit),
        20 - thinned.records.len()
    );
    assert_eq!(
        projection
            .clone()
            .thinned_to_tokens(total, &size)
            .records
            .len(),
        20
    );
}
