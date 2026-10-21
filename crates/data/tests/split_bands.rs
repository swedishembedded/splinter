// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for leakage-free evaluation of
// models trained on longitudinal data, for its clients. If your team needs
// expertise in train/test hygiene, temporal validation or external
// validation designs, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: a temporal split in three bands (oldest train, then validation, the
//! newest test) keeps every group in one band, is a pure function of its
//! boundaries and has a content address; and a record of one part whose
//! content duplicates or nearly duplicates a record of another part is found,
//! reported by pair, and refused unless it is left out.

#![allow(clippy::unwrap_used)]

use std::collections::{BTreeMap, BTreeSet};

use splinter_core::digest::Digest;
use splinter_data::partition::Unit;
use splinter_data::split::{
    content_overlaps, temporal_bands, LeakageError, Member, OverlapRule, Part, SplitKind,
};

fn member(id: &str, group: &str, at: f64) -> Member {
    Member {
        unit: Unit {
            id: id.into(),
            group: group.into(),
            stratum: String::new(),
        },
        source: "history".into(),
        at,
    }
}

/// Twenty commits at positions 0..20, two records each (one per style), a
/// commit being the group.
fn commits() -> Vec<Member> {
    (0..20)
        .flat_map(|at| {
            ["terse", "detailed"]
                .map(|style| member(&format!("c{at}/{style}"), &format!("c{at}"), f64::from(at)))
        })
        .collect()
}

fn data() -> Digest {
    Digest::of(b"history")
}

#[test]
fn the_bands_are_cut_by_position_and_a_commit_stays_in_one_band() {
    let m = commits();
    // Positions 0..=13 train, 14..=16 validation, 17.. test.
    let split = temporal_bands(&m, 13.5, 16.5, 1, &data()).unwrap();
    assert_eq!(split.ids(Part::Train).len(), 28);
    assert_eq!(split.ids(Part::Validation).len(), 6);
    assert_eq!(split.ids(Part::Test).len(), 6);
    for part in Part::ALL {
        for id in split.ids(part) {
            let (commit, _) = id.split_once('/').unwrap();
            let sibling = |style: &str| format!("{commit}/{style}");
            assert!(split.ids(part).contains(&sibling("terse")));
            assert!(split.ids(part).contains(&sibling("detailed")));
        }
    }
    assert_eq!(split.verify_members(&m), Ok(()));
    assert!(matches!(split.kind, SplitKind::TemporalBands { .. }));
}

#[test]
fn the_same_boundaries_give_the_same_content_address_and_other_boundaries_another() {
    let m = commits();
    let a = temporal_bands(&m, 13.5, 16.5, 1, &data()).unwrap();
    let b = temporal_bands(&m, 13.5, 16.5, 1, &data()).unwrap();
    let c = temporal_bands(&m, 12.5, 16.5, 1, &data()).unwrap();
    assert_eq!(a.digest().unwrap(), b.digest().unwrap());
    assert_ne!(a.digest().unwrap(), c.digest().unwrap());
}

#[test]
fn a_group_whose_members_fall_in_two_bands_is_left_out_and_counted() {
    let mut m = commits();
    // One commit whose second record is dated after the train boundary.
    m.push(member("late/extra", "c3", 15.0));
    let split = temporal_bands(&m, 13.5, 16.5, 1, &data()).unwrap();
    assert_eq!(split.excluded.get("straddles_cutoff"), Some(&3));
    assert!(!split
        .ids(Part::Train)
        .iter()
        .any(|id| id.starts_with("c3/")));
    assert_eq!(split.verify_members(&m), Ok(()));
}

#[test]
fn boundaries_that_are_not_ordered_are_refused() {
    let m = commits();
    assert!(temporal_bands(&m, 16.5, 13.5, 1, &data()).is_err());
    assert!(temporal_bands(&m, f64::NAN, 13.5, 1, &data()).is_err());
}

fn features(lines: &[&str]) -> BTreeSet<String> {
    lines.iter().map(|l| (*l).to_string()).collect()
}

const RULE: OverlapRule = OverlapRule {
    min_similarity: 0.8,
    min_features: 3,
};

fn with_features(entries: &[(&str, &[&str])]) -> BTreeMap<String, BTreeSet<String>> {
    entries
        .iter()
        .map(|(id, lines)| ((*id).to_string(), features(lines)))
        .collect()
}

#[test]
fn a_test_record_that_repeats_a_training_record_is_reported_as_a_pair() {
    let m = commits();
    let split = temporal_bands(&m, 13.5, 16.5, 1, &data()).unwrap();
    let same = ["a = 1", "b = 2", "c = 3", "d = 4", "e = 5"];
    let other = ["x", "y", "z", "w", "v"];
    let mut table: BTreeMap<String, BTreeSet<String>> = m
        .iter()
        .map(|m| {
            (
                m.unit.id.clone(),
                features(&[&format!("own {}", m.unit.id), "p", "q"]),
            )
        })
        .collect();
    table.extend(with_features(&[
        ("c2/terse", &same),
        ("c18/terse", &same),
        ("c5/terse", &other),
    ]));
    let found = content_overlaps(&split, &table, &RULE);
    assert_eq!(found.len(), 1, "{found:?}");
    assert_eq!(
        (found[0].a.as_str(), found[0].b.as_str()),
        ("c2/terse", "c18/terse")
    );
    assert_eq!(
        (found[0].part_a, found[0].part_b),
        (Part::Train, Part::Test)
    );
    assert!((found[0].similarity - 1.0).abs() < 1e-9);
}

#[test]
fn near_duplicates_count_and_dissimilar_records_do_not() {
    let m = commits();
    let split = temporal_bands(&m, 13.5, 16.5, 1, &data()).unwrap();
    let nine: Vec<String> = (0..9).map(|i| format!("line {i}")).collect();
    let mut near = nine.clone();
    near.push("one more line".into());
    let table = with_features(&[
        (
            "c2/terse",
            &nine.iter().map(String::as_str).collect::<Vec<_>>(),
        ),
        (
            "c18/terse",
            &near.iter().map(String::as_str).collect::<Vec<_>>(),
        ),
        (
            "c19/terse",
            &["line 0", "line 1", "something else", "and more", "also"],
        ),
    ]);
    let found = content_overlaps(&split, &table, &RULE);
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(found[0].similarity >= 0.8 && found[0].similarity < 1.0);
}

#[test]
fn records_in_the_same_part_may_repeat_each_other_and_too_small_records_are_not_compared() {
    let m = commits();
    let split = temporal_bands(&m, 13.5, 16.5, 1, &data()).unwrap();
    let same = ["a", "b", "c", "d"];
    let table = with_features(&[
        ("c1/terse", &same),
        ("c2/terse", &same),
        ("c18/terse", &["tiny"]),
        ("c19/terse", &["tiny"]),
    ]);
    assert!(content_overlaps(&split, &table, &RULE).is_empty());
}

#[test]
fn a_split_with_an_overlap_is_refused_by_count_until_the_overlapping_side_is_left_out() {
    let m = commits();
    let split = temporal_bands(&m, 13.5, 16.5, 1, &data()).unwrap();
    let same = ["a", "b", "c", "d"];
    let table = with_features(&[("c2/terse", &same), ("c18/terse", &same)]);
    assert_eq!(
        split.verify_no_overlap(&table, &RULE),
        Err(LeakageError::ContentOverlap { pairs: 1 })
    );
    let cleaned = split.without_overlaps(&content_overlaps(&split, &table, &RULE));
    // The record that is nearer the training side goes; the held-out one stays.
    assert!(!cleaned.ids(Part::Train).contains(&"c2/terse".to_string()));
    assert!(cleaned.ids(Part::Test).contains(&"c18/terse".to_string()));
    assert_eq!(cleaned.excluded.get("content_overlap"), Some(&1));
    assert_eq!(cleaned.verify_no_overlap(&table, &RULE), Ok(()));
    assert_ne!(cleaned.digest().unwrap(), split.digest().unwrap());
}
