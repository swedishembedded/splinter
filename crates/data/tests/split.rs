// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for leakage-free evaluation of
// models trained on longitudinal data, for its clients. If your team needs
// expertise in train/test hygiene, temporal validation or external
// validation designs, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: no group is ever in two parts of a split, a temporal split keeps
//! the future out of training, a held-out source is absent from training,
//! statistics fitted beyond the training units are refused, every split is a
//! pure function of its seed, and each gate says how many, never who.
#![allow(clippy::unwrap_used)]

use std::collections::HashMap;

use splinter_core::digest::Digest;
use splinter_data::partition::{partition, PartitionSpec, Unit};
use splinter_data::split::{
    leave_one_source_out, temporal_split, verify_disjoint, DataSplit, FitLedger, LeakageError,
    Member, Part,
};

fn member(id: usize, group: usize, source: &str, at: f64) -> Member {
    Member {
        unit: Unit {
            id: format!("u{id}"),
            group: format!("g{group}"),
            stratum: String::new(),
        },
        source: source.into(),
        at,
    }
}

/// Three sources of 200 people, households of two within a source,
/// examined across 2000-2009.
fn members() -> Vec<Member> {
    (0..600)
        .map(|i| {
            let source = ["a", "b", "c"][i / 200];
            member(i, i / 2, source, 2000.0 + (i % 10) as f64 + 0.5)
        })
        .collect()
}

fn data() -> Digest {
    Digest::of(b"dataset")
}

/// Moves the first test unit and the rest of its group to training: a
/// planted mistake that keeps every group whole.
fn plant_test_group_in_training(split: &mut DataSplit, members: &[Member]) -> usize {
    let first = split.ids(Part::Test)[0].clone();
    let group = &members
        .iter()
        .find(|m| m.unit.id == first)
        .unwrap()
        .unit
        .group;
    let moved: Vec<String> = members
        .iter()
        .filter(|m| &m.unit.group == group)
        .map(|m| m.unit.id.clone())
        .collect();
    split
        .parts
        .get_mut(&Part::Test)
        .unwrap()
        .retain(|id| !moved.contains(id));
    split
        .parts
        .get_mut(&Part::Train)
        .unwrap()
        .extend(moved.iter().cloned());
    moved.len()
}

#[test]
fn a_planted_group_in_two_parts_is_caught_by_count_not_by_name() {
    let m = members();
    let units: Vec<Unit> = m.iter().map(|x| x.unit.clone()).collect();
    let train = &units[..100];
    let test = &units[100..200];
    assert_eq!(
        verify_disjoint(&[(Part::Train, train), (Part::Test, test)]),
        Ok(())
    );
    // u99 and u100 share the household g50 only if split mid-pair: plant it.
    let mut planted = test.to_vec();
    planted.push(units[3].clone());
    let error = verify_disjoint(&[(Part::Train, train), (Part::Test, &planted)]).unwrap_err();
    assert_eq!(
        error,
        LeakageError::SharedGroups {
            groups: 1,
            units: 1
        }
    );
    let text = error.to_string();
    assert!(text.contains("1 group") && !text.contains("g1") && !text.contains("u3"));
}

#[test]
fn temporal_split_puts_the_cutoff_between_training_and_test() {
    let m = members();
    let split = temporal_split(&m, 2004.9, 0.2, 7, &data()).unwrap();
    let at: HashMap<&str, f64> = m.iter().map(|x| (x.unit.id.as_str(), x.at)).collect();
    assert!(split
        .ids(Part::Train)
        .iter()
        .all(|id| at[id.as_str()] <= 2004.9));
    assert!(split
        .ids(Part::Validation)
        .iter()
        .all(|id| at[id.as_str()] <= 2004.9));
    assert!(split
        .ids(Part::Test)
        .iter()
        .all(|id| at[id.as_str()] > 2004.9));
    assert!(!split.ids(Part::Validation).is_empty());
    // Households of two whose members straddle the cutoff are left out.
    let straddling = split.excluded.get("straddles_cutoff").copied().unwrap_or(0);
    let kept: usize = Part::ALL.iter().map(|p| split.ids(*p).len()).sum();
    assert_eq!(kept + straddling, m.len());
    assert_eq!(split.verify_members(&m), Ok(()));
    // A planted unit after the cutoff in training is caught.
    let mut bad = split.clone();
    let moved = plant_test_group_in_training(&mut bad, &m);
    assert_eq!(
        bad.verify_members(&m),
        Err(LeakageError::WrongSideOfCutoff { units: moved })
    );
}

#[test]
fn a_held_out_source_is_absent_from_training_and_validation() {
    let m = members();
    let split = leave_one_source_out(&m, "b", 0.2, 3, &data()).unwrap();
    let source: HashMap<&str, &str> = m
        .iter()
        .map(|x| (x.unit.id.as_str(), x.source.as_str()))
        .collect();
    for part in [Part::Train, Part::Validation] {
        assert!(split.ids(part).iter().all(|id| source[id.as_str()] != "b"));
    }
    assert_eq!(split.ids(Part::Test).len(), 200);
    assert!(split
        .ids(Part::Test)
        .iter()
        .all(|id| source[id.as_str()] == "b"));
    let mut bad = split.clone();
    let moved = plant_test_group_in_training(&mut bad, &m);
    assert_eq!(
        bad.verify_members(&m),
        Err(LeakageError::SourceNotHeldOut { units: moved })
    );
    assert!(leave_one_source_out(&m, "zzz", 0.2, 3, &data()).is_err());
}

#[test]
fn identical_seeds_make_identical_splits_and_other_seeds_other_ones() {
    let m = members();
    let a = temporal_split(&m, 2004.9, 0.2, 7, &data()).unwrap();
    let mut reversed = m.clone();
    reversed.reverse();
    let b = temporal_split(&reversed, 2004.9, 0.2, 7, &data()).unwrap();
    assert_eq!(a.digest().unwrap(), b.digest().unwrap());
    let c = temporal_split(&m, 2004.9, 0.2, 8, &data()).unwrap();
    assert_ne!(a.digest().unwrap(), c.digest().unwrap());
    let x = leave_one_source_out(&m, "a", 0.2, 1, &data()).unwrap();
    let y = leave_one_source_out(&m, "a", 0.2, 1, &data()).unwrap();
    assert_eq!(x.digest().unwrap(), y.digest().unwrap());
    assert_ne!(x.digest().unwrap(), a.digest().unwrap());
}

#[test]
fn statistics_fitted_beyond_the_training_units_are_refused_with_counts() {
    let m = members();
    let split = temporal_split(&m, 2004.9, 0.2, 7, &data()).unwrap();
    let mut honest = FitLedger::new();
    split
        .ids(Part::Train)
        .iter()
        .for_each(|id| honest.consume(id));
    let certificate = split.certify_fit(&honest).unwrap();
    assert_eq!(certificate.units, split.ids(Part::Train).len());
    assert_eq!(certificate.split, split.digest().unwrap());

    let mut leaky = honest.clone();
    leaky.consume(&split.ids(Part::Test)[0]);
    leaky.consume(&split.ids(Part::Validation)[0]);
    assert_eq!(
        split.certify_fit(&leaky),
        Err(LeakageError::FittedOutsideTraining {
            units: 2,
            held_out: 2
        })
    );
    assert_eq!(
        split.certify_fit(&FitLedger::new()),
        Err(LeakageError::FittedOnNothing)
    );
}

#[test]
fn a_participant_exclusive_split_comes_from_a_partition() {
    let m = members();
    let units: Vec<Unit> = m.iter().map(|x| x.unit.clone()).collect();
    let p = partition(&units, data(), &PartitionSpec::default()).unwrap();
    let split = DataSplit::from_partition(&p, &units, 1, 2).unwrap();
    assert_eq!(split.ids(Part::Test), p.locked.as_slice());
    let total: usize = Part::ALL.iter().map(|x| split.ids(*x).len()).sum();
    assert_eq!(total, units.len());
    assert!(DataSplit::from_partition(&p, &units, 9, 0).is_err());
}
