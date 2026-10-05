// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Partitions of a record dataset for honest evaluation, decided once and
//! pinned before anything is trained.
//!
//! - A **locked test** share, drawn per stratum, that only a candidate
//!   declared final is ever scored on. This is where a claim is made.
//! - **Repeated grouped stratified K-fold cross-validation** over the rest:
//!   every repeat reshuffles, every fold's validation part is held out from
//!   everything that fold does, so the spread across folds is the measure of
//!   how stable a result is.
//!
//! A group (a household, a site, a person seen twice) is never divided
//! between parts, and every stratum (an outcome, a survey cycle, an age band)
//! is spread evenly across them. The partition is a pure function of the
//! units, the dataset digest and the spec; its canonical JSON has a digest,
//! so a frozen ledger can pin it and every result can name it.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use splinter_core::digest::{canonical_json, Digest};
use thiserror::Error;

/// One record as the partition sees it.
#[derive(Clone, Debug, PartialEq)]
pub struct Unit {
    /// The record's unique id.
    pub id: String,
    /// Records sharing a group stay together.
    pub group: String,
    /// The stratum to balance across parts.
    pub stratum: String,
}

/// How to partition.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PartitionSpec {
    /// Share of the units (per stratum) held out as the locked test.
    pub locked_share: f64,
    /// Cross-validation repeats over the rest.
    pub repeats: u32,
    /// Folds per repeat.
    pub folds: u32,
    /// Seed of every shuffle.
    pub seed: u64,
}

impl Default for PartitionSpec {
    /// The evaluation protocol's: 15% locked, five repeats of five folds.
    fn default() -> Self {
        PartitionSpec {
            locked_share: 0.15,
            repeats: 5,
            folds: 5,
            seed: 20_261_005,
        }
    }
}

/// Why units cannot be partitioned.
#[derive(Debug, Error, PartialEq)]
pub enum PartitionError {
    /// Two units share an id.
    #[error("unit id {0} appears more than once")]
    DuplicateId(String),
    /// The spec cannot be met.
    #[error("invalid partition spec: {0}")]
    Spec(String),
}

/// A partition: the locked test ids, and per repeat and fold the ids of that
/// fold's validation part (its training part is every other non-locked id).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Partition {
    /// The dataset the units came from.
    pub dataset: Digest,
    /// The spec that produced it.
    pub spec: PartitionSpec,
    /// Locked test ids, sorted.
    pub locked: Vec<String>,
    /// `repeats[r][k]`: validation ids of fold `k` in repeat `r`, sorted.
    pub repeats: Vec<Vec<Vec<String>>>,
}

/// splitmix64: a small, well-mixed generator, so a partition depends on
/// nothing outside this file.
struct Mix(u64);

impl Mix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = (self.next() % (i as u64 + 1)) as usize;
            v.swap(i, j);
        }
    }
}

struct Group {
    ids: Vec<String>,
}

/// Groups by stratum (a group's stratum is its most common unit stratum,
/// ties to the smaller name), each stratum's groups in name order.
fn groups(units: &[Unit]) -> Result<BTreeMap<String, Vec<Group>>, PartitionError> {
    let mut seen = std::collections::HashSet::new();
    let mut by_group: BTreeMap<&str, Vec<&Unit>> = BTreeMap::new();
    for u in units {
        if !seen.insert(u.id.as_str()) {
            return Err(PartitionError::DuplicateId(u.id.clone()));
        }
        by_group.entry(&u.group).or_default().push(u);
    }
    let mut strata: BTreeMap<String, Vec<Group>> = BTreeMap::new();
    for members in by_group.into_values() {
        let mut count: BTreeMap<&str, usize> = BTreeMap::new();
        for m in &members {
            *count.entry(&m.stratum).or_default() += 1;
        }
        let stratum = count
            .iter()
            .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
            .map(|(s, _)| s.to_string())
            .unwrap_or_default(); // every group has at least one member
        let mut ids: Vec<String> = members.iter().map(|m| m.id.clone()).collect();
        ids.sort();
        strata.entry(stratum).or_default().push(Group { ids });
    }
    Ok(strata)
}

/// Partition `units` of the dataset `dataset` by `spec`.
pub fn partition(
    units: &[Unit],
    dataset: Digest,
    spec: &PartitionSpec,
) -> Result<Partition, PartitionError> {
    if !(0.0..1.0).contains(&spec.locked_share) || spec.folds < 2 || spec.repeats == 0 {
        return Err(PartitionError::Spec(format!(
            "locked share in [0, 1), at least two folds and one repeat, got {spec:?}"
        )));
    }
    let strata = groups(units)?;
    let mut mix = Mix(spec.seed);
    let mut locked = Vec::new();
    let mut rest: Vec<Vec<&Group>> = Vec::new();
    for groups in strata.values() {
        let mut order: Vec<&Group> = groups.iter().collect();
        mix.shuffle(&mut order);
        let total: usize = order.iter().map(|g| g.ids.len()).sum();
        let target = spec.locked_share * total as f64;
        let mut taken = 0usize;
        let mut kept = Vec::new();
        for g in order {
            // Take a group while that brings the locked count closer to its target.
            if (taken as f64 + g.ids.len() as f64 - target).abs() < (taken as f64 - target).abs() {
                taken += g.ids.len();
                locked.extend(g.ids.iter().cloned());
            } else {
                kept.push(g);
            }
        }
        rest.push(kept);
    }
    let k = spec.folds as usize;
    let mut repeats = Vec::with_capacity(spec.repeats as usize);
    for r in 0..spec.repeats {
        let mut mix = Mix(spec.seed ^ (0xA076_1D64_78BD_642F_u64.wrapping_mul(r as u64 + 1)));
        let mut folds: Vec<Vec<String>> = vec![Vec::new(); k];
        for groups in &rest {
            let mut order = groups.clone();
            mix.shuffle(&mut order);
            // Largest groups first, then the least-filled fold of this stratum.
            order.sort_by_key(|g| std::cmp::Reverse(g.ids.len()));
            let mut fill = vec![0usize; k];
            for g in order {
                let f = (0..k).min_by_key(|&f| (fill[f], f)).unwrap_or(0); // k >= 2
                fill[f] += g.ids.len();
                folds[f].extend(g.ids.iter().cloned());
            }
        }
        for f in &mut folds {
            f.sort();
        }
        repeats.push(folds);
    }
    locked.sort();
    Ok(Partition {
        dataset,
        spec: spec.clone(),
        locked,
        repeats,
    })
}

impl Partition {
    /// The ids of fold `k` of repeat `r`: `(training, validation)`.
    pub fn fold(&self, r: usize, k: usize) -> (Vec<&str>, Vec<&str>) {
        let validation: Vec<&str> = self.repeats[r][k].iter().map(String::as_str).collect();
        let training = self.repeats[r]
            .iter()
            .enumerate()
            .filter(|(f, _)| *f != k)
            .flat_map(|(_, ids)| ids.iter().map(String::as_str))
            .collect();
        (training, validation)
    }

    /// The canonical JSON a frozen ledger pins.
    pub fn canonical(&self) -> Result<Vec<u8>, serde_json::Error> {
        canonical_json(self)
    }

    /// The partition's own content address.
    pub fn digest(&self) -> Result<Digest, serde_json::Error> {
        Ok(Digest::of(&self.canonical()?))
    }
}

/// Units from JSON lines that carry `subject_id` and optionally `group_id`
/// (the `timeline-v1` fields), with `stratum` computed from each record.
pub fn units_from_json_lines(
    text: &str,
    stratum: impl Fn(&serde_json::Value) -> String,
) -> Result<Vec<Unit>, String> {
    text.lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(n, l)| {
            let v: serde_json::Value =
                serde_json::from_str(l).map_err(|e| format!("line {}: {e}", n + 1))?;
            let id = v["subject_id"]
                .as_str()
                .ok_or_else(|| format!("line {}: no subject_id", n + 1))?
                .to_string();
            let group = v["group_id"]
                .as_str()
                .map(str::to_string)
                .unwrap_or_else(|| id.clone());
            Ok(Unit {
                stratum: stratum(&v),
                id,
                group,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn units(n: usize) -> Vec<Unit> {
        (0..n)
            .map(|i| Unit {
                id: format!("u{i}"),
                group: format!("g{}", i / 3), // groups of three
                // A stratum is a property of the group (the same person's
                // records share an outcome), so every seventh group is one.
                stratum: if (i / 3) % 7 == 0 {
                    "event".into()
                } else {
                    "none".into()
                },
            })
            .collect()
    }

    #[test]
    fn every_unit_lands_once_groups_stay_whole_and_strata_balance() {
        let u = units(3000);
        let p = partition(&u, Digest::of(b"data"), &PartitionSpec::default()).unwrap();
        let group_of: BTreeMap<&str, &str> = u
            .iter()
            .map(|x| (x.id.as_str(), x.group.as_str()))
            .collect();
        let stratum_of: BTreeMap<&str, &str> = u
            .iter()
            .map(|x| (x.id.as_str(), x.stratum.as_str()))
            .collect();
        let expected = u.iter().filter(|x| x.stratum == "event").count() as f64 / u.len() as f64;
        for r in 0..5 {
            let mut all: Vec<&str> = p.locked.iter().map(String::as_str).collect();
            for k in 0..5 {
                let (train, val) = p.fold(r, k);
                assert_eq!(train.len() + val.len() + p.locked.len(), u.len());
                all.extend(&val);
                let vg: std::collections::HashSet<&str> =
                    val.iter().map(|id| group_of[id]).collect();
                assert!(
                    train.iter().all(|id| !vg.contains(group_of[id])),
                    "a group was divided"
                );
                let events = val.iter().filter(|id| stratum_of[*id] == "event").count() as f64
                    / val.len() as f64;
                assert!(
                    (events - expected).abs() < 0.01,
                    "fold {r}/{k} event share {events} vs {expected}"
                );
            }
            all.sort();
            all.dedup();
            assert_eq!(
                all.len(),
                u.len(),
                "locked and folds cover every unit exactly once"
            );
        }
        let share = p.locked.len() as f64 / u.len() as f64;
        assert!((share - 0.15).abs() < 0.01, "locked share {share}");
        let locked_events = p
            .locked
            .iter()
            .filter(|id| stratum_of[id.as_str()] == "event")
            .count() as f64
            / p.locked.len() as f64;
        assert!((locked_events - expected).abs() < 0.01);
        assert_ne!(p.repeats[0], p.repeats[1], "every repeat reshuffles");
    }

    #[test]
    fn the_partition_is_a_pure_function_of_its_inputs() {
        let u = units(600);
        let a = partition(&u, Digest::of(b"d"), &PartitionSpec::default()).unwrap();
        let mut shuffled = u.clone();
        shuffled.reverse();
        let b = partition(&shuffled, Digest::of(b"d"), &PartitionSpec::default()).unwrap();
        assert_eq!(
            a.digest().unwrap(),
            b.digest().unwrap(),
            "input order does not matter"
        );
        let c = partition(
            &u,
            Digest::of(b"d"),
            &PartitionSpec {
                seed: 1,
                ..Default::default()
            },
        )
        .unwrap();
        assert_ne!(a.digest().unwrap(), c.digest().unwrap());
    }

    #[test]
    fn duplicates_and_bad_specs_are_refused() {
        let mut u = units(10);
        u.push(u[0].clone());
        assert_eq!(
            partition(&u, Digest::of(b"d"), &PartitionSpec::default()),
            Err(PartitionError::DuplicateId("u0".into()))
        );
        let spec = PartitionSpec {
            folds: 1,
            ..Default::default()
        };
        assert!(matches!(
            partition(&units(10), Digest::of(b"d"), &spec),
            Err(PartitionError::Spec(_))
        ));
    }

    #[test]
    fn units_read_from_timeline_lines() {
        let text = "{\"subject_id\":\"a\",\"group_id\":\"h1\",\"x\":1}\n\n{\"subject_id\":\"b\",\"x\":2}\n";
        let u = units_from_json_lines(text, |v| v["x"].to_string()).unwrap();
        assert_eq!(
            u[0],
            Unit {
                id: "a".into(),
                group: "h1".into(),
                stratum: "1".into()
            }
        );
        assert_eq!(u[1].group, "b", "no group: the record is its own group");
        assert!(units_from_json_lines("{\"x\":1}", |_| String::new())
            .unwrap_err()
            .contains("subject_id"));
    }
}
