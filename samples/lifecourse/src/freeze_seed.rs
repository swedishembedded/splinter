// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for held-out evaluation that
// cannot be regenerated after the fact for its clients. If your team needs
// expertise in preregistered, leakage-free evaluation of predictive models,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! The seed the partition of a data directory was frozen with.
//!
//! The locked test and the folds are a function of a seed and are pinned by
//! content. The seed is recorded beside them when the partition is first
//! made, and a later `freeze` that is given no seed reads it back. A directory
//! partitioned before seed manifests existed states its seed in its
//! `partition.json`; that is the seed recorded for it.

use std::path::Path;

use anyhow::{Context, Result};
use splinter_sdk::data::partition::{Partition, PartitionSpec};
use splinter_sdk::seed::FrozenSeed;

use crate::commands::PARTITION;

const MANIFEST: &str = "SEED.json";

/// The seed to partition `data` with: the recorded one, else `given`, else the
/// protocol's. A `given` that differs from the recorded one is refused. The
/// caller records it with [`FrozenSeed::record`] once the partition is made.
pub fn partition_seed(data: &Path, given: Option<u64>) -> Result<FrozenSeed> {
    let existing = data.join(PARTITION);
    let legacy = if existing.exists() {
        let partition: Partition = serde_json::from_slice(&std::fs::read(&existing)?)
            .with_context(|| format!("{}: not a partition", existing.display()))?;
        Some(partition.spec.seed)
    } else {
        None
    };
    Ok(FrozenSeed::decide(
        &data.join(MANIFEST),
        given,
        PartitionSpec::default().seed,
        legacy,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use splinter_sdk::data::partition::{partition, Unit};
    use splinter_sdk::vocabulary::digest::Digest;

    fn recorded(data: &Path, given: Option<u64>) -> u64 {
        partition_seed(data, given).unwrap().record().unwrap()
    }

    #[test]
    fn a_second_freeze_without_a_seed_is_accepted_and_another_seed_refused() {
        let dir = tempfile::tempdir().unwrap();
        // seed: any value but the protocol's shows the recorded one is read back
        assert_eq!(recorded(dir.path(), Some(9)), 9);
        assert_eq!(recorded(dir.path(), None), 9);
        assert!(partition_seed(dir.path(), Some(10)).is_err());
    }

    #[test]
    fn a_partition_made_before_manifests_gains_one_with_its_own_seed() {
        let dir = tempfile::tempdir().unwrap();
        let units: Vec<Unit> = (0..40)
            .map(|n| Unit {
                id: format!("u{n}"),
                group: format!("u{n}"),
                stratum: "a".into(),
            })
            .collect();
        let spec = PartitionSpec {
            seed: 7,
            ..PartitionSpec::default()
        };
        let made = partition(&units, Digest::of(b"data"), &spec).unwrap();
        std::fs::write(dir.path().join(PARTITION), made.canonical().unwrap()).unwrap();

        assert_eq!(recorded(dir.path(), None), 7);
        let manifest = std::fs::read_to_string(dir.path().join(MANIFEST)).unwrap();
        assert!(manifest.contains("legacy"), "{manifest}");
        assert!(partition_seed(dir.path(), Some(8)).is_err());
    }
}
