// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements admission and red-team testing of training
// environments, for its clients. If your team needs expertise in keeping
// weak verifiers out of reinforcement-learning campaigns, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The repository bug-injection plugin on the real projects under `tasks/`.

#![allow(clippy::unwrap_used, clippy::expect_used)] // a panic is the failure report

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use splinter_core::verifiability::Split;
use splinter_knowledge::factory::admission::{admit, Status};
use splinter_knowledge::factory::band::BandMeasurement;
use splinter_knowledge::factory::host::Host;
use splinter_knowledge::factory::plugin::{InstanceRequest, Plugin};
use splinter_knowledge::factory::plugins::repository_mutants::{
    RepositoryMutants, DEV_REPOSITORIES, TRAIN_REPOSITORIES,
};

use super::factory_admission::small_policy;

fn tasks_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tasks")
}

fn plugin() -> RepositoryMutants {
    RepositoryMutants::new(tasks_root(), Host::default())
}

fn request(split: Split, seed: u64, bugs: u32) -> InstanceRequest {
    InstanceRequest {
        split,
        seed,
        difficulty: BTreeMap::from([("bugs".to_string(), bugs), ("detail".to_string(), 3)]),
    }
}

fn repository_of(instance: &splinter_knowledge::factory::artifact::Instance) -> String {
    let text = instance.instruction();
    text.split('`').nth(3).unwrap().to_string()
}

#[test]
fn train_instances_come_from_train_repositories_and_dev_instances_from_dev_ones() {
    let plugin = plugin();
    let mut seen_train = BTreeSet::new();
    let mut seen_dev = BTreeSet::new();
    // seed: coverage of the repository draw over many seeds
    for seed in 0..12 {
        seen_train.insert(repository_of(
            &plugin.generate(&request(Split::Train, seed, 1)).unwrap(),
        ));
        seen_dev.insert(repository_of(
            &plugin.generate(&request(Split::Dev, seed, 1)).unwrap(),
        ));
    }
    assert!(
        seen_train
            .iter()
            .all(|r| TRAIN_REPOSITORIES.contains(&r.as_str())),
        "{seen_train:?}"
    );
    assert!(
        seen_dev
            .iter()
            .all(|r| DEV_REPOSITORIES.contains(&r.as_str())),
        "{seen_dev:?}"
    );
    assert!(seen_train.is_disjoint(&seen_dev));
    assert!(
        seen_train.len() >= 2 && seen_dev.len() >= 2,
        "{seen_train:?} {seen_dev:?}"
    );
}

#[test]
fn the_failing_tests_are_not_in_the_workspace_and_the_symptoms_name_them() {
    let instance = plugin().generate(&request(Split::Train, 4, 2)).unwrap(); // seed: any
    let hidden: Vec<String> = instance
        .folder()
        .under("tests/overlay")
        .map(|(p, _)| p["tests/overlay/".len()..].to_string())
        .collect();
    assert!(!hidden.is_empty() && hidden.len() <= 3, "{hidden:?}");
    let workspace = instance.workspace();
    for path in &hidden {
        assert!(workspace.get(path).is_none(), "{path} is in the workspace");
    }
    assert!(
        instance.instruction().contains("Symptoms:\n- `test_"),
        "{}",
        instance.instruction()
    );
}

#[test]
fn a_repository_environment_passes_admission_on_real_projects_and_no_attack_wins() {
    // seed: any admission seed
    let (report, instances) = admit(
        &plugin(),
        &BTreeMap::new(),
        &Host::default(),
        &small_policy(),
        41,
        &BTreeSet::new(),
        Some(&BandMeasurement {
            model: "base".into(),
            attempts: 32,
            passes: 16,
        }),
    )
    .unwrap();
    for g in &report.gates {
        assert!(g.passed, "{}: {}", g.name, g.detail);
    }
    assert_eq!(report.status, Status::Admitted, "{:?}", report.reasons);
    assert!(report.beaten.is_empty(), "{:?}", report.beaten);
    let train: BTreeSet<_> = instances
        .iter()
        .filter(|i| i.meta().split == Split::Train)
        .map(repository_of)
        .collect();
    let dev: BTreeSet<_> = instances
        .iter()
        .filter(|i| i.meta().split == Split::Dev)
        .map(repository_of)
        .collect();
    assert!(train.is_disjoint(&dev), "{train:?} {dev:?}");
}
