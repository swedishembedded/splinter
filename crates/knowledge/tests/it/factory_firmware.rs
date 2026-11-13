// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements admission and red-team testing of training
// environments, for its clients. If your team needs expertise in keeping
// weak verifiers out of reinforcement-learning campaigns, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The firmware plugin against the real emulator. These tests need the `vemu`
//! binary (read only, run as a command) and `clang` with `lld`: they are
//! ignored by default and fail with the reason when a prerequisite is missing.

#![allow(clippy::unwrap_used, clippy::expect_used)] // a panic is the failure report

use std::collections::{BTreeMap, BTreeSet};

use splinter_core::verifiability::Split;
use splinter_knowledge::factory::admission::{admit, Status};
use splinter_knowledge::factory::band::BandMeasurement;
use splinter_knowledge::factory::host::{Host, Verdict};
use splinter_knowledge::factory::plugin::{InstanceRequest, Plugin};
use splinter_knowledge::factory::plugins::firmware_emulator::{
    FirmwareEmulator, DEV_MODULES, TRAIN_MODULES,
};

use super::factory_admission::small_policy;

/// The emulator binary: `$VEMU`, else `vemu` on the path. Found by a shell, so
/// that the answer is the one the person running the tests gave.
fn vemu() -> String {
    let found = std::process::Command::new("sh")
        .args(["-c", "command -v \"${VEMU:-vemu}\""])
        .output()
        .unwrap();
    let path = String::from_utf8(found.stdout).unwrap().trim().to_string();
    assert!(
        !path.is_empty(),
        "no vemu binary: set VEMU to its path or put vemu on PATH"
    );
    path
}

fn host() -> Host {
    Host::default()
        .with_variable("VEMU", &vemu())
        .with_timeout(std::time::Duration::from_secs(120))
}

fn request(split: Split, seed: u64, faults: u32) -> InstanceRequest {
    InstanceRequest {
        split,
        seed,
        difficulty: BTreeMap::from([("faults".to_string(), faults), ("detail".to_string(), 3)]),
    }
}

fn modules_of(instance: &splinter_knowledge::factory::artifact::Instance) -> BTreeSet<String> {
    instance
        .folder()
        .under("solution/files")
        .map(|(_, e)| String::from_utf8_lossy(&e.bytes).into_owned())
        .collect()
}

#[test]
#[ignore = "needs the vemu emulator and clang: set VEMU or put vemu on PATH, run with --ignored"]
fn faults_are_injected_in_train_modules_for_train_and_in_other_modules_for_dev() {
    let plugin = FirmwareEmulator::new(host());
    // seed: coverage of the module draw over a few seeds
    for seed in 0..4 {
        let train = plugin.generate(&request(Split::Train, seed, 2)).unwrap();
        let dev = plugin.generate(&request(Split::Dev, seed, 2)).unwrap();
        let hit = |instance: &splinter_knowledge::factory::artifact::Instance, modules: &[&str]| {
            let solve = instance
                .folder()
                .text("solution/solve.sh")
                .unwrap()
                .to_string();
            modules
                .iter()
                .filter(|m| solve.contains(&format!("src/{m}.c")))
                .count()
        };
        assert!(hit(&train, &TRAIN_MODULES) >= 1 && hit(&train, &DEV_MODULES) == 0);
        assert!(hit(&dev, &DEV_MODULES) >= 1 && hit(&dev, &TRAIN_MODULES) == 0);
        assert!(!modules_of(&train).is_empty());
        assert!(train
            .instruction()
            .contains("Symptoms on the sample inputs:\n- console line"));
    }
}

#[test]
#[ignore = "needs the vemu emulator and clang: set VEMU or put vemu on PATH, run with --ignored"]
fn the_firmware_environment_passes_admission_and_a_table_printing_firmware_does_not_win() {
    // seed: any admission seed
    let (report, _) = admit(
        &FirmwareEmulator::new(host()),
        &BTreeMap::new(),
        &host(),
        &small_policy(),
        77,
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
}

#[test]
#[ignore = "needs the vemu emulator and clang: set VEMU or put vemu on PATH, run with --ignored"]
fn the_reference_solves_it_and_the_faulty_workspace_does_not() {
    let plugin = FirmwareEmulator::new(host());
    let instance = plugin.generate(&request(Split::Dev, 5, 1)).unwrap(); // seed: any
    let session = host().open(&instance).unwrap();
    let before = session.verify().unwrap();
    let Verdict::Graded(vector) = before else {
        panic!("graded")
    };
    assert_eq!(vector["reward"], 0.0, "{vector:?}");
    assert!(session.solve().unwrap().succeeded());
    assert!(session.verify().unwrap().solved());
}
