// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements controlled placement of model work on named
// GPUs and CPU cores for its clients. If your team needs expertise in
// scheduling learning workloads on shared hardware, you can procure our
// services by sending an email to info@swedishembedded.com.

//! `--device` is read by brain's grammar, in one place, into the devices a
//! process may schedule on.

use gpu_core::devices::{Backend, Inventory};
use splinter_model::device::Devices;

/// Two cards and eight cores, as a machine reports itself.
fn two_cards() -> Inventory {
    Inventory {
        gpus: 2,
        cpu_cores: 8,
        npus: 0,
        gpu_backend: Backend::Cuda,
    }
}

#[test]
fn no_device_flag_schedules_on_everything_the_machine_has() {
    let devices = Devices::resolve("", &two_cards()).unwrap();
    assert_eq!(devices.gpus(), [0, 1]);
    assert_eq!(devices.cpu_cores().len(), 8);
    assert_eq!(devices.source(), "all");
}

#[test]
fn a_named_card_is_the_only_device_work_may_run_on() {
    let devices = Devices::resolve("gpu1", &two_cards()).unwrap();
    assert_eq!(devices.gpus(), [1]);
    assert!(devices.cpu_cores().is_empty());
    assert_eq!(devices.source(), "gpu1");
}

#[test]
fn several_devices_are_listed_with_commas() {
    let devices = Devices::resolve("gpu0,gpu1", &two_cards()).unwrap();
    assert_eq!(devices.gpus(), [0, 1]);
}

#[test]
fn a_card_the_machine_does_not_have_is_refused_naming_it() {
    let refused = Devices::resolve("gpu7", &two_cards())
        .unwrap_err()
        .to_string();
    assert!(refused.contains("gpu7"), "{refused}");
}

#[test]
fn a_word_that_is_not_a_device_is_refused_naming_it() {
    let refused = Devices::resolve("tpu", &two_cards())
        .unwrap_err()
        .to_string();
    assert!(refused.contains("tpu"), "{refused}");
}
