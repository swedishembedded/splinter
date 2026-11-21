// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `doctor` answers whether a run can start, and says why not, without
//! writing anything under the state root.

#![allow(clippy::unwrap_used)]

use crate::common;

use common::{scratch_context, Scripted};
use splinter_pipelines::doctor::{doctor, Verdict};

fn verdict(report: &splinter_pipelines::doctor::Doctor, name: &str) -> Verdict {
    report
        .checks
        .iter()
        .find(|c| c.name == name)
        .unwrap_or_else(|| panic!("no check named {name}: {:?}", report.checks))
        .verdict
}

#[test]
fn a_scratch_state_root_is_ready_and_stays_uncreated() {
    let (scratch, ctx) = scratch_context("doctor-ready", Scripted::new(|_| String::new()), false);
    std::fs::create_dir_all(&common::config(&scratch).policy_base).unwrap();
    let report = doctor(&ctx);
    assert_eq!(verdict(&report, "state-writable"), Verdict::Ok);
    assert!(
        matches!(verdict(&report, "free-space"), Verdict::Ok | Verdict::Warn),
        "{:?}",
        report.checks
    );
    assert_eq!(verdict(&report, "role:policy"), Verdict::Ok);
    assert!(report.ready());
    assert!(
        !ctx.root().path().exists(),
        "checking creates nothing under the state root"
    );
}

#[cfg(unix)]
#[test]
fn an_unwritable_state_root_fails_with_the_fix() {
    use std::os::unix::fs::PermissionsExt;
    let (scratch, ctx) =
        scratch_context("doctor-readonly", Scripted::new(|_| String::new()), false);
    std::fs::set_permissions(&scratch.0, std::fs::Permissions::from_mode(0o555)).unwrap();
    // Root ignores the mode, and then nothing here can be provoked: the test
    // fails rather than pass without having checked anything.
    let ignored = std::fs::write(scratch.0.join("probe"), b"x").is_ok();
    if ignored {
        std::fs::set_permissions(&scratch.0, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    assert!(
        !ignored,
        "the file mode is ignored for this user (root?): run this test as an ordinary user"
    );
    let report = doctor(&ctx);
    std::fs::set_permissions(&scratch.0, std::fs::Permissions::from_mode(0o755)).unwrap();

    let writable = report
        .checks
        .iter()
        .find(|c| c.name == "state-writable")
        .unwrap();
    assert_eq!(writable.verdict, Verdict::Fail);
    assert!(writable.fix.as_deref().unwrap().contains("--state"));
    assert!(!report.ready());
    assert!(
        report.checks.iter().all(|c| c.name != "free-space"),
        "no free-space reading is taken where nothing can be written"
    );
}

#[test]
fn a_role_whose_model_does_not_resolve_fails_the_report() {
    // Without the scripted policy the configured base does not exist.
    let scratch = common::Scratch::new("doctor-nomodel");
    let ctx = splinter_orchestrator::Context::new(common::config(&scratch), false).unwrap();
    let report = doctor(&ctx);
    assert_eq!(verdict(&report, "role:policy"), Verdict::Fail);
    assert!(!report.ready());
}

mod gpu {
    use splinter_model::device::GpuCard;
    use splinter_pipelines::doctor::{gpu_fit, Verdict};

    const GIB: u64 = 1 << 30;

    fn card(index: u32, free_gib: u64, measured: bool) -> GpuCard {
        GpuCard {
            endpoint: 0,
            index,
            total_bytes: 24 * GIB,
            free_bytes: free_gib * GIB,
            measured,
        }
    }

    #[test]
    fn a_machine_without_a_gpu_warns_that_the_run_would_use_the_cpu() {
        let check = gpu_fit(&[], GIB);
        assert_eq!(check.verdict, Verdict::Warn);
        assert!(check.detail.contains("no GPU") && check.detail.contains("CPU"));
        assert!(check.fix.is_some());
    }

    #[test]
    fn a_card_with_room_for_the_weights_is_fine() {
        let check = gpu_fit(&[card(0, 2, true), card(1, 20, true)], 4 * GIB);
        assert_eq!(check.verdict, Verdict::Ok, "{check:?}");
        assert!(check.detail.contains("gpu1"), "{}", check.detail);
    }

    #[test]
    fn the_cards_of_every_endpoint_are_counted_and_named_by_it() {
        let elsewhere = GpuCard {
            endpoint: 1,
            ..card(0, 20, true)
        };
        let check = gpu_fit(&[card(0, 2, true), elsewhere], 4 * GIB);
        assert_eq!(check.verdict, Verdict::Ok, "{check:?}");
        assert!(
            check.detail.contains("fit on endpoint 1 gpu0"),
            "{}",
            check.detail
        );
    }

    #[test]
    fn cards_too_full_for_the_weights_warn_that_brain_will_fall_back_to_the_cpu() {
        let check = gpu_fit(&[card(0, 2, true), card(1, 3, true)], 4 * GIB);
        assert_eq!(check.verdict, Verdict::Warn);
        assert!(check.detail.contains("CPU"), "{}", check.detail);
        assert!(check.fix.is_some());
    }

    #[test]
    fn free_memory_that_could_not_be_read_is_never_presented_as_a_measurement() {
        let check = gpu_fit(&[card(0, 24, false)], GIB);
        assert_eq!(check.verdict, Verdict::Warn);
        assert!(
            check.detail.contains("could not be read"),
            "{}",
            check.detail
        );
    }
}

mod cluster {
    use splinter_orchestrator::cluster::{ClusterAllocation, ClusterMode, ClusterSettings};
    use splinter_pipelines::doctor::{cluster_check, Verdict};

    fn inside(nodes: &[&str]) -> ClusterSettings {
        ClusterSettings {
            detected: Ok(Some(ClusterAllocation {
                job_id: "5600435".into(),
                nodes: nodes.iter().map(|n| (*n).to_string()).collect(),
                gpus_per_node: 4,
                this_node: Some("n424".into()),
            })),
            srun: Some("/usr/bin/srun".into()),
            ..ClusterSettings::default()
        }
    }

    #[test]
    fn doctor_shows_the_allocation_and_the_nodes_a_run_uses() {
        let check = cluster_check(&ClusterSettings {
            nodes: Some("n442".into()),
            ..inside(&["n424", "n442", "n445"])
        });
        assert_eq!(check.verdict, Verdict::Ok, "{check:?}");
        for shown in [
            "5600435",
            "n424,n442,n445",
            "4 GPUs each",
            "this node n424",
            "using n442",
            "--cluster-nodes",
        ] {
            assert!(check.detail.contains(shown), "{shown}: {}", check.detail);
        }
    }

    #[test]
    fn no_allocation_off_and_one_node_keep_the_run_on_this_machine() {
        for settings in [
            ClusterSettings::default(),
            ClusterSettings {
                mode: ClusterMode::Off,
                ..inside(&["n1", "n2"])
            },
            inside(&["n424"]),
        ] {
            let check = cluster_check(&settings);
            assert_eq!(check.verdict, Verdict::Ok);
            assert!(check.detail.contains("this machine"), "{}", check.detail);
        }
    }

    #[test]
    fn an_unusable_allocation_fails_by_what_is_wrong() {
        let stranger = cluster_check(&ClusterSettings {
            nodes: Some("n999".into()),
            ..inside(&["n1", "n2"])
        });
        assert_eq!(stranger.verdict, Verdict::Fail);
        assert!(stranger.detail.contains("n999"), "{}", stranger.detail);
        let no_srun = cluster_check(&ClusterSettings {
            srun: None,
            ..inside(&["n1", "n2"])
        });
        assert_eq!(no_srun.verdict, Verdict::Fail);
        assert!(no_srun.fix.unwrap().contains("--slurm-srun"));
    }
}

/// A policy left to plan, write, teach, judge and read sentences for itself
/// can start a run but rarely trains anything: each such role is a warning
/// that names the way out, and naming a stronger model clears it.
#[test]
fn a_policy_left_to_teach_itself_is_a_warning_naming_the_way_out() {
    let (scratch, ctx) = scratch_context("doctor-weak", Scripted::new(|_| String::new()), false);
    std::fs::create_dir_all(&common::config(&scratch).policy_base).unwrap();
    let report = doctor(&ctx);
    for role in ["teacher", "generator", "planner", "judge", "router"] {
        let check = report
            .checks
            .iter()
            .find(|c| c.name == format!("role:{role}"))
            .unwrap();
        assert_eq!(check.verdict, Verdict::Warn, "{check:?}");
        assert!(
            check.fix.as_deref().unwrap().contains("--assistant-model"),
            "{check:?}"
        );
    }
    assert_eq!(verdict(&report, "role:policy"), Verdict::Ok);
    assert_eq!(
        verdict(&report, "role:critic"),
        Verdict::Ok,
        "the critic is the policy by design"
    );
    assert!(report.ready(), "a warning does not stop a run");

    let helper = scratch.0.join("helper");
    std::fs::create_dir_all(&helper).unwrap();
    let mut config = common::config(&scratch);
    config.assistant_model = Some(format!("local:{}", helper.display()));
    let ctx = splinter_orchestrator::Context::new(config, false).unwrap();
    let report = doctor(&ctx);
    assert_eq!(
        verdict(&report, "role:teacher"),
        Verdict::Ok,
        "{:?}",
        report.checks
    );
}
