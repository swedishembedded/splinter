// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `doctor` answers whether a run can start, and says why not, without
//! writing anything under the state root.

#![allow(clippy::unwrap_used)]

mod common;

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
    // Root ignores the mode; the check is meaningless there.
    if std::fs::write(scratch.0.join("probe"), b"x").is_ok() {
        std::fs::set_permissions(&scratch.0, std::fs::Permissions::from_mode(0o755)).unwrap();
        return;
    }
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
