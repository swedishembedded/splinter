// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec: the sample's gate accepts honest work and refuses each kind of
//! cheating for the right reason, over a real history, with no model.
//!
//! The scenarios live in `splinter_repo_replay::audit` (the `audit` verb runs
//! the same ones); this spec runs them and holds every named check to account.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use splinter_repo_replay::audit::audit;

#[test]
fn the_gate_passes_honest_work_and_catches_every_cheat() {
    let work = tempfile::tempdir().unwrap();
    let checks = audit(
        work.path(),
        std::path::Path::new(env!("CARGO_BIN_EXE_repo-replay")),
    )
    .unwrap();
    let failed: Vec<String> = checks
        .iter()
        .filter(|c| !c.ok)
        .map(|c| format!("{} ({})", c.name, c.detail))
        .collect();
    assert!(failed.is_empty(), "failed checks:\n{}", failed.join("\n"));
    // The audit is not vacuous: honest history, each cheater, determinism,
    // dataset, the unmotivated edit and the untouched source.
    assert!(checks.len() >= 18, "{} checks ran", checks.len());
}
