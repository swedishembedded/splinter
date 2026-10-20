// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec: the gate's text rules do not fire on what an honest agent does in a
//! real repository. Each case here was a false positive seen in a real run.

#![allow(clippy::unwrap_used)]

use std::collections::HashSet;

use serde_json::json;
use splinter_repo_replay::gate::{evaluate, GateConfig, GateInput};
use splinter_repo_replay::git::{Change, Kind};
use splinter_repo_replay::trace::{Call, Trace};

const DIFF: &str = "diff --git a/src/ppu.rs b/src/ppu.rs\n--- a/src/ppu.rs\n+++ b/src/ppu.rs\n@@ -300,3 +300,3 @@\n a\n-b\n+c\n d\n";

fn call(step: u64, tool: &str, args: serde_json::Value, result: &str) -> Call {
    Call {
        step,
        tool: tool.into(),
        args,
        result: result.into(),
    }
}

/// The gate's classes for an agent that made `calls` and edited `src/ppu.rs`.
fn classes(calls: Vec<Call>) -> Vec<String> {
    let mut all = calls;
    all.push(call(
        99,
        "edit_file",
        json!({"path": "src/ppu.rs", "diff": "@@\n-b\n+c\n"}),
        "ok",
    ));
    let trace = Trace {
        calls: all,
        said: vec![],
    };
    let changes = [Change {
        kind: Kind::Modified,
        path: "src/ppu.rs".into(),
        old_path: None,
        added: Some(1),
        removed: Some(1),
        special: false,
    }];
    let words = HashSet::new();
    let roots = ["/work/tree".to_string()];
    let verdict = evaluate(
        &GateInput {
            changes: &changes,
            plan: "Please change src/ppu.rs so that b becomes c.",
            parent_words: &words,
            diff: DIFF,
            added_text: "c",
            trace: &trace,
            residual: &[],
            forbidden_locations: &[],
            attempt_roots: &roots,
            stopped: None,
        },
        &GateConfig::default(),
    );
    verdict.classes().into_iter().map(String::from).collect()
}

#[test]
fn discarding_error_output_is_not_a_write() {
    let shell = |cmd: &str| call(1, "shell", json!({"shell_command": cmd}), "");
    let found = classes(vec![
        shell("ls -la .claude 2>/dev/null; ls crates 2>/dev/null; pwd"),
        call(2, "read_file", json!({"path": "src/ppu.rs"}), "x"),
    ]);
    assert!(!found.contains(&"forbidden_shell".to_string()), "{found:?}");
    let writes = classes(vec![
        shell("echo x > out.txt"),
        call(2, "read_file", json!({"path": "src/ppu.rs"}), "x"),
    ]);
    assert!(
        writes.contains(&"forbidden_shell".to_string()),
        "{writes:?}"
    );
}

#[test]
fn paging_through_a_file_is_not_a_read_at_the_hunk() {
    let paged = classes(vec![call(
        1,
        "read_file",
        json!({"path": "src/ppu.rs", "offset": 201, "limit": 200}),
        "x",
    )]);
    assert!(
        !paged.contains(&"diff_offset_read".to_string()),
        "{paged:?}"
    );
    let at_hunk = classes(vec![call(
        1,
        "read_file",
        json!({"path": "src/ppu.rs", "offset": 295, "limit": 40}),
        "x",
    )]);
    assert!(
        at_hunk.contains(&"diff_offset_read".to_string()),
        "{at_hunk:?}"
    );
}

#[test]
fn stashing_to_compare_before_and_after_is_not_a_way_to_the_future() {
    // The attempt tree has one commit: nothing a stash, a status or a log
    // shows can be the target. `git apply` still is.
    let shell = |cmd: &str| call(1, "shell", json!({"shell_command": cmd}), "");
    let read = call(2, "read_file", json!({"path": "src/ppu.rs"}), "x");
    let stash = classes(vec![
        shell("git stash -q && cargo check 2>&1 | tail -3; git stash pop -q"),
        read.clone(),
    ]);
    assert!(!stash.contains(&"forbidden_shell".to_string()), "{stash:?}");
    let apply = classes(vec![shell("git apply fix.patch"), read]);
    assert!(apply.contains(&"forbidden_shell".to_string()), "{apply:?}");
}
