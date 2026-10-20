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
    classes_saying(calls, &[])
}

/// [`classes`] for an agent that also said `said`.
fn classes_saying(calls: Vec<Call>, said: &[&str]) -> Vec<String> {
    let mut all = calls;
    all.push(call(
        99,
        "edit_file",
        json!({"path": "src/ppu.rs", "diff": "@@\n-b\n+c\n"}),
        "ok",
    ));
    let trace = Trace {
        calls: all,
        said: said.iter().map(|s| (*s).to_string()).collect(),
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

#[test]
fn talking_about_its_own_git_diff_is_not_a_reference_to_the_given_change() {
    let read = || vec![call(1, "read_file", json!({"path": "src/ppu.rs"}), "x")];
    let own = classes_saying(
        read(),
        &["The diff output looks odd at the end - let me verify the file end is intact."],
    );
    assert!(!own.contains(&"diff_reference".to_string()), "{own:?}");
    for leak in [
        "The diff matches the intended change exactly.",
        "Let me verify my hunks against the diff.",
        "I will follow the system prompt.",
    ] {
        let found = classes_saying(read(), &[leak]);
        assert!(
            found.contains(&"diff_reference".to_string()),
            "{leak}: {found:?}"
        );
    }
}

#[test]
fn comparing_work_with_the_change_it_was_shown_is_a_reference_but_comparing_with_the_task_is_not() {
    let read = || vec![call(1, "read_file", json!({"path": "src/ppu.rs"}), "x")];
    for ok in [
        "My git diff covers everything the task asked for and nothing else.",
        "Comparing my git diff with the task, one requirement is still open.",
    ] {
        let found = classes_saying(read(), &[ok]);
        assert!(
            !found.contains(&"diff_reference".to_string()),
            "{ok}: {found:?}"
        );
    }
    for leak in [
        "My edits match the change above.",
        "Checking my work against the change shown to me.",
        "I compared the result with the provided diff line by line.",
    ] {
        let found = classes_saying(read(), &[leak]);
        assert!(
            found.contains(&"diff_reference".to_string()),
            "{leak}: {found:?}"
        );
    }
}

#[test]
fn an_interpreter_reading_a_file_is_not_writing_one() {
    let read = call(2, "read_file", json!({"path": "src/ppu.rs"}), "x");
    let shell = |cmd: &str| call(1, "shell", json!({"shell_command": cmd}), "");
    let checks = classes(vec![
        shell("python3 -c \"import yaml; d=yaml.safe_load(open('board.yaml')); print(d)\""),
        read.clone(),
    ]);
    assert!(
        !checks.contains(&"forbidden_shell".to_string()),
        "{checks:?}"
    );
    for write in [
        "python3 -c \"open('out.txt','w').write('x')\"",
        "python3 -c \"import pathlib; pathlib.Path('a').write_text('x')\"",
        "node -e \"require('fs').writeFileSync('a','x')\"",
    ] {
        let found = classes(vec![shell(write), read.clone()]);
        assert!(
            found.contains(&"forbidden_shell".to_string()),
            "{write}: {found:?}"
        );
    }
}

#[test]
fn a_crate_name_the_parent_spells_with_dashes_is_not_a_name_only_the_change_has() {
    // The parent's manifest says `acme-core-m`; the change spells it `acme_core_m` in code.
    let parent = splinter_repo_replay::tree::vocabulary(
        &[(
            "Cargo.toml".to_string(),
            splinter_repo_replay::tree::Entry::File {
                bytes: b"acme-core-m = { path = \"../acme-core-m\" }\n".to_vec(),
                executable: false,
            },
        )]
        .into_iter()
        .collect(),
    );
    let trace = Trace {
        calls: vec![
            call(1, "read_file", json!({"path": "src/ppu.rs"}), "x"),
            call(
                2,
                "shell",
                json!({"shell_command": "grep -n acme_core_m Cargo.toml"}),
                "",
            ),
            call(
                3,
                "edit_file",
                json!({"path": "src/ppu.rs", "diff": "@@\n-b\n+use acme_core_m;\n"}),
                "ok",
            ),
        ],
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
    let roots = ["/work/tree".to_string()];
    let verdict = evaluate(
        &GateInput {
            changes: &changes,
            plan: "Please change src/ppu.rs so that b links the cortex-m crate.",
            parent_words: &parent,
            diff: DIFF,
            added_text: "use acme_core_m;",
            trace: &trace,
            residual: &[],
            forbidden_locations: &[],
            attempt_roots: &roots,
            stopped: None,
        },
        &GateConfig::default(),
    );
    assert!(!verdict.has("leaked_target"), "{:?}", verdict.reasons);
}

#[test]
fn a_search_with_context_lines_in_the_file_shows_the_file() {
    let grep = |context: serde_json::Value| {
        call(
            1,
            "grep",
            json!({"path": "src/ppu.rs", "pattern": "b", "context_lines": context}),
            "src/ppu.rs-299-a\nsrc/ppu.rs:300:b\nsrc/ppu.rs-301-d",
        )
    };
    let with = classes(vec![grep(json!(15))]);
    assert!(!with.contains(&"unread_edit".to_string()), "{with:?}");
    let without = classes(vec![grep(json!(0))]);
    assert!(without.contains(&"unread_edit".to_string()), "{without:?}");
}

#[test]
fn words_inside_quotes_that_are_only_searched_for_are_not_commands() {
    let read = call(2, "read_file", json!({"path": "src/ppu.rs"}), "x");
    let shell = |cmd: &str| call(1, "shell", json!({"shell_command": cmd}), "");
    for searching in [
        "grep -n 'x << 9\\|y' src/ppu.rs",
        "grep -rn \"curl\\|patch\" src",
        "grep -n tee src/ppu.rs | head",
        "grep -rn patch src/",
    ] {
        let found = classes(vec![shell(searching), read.clone()]);
        assert!(
            !found.contains(&"forbidden_shell".to_string()),
            "{searching}: {found:?}"
        );
    }
    for doing in [
        "cat <<'EOF' > a.txt\nx\nEOF",
        "patch -p1 < fix.diff",
        "echo x | tee out.txt",
        "curl https://example.org",
        "bash -c \"cat <<EOF\nx\nEOF\"",
    ] {
        let found = classes(vec![shell(doing), read.clone()]);
        assert!(
            found.contains(&"forbidden_shell".to_string()),
            "{doing}: {found:?}"
        );
    }
}
