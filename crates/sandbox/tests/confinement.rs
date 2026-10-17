// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements filesystem confinement for code-editing
// agents, for its clients. If your team needs expertise in keeping a
// language-model agent away from material it must not read, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: a command run in a workspace tree reaches the tree, its own private
//! directory and the system, and nothing else: not a file elsewhere in the
//! user's directories, not a sibling tree, not the listing of a home
//! directory. It still runs the system's programs. Where the kernel cannot
//! confine a command it is refused, unless running unconfined was allowed,
//! and then the environment record says so.

#![allow(clippy::unwrap_used)]

use std::path::Path;

use splinter_sandbox::{CommandCall, Limits, ProcessSandbox, Sandbox, SandboxError};

struct Rig {
    outside: tempfile::TempDir,
    scratch: tempfile::TempDir,
    tree: std::path::PathBuf,
}

impl Rig {
    fn new() -> Self {
        let rig = Self {
            outside: tempfile::tempdir().unwrap(),
            scratch: tempfile::tempdir().unwrap(),
            tree: tempfile::tempdir().unwrap().keep(),
        };
        std::fs::write(rig.outside.path().join("secret.py"), "SECRET = 1\n").unwrap();
        std::fs::write(rig.tree.join("inside.py"), "VISIBLE = 1\n").unwrap();
        rig
    }

    fn sandbox(&self) -> ProcessSandbox {
        ProcessSandbox::new(
            self.scratch.path().join("calls"),
            [],
            &[],
            Limits::default(),
        )
    }

    fn sh(&self, sandbox: &ProcessSandbox, script: &str) -> splinter_sandbox::CodeResult {
        let call = CommandCall {
            argv: vec!["sh".into(), "-c".into(), script.into()],
            stdin: None,
        };
        sandbox.run_command(&self.tree, &call).unwrap()
    }
}

impl Drop for Rig {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.tree);
    }
}

#[test]
fn a_command_cannot_read_a_file_outside_its_tree_or_list_a_home_directory() {
    let rig = Rig::new();
    let sandbox = rig.sandbox();
    let secret = rig.outside.path().join("secret.py");

    let read = rig.sh(&sandbox, &format!("cat {}", secret.display()));
    assert_ne!(read.exit_code, Some(0), "{read:?}");
    assert!(!read.stdout.contains("SECRET"));

    for dir in ["/home", "/nobackup"] {
        if Path::new(dir).exists() {
            let listed = rig.sh(&sandbox, &format!("ls {dir}"));
            assert_ne!(listed.exit_code, Some(0), "{dir}: {listed:?}");
        }
    }
    let sibling = rig.sh(&sandbox, &format!("ls {}", rig.scratch.path().display()));
    assert_ne!(
        sibling.exit_code,
        Some(0),
        "the scratch root is not the tree's"
    );
}

#[test]
fn a_command_cannot_write_outside_its_tree_but_can_inside_it_and_in_its_tmp() {
    let rig = Rig::new();
    let sandbox = rig.sandbox();
    let target = rig.outside.path().join("planted");
    let planted = rig.sh(&sandbox, &format!("echo x > {}", target.display()));
    assert_ne!(planted.exit_code, Some(0), "{planted:?}");
    assert!(!target.exists());

    let inside = rig.sh(
        &sandbox,
        "echo y > new.txt && cat inside.py && echo z > \"$TMPDIR/t\" && cat \"$TMPDIR/t\"",
    );
    assert_eq!(inside.exit_code, Some(0), "{inside:?}");
    assert!(inside.stdout.contains("VISIBLE") && inside.stdout.contains('z'));
    assert!(rig.tree.join("new.txt").exists());
}

#[test]
fn a_command_still_reads_the_system_and_runs_python() {
    let rig = Rig::new();
    let sandbox = rig.sandbox();
    let ran = rig.sh(
        &sandbox,
        "ls /usr >/dev/null && python3 -c 'import json, sys; print(json.dumps([1]))'",
    );
    assert_eq!(ran.exit_code, Some(0), "{ran:?}");
    assert!(ran.stdout.contains("[1]"));
}

#[test]
fn without_kernel_confinement_a_command_is_refused_unless_unconfined_was_allowed() {
    let rig = Rig::new();
    let strict = rig.sandbox().without_landlock();
    let call = CommandCall {
        argv: vec!["true".into()],
        stdin: None,
    };
    assert!(matches!(
        strict.run_command(&rig.tree, &call),
        Err(SandboxError::Unconfined)
    ));
    assert!(!strict.spec().unconfined_commands);

    let allowed = rig.sandbox().without_landlock().allow_unconfined(true);
    let secret = rig.outside.path().join("secret.py");
    let read = rig.sh(&allowed, &format!("cat {}", secret.display()));
    assert!(
        read.stdout.contains("SECRET"),
        "unconfined means unconfined"
    );
    assert!(allowed.spec().unconfined_commands, "the record says so");
}
