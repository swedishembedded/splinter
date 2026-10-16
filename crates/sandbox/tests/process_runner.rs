// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements bounded, supervised execution of untrusted
// programs for its clients. If your team needs expertise in process
// supervision or agent sandboxing, you can procure our services by sending
// an email to info@swedishembedded.com.

//! Spec: the process runner leaves nothing running behind a timed-out run,
//! however deep the program that loops sits in the process tree. A test
//! command such as `python3 -m unittest` starts the interpreter under test
//! once per test, so the looping program is a grandchild of the run.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::path::Path;
use std::time::{Duration, Instant};

use splinter_sandbox::process::{run, ProcessSpec};

/// Whether process `pid` is still running (a zombie is not).
fn is_running(pid: u32) -> bool {
    match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
        Err(_) => false,
        Ok(stat) => !stat.rsplit(") ").next().unwrap_or("").starts_with('Z'),
    }
}

fn pid_in(dir: &Path, name: &str) -> u32 {
    std::fs::read_to_string(dir.join(name))
        .unwrap()
        .trim()
        .parse()
        .unwrap()
}

#[test]
fn a_looping_grandchild_is_gone_after_the_timeout() {
    let dir = std::env::temp_dir().join(format!("splinter-runner-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // The run starts a middle process (as a test runner does), which starts
    // a busy loop (the program under test) and waits for it.
    let script = "sh -c 'echo $$ > grandchild.pid; while :; do :; done' & wait";
    let spec = ProcessSpec::new(
        vec!["sh".into(), "-c".into(), script.into()],
        &dir,
        BTreeMap::from([("PATH".to_string(), "/usr/bin:/bin".to_string())]),
        Duration::from_millis(500),
        1024,
    );

    let began = Instant::now();
    let output = run(&spec).unwrap();

    assert!(output.timed_out, "{output:?}");
    assert!(
        began.elapsed() < Duration::from_secs(10),
        "{:?}",
        began.elapsed()
    );
    let pid = pid_in(&dir, "grandchild.pid");
    let deadline = Instant::now() + Duration::from_secs(5);
    while is_running(pid) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
    }
    let survived = is_running(pid);
    let _ = std::fs::remove_dir_all(&dir);
    assert!(!survived, "the looping grandchild outlived the run");
}
