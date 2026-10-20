// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements bounded, supervised execution of untrusted
// programs for its clients. If your team needs expertise in process
// supervision or agent sandboxing, you can procure our services by sending
// an email to info@swedishembedded.com.

//! Spec: the process runner leaves nothing running behind a run, however deep
//! the program that loops sits in the process tree, however it detached, and
//! whether the run timed out, ended on its own or lost its supervisor. A test
//! command such as `python3 -m unittest` starts the interpreter under test
//! once per test, so the looping program is a grandchild of the run. The
//! memory a run holds is bounded as a whole.
//!
//! A run may live in a pid namespace of its own, so the pid a program reports
//! for itself is not the one the host knows: the programs below carry a marker
//! argument, which the host finds in the command lines of its processes.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::path::Path;
use std::time::{Duration, Instant};

use splinter_sandbox::process::{run, ProcessSpec, ResourceLimits};

/// Whether process `pid` is still running (a zombie is not).
fn is_running(pid: u32) -> bool {
    match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
        Err(_) => false,
        Ok(stat) => !stat.rsplit(") ").next().unwrap_or("").starts_with('Z'),
    }
}

/// The running processes of the host with `marker` in their command line.
fn marked(marker: &str) -> Vec<u32> {
    let own = std::process::id();
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return Vec::new();
    };
    entries
        .flatten()
        .filter_map(|e| e.file_name().to_str()?.parse::<u32>().ok())
        .filter(|pid| *pid != own && is_running(*pid))
        .filter(|pid| {
            std::fs::read(format!("/proc/{pid}/cmdline"))
                .is_ok_and(|c| String::from_utf8_lossy(&c).contains(marker))
        })
        .collect()
}

/// A failing spec must not leave the busy loop it exists to catch.
fn kill_all(pids: &[u32]) {
    for pid in pids {
        let _ = std::process::Command::new("kill")
            .args(["-9", &pid.to_string()])
            .status();
    }
}

/// Waits until nothing carries `marker`, up to `within`; the survivors, if
/// any, are killed and returned.
fn survivors(marker: &str, within: Duration) -> Vec<u32> {
    let deadline = Instant::now() + within;
    while !marked(marker).is_empty() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
    }
    let left = marked(marker);
    kill_all(&left);
    left
}

/// Waits until something carries `marker`.
fn wait_started(marker: &str) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while marked(marker).is_empty() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(!marked(marker).is_empty(), "{marker} never started");
}

/// A marker unique to this test and process. Scripts build it from a variable
/// so that their own command line does not carry it.
fn marker(name: &str) -> String {
    format!("splinter-marker-{name}-{}", std::process::id())
}

/// `script` with `$M` standing for the marker's text and `LOOP` for a busy
/// loop that carries it.
fn marked_script(marker: &str, script: &str) -> String {
    let (prefix, rest) = marker.split_at(8);
    format!("p={prefix}; M=$p{rest}; {script}")
}

const LOOP: &str = "sh -c 'while :; do :; done' $M";

fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("splinter-runner-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn plain_env() -> BTreeMap<String, String> {
    BTreeMap::from([("PATH".to_string(), "/usr/bin:/bin".to_string())])
}

fn spec(script: String, dir: &Path, timeout: Duration) -> ProcessSpec {
    ProcessSpec::new(
        vec!["sh".into(), "-c".into(), script],
        dir,
        plain_env(),
        timeout,
        1024,
    )
}

/// Runs `script` under a 500 ms timeout and says what outlived the run.
fn outlived_the_run(name: &str, script: &str) -> Vec<u32> {
    let dir = scratch(name);
    let marker = marker(name);
    let began = Instant::now();
    let ran = run(&spec(
        marked_script(&marker, script),
        &dir,
        Duration::from_millis(500),
    ));
    let output = match ran {
        Ok(output) => output,
        Err(e) => {
            // The escaped process holds the run's output open.
            kill_all(&marked(&marker));
            panic!("{e}");
        }
    };
    assert!(output.timed_out, "{output:?}");
    assert!(
        began.elapsed() < Duration::from_secs(10),
        "{:?}",
        began.elapsed()
    );
    let _ = std::fs::remove_dir_all(&dir);
    survivors(&marker, Duration::from_secs(5))
}

/// A middle process (as a test runner is) starts a busy loop (the program
/// under test) and waits for it.
#[test]
fn a_looping_grandchild_is_gone_after_the_timeout() {
    let left = outlived_the_run("group", &format!("{LOOP} & wait"));
    assert!(left.is_empty(), "the looping grandchild outlived the run");
}

/// A program that puts itself in a session or group of its own (`setsid`,
/// `timeout`) leaves the run's process group, but not its descendants.
#[test]
fn a_looping_descendant_that_left_the_process_group_is_gone_too() {
    let left = outlived_the_run("session", &format!("setsid {LOOP} & wait"));
    assert!(left.is_empty(), "the descendant outlived the run");
}

/// A program that detaches a process into a session of its own and exits
/// before the run is over leaves that process an orphan of init: nothing links
/// it to the run by parent any more. It must still be gone when the run ends,
/// whether the run timed out or not (an orphaned test interpreter once grew to
/// 145 GB, half an hour after its run).
#[test]
fn a_process_orphaned_during_a_run_that_ends_normally_is_gone_with_the_run() {
    let dir = scratch("orphan");
    let marker = marker("orphan");
    let script = marked_script(&marker, &format!("setsid {LOOP} & sleep 0.3"));
    let output = run(&spec(script, &dir, Duration::from_secs(20)));
    let left = survivors(&marker, Duration::from_secs(5));
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(output.unwrap().exit_code, Some(0));
    assert!(left.is_empty(), "the orphan outlived the run");
}

/// The supervisor of a run dies (killed, crashed, container restarted) while
/// the run is in flight: nothing it started may keep running.
#[test]
fn nothing_outlives_a_supervisor_that_is_killed_mid_run() {
    let dir = scratch("supervisor");
    let marker = marker("supervisor");
    // The helper finds its orders in a file named for its parent (this test).
    std::fs::write(dir.join("marker"), &marker).unwrap();
    let mut supervisor = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "supervisor_for_the_death_test", "--nocapture"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    wait_started(&marker);
    supervisor.kill().unwrap();
    let _ = supervisor.wait();
    let left = survivors(&marker, Duration::from_secs(5));
    let _ = std::fs::remove_dir_all(&dir);
    assert!(left.is_empty(), "the run's process outlived its supervisor");
}

/// The supervisor of `nothing_outlives_a_supervisor_that_is_killed_mid_run`:
/// a no-op unless that test started this process for the purpose.
#[test]
fn supervisor_for_the_death_test() {
    // SAFETY: `getppid` takes no arguments and cannot fail.
    let parent = unsafe { libc::getppid() };
    let dir = std::env::temp_dir().join(format!("splinter-runner-supervisor-{parent}"));
    let Ok(marker) = std::fs::read_to_string(dir.join("marker")) else {
        return;
    };
    let script = marked_script(&marker, &format!("{LOOP} & wait"));
    let _ = run(&spec(script, &dir, Duration::from_secs(120)));
}

fn tree_bound(bytes: u64) -> ResourceLimits {
    ResourceLimits {
        cpu_seconds: 120,
        address_space_bytes: 8 << 30,
        file_size_bytes: 1 << 30,
        max_processes: None,
        tree_memory_bytes: Some(bytes),
    }
}

/// Holds `mib` MiB resident for `seconds`.
fn holder(mib: u32, seconds: u32) -> String {
    format!("python3 -c \"b = b'x' * ({mib} << 20); import time; time.sleep({seconds})\"")
}

/// The address space limit binds each process on its own, so a test command
/// that starts many interpreters is bounded by none of them. The bound on the
/// whole tree stops it, and the result says a resource limit did, not the
/// program.
#[test]
fn a_tree_that_holds_more_memory_than_its_bound_is_killed_and_reported() {
    let dir = scratch("memory");
    let script = format!("{} & {} & wait", holder(160, 60), holder(160, 60));
    let mut spec = ProcessSpec::new(
        vec!["sh".into(), "-c".into(), script],
        &dir,
        plain_env(),
        Duration::from_secs(60),
        1024,
    );
    spec.resources = Some(tree_bound(256 << 20));
    let began = Instant::now();
    let output = run(&spec).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(output.memory_exceeded, "{output:?}");
    assert!(!output.timed_out, "{output:?}");
    assert!(
        began.elapsed() < Duration::from_secs(20),
        "{:?}",
        began.elapsed()
    );
}

#[test]
fn a_tree_within_its_memory_bound_runs_to_the_end_unreported() {
    let dir = scratch("memory-ok");
    let mut spec = ProcessSpec::new(
        vec!["sh".into(), "-c".into(), holder(100, 1)],
        &dir,
        plain_env(),
        Duration::from_secs(60),
        1024,
    );
    spec.resources = Some(tree_bound(256 << 20));
    let output = run(&spec).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(output.exit_code, Some(0), "{output:?}");
    assert!(!output.memory_exceeded, "{output:?}");
}
