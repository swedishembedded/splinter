// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! `splinter-confine`, the launcher a sandbox wraps a command in where the
//! host gives neither containers nor namespaces: it says what the kernel can
//! enforce, and a command it runs writes only where it may and reaches only
//! the network it may.

use std::net::TcpListener;
use std::process::Command;

fn confine() -> Command {
    Command::new(env!("CARGO_BIN_EXE_splinter-confine"))
}

fn landlock_network() -> bool {
    splinter_sandbox::landlock::abi_version().is_some_and(|abi| abi >= 4)
}

#[test]
fn the_probe_says_what_the_kernel_enforces() {
    let out = confine().arg("--probe").output().unwrap();
    assert!(out.status.success());
    let probe: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(
        probe["landlock_abi"].as_u64(),
        splinter_sandbox::landlock::abi_version().map(u64::from)
    );
    assert_eq!(probe["network"].as_bool(), Some(landlock_network()));
}

#[test]
fn a_confined_command_writes_only_where_it_may() {
    if splinter_sandbox::landlock::abi_version().is_none() {
        // The probe test above is the check on such a kernel; the launcher
        // refuses (below) rather than run unconfined.
        let out = confine()
            .args(["--read-write", "/tmp", "--", "true"])
            .output()
            .unwrap();
        assert!(!out.status.success());
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (allowed, denied) = (dir.path().join("allowed"), dir.path().join("denied"));
    std::fs::create_dir(&allowed).unwrap();
    std::fs::create_dir(&denied).unwrap();
    let script = format!(
        "echo ok > {a}/f && ! echo no > {d}/f 2>/dev/null",
        a = allowed.display(),
        d = denied.display()
    );
    let status = confine()
        .args(["--read-only", "/", "--read-write"])
        .arg(&allowed)
        .args(["--network", "none", "--", "sh", "-c", &script])
        .status()
        .unwrap();
    assert!(status.success());
    assert!(allowed.join("f").is_file() && !denied.join("f").exists());
}

#[test]
fn a_confined_command_reaches_only_the_port_it_may() {
    if !landlock_network() {
        return;
    }
    let open = TcpListener::bind("127.0.0.1:0").unwrap();
    let closed = TcpListener::bind("127.0.0.1:0").unwrap();
    let (open_port, closed_port) = (
        open.local_addr().unwrap().port(),
        closed.local_addr().unwrap().port(),
    );
    // bash's /dev/tcp connects with connect(2).
    let reach = |port: u16, network: &str| {
        confine()
            .args(["--read-only", "/", "--network", network, "--", "bash", "-c"])
            .arg(format!("exec 3<>/dev/tcp/127.0.0.1/{port}"))
            .status()
            .unwrap()
            .success()
    };
    let allowed = format!("tcp:{open_port}");
    assert!(reach(open_port, &allowed));
    assert!(!reach(closed_port, &allowed));
    assert!(!reach(open_port, "none"));
    assert!(reach(closed_port, "open"));
    // No UDP under a port allowance: a datagram socket cannot even be made.
    let udp = confine()
        .args([
            "--read-only",
            "/",
            "--network",
            &allowed,
            "--",
            "bash",
            "-c",
            "exec 3<>/dev/udp/127.0.0.1/53",
        ])
        .status()
        .unwrap();
    assert!(!udp.success());
}

#[test]
fn a_launcher_given_nothing_to_run_or_an_unknown_option_refuses() {
    assert_eq!(
        confine()
            .args(["--read-only", "/"])
            .status()
            .unwrap()
            .code(),
        Some(2)
    );
    assert_eq!(
        confine()
            .args(["--writable", "/", "--", "true"])
            .status()
            .unwrap()
            .code(),
        Some(2)
    );
    assert_eq!(
        confine()
            .args(["--network", "maybe", "--", "true"])
            .status()
            .unwrap()
            .code(),
        Some(2)
    );
}

#[test]
fn a_skipped_system_call_is_answered_not_killed() {
    if splinter_sandbox::landlock::abi_version().is_none() {
        return;
    }
    // A tracer (proot) skips a call by setting its number to -1, which the
    // kernel answers with ENOSYS; the socket filter must not kill it.
    let skip = "import ctypes, sys; sys.exit(0 if ctypes.CDLL(None).syscall(-1) == -1 else 1)";
    for network in ["none", "tcp:9"] {
        let status = confine()
            .args([
                "--read-only",
                "/",
                "--network",
                network,
                "--",
                "python3",
                "-c",
                skip,
            ])
            .status()
            .unwrap();
        assert!(status.success(), "{network}: {status}");
    }
}
