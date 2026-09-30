// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements bounded execution of model-written code for
// its clients. If your team needs expertise in agent sandboxing or process
// supervision, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: the process sandbox runs a runtime directly, bounded - a fresh
//! directory per call that is gone afterwards, an explicit environment, a
//! wall-clock timeout and resource limits that stop runaway code, and
//! output caps whose truncation is reported. It is not isolation, and says
//! so.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use splinter_sandbox::{
    CodeCall, Isolation, Limits, ProcessSandbox, RuntimeEnvironment, RuntimeRegistry, Sandbox,
};

/// A fresh scratch root per test, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        static N: AtomicU32 = AtomicU32::new(0);
        let path = std::env::temp_dir().join(format!(
            "splinter-sandbox-{name}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    /// Whatever the sandbox left behind in the scratch root.
    fn leftovers(&self) -> Vec<String> {
        std::fs::read_dir(&self.0)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect()
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The caller's environment as a caller would read it: a map passed in,
/// never the test process's own.
fn caller_env() -> BTreeMap<String, String> {
    [
        ("SPLINTER_TEST_SECRET", "must-not-leak"),
        ("SPLINTER_TEST_SHARED", "passed-through"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect()
}

fn python(scratch: &Scratch, limits: Limits, allowlist: &[&str]) -> RuntimeEnvironment {
    let sandbox = ProcessSandbox::new(&scratch.0, caller_env(), allowlist, limits);
    assert_eq!(sandbox.isolation(), Isolation::None);
    RuntimeEnvironment::new(&RuntimeRegistry::builtin(), "python3", Arc::new(sandbox)).unwrap()
}

fn call(code: &str) -> CodeCall {
    CodeCall {
        code: code.into(),
        stdin: None,
    }
}

#[test]
fn python_prints_and_exits_in_a_directory_that_is_gone_afterwards() {
    let scratch = Scratch::new("print");
    let env = python(&scratch, Limits::default(), &[]);
    let result = env
        .run(&CodeCall {
            code: "import os, sys\nprint(sys.stdin.read().upper())\nprint(os.getcwd())\n\
                   sys.stderr.write('warned')\nsys.exit(3)\n"
                .into(),
            stdin: Some("from stdin".into()),
        })
        .unwrap();
    assert_eq!(result.exit_code, Some(3), "{result:?}");
    assert!(!result.timed_out);
    assert_eq!(result.stderr, "warned");
    let mut lines = result.stdout.lines();
    assert_eq!(lines.next(), Some("FROM STDIN"));
    let cwd = PathBuf::from(lines.next().unwrap());
    assert!(
        cwd.starts_with(scratch.0.canonicalize().unwrap()),
        "{cwd:?}"
    );
    assert!(!cwd.exists(), "the call's directory was removed");
    assert!(scratch.leftovers().is_empty(), "{:?}", scratch.leftovers());
}

#[test]
fn a_busy_loop_is_stopped_by_the_cpu_limit_and_reported_timed_out() {
    let scratch = Scratch::new("cpu");
    let limits = Limits {
        cpu_seconds: 1,
        wall_time_ms: 20_000,
        ..Limits::default()
    };
    let env = python(&scratch, limits, &[]);
    let began = Instant::now();
    let result = env.run(&call("while True:\n    pass\n")).unwrap();
    assert!(result.timed_out, "{result:?}");
    assert_eq!(result.exit_code, None, "a killed process has no exit code");
    assert!(
        began.elapsed() < Duration::from_secs(15),
        "the CPU limit, not the wall clock, stopped it: {:?}",
        began.elapsed()
    );
    assert!(scratch.leftovers().is_empty());
}

#[test]
fn a_sleeping_call_is_stopped_by_the_wall_clock() {
    let scratch = Scratch::new("wall");
    let limits = Limits {
        wall_time_ms: 300,
        ..Limits::default()
    };
    let env = python(&scratch, limits, &[]);
    let began = Instant::now();
    let result = env
        .run(&call(
            "import subprocess, time\nprint('started', flush=True)\n\
             subprocess.Popen(['sleep', '60'])\ntime.sleep(60)\n",
        ))
        .unwrap();
    assert!(result.timed_out, "{result:?}");
    assert_eq!(result.stdout, "started\n");
    assert!(
        began.elapsed() < Duration::from_secs(10),
        "the call waited for a sleeping process: {:?}",
        began.elapsed()
    );
    assert!(scratch.leftovers().is_empty());
}

#[test]
fn output_beyond_the_cap_is_truncated_and_flagged() {
    let scratch = Scratch::new("cap");
    let limits = Limits {
        output_cap_bytes: 64,
        ..Limits::default()
    };
    let env = python(&scratch, limits, &[]);
    let result = env
        .run(&call("for _ in range(200):\n    print('0123456789')\n"))
        .unwrap();
    assert_eq!(result.exit_code, Some(0), "the process ran to completion");
    assert!(result.stdout_truncated && !result.stderr_truncated);
    assert_eq!(result.stdout.len(), 64);
    assert!(result.stdout.starts_with("0123456789\n0123456789\n"));
}

#[test]
fn the_child_sees_only_allowlisted_variables_from_the_callers_map() {
    let scratch = Scratch::new("env");
    let env = python(&scratch, Limits::default(), &["SPLINTER_TEST_SHARED"]);
    let result = env
        .run(&call(
            "import os\nfor k in sorted(os.environ):\n    print(k + '=' + os.environ[k])\n",
        ))
        .unwrap();
    assert_eq!(result.exit_code, Some(0), "{result:?}");
    assert!(
        result
            .stdout
            .contains("SPLINTER_TEST_SHARED=passed-through\n"),
        "{}",
        result.stdout
    );
    assert!(
        !result.stdout.contains("SPLINTER_TEST_SECRET"),
        "{}",
        result.stdout
    );
    let names: Vec<&str> = result
        .stdout
        .lines()
        .filter_map(|l| l.split_once('=').map(|(k, _)| k))
        .collect();
    for name in &names {
        assert!(
            ["HOME", "LANG", "PATH", "SPLINTER_TEST_SHARED", "TERM", "TMPDIR"].contains(name)
                // The interpreter may add its own locale variable.
                || name.starts_with("LC_"),
            "unexpected variable {name}: {}",
            result.stdout
        );
    }
}
