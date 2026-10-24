// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements detached job control for learning campaigns,
// for its clients. If your team needs expertise in operator commands that
// reconcile work after a restart, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Spec: `splinter jobs` submits one detached local job per specification,
//! lists it, reconciles it from another process after its worker exits, shows
//! it and cancels it. The workers are real short-lived processes, so no model
//! and no GPU is needed.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use serde_json::Value;

/// A scratch directory standing in for the state root's parent.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("splinter-jobs-cli-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn state(&self) -> PathBuf {
        self.0.join("state")
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// `splinter --json --state <state> args...`; its exit code and stdout as JSON.
fn splinter(state: &Path, args: &[&str]) -> (i32, Value) {
    let output = Command::new(env!("CARGO_BIN_EXE_splinter"))
        .arg("--json")
        .arg("--state")
        .arg(state)
        .args(args)
        .output()
        .unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    let json = serde_json::from_str(&stdout).unwrap_or_else(|e| {
        panic!(
            "{args:?} printed no JSON ({e}): {stdout}\n{}",
            String::from_utf8_lossy(&output.stderr)
        )
    });
    (output.status.code().unwrap_or(-1), json)
}

/// The job in `report` whose specification is `digest`.
fn job_with<'a>(report: &'a Value, digest: &str) -> Option<&'a Value> {
    report["jobs"]
        .as_array()?
        .iter()
        .find(|job| job["spec"] == digest)
}

#[test]
fn a_submitted_job_runs_detached_and_a_resubmission_starts_nothing() {
    let scratch = Scratch::new("submit");
    let submit = ["jobs", "submit", "--", "/bin/sh", "-c", "sleep 0.1"];
    let (code, first) = splinter(&scratch.state(), &submit);
    assert_eq!(code, 0, "{first}");
    assert_eq!(first["launched"], true, "{first}");
    assert_eq!(first["job"]["state"], "running", "{first}");
    assert_eq!(first["job"]["attempt"], 1, "{first}");
    assert_eq!(first["job"]["launcher"], "local", "{first}");
    let digest = first["job"]["spec"].as_str().unwrap().to_owned();

    // The same specification is the same job: nothing is launched again.
    let (code, second) = splinter(&scratch.state(), &submit);
    assert_eq!(code, 0, "{second}");
    assert_eq!(second["launched"], false, "{second}");
    assert_eq!(second["job"]["spec"], digest.as_str(), "{second}");

    let (code, listed) = splinter(&scratch.state(), &["jobs", "list"]);
    assert_eq!(code, 0, "{listed}");
    let jobs = listed["jobs"].as_array().unwrap();
    assert_eq!(jobs.len(), 1, "one job, not one per submission: {listed}");
    assert_eq!(jobs[0]["spec"], digest.as_str(), "{listed}");
    assert_eq!(
        jobs[0]["state"], "running",
        "the record says what the submitting controller last saw: {listed}"
    );

    // Both the whole digest and a prefix of its hex name it.
    for id in [digest.clone(), digest[3..11].to_string()] {
        let (code, shown) = splinter(&scratch.state(), &["jobs", "show", &id]);
        assert_eq!(code, 0, "{shown}");
        assert_eq!(shown["job"]["spec"], digest.as_str(), "{shown}");
        assert_eq!(shown["spec"]["command"][0], "/bin/sh", "{shown}");
        assert!(
            shown["job"]["launch"].as_str().unwrap().contains("/jobs/"),
            "the attempt directory is under the state root's jobs area: {shown}"
        );
    }

    let (code, missing) = splinter(&scratch.state(), &["jobs", "show", "000000000000"]);
    assert_eq!(code, 2, "an unknown job is refused: {missing}");
    assert!(missing["refused"].as_bool().unwrap(), "{missing}");
}

#[test]
fn reconcile_completes_a_clean_exit_and_fails_a_nonzero_one() {
    let scratch = Scratch::new("reconcile");
    let clean = ["jobs", "submit", "--", "/bin/sh", "-c", "sleep 0.1"];
    let failed = [
        "jobs",
        "submit",
        "--attempts",
        "1",
        "--",
        "/bin/sh",
        "-c",
        "exit 3",
    ];
    let (_, clean_submission) = splinter(&scratch.state(), &clean);
    let clean_digest = clean_submission["job"]["spec"].as_str().unwrap().to_owned();
    let (_, failed_submission) = splinter(&scratch.state(), &failed);
    let failed_digest = failed_submission["job"]["spec"]
        .as_str()
        .unwrap()
        .to_owned();

    // A later process reconciles what the workers did after the submitter left.
    let deadline = Instant::now() + Duration::from_secs(20);
    let (mut clean_state, mut failed_state) = (None, None);
    while Instant::now() < deadline && (clean_state.is_none() || failed_state.is_none()) {
        let (_, reconciled) = splinter(&scratch.state(), &["jobs", "reconcile"]);
        for job in reconciled["jobs"].as_array().unwrap() {
            if job["spec"] == clean_digest.as_str() {
                clean_state = Some(job.clone());
            }
            if job["spec"] == failed_digest.as_str() {
                failed_state = Some(job.clone());
            }
        }
        if clean_state.is_none() || failed_state.is_none() {
            std::thread::sleep(Duration::from_millis(50));
        }
    }
    let clean_job = clean_state.expect("the clean job was reconciled");
    assert_eq!(clean_job["state"], "completed", "{clean_job}");
    assert_eq!(clean_job["exit_code"], 0, "{clean_job}");
    let failed_job = failed_state.expect("the failed job was reconciled");
    assert_eq!(failed_job["state"], "failed", "{failed_job}");
    assert_eq!(failed_job["exit_code"], 3, "{failed_job}");

    // Nothing is left to reconcile once both are terminal.
    let (code, reconciled) = splinter(&scratch.state(), &["jobs", "reconcile"]);
    assert_eq!(code, 0, "{reconciled}");
    assert!(
        job_with(&reconciled, &clean_digest).is_none()
            && job_with(&reconciled, &failed_digest).is_none(),
        "{reconciled}"
    );
}

/// Reconcile until `digest`'s record reaches `wanted`, returning it.
fn until_reconciled(state: &Path, digest: &str, wanted: &str) -> Value {
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let (_, reconciled) = splinter(state, &["jobs", "reconcile"]);
        if let Some(job) = job_with(&reconciled, digest) {
            if job["state"] == wanted {
                return job.clone();
            }
        }
        assert!(
            Instant::now() < deadline,
            "job {digest} never became {wanted}: {reconciled}"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[test]
fn a_failed_attempt_is_retried_while_the_policy_allows_one() {
    let scratch = Scratch::new("retry");
    let submit = [
        "jobs",
        "submit",
        "--attempts",
        "2",
        "--",
        "/bin/sh",
        "-c",
        "exit 4",
    ];
    let (_, submitted) = splinter(&scratch.state(), &submit);
    let digest = submitted["job"]["spec"].as_str().unwrap().to_owned();

    let retrying = until_reconciled(&scratch.state(), &digest, "retrying");
    assert_eq!(retrying["attempt"], 1, "{retrying}");
    assert_eq!(retrying["exit_code"], 4, "{retrying}");

    let (code, retried) = splinter(&scratch.state(), &["jobs", "retry"]);
    assert_eq!(code, 0, "{retried}");
    let launched = job_with(&retried, &digest).expect("the second attempt was launched");
    assert_eq!(launched["attempt"], 2, "{retried}");
    assert_eq!(launched["state"], "running", "{retried}");

    let failed = until_reconciled(&scratch.state(), &digest, "failed");
    assert_eq!(failed["attempt"], 2, "the policy was spent: {failed}");
    assert_eq!(failed["exit_code"], 4, "{failed}");

    // With no attempt left, a retry launches nothing.
    let (code, retried) = splinter(&scratch.state(), &["jobs", "retry"]);
    assert_eq!(code, 0, "{retried}");
    assert!(job_with(&retried, &digest).is_none(), "{retried}");
}

#[test]
fn a_wall_clock_bound_stops_a_worker_that_outlives_it() {
    let scratch = Scratch::new("wall-clock");
    let started = Instant::now();
    let submit = [
        "jobs",
        "submit",
        "--wall-clock",
        "1",
        "--",
        "/bin/sh",
        "-c",
        "sleep 30",
    ];
    let (_, submitted) = splinter(&scratch.state(), &submit);
    let digest = submitted["job"]["spec"].as_str().unwrap().to_owned();

    let failed = until_reconciled(&scratch.state(), &digest, "failed");
    assert_eq!(
        failed["exit_code"], 124,
        "the wall clock stopped it: {failed}"
    );
    assert!(
        started.elapsed() < Duration::from_secs(15),
        "the bound was enforced, not waited out"
    );
}

#[test]
fn cancelling_stops_the_workers_and_records_the_job_cancelled() {
    let scratch = Scratch::new("cancel");
    let sleep = ["jobs", "submit", "--", "/bin/sh", "-c", "sleep 30"];
    let (_, submitted) = splinter(&scratch.state(), &sleep);
    let digest = submitted["job"]["spec"].as_str().unwrap().to_owned();

    let (code, cancelled) = splinter(&scratch.state(), &["jobs", "cancel", &digest]);
    assert_eq!(code, 0, "{cancelled}");
    assert_eq!(cancelled["job"]["state"], "cancelled", "{cancelled}");
    assert_eq!(cancelled["job"]["spec"], digest.as_str(), "{cancelled}");
}

#[test]
fn plain_jobs_lists_its_subcommands_and_the_help_names_what_submit_takes() {
    let output = Command::new(env!("CARGO_BIN_EXE_splinter"))
        .args(["jobs"])
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    let listing = String::from_utf8(output.stdout).unwrap();
    for subcommand in ["submit", "list", "reconcile", "retry", "show", "cancel"] {
        assert!(
            listing.contains(subcommand),
            "{subcommand:?} in:\n{listing}"
        );
    }

    let output = Command::new(env!("CARGO_BIN_EXE_splinter"))
        .args(["jobs", "submit", "--help"])
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    let help = String::from_utf8(output.stdout).unwrap();
    for wanted in [
        "<PROGRAM ARG...>",
        "--distributed",
        "--nodes <N>",
        "--ranks-per-node <N>",
        "--gpus-per-rank <N>",
        "--wall-clock <SECONDS>",
        "--attempts <N>",
    ] {
        assert!(help.contains(wanted), "{wanted:?} is not in:\n{help}");
    }
}
