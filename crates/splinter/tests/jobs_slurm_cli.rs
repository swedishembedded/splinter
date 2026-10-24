// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements selectable job launching for learning
// campaigns, for its clients. If your team needs expertise in batch-system
// integration or operator commands that reconcile work after a restart, you
// can procure our services by sending an email to info@swedishembedded.com.

//! Spec: `splinter jobs` submits through Slurm when `--job-launcher slurm`
//! names the scheduler's programs, records the submitted job, shows it running
//! and records it completed after the scheduler reports it. The scheduler is
//! stand-in programs that record what the launcher asked and answer from a
//! state file, so no scheduler, model or GPU is needed.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

/// A scratch directory holding the state root and the stand-in scheduler.
struct Scratch {
    root: PathBuf,
    state: PathBuf,
    sbatch: PathBuf,
    squeue: PathBuf,
    scancel: PathBuf,
}

impl Scratch {
    fn new(name: &str) -> Self {
        let root = std::env::temp_dir().join(format!(
            "splinter-jobs-slurm-cli-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let bin = root.join("bin");
        let state = root.join("state");
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::create_dir_all(&state).unwrap();
        let recorded = state.display().to_string();
        write_program(
            &bin.join("sbatch"),
            &format!(
                "#!/bin/sh\n\
                 printf '%s\\n' \"$@\" > '{recorded}/sbatch.args'\n\
                 printf 'call\\n' >> '{recorded}/sbatch.calls'\n\
                 for last do :; done\n\
                 cp \"$last\" '{recorded}/script'\n\
                 cat '{recorded}/sbatch.output'\n"
            ),
        );
        write_program(
            &bin.join("squeue"),
            &format!(
                "#!/bin/sh\n\
                 printf '%s\\n' \"$@\" > '{recorded}/squeue.args'\n\
                 cat '{recorded}/squeue.state'\n"
            ),
        );
        write_program(
            &bin.join("scancel"),
            &format!(
                "#!/bin/sh\n\
                 printf '%s\\n' \"$@\" > '{recorded}/scancel.args'\n"
            ),
        );
        let scratch = Self {
            root,
            state,
            sbatch: bin.join("sbatch"),
            squeue: bin.join("squeue"),
            scancel: bin.join("scancel"),
        };
        scratch.write("sbatch.output", "4242;test\n");
        scratch.write("squeue.state", "RUNNING\n");
        scratch
    }

    fn write(&self, name: &str, body: &str) {
        std::fs::write(self.state.join(name), body).unwrap();
    }

    fn read(&self, name: &str) -> String {
        std::fs::read_to_string(self.state.join(name)).unwrap()
    }

    /// The flags that select the stand-in scheduler.
    fn launcher_flags(&self) -> Vec<String> {
        vec![
            "--job-launcher".into(),
            "slurm".into(),
            "--slurm-sbatch".into(),
            self.sbatch.display().to_string(),
            "--slurm-squeue".into(),
            self.squeue.display().to_string(),
            "--slurm-scancel".into(),
            self.scancel.display().to_string(),
            "--slurm-arg".into(),
            "--partition=debug".into(),
        ]
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn write_program(path: &Path, body: &str) {
    std::fs::write(path, body).unwrap();
    let mut permissions = std::fs::metadata(path).unwrap().permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(path, permissions).unwrap();
}

/// `splinter --json --state <state> <global...> <args...>`; exit code and
/// stdout as JSON.
fn splinter(state: &Path, global: &[String], args: &[&str]) -> (i32, Value) {
    let output = Command::new(env!("CARGO_BIN_EXE_splinter"))
        .arg("--json")
        .arg("--state")
        .arg(state)
        .args(global)
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

fn job_with<'a>(report: &'a Value, digest: &str) -> Option<&'a Value> {
    report["jobs"]
        .as_array()?
        .iter()
        .find(|job| job["spec"] == digest)
}

#[test]
fn a_slurm_job_is_submitted_shown_running_and_reconciled_completed() {
    let scratch = Scratch::new("lifecycle");
    let flags = scratch.launcher_flags();
    let submit = ["jobs", "submit", "--", "/bin/echo", "hello"];

    let (code, submitted) = splinter(&scratch.state, &flags, &submit);
    assert_eq!(code, 0, "{submitted}");
    assert_eq!(submitted["launched"], true, "{submitted}");
    assert_eq!(submitted["job"]["state"], "running", "{submitted}");
    assert_eq!(submitted["job"]["launcher"], "slurm", "{submitted}");
    let digest = submitted["job"]["spec"].as_str().unwrap().to_owned();

    // sbatch received the configured extra argument, then --parsable and the
    // batch script, and the script carries the job's plan.
    let arguments: Vec<String> = scratch
        .read("sbatch.args")
        .lines()
        .map(str::to_string)
        .collect();
    assert_eq!(arguments[0], "--partition=debug", "{arguments:?}");
    assert_eq!(arguments[1], "--parsable", "{arguments:?}");
    assert!(
        arguments[2].ends_with("job.sbatch"),
        "the script is submitted: {arguments:?}"
    );
    let script = scratch.read("script");
    assert!(script.contains("#SBATCH --nodes=1\n"), "{script}");
    assert!(script.contains("#SBATCH --ntasks-per-node="), "{script}");
    assert!(script.contains("exec srun "), "{script}");

    // The attempt directory records the scheduler's job id.
    let launch = Path::new(submitted["job"]["launch"].as_str().unwrap());
    assert_eq!(
        std::fs::read_to_string(launch.join("scheduler.job"))
            .unwrap()
            .trim(),
        "4242"
    );

    let (code, listed) = splinter(&scratch.state, &flags, &["jobs", "list"]);
    assert_eq!(code, 0, "{listed}");
    let job = job_with(&listed, &digest).expect("the submitted job is listed");
    assert_eq!(job["state"], "running", "{listed}");
    assert_eq!(job["launcher"], "slurm", "{listed}");

    // The scheduler finishes the job; reconciliation records what it proves.
    scratch.write("squeue.state", "COMPLETED\n");
    let (code, reconciled) = splinter(&scratch.state, &flags, &["jobs", "reconcile"]);
    assert_eq!(code, 0, "{reconciled}");
    let job = job_with(&reconciled, &digest).expect("the job changed state");
    assert_eq!(job["state"], "completed", "{reconciled}");
    assert_eq!(job["exit_code"], 0, "{reconciled}");

    let (code, listed) = splinter(&scratch.state, &flags, &["jobs", "list"]);
    assert_eq!(code, 0, "{listed}");
    assert_eq!(job_with(&listed, &digest).unwrap()["state"], "completed");

    // The scheduler was asked about the recorded id.
    let arguments: Vec<String> = scratch
        .read("squeue.args")
        .lines()
        .map(str::to_string)
        .collect();
    assert_eq!(arguments, ["-h", "-j", "4242", "-o", "%T"]);
}

#[test]
fn a_slurm_launcher_without_its_programs_is_refused_by_name() {
    let scratch = Scratch::new("missing");
    let (code, refused) = splinter(
        &scratch.state,
        &[
            "--job-launcher".into(),
            "slurm".into(),
            "--slurm-squeue".into(),
            "squeue".into(),
        ],
        &["status"],
    );
    assert_eq!(
        code, 2,
        "a missing launcher setting is a refusal: {refused}"
    );
    assert_eq!(refused["refused"], true, "{refused}");
    assert!(
        refused["error"]
            .as_str()
            .unwrap()
            .contains("--slurm-sbatch"),
        "the refusal names the setting: {refused}"
    );
}

#[test]
fn an_unknown_launcher_kind_is_refused_by_the_grammar() {
    let scratch = Scratch::new("unknown");
    let output = Command::new(env!("CARGO_BIN_EXE_splinter"))
        .arg("--json")
        .arg("--state")
        .arg(&scratch.state)
        .args(["--job-launcher", "unknown", "status"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2), "{output:?}");
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains("unknown"), "{error}");
    assert!(error.contains("local"), "the kinds are named: {error}");
}
