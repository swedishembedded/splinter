// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements end-to-end submission of distributed training
// jobs, for its clients. If your team needs expertise in trainer contracts or
// checkpoint lineage from multi-rank runs, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec: `jobs submit --distributed` prepares a training job on this machine.
//! It writes the plan every rank reads into the attempt directory, appends
//! `--job-plan`/`--job-rank` to the worker command, and when two stand-in ranks
//! commit checkpoints the controller reconciles the job to completed and keeps
//! their files as artifacts whose checksums match.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use serde_json::Value;
use splinter_sdk::store::artifacts::ArtifactStore;
use splinter_sdk::store::workspace::Workspace;
use splinter_sdk::store::StateRoot;
use splinter_sdk::vocabulary::checkpoint::JobPlan;
use splinter_sdk::vocabulary::digest::Digest;

/// A scratch directory holding the state root and the stand-in worker.
struct Scratch {
    root: PathBuf,
    state: PathBuf,
    worker: PathBuf,
}

/// The stand-in rank: it records which plan and rank it was given, writes its
/// own output file, and rank 0 publishes the commit manifest for both.
const WORKER: &str = r#"#!/bin/sh
set -eu
plan=""
rank=""
while [ $# -gt 0 ]; do
  case "$1" in
    --job-plan) plan=$2; shift 2 ;;
    --job-rank) rank=$2; shift 2 ;;
    *) shift ;;
  esac
done
sum=$(sha256sum "$plan" | cut -d' ' -f1)
printf 'rank=%s\nplan=%s\nsha256=%s\n' "$rank" "$plan" "$sum" > "record-$rank.txt"
printf 'rank %s data\n' "$rank" > "rank-$rank.bin"
if [ "$rank" = 0 ]; then
  tries=0
  while [ ! -f rank-1.bin ]; do
    tries=$((tries + 1))
    if [ "$tries" -ge 500 ]; then
      echo "rank 1 never wrote its output" >&2
      exit 1
    fi
    sleep 0.01
  done
  job=$(sed -n 's/.*"job":"\([^"]*\)".*/\1/p' "$plan")
  plan_json=$(sed 's/\\/\\\\/g; s/"/\\"/g; s/$/\\n/' "$plan" | tr -d '\n')
  sum0=$(sha256sum rank-0.bin | cut -d' ' -f1)
  len0=$(wc -c < rank-0.bin | tr -d ' ')
  sum1=$(sha256sum rank-1.bin | cut -d' ' -f1)
  len1=$(wc -c < rank-1.bin | tr -d ' ')
  printf '{"job":"%s","plan":"%s","step":2,"world_size":2,"files":[{"role":"weights","path":"rank-0.bin","len":%s,"sha256":"sha256:%s"},{"role":"weights","path":"rank-1.bin","len":%s,"sha256":"sha256:%s"}]}\n' \
    "$job" "$plan_json" "$len0" "$sum0" "$len1" "$sum1" > commit-00000002.json
fi
"#;

impl Scratch {
    fn new(name: &str) -> Self {
        let root = std::env::temp_dir().join(format!(
            "splinter-jobs-distributed-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let bin = root.join("bin");
        std::fs::create_dir_all(&bin).unwrap();
        let worker = bin.join("worker.sh");
        std::fs::write(&worker, WORKER).unwrap();
        let mut permissions = std::fs::metadata(&worker).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&worker, permissions).unwrap();
        Self {
            state: root.join("state"),
            root,
            worker,
        }
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
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

fn job_with<'a>(report: &'a Value, digest: &str) -> Option<&'a Value> {
    report["jobs"]
        .as_array()?
        .iter()
        .find(|job| job["spec"] == digest)
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

/// The value of `key` in a rank's record file.
fn recorded<'a>(record: &'a str, key: &str) -> &'a str {
    record
        .lines()
        .find_map(|line| line.strip_prefix(key))
        .unwrap_or_else(|| panic!("{record:?} names no {key}"))
}

#[test]
fn a_distributed_submission_writes_the_plan_and_registers_each_ranks_output() {
    let scratch = Scratch::new("submit");
    let worker = scratch.worker.display().to_string();
    let (code, submitted) = splinter(
        &scratch.state,
        &[
            "jobs",
            "submit",
            "--distributed",
            "--nodes",
            "1",
            "--ranks-per-node",
            "2",
            "--gpus-per-rank",
            "1",
            "--",
            &worker,
        ],
    );
    assert_eq!(code, 0, "{submitted}");
    assert_eq!(submitted["launched"], true, "{submitted}");
    assert_eq!(submitted["job"]["state"], "running", "{submitted}");
    let digest = submitted["job"]["spec"].as_str().unwrap().to_owned();
    let hex = digest.rsplit(':').next().unwrap().to_owned();
    let launch = PathBuf::from(submitted["job"]["launch"].as_str().unwrap());

    // The plan is in the attempt directory, parses with the contract, and names
    // the job by its specification digest.
    let plan_bytes = std::fs::read(launch.join("job-plan.json")).unwrap();
    let plan = JobPlan::parse(std::str::from_utf8(&plan_bytes).unwrap()).unwrap();
    assert_eq!(plan.job, hex);
    assert_eq!(plan.world_size, 2);
    assert_eq!((plan.tp, plan.pp, plan.dp), (1, 1, 2));
    assert!(
        plan.coordinator.starts_with("127.0.0.1:"),
        "{}",
        plan.coordinator
    );
    assert_eq!(
        (plan.handshake_timeout_ms, plan.op_timeout_ms),
        (120_000, 600_000)
    );

    let job = until_reconciled(&scratch.state, &digest, "completed");
    assert_eq!(job["exit_code"], 0, "{job}");

    // Both ranks ran under their own rank and the same plan bytes.
    let records: Vec<String> = (0..2)
        .map(|rank| std::fs::read_to_string(launch.join(format!("record-{rank}.txt"))).unwrap())
        .collect();
    assert_ne!(records[0], records[1], "the rank records are disjoint");
    for (rank, record) in records.iter().enumerate() {
        assert_eq!(recorded(record, "rank="), rank.to_string(), "{record}");
        assert_eq!(recorded(record, "plan="), "job-plan.json", "{record}");
    }
    assert_eq!(
        recorded(&records[0], "sha256="),
        Digest::sha256_of(&plan_bytes).hex(),
        "a rank records the plan it read"
    );
    assert_eq!(
        recorded(&records[0], "sha256="),
        recorded(&records[1], "sha256="),
        "both ranks read the same plan bytes"
    );

    // The commit the ranks published became artifacts with the checksums the
    // manifest declared, and one lineage document links them to the job.
    let root = StateRoot::new(&scratch.state);
    let workspace = Workspace::at(&root);
    let artifacts = ArtifactStore::new(&workspace, &root);
    let outputs = workspace.job_outputs().unwrap();
    assert_eq!(outputs.len(), 1, "{outputs:?}");
    let output = &outputs[0];
    assert_eq!(output.job.as_str(), digest.as_str());
    assert_eq!(output.attempt, 1);
    assert_eq!(output.step, 2);
    assert_eq!(output.world_size, 2);
    assert_eq!(output.plan, std::str::from_utf8(&plan_bytes).unwrap());
    assert_eq!(output.files.len(), 2);
    for (rank, file) in output.files.iter().enumerate() {
        let bytes = format!("rank {rank} data\n");
        assert_eq!(file.role, "weights");
        assert_eq!(file.len, bytes.len() as u64);
        assert_eq!(file.sha256, Digest::sha256_of(bytes.as_bytes()));
        let kept = artifacts.get(&file.digest).unwrap();
        assert_eq!(kept.sha256.as_ref(), Some(&file.sha256));
        assert_eq!(
            std::fs::read(artifacts.path(&file.digest).unwrap()).unwrap(),
            bytes.as_bytes()
        );
    }
}

#[test]
fn the_help_names_the_distributed_mode() {
    let output = Command::new(env!("CARGO_BIN_EXE_splinter"))
        .args(["jobs", "submit", "--help"])
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    let help = String::from_utf8(output.stdout).unwrap();
    assert!(help.contains("--distributed"), "{help}");
}
