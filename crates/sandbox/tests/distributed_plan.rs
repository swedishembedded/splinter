// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements coordinator handoff between attempts of a
// distributed trainer, for its clients. If your team needs expertise in
// multi-rank launch that survives retries, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec: a job launched under a plan has the plan written into its attempt
//! directory before any rank starts, every rank reads the same bytes, and each
//! attempt gets its own plan naming the same job identity.

#![allow(clippy::unwrap_used)]

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use splinter_core::checkpoint::JobPlan;
use splinter_core::digest::Digest;
use splinter_core::job::{JobSpec, PlanPolicy, WorkerPlan};
use splinter_sandbox::job::{JobLauncher, Launch, LocalLauncher, Observation};

/// A worker that records which plan and rank it was given, then exits.
fn worker(dir: &Path) -> PathBuf {
    let path = dir.join("worker.sh");
    std::fs::write(
        &path,
        "#!/bin/sh\n\
         set -eu\n\
         plan=\"\"\n\
         rank=\"\"\n\
         while [ $# -gt 0 ]; do\n\
           case \"$1\" in\n\
             --job-plan) plan=$2; shift 2 ;;\n\
             --job-rank) rank=$2; shift 2 ;;\n\
             *) shift ;;\n\
           esac\n\
         done\n\
         sum=$(sha256sum \"$plan\" | cut -d' ' -f1)\n\
         printf 'rank=%s\\nplan=%s\\nsha256=%s\\n' \"$rank\" \"$plan\" \"$sum\" > \"record-$rank.txt\"\n",
    )
    .unwrap();
    let mut permissions = std::fs::metadata(&path).unwrap().permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&path, permissions).unwrap();
    path
}

fn two_rank(worker: &Path) -> JobSpec {
    let mut spec =
        JobSpec::single(vec![worker.display().to_string()]).with_plan(PlanPolicy::default());
    spec.workers = WorkerPlan {
        nodes: 1,
        ranks_per_node: 2,
        gpus_per_rank: 0,
    };
    spec
}

fn wait_for(path: &Path) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !path.exists() {
        assert!(
            Instant::now() < deadline,
            "{} never appeared",
            path.display()
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn wait_finished(launcher: &LocalLauncher, launch: &Launch) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !matches!(launcher.observe(launch), Observation::Finished { .. }) {
        assert!(Instant::now() < deadline, "the ranks never exited");
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// The plan file in `dir` and the plan it parses to.
fn read_plan(dir: &Path) -> (Vec<u8>, JobPlan) {
    let bytes = std::fs::read(dir.join("job-plan.json")).unwrap();
    let text = std::str::from_utf8(&bytes).unwrap();
    let plan = JobPlan::parse(text).unwrap();
    (bytes, plan)
}

/// The plan checksum a rank recorded.
fn recorded_sum(record: &str) -> &str {
    record
        .lines()
        .find_map(|line| line.strip_prefix("sha256="))
        .unwrap_or_else(|| panic!("the record names no plan checksum: {record:?}"))
}

#[test]
fn a_plan_is_written_before_the_ranks_start_and_both_read_the_same_bytes() {
    let scratch = tempfile::tempdir().unwrap();
    let worker = worker(scratch.path());
    let job = two_rank(&worker);
    let dir = scratch.path().join("attempt-1");
    let launcher = LocalLauncher::default();
    let launch = launcher.launch(&job, 1, &dir).expect("launch");
    for rank in 0..2 {
        wait_for(&dir.join(format!("record-{rank}.txt")));
    }
    wait_finished(&launcher, &launch);

    let (bytes, plan) = read_plan(&dir);
    assert_eq!(
        plan.job,
        job.digest().hex(),
        "the plan names the job digest"
    );
    assert_eq!(plan.world_size, 2);
    assert_eq!((plan.tp, plan.pp, plan.dp), (1, 1, 2));
    assert!(
        plan.coordinator.starts_with("127.0.0.1:"),
        "the coordinator is on the loopback address: {}",
        plan.coordinator
    );
    let port: u16 = plan
        .coordinator
        .rsplit(':')
        .next()
        .unwrap()
        .parse()
        .expect("a coordinator port");
    assert!(port > 0);
    assert_eq!(
        (plan.handshake_timeout_ms, plan.op_timeout_ms),
        (120_000, 600_000)
    );

    let records: Vec<String> = (0..2)
        .map(|rank| std::fs::read_to_string(dir.join(format!("record-{rank}.txt"))).unwrap())
        .collect();
    assert_ne!(
        records[0], records[1],
        "each rank records its own invocation"
    );
    for (rank, record) in records.iter().enumerate() {
        assert!(record.contains(&format!("rank={rank}\n")), "{record}");
        assert!(record.contains("plan=job-plan.json\n"), "{record}");
    }
    let expected = Digest::sha256_of(&bytes).hex().to_owned();
    assert_eq!(recorded_sum(&records[0]), expected);
    assert_eq!(
        recorded_sum(&records[0]),
        recorded_sum(&records[1]),
        "both ranks read the same plan bytes"
    );
}

#[test]
fn a_retry_writes_its_own_plan_naming_the_same_job() {
    let scratch = tempfile::tempdir().unwrap();
    let worker = worker(scratch.path());
    let job = two_rank(&worker);
    let launcher = LocalLauncher::default();

    let first_dir = scratch.path().join("attempt-1");
    let first = launcher.launch(&job, 1, &first_dir).expect("first launch");
    wait_finished(&launcher, &first);
    let (first_bytes, first_plan) = read_plan(&first_dir);

    let second_dir = scratch.path().join("attempt-2");
    let second = launcher
        .launch(&job, 2, &second_dir)
        .expect("second launch");
    wait_finished(&launcher, &second);
    let (_, second_plan) = read_plan(&second_dir);

    assert_eq!(first_plan.job, second_plan.job);
    assert_eq!(first_plan.job, job.digest().hex());
    assert!(
        first_plan.coordinator.starts_with("127.0.0.1:")
            && second_plan.coordinator.starts_with("127.0.0.1:"),
        "each attempt has its own coordinator"
    );
    assert_eq!(
        std::fs::read(first_dir.join("job-plan.json")).unwrap(),
        first_bytes,
        "the next attempt does not rewrite the plan it replaced"
    );
    assert!(
        second_dir.join("job-plan.json").exists(),
        "the retry has its own plan file"
    );
}

#[test]
fn a_job_that_asks_for_no_plan_writes_none() {
    let scratch = tempfile::tempdir().unwrap();
    let dir = scratch.path().join("attempt-1");
    let launcher = LocalLauncher::default();
    let job = JobSpec::single(vec!["/bin/true".into()]);
    let launch = launcher.launch(&job, 1, &dir).expect("launch");
    wait_finished(&launcher, &launch);
    assert!(
        !dir.join("job-plan.json").exists(),
        "a job without a plan policy writes no plan file"
    );
}
