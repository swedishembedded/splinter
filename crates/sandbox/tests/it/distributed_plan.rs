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
    // Every attempt has its own owner-only secret for its ranks.
    let secret = |dir: &Path| std::fs::read_to_string(dir.join("job.secret")).unwrap();
    assert_eq!(secret(&first_dir).trim().len(), 64);
    assert_ne!(secret(&first_dir), secret(&second_dir));
    let mode = std::fs::metadata(first_dir.join("job.secret"))
        .unwrap()
        .permissions()
        .mode();
    assert_eq!(mode & 0o777, 0o600);
}

/// A stand-in `srun`: records its arguments, then runs the step's script
/// once per task of a one-node step, as the scheduler's task variables say.
const SRUN: &str = r#"#!/bin/sh
echo "$@" > "$0.args"
tasks=1
for a in "$@"; do case "$a" in --ntasks=*) tasks=${a#--ntasks=} ;; esac; done
while [ "$1" != sh ]; do shift; done
script=$2
i=0
pids=""
while [ "$i" -lt "$tasks" ]; do
  SLURM_JOB_ID=9 SLURM_STEP_ID=4 SLURM_PROCID=$i SLURM_LOCALID=$i SLURM_NTASKS=$tasks sh "$script" &
  pids="$pids $!"
  i=$((i + 1))
done
status=0
for p in $pids; do wait "$p" || status=1; done
exit $status
"#;

#[test]
fn a_step_launch_runs_every_rank_as_one_step_with_the_coordinator_on_rank_0s_node() {
    use splinter_sandbox::job::NodeStepLauncher;
    use splinter_sandbox::step::NodeSteps;
    let scratch = tempfile::tempdir().unwrap();
    let worker = worker(scratch.path());
    let srun = scratch.path().join("srun");
    std::fs::write(&srun, SRUN).unwrap();
    std::fs::set_permissions(&srun, std::fs::Permissions::from_mode(0o755)).unwrap();
    let launcher = NodeStepLauncher::new(
        NodeSteps {
            srun: srun.clone(),
            scancel: None,
            args: vec!["--mem=1G".into()],
            single_node_args: vec!["--network=single_node_vni".into()],
            exec_prefix: Vec::new(),
        },
        vec!["localhost".into(), "unused".into()],
        None,
    );
    let mut job = two_rank(&worker);
    job.workers.gpus_per_rank = 1;
    job.command.push("--card=gpu{local_rank}".into());
    let dir = scratch.path().join("attempt-1");
    let launch = launcher.launch(&job, 1, &dir).expect("launch");
    assert_eq!(launch.world_size, 1, "one srun carries every rank");
    for rank in 0..2 {
        wait_for(&dir.join(format!("record-{rank}.txt")));
    }
    let deadline = Instant::now() + Duration::from_secs(20);
    let outcome = loop {
        match launcher.observe(&launch) {
            Observation::Finished { outcome, .. } => break outcome,
            Observation::Running { ranks_alive } => {
                assert_eq!(ranks_alive, [0, 1], "every rank of the step is alive");
            }
        }
        assert!(Instant::now() < deadline, "the step did not end");
        std::thread::sleep(Duration::from_millis(50));
    };
    assert_eq!(outcome, splinter_sandbox::job::JobOutcome::Completed);

    let (_, plan) = read_plan(&dir);
    assert!(
        plan.coordinator.starts_with("127.0.0.1:"),
        "rank 0's node is localhost: {}",
        plan.coordinator
    );
    let args = std::fs::read_to_string(scratch.path().join("srun.args")).unwrap();
    for wanted in [
        "--overlap",
        "--nodelist=localhost",
        "--nodes=1",
        "--ntasks-per-node=2",
        "--gpus-per-node=2",
        "--network=single_node_vni",
        "--mem=1G",
    ] {
        assert!(args.contains(wanted), "{wanted} in {args}");
    }
    assert_eq!(
        std::fs::read_to_string(dir.join("step.id")).unwrap().trim(),
        "9.4"
    );
    let script = std::fs::read_to_string(dir.join("job.step.sh")).unwrap();
    assert!(script.contains("'--card=gpu'${SLURM_LOCALID}"), "{script}");

    let mut too_wide = two_rank(&worker);
    too_wide.workers.nodes = 3;
    too_wide.workers.ranks_per_node = 1;
    let error = launcher
        .launch(&too_wide, 1, &scratch.path().join("attempt-wide"))
        .unwrap_err()
        .to_string();
    assert!(error.contains("3 nodes"), "{error}");
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
