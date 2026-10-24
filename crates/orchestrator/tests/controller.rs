// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements crash-tolerant control of jobs that outlive
// the process that submitted them, for its clients. If your team needs
// expertise in lease-fenced reconciliation or verified checkpoint lineage,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: the controller submits once, reconciles only what it can prove, and
//! turns a completed attempt's commit manifest into artifacts and lineage.

#![allow(clippy::unwrap_used)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use splinter_core::clock::SystemClock;
use splinter_core::job::{JobSpec, JobState};
use splinter_orchestrator::jobs::{JobController, Submission};
use splinter_orchestrator::OrchestratorError;
use splinter_sandbox::job::{JobError, JobLauncher, JobOutcome, Launch, Observation};
use splinter_store::workspace::Workspace;
use splinter_store::StateRoot;

/// A launcher whose workers are whatever the test says they are.
#[derive(Default)]
struct ScriptedLauncher {
    running: Mutex<HashMap<PathBuf, Vec<u32>>>,
    finished: Mutex<HashMap<PathBuf, (JobOutcome, Option<i32>)>>,
    launches: Mutex<Vec<PathBuf>>,
}

impl ScriptedLauncher {
    fn set_exit(&self, dir: &Path, code: i32) {
        let outcome = if code == 0 {
            JobOutcome::Completed
        } else {
            JobOutcome::Failed
        };
        self.set_finished(dir, outcome, Some(code));
    }

    fn set_finished(&self, dir: &Path, outcome: JobOutcome, exit_code: Option<i32>) {
        self.running.lock().unwrap().remove(dir);
        self.finished
            .lock()
            .unwrap()
            .insert(dir.to_path_buf(), (outcome, exit_code));
    }
}

impl JobLauncher for ScriptedLauncher {
    fn name(&self) -> &'static str {
        "scripted"
    }

    fn launch(&self, spec: &JobSpec, attempt: u32, dir: &Path) -> Result<Launch, JobError> {
        std::fs::create_dir_all(dir).map_err(|source| JobError::Io {
            context: "creating the attempt".into(),
            source,
        })?;
        self.launches.lock().unwrap().push(dir.to_path_buf());
        self.running
            .lock()
            .unwrap()
            .insert(dir.to_path_buf(), vec![42]);
        let launch = Launch {
            digest: spec.digest(),
            attempt,
            world_size: spec.workers.world_size(),
            pids: vec![42],
            dir: dir.to_path_buf(),
        };
        std::fs::write(
            dir.join("launch.json"),
            serde_json::to_vec(&launch).unwrap(),
        )
        .unwrap();
        Ok(launch)
    }

    fn observe(&self, launch: &Launch) -> Observation {
        if let Some((outcome, exit_code)) = self.finished.lock().unwrap().get(&launch.dir) {
            return Observation::Finished {
                outcome: *outcome,
                exit_code: *exit_code,
            };
        }
        Observation::Running {
            ranks_alive: self
                .running
                .lock()
                .unwrap()
                .get(&launch.dir)
                .cloned()
                .unwrap_or_default(),
        }
    }

    fn stop(&self, launch: &Launch, _rank: Option<u32>) -> Result<(), JobError> {
        self.running.lock().unwrap().remove(&launch.dir);
        Ok(())
    }
}

fn workspace() -> (tempfile::TempDir, Workspace) {
    let dir = tempfile::tempdir().unwrap();
    let workspace = Workspace::at(&StateRoot::new(dir.path()));
    (dir, workspace)
}

fn spec() -> JobSpec {
    JobSpec::single(vec!["brain".into(), "train".into()])
}

#[test]
fn submitting_twice_launches_one_attempt() {
    let (_dir, workspace) = workspace();
    let launcher = ScriptedLauncher::default();
    let clock = SystemClock;
    let root =
        std::env::temp_dir().join(format!("splinter-controller-{}-submit", std::process::id()));
    let controller = JobController::new(&workspace, &launcher, root, &clock, "test-controller");
    let first = controller.submit(&spec()).unwrap();
    let Submission::Launched(record) = first else {
        panic!("the first submission must launch")
    };
    assert_eq!(record.state, JobState::Running);
    assert_eq!(record.attempt, 1);
    let second = controller.submit(&spec()).unwrap();
    let Submission::Existing(existing) = second else {
        panic!("the second submission must find the job")
    };
    assert_eq!(existing, record);
    assert_eq!(launcher.launches.lock().unwrap().len(), 1);
}

#[test]
fn a_clean_exit_completes_the_job_and_a_failed_one_retries() {
    let (_dir, workspace) = workspace();
    let launcher = ScriptedLauncher::default();
    let clock = SystemClock;
    let root = std::env::temp_dir().join(format!(
        "splinter-controller-{}-reconcile",
        std::process::id()
    ));
    let controller = JobController::new(&workspace, &launcher, root, &clock, "test-controller");
    let mut job = spec();
    job.retry.attempts = 2;
    let Submission::Launched(record) = controller.submit(&job).unwrap() else {
        panic!("launched")
    };

    launcher.set_exit(Path::new(record.launch.as_ref().unwrap()), 3);
    let updated = controller.reconcile().unwrap();
    assert_eq!(updated.len(), 1);
    assert_eq!(updated[0].state, JobState::Retrying);
    assert_eq!(updated[0].exit_code, Some(3));

    let retried = controller.retry_ready().unwrap();
    assert_eq!(retried.len(), 1);
    assert_eq!(retried[0].attempt, 2);
    assert_eq!(retried[0].state, JobState::Running);

    launcher.set_exit(Path::new(retried[0].launch.as_ref().unwrap()), 0);
    let updated = controller.reconcile().unwrap();
    assert_eq!(updated[0].state, JobState::Completed);
    assert!(workspace.jobs_needing_reconciliation().unwrap().is_empty());
}

/// The state one reconcile gives a job whose launcher reports `outcome`.
fn outcome_state(
    outcome: JobOutcome,
    exit_code: Option<i32>,
    attempts: u32,
    retry_on_preemption: bool,
) -> (JobState, Option<i32>) {
    let (_state, workspace) = workspace();
    let launcher = ScriptedLauncher::default();
    let clock = SystemClock;
    let root = tempfile::tempdir().unwrap();
    let controller = JobController::new(
        &workspace,
        &launcher,
        root.path().to_path_buf(),
        &clock,
        "test",
    );
    let mut job = spec();
    job.retry.attempts = attempts;
    job.retry.retry_on_preemption = retry_on_preemption;
    let Submission::Launched(record) = controller.submit(&job).unwrap() else {
        panic!("launched")
    };
    launcher.set_finished(
        Path::new(record.launch.as_ref().unwrap()),
        outcome,
        exit_code,
    );
    let updated = controller.reconcile().unwrap();
    (updated[0].state, updated[0].exit_code)
}

#[test]
fn each_outcome_maps_to_the_state_the_controller_states() {
    assert_eq!(
        outcome_state(JobOutcome::Completed, Some(0), 1, true),
        (JobState::Completed, Some(0))
    );
    // The outcome decides, not the code: a launcher that says a job failed
    // fails it even if it reports a zero exit.
    assert_eq!(
        outcome_state(JobOutcome::Failed, Some(0), 1, true),
        (JobState::Failed, Some(0))
    );
    assert_eq!(
        outcome_state(JobOutcome::Failed, Some(2), 1, true),
        (JobState::Failed, Some(2))
    );
    // A cancel is terminal however many attempts remain.
    assert_eq!(
        outcome_state(JobOutcome::Cancelled, None, 5, true),
        (JobState::Cancelled, None)
    );
    assert_eq!(
        outcome_state(JobOutcome::Lost, None, 1, true),
        (JobState::Lost, None)
    );
}

#[test]
fn a_preempted_attempt_retries_only_when_the_policy_allows() {
    assert_eq!(
        outcome_state(JobOutcome::Preempted, None, 2, true),
        (JobState::Retrying, None)
    );
    assert_eq!(
        outcome_state(JobOutcome::Preempted, None, 3, false),
        (JobState::Failed, None)
    );
    assert_eq!(
        outcome_state(JobOutcome::Preempted, None, 1, true),
        (JobState::Failed, None)
    );
}

#[test]
fn a_retrying_preempted_job_launches_its_next_attempt() {
    let (_state, workspace) = workspace();
    let launcher = ScriptedLauncher::default();
    let clock = SystemClock;
    let root = tempfile::tempdir().unwrap();
    let controller = JobController::new(
        &workspace,
        &launcher,
        root.path().to_path_buf(),
        &clock,
        "test",
    );
    let mut job = spec();
    job.retry.attempts = 2;
    let Submission::Launched(record) = controller.submit(&job).unwrap() else {
        panic!("launched")
    };
    launcher.set_finished(
        Path::new(record.launch.as_ref().unwrap()),
        JobOutcome::Preempted,
        None,
    );
    let updated = controller.reconcile().unwrap();
    assert_eq!(updated[0].state, JobState::Retrying);

    let retried = controller.retry_ready().unwrap();
    assert_eq!(retried.len(), 1);
    assert_eq!(retried[0].attempt, 2);
    assert_eq!(retried[0].state, JobState::Running);
}

/// Writes `name` in `dir` and the manifest entry naming it.
fn commit_file(dir: &Path, role: &str, name: &str, bytes: &[u8]) -> serde_json::Value {
    let path = dir.join(name);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(&path, bytes).unwrap();
    serde_json::json!({
        "role": role,
        "path": name,
        "len": bytes.len(),
        "sha256": splinter_core::digest::Digest::sha256_of(bytes).as_str(),
    })
}

fn write_commit(dir: &Path, step: u32, files: &[serde_json::Value]) {
    let manifest = serde_json::json!({
        "job": "qwen-coding-v1",
        "plan": "{\"job\":\"qwen-coding-v1\",\"world_size\":1}",
        "step": step,
        "world_size": 1,
        "files": files,
    });
    std::fs::write(
        dir.join(format!("commit-{step:08}.json")),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
}

#[test]
fn a_completed_job_registers_its_outputs_once() {
    let (_state, workspace) = workspace();
    let launcher = ScriptedLauncher::default();
    let clock = SystemClock;
    let root = tempfile::tempdir().unwrap();
    let controller = JobController::new(
        &workspace,
        &launcher,
        root.path().to_path_buf(),
        &clock,
        "test",
    );
    let Submission::Launched(record) = controller.submit(&spec()).unwrap() else {
        panic!("launched")
    };
    let dir = Path::new(record.launch.as_ref().unwrap());
    let weights = commit_file(dir, "weights", "weights.safetensors", b"weights\n");
    let state = commit_file(dir, "state", "state/train.state", b"train state");
    write_commit(dir, 500, &[weights, state]);
    launcher.set_exit(dir, 0);

    let updated = controller.reconcile().unwrap();
    assert_eq!(updated[0].state, JobState::Completed);
    let outputs = workspace.job_outputs().unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].step, 500);
    assert_eq!(outputs[0].attempt, 1);
    assert_eq!(outputs[0].job, record.spec);
    assert_eq!(outputs[0].files.len(), 2);
    assert_eq!(
        workspace
            .job_output(&record.spec, 1)
            .unwrap()
            .expect("the lineage document")
            .files[0]
            .role,
        "weights"
    );

    // A second reconciliation has nothing left to do and adds nothing.
    assert!(controller.reconcile().unwrap().is_empty());
    assert_eq!(workspace.job_outputs().unwrap().len(), 1);
}

#[test]
fn a_tampered_checkpoint_fails_the_job() {
    let (_state, workspace) = workspace();
    let launcher = ScriptedLauncher::default();
    let clock = SystemClock;
    let root = tempfile::tempdir().unwrap();
    let controller = JobController::new(
        &workspace,
        &launcher,
        root.path().to_path_buf(),
        &clock,
        "test",
    );
    let Submission::Launched(record) = controller.submit(&spec()).unwrap() else {
        panic!("launched")
    };
    let dir = Path::new(record.launch.as_ref().unwrap());
    let weights = commit_file(dir, "weights", "weights.safetensors", b"weights\n");
    write_commit(dir, 500, &[weights]);
    // Same length, other bytes: the checksum is what refuses it.
    std::fs::write(dir.join("weights.safetensors"), b"WEIGHTS\n").unwrap();
    launcher.set_exit(dir, 0);

    let updated = controller.reconcile().unwrap();
    assert_eq!(updated[0].state, JobState::Failed);
    assert!(
        updated[0]
            .detail
            .as_deref()
            .is_some_and(|detail| detail.contains("hashes to")),
        "{:?}",
        updated[0].detail
    );
    assert!(workspace.job_outputs().unwrap().is_empty());
}

#[test]
fn a_manifest_that_names_a_path_outside_the_attempt_fails_the_job() {
    let (_state, workspace) = workspace();
    let launcher = ScriptedLauncher::default();
    let clock = SystemClock;
    let root = tempfile::tempdir().unwrap();
    let controller = JobController::new(
        &workspace,
        &launcher,
        root.path().to_path_buf(),
        &clock,
        "test",
    );
    let Submission::Launched(record) = controller.submit(&spec()).unwrap() else {
        panic!("launched")
    };
    let dir = Path::new(record.launch.as_ref().unwrap());
    let sha = splinter_core::digest::Digest::sha256_of(b"outside");
    let manifest = serde_json::json!({
        "job": "qwen-coding-v1",
        "plan": "{}",
        "step": 1,
        "world_size": 1,
        "files": [{
            "role": "weights",
            "path": "../outside.bin",
            "len": 7,
            "sha256": sha.as_str(),
        }],
    });
    std::fs::write(
        dir.join("commit-00000001.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    launcher.set_exit(dir, 0);

    let updated = controller.reconcile().unwrap();
    assert_eq!(updated[0].state, JobState::Failed);
    assert!(
        updated[0]
            .detail
            .as_deref()
            .is_some_and(|detail| detail.contains("relative path")),
        "{:?}",
        updated[0].detail
    );
}

#[test]
fn a_job_is_listed_and_cancel_stops_it_once() {
    let (_dir, workspace) = workspace();
    let launcher = ScriptedLauncher::default();
    let clock = SystemClock;
    let root =
        std::env::temp_dir().join(format!("splinter-controller-{}-cancel", std::process::id()));
    let controller = JobController::new(&workspace, &launcher, root, &clock, "test-controller");
    let Submission::Launched(record) = controller.submit(&spec()).unwrap() else {
        panic!("launched")
    };
    assert_eq!(controller.jobs().unwrap(), vec![record.clone()]);
    assert_eq!(
        launcher.observe(
            &Launch::open(Path::new(record.launch.as_ref().unwrap()))
                .unwrap()
                .unwrap()
        ),
        Observation::Running {
            ranks_alive: vec![42]
        }
    );

    let cancelled = controller.cancel(&record.spec).unwrap();
    assert_eq!(cancelled.state, JobState::Cancelled);
    assert!(cancelled.lease.is_none(), "a terminal job holds no claim");
    assert_eq!(
        launcher.observe(
            &Launch::open(Path::new(record.launch.as_ref().unwrap()))
                .unwrap()
                .unwrap()
        ),
        Observation::Running {
            ranks_alive: vec![]
        },
        "the workers were stopped"
    );
    assert!(workspace.jobs_needing_reconciliation().unwrap().is_empty());

    // Cancelling a finished job returns it as it stands, and a job that
    // was never submitted is refused.
    assert_eq!(controller.cancel(&record.spec).unwrap(), cancelled);
    assert!(matches!(
        controller.cancel(&JobSpec::single(vec!["/bin/sh".into()]).digest()),
        Err(OrchestratorError::NotFound { .. })
    ));
}

#[test]
fn a_live_lease_blocks_another_controller_until_it_expires() {
    let (_dir, workspace) = workspace();
    let launcher = ScriptedLauncher::default();
    let clock = SystemClock;
    let root =
        std::env::temp_dir().join(format!("splinter-controller-{}-lease", std::process::id()));
    let first = JobController::new(&workspace, &launcher, root.clone(), &clock, "controller-a")
        .with_lease(100);
    let second =
        JobController::new(&workspace, &launcher, root, &clock, "controller-b").with_lease(100);

    let Submission::Launched(record) = first.submit_at(&spec(), 1_000).unwrap() else {
        panic!("launched")
    };
    let lease = record.lease.as_ref().expect("a submission claims the job");
    assert_eq!(lease.owner, "controller-a");
    assert_eq!(lease.expires_unix, 1_100);

    // The first controller's claim is live: the second leaves the job alone.
    assert!(second.reconcile_at(1_050).unwrap().is_empty());

    // The claim expired: the second takes the job over.
    let taken = second.reconcile_at(1_101).unwrap();
    assert_eq!(taken.len(), 1);
    let lease = taken[0].lease.as_ref().expect("the takeover claims it");
    assert_eq!(lease.owner, "controller-b");
    assert_eq!(lease.expires_unix, 1_201);

    // The first controller must not touch it now.
    assert!(first.reconcile_at(1_150).unwrap().is_empty());
}

#[test]
fn a_job_whose_workers_are_gone_without_an_exit_code_is_lost_not_completed() {
    let (_dir, workspace) = workspace();
    let launcher = ScriptedLauncher::default();
    let clock = SystemClock;
    let root =
        std::env::temp_dir().join(format!("splinter-controller-{}-lost", std::process::id()));
    let controller = JobController::new(&workspace, &launcher, root, &clock, "test-controller");
    let Submission::Launched(record) = controller.submit(&spec()).unwrap() else {
        panic!("launched")
    };
    // The controller restarted: it has no children, so the exit is unknown.
    std::fs::remove_file(Path::new(record.launch.as_ref().unwrap()).join("launch.json")).unwrap();
    let updated = controller.reconcile().unwrap();
    assert_eq!(updated[0].state, JobState::Lost);
    assert_ne!(updated[0].state, JobState::Completed);
}
