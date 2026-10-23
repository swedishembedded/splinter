// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements detached job control for self-improving
// systems, for its clients. If your team needs expertise in scheduler
// reconciliation or crash-tolerant run supervision, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The controller's side of a job: submit it once, observe it, and reconcile a
//! controller that restarted against the workers that are still running.
//!
//! The durable record decides, not this process's memory. Submitting the same
//! specification twice starts one job; an attempt is launched into a directory
//! that is written once, so a retried submission cannot start a second set of
//! workers; a state update names the record it replaces, so two controllers
//! cannot both move a job. Reconciliation asks the launcher, then the launch the
//! record names, and states only what it can prove: workers that are gone with
//! an exit code the controller saw are completed or failed, and everything else
//! that cannot be accounted for is `Lost`, never a claim of success.

use std::path::{Path, PathBuf};

use splinter_core::clock::Clock;
use splinter_core::digest::Digest;
use splinter_core::job::{JobSpec, JobState};
use splinter_sandbox::job::{JobLauncher, Launch, Observation};
use splinter_store::jobs::{JobLease, JobRecord};
use splinter_store::workspace::Workspace;

use crate::error::OrchestratorError;

/// How long a controller's claim on a job lasts, unless it says otherwise.
pub const DEFAULT_JOB_LEASE_SECS: u64 = 300;

/// What a submission did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Submission {
    /// The job was submitted now and its first attempt was launched.
    Launched(JobRecord),
    /// The job was already submitted; its current record is returned and
    /// nothing was launched.
    Existing(JobRecord),
}

/// Submits, observes and reconciles jobs of one campaign.
pub struct JobController<'a> {
    workspace: &'a Workspace,
    launcher: &'a dyn JobLauncher,
    root: PathBuf,
    clock: &'a dyn Clock,
    owner: String,
    lease_seconds: u64,
}

impl<'a> JobController<'a> {
    /// A controller writing attempt directories under `root`.
    #[must_use]
    pub fn new(
        workspace: &'a Workspace,
        launcher: &'a dyn JobLauncher,
        root: PathBuf,
        clock: &'a dyn Clock,
        owner: impl Into<String>,
    ) -> JobController<'a> {
        JobController {
            workspace,
            launcher,
            root,
            clock,
            owner: owner.into(),
            lease_seconds: DEFAULT_JOB_LEASE_SECS,
        }
    }

    /// How long this controller's claim on a job lasts.
    #[must_use]
    pub fn with_lease(mut self, seconds: u64) -> JobController<'a> {
        self.lease_seconds = seconds.max(1);
        self
    }

    /// Submit `spec`, launching its first attempt unless the job already
    /// exists.
    pub fn submit(&self, spec: &JobSpec) -> Result<Submission, OrchestratorError> {
        self.submit_at(spec, splinter_core::clock::unix_now())
    }

    /// [`Self::submit`], with the claim's expiry measured from `now_unix`.
    pub fn submit_at(
        &self,
        spec: &JobSpec,
        now_unix: u64,
    ) -> Result<Submission, OrchestratorError> {
        spec.validate()
            .map_err(|error| OrchestratorError::Refused(format!("job specification: {error}")))?;
        let Some(record) = self.workspace.submit_job(
            spec,
            self.launcher.name(),
            &self.clock.utc_now(),
            Some(self.lease_at(now_unix)),
        )?
        else {
            let existing = self.workspace.job(&spec.digest())?.ok_or_else(|| {
                OrchestratorError::Refused(format!(
                    "job {} is submitted but has no record",
                    spec.digest().as_str()
                ))
            })?;
            return Ok(Submission::Existing(existing));
        };
        Ok(Submission::Launched(
            self.launch_attempt(spec, record, 1, now_unix)?,
        ))
    }

    /// Every submitted job's current record, in specification order.
    pub fn jobs(&self) -> Result<Vec<JobRecord>, OrchestratorError> {
        Ok(self.workspace.jobs()?)
    }

    /// Stop the job `spec` names and record it `Cancelled`.
    ///
    /// Workers still running are stopped through the launcher. A job that
    /// already reached a terminal state is returned as it stands; a
    /// specification that names no job is refused. The record is replaced with
    /// one that names the record it replaces, so a concurrent reconciliation
    /// cannot be silently overwritten.
    pub fn cancel(&self, spec: &Digest) -> Result<JobRecord, OrchestratorError> {
        let record = self
            .workspace
            .job(spec)?
            .ok_or_else(|| OrchestratorError::NotFound {
                what: "job",
                id: spec.as_str().to_owned(),
            })?;
        if record.state.is_terminal() {
            return Ok(record);
        }
        if let Some(dir) = &record.launch {
            let stopped = Launch::open(Path::new(dir))
                .map_err(|error| {
                    OrchestratorError::Refused(format!("job {}: {error}", spec.as_str()))
                })?
                .map(|launch| self.launcher.stop(&launch, None))
                .transpose();
            stopped.map_err(|error| {
                OrchestratorError::Refused(format!("job {}: {error}", spec.as_str()))
            })?;
        }
        let next = JobRecord {
            state: JobState::Cancelled,
            detail: Some("cancelled on request".to_string()),
            updated_at: self.clock.utc_now(),
            lease: None,
            ..record.clone()
        };
        Ok(self.workspace.update_job(&record, &next)?)
    }

    /// Bring every unfinished job up to date with its launcher. Records another
    /// controller moved first are left alone.
    pub fn reconcile(&self) -> Result<Vec<JobRecord>, OrchestratorError> {
        self.reconcile_at(splinter_core::clock::unix_now())
    }

    /// [`Self::reconcile`], judging claims as of `now_unix`. A job another
    /// controller holds under a claim that has not expired is left alone; one
    /// whose claim expired is taken over, and the takeover is fenced by the
    /// record compare-and-set.
    pub fn reconcile_at(&self, now_unix: u64) -> Result<Vec<JobRecord>, OrchestratorError> {
        let mut updated = Vec::new();
        for record in self.workspace.jobs_needing_reconciliation()? {
            if !self.owns(&record, now_unix) {
                continue;
            }
            let Some(next) = self.observe(&record, now_unix)? else {
                continue;
            };
            if let Ok(record) = self.workspace.update_job(&record, &next) {
                updated.push(record);
            }
        }
        Ok(updated)
    }

    /// Launch the next attempt of every job whose last attempt failed and whose
    /// policy allows another.
    pub fn retry_ready(&self) -> Result<Vec<JobRecord>, OrchestratorError> {
        self.retry_ready_at(splinter_core::clock::unix_now())
    }

    /// [`Self::retry_ready`], taking jobs whose claim expired at `now_unix`.
    pub fn retry_ready_at(&self, now_unix: u64) -> Result<Vec<JobRecord>, OrchestratorError> {
        let mut retried = Vec::new();
        for record in self.workspace.jobs_needing_reconciliation()? {
            if record.state != JobState::Retrying || !self.owns(&record, now_unix) {
                continue;
            }
            let Some(spec) = self.workspace.job_spec(&record.spec)? else {
                continue;
            };
            if record.attempt >= spec.retry.attempts {
                continue;
            }
            let attempt = record.attempt + 1;
            let next = self.launch_attempt(&spec, record, attempt, now_unix)?;
            retried.push(next);
        }
        Ok(retried)
    }

    /// The record after observing `record` once, or `None` when another
    /// controller's update already accounts for it.
    fn observe(
        &self,
        record: &JobRecord,
        now_unix: u64,
    ) -> Result<Option<JobRecord>, OrchestratorError> {
        let state = match &record.launch {
            None => (
                JobState::Lost,
                Some("no launch was started for this job".to_string()),
                record.exit_code,
            ),
            Some(dir) => match Launch::open(Path::new(dir)).map_err(|error| {
                OrchestratorError::Refused(format!("job {}: {error}", record.spec.as_str()))
            })? {
                None => (
                    JobState::Lost,
                    Some(format!("{dir} holds no launch record")),
                    record.exit_code,
                ),
                Some(launch) => match self.launcher.observe(&launch) {
                    Observation::Running { ranks_alive } => (
                        JobState::Running,
                        Some(format!(
                            "{} of {} ranks alive",
                            ranks_alive.len(),
                            launch.world_size
                        )),
                        record.exit_code,
                    ),
                    Observation::Exited { exit_code: Some(0) } => (
                        JobState::Completed,
                        Some("the workers exited successfully".to_string()),
                        Some(0),
                    ),
                    Observation::Exited {
                        exit_code: Some(code),
                    } => match self.failed_state(record)? {
                        JobState::Retrying => (
                            JobState::Retrying,
                            Some(format!(
                                "attempt {} exited {code}; a retry is ready",
                                record.attempt
                            )),
                            Some(code),
                        ),
                        state => (
                            state,
                            Some(format!("attempt {} exited {code}", record.attempt)),
                            Some(code),
                        ),
                    },
                    Observation::Exited { exit_code: None } => (
                        JobState::Lost,
                        Some(
                            "the workers are gone and the observing process did not see their exit"
                                .to_string(),
                        ),
                        record.exit_code,
                    ),
                },
            },
        };
        Ok(Some(JobRecord {
            state: state.0,
            detail: state.1,
            exit_code: state.2,
            updated_at: self.clock.utc_now(),
            lease: Some(self.lease_at(now_unix)),
            ..record.clone()
        }))
    }

    /// Whether this controller may act on `record` at `now_unix`: it holds the
    /// claim, no claim was made, or the claim expired.
    fn owns(&self, record: &JobRecord, now_unix: u64) -> bool {
        match &record.lease {
            None => true,
            Some(lease) => lease.owner == self.owner || !lease.is_valid(now_unix),
        }
    }

    /// This controller's claim, valid from `now_unix`.
    fn lease_at(&self, now_unix: u64) -> JobLease {
        JobLease {
            owner: self.owner.clone(),
            expires_unix: now_unix.saturating_add(self.lease_seconds),
        }
    }

    /// What a failed attempt becomes: `Retrying` while the policy allows another
    /// attempt, `Failed` otherwise.
    fn failed_state(&self, record: &JobRecord) -> Result<JobState, OrchestratorError> {
        let retries_left = self
            .workspace
            .job_spec(&record.spec)?
            .is_some_and(|spec| record.attempt < spec.retry.attempts);
        Ok(if retries_left {
            JobState::Retrying
        } else {
            JobState::Failed
        })
    }

    /// Launch attempt `attempt` of `spec` and record it.
    fn launch_attempt(
        &self,
        spec: &JobSpec,
        record: JobRecord,
        attempt: u32,
        now_unix: u64,
    ) -> Result<JobRecord, OrchestratorError> {
        let dir = self.attempt_dir(&record.spec, attempt);
        let launch = self.launcher.launch(spec, attempt, &dir).map_err(|error| {
            let next = JobRecord {
                state: JobState::Failed,
                detail: Some(error.to_string()),
                updated_at: self.clock.utc_now(),
                lease: Some(self.lease_at(now_unix)),
                ..record.clone()
            };
            let _ = self.workspace.update_job(&record, &next);
            OrchestratorError::Refused(format!(
                "job {} attempt {attempt}: {error}",
                record.spec.as_str()
            ))
        })?;
        let next = JobRecord {
            attempt,
            state: JobState::Running,
            launch: Some(dir.to_string_lossy().into_owned()),
            detail: Some(format!("{} ranks launched", launch.world_size)),
            exit_code: None,
            updated_at: self.clock.utc_now(),
            lease: Some(self.lease_at(now_unix)),
            ..record.clone()
        };
        Ok(self.workspace.update_job(&record, &next)?)
    }

    fn attempt_dir(&self, spec: &Digest, attempt: u32) -> PathBuf {
        let hex = spec.as_str().rsplit(':').next().unwrap_or(spec.as_str());
        self.root.join(hex).join(format!("attempt-{attempt}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use splinter_core::clock::SystemClock;
    use splinter_sandbox::job::JobError;
    use splinter_store::StateRoot;
    use std::collections::HashMap;
    use std::sync::Mutex;

    /// A launcher whose workers are whatever the test says they are.
    #[derive(Default)]
    struct ScriptedLauncher {
        running: Mutex<HashMap<PathBuf, Vec<u32>>>,
        exits: Mutex<HashMap<PathBuf, i32>>,
        launches: Mutex<Vec<PathBuf>>,
    }

    impl ScriptedLauncher {
        fn set_exit(&self, dir: &Path, code: i32) {
            self.running.lock().unwrap().remove(dir);
            self.exits.lock().unwrap().insert(dir.to_path_buf(), code);
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
            if let Some(code) = self.exits.lock().unwrap().get(&launch.dir) {
                return Observation::Exited {
                    exit_code: Some(*code),
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
        std::fs::remove_file(Path::new(record.launch.as_ref().unwrap()).join("launch.json"))
            .unwrap();
        let updated = controller.reconcile().unwrap();
        assert_eq!(updated[0].state, JobState::Lost);
        assert_ne!(updated[0].state, JobState::Completed);
    }
}
