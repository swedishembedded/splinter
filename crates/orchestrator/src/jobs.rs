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
use splinter_sandbox::job::{JobLauncher, JobOutcome, Launch, Observation};
use splinter_store::artifacts::ArtifactStore;
use splinter_store::job_outputs::register_job_outputs;
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
                    Observation::Finished { outcome, exit_code } => {
                        self.finished(record, &launch, outcome, exit_code)?
                    }
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

    /// What a launcher's finished outcome becomes.
    ///
    /// Only a completed outcome may complete the job. A failed attempt retries
    /// while the policy allows it, a preempted one only when the policy asks
    /// for that, a cancelled one is terminal however many attempts remain, and
    /// an outcome the launcher could not determine leaves the job lost rather
    /// than a claim of success.
    fn finished(
        &self,
        record: &JobRecord,
        launch: &Launch,
        outcome: JobOutcome,
        exit_code: Option<i32>,
    ) -> Result<(JobState, Option<String>, Option<i32>), OrchestratorError> {
        Ok(match outcome {
            JobOutcome::Completed => self.register_outputs(record, launch, exit_code),
            JobOutcome::Failed => match self.failed_state(record)? {
                JobState::Retrying => (
                    JobState::Retrying,
                    Some(format!(
                        "attempt {} failed; a retry is ready",
                        record.attempt
                    )),
                    exit_code,
                ),
                state => (
                    state,
                    Some(match exit_code {
                        Some(code) => format!("attempt {} exited {code}", record.attempt),
                        None => format!("attempt {} failed", record.attempt),
                    }),
                    exit_code,
                ),
            },
            JobOutcome::Cancelled => (
                JobState::Cancelled,
                Some(format!("attempt {} was cancelled", record.attempt)),
                exit_code,
            ),
            JobOutcome::Preempted => match self.preempted_state(record)? {
                JobState::Retrying => (
                    JobState::Retrying,
                    Some(format!(
                        "attempt {} was preempted; a retry is ready",
                        record.attempt
                    )),
                    exit_code,
                ),
                state => (
                    state,
                    Some(format!("attempt {} was preempted", record.attempt)),
                    exit_code,
                ),
            },
            JobOutcome::Lost => (
                JobState::Lost,
                Some(
                    "the workers are gone and no exit or scheduler state says how they ended"
                        .to_string(),
                ),
                exit_code.or(record.exit_code),
            ),
        })
    }

    /// What a clean finish becomes: `Completed` once the attempt's outputs, if
    /// it wrote any, are registered as artifacts and lineage.
    ///
    /// No commit manifest means the job produced no durable output, which is
    /// not a failure. A manifest that is malformed, names a missing or altered
    /// file, or names a path outside the attempt directory is a failure of the
    /// job, never a claim that it completed.
    fn register_outputs(
        &self,
        record: &JobRecord,
        launch: &Launch,
        exit_code: Option<i32>,
    ) -> (JobState, Option<String>, Option<i32>) {
        let artifacts = ArtifactStore::new(self.workspace, self.workspace.state_root());
        match register_job_outputs(
            self.workspace,
            &artifacts,
            &launch.dir,
            &record.spec,
            record.attempt,
        ) {
            Ok(Some(output)) => (
                JobState::Completed,
                Some(format!(
                    "the workers exited successfully; {} artifacts registered for step {}",
                    output.files.len(),
                    output.step
                )),
                exit_code,
            ),
            Ok(None) => (
                JobState::Completed,
                Some("the workers exited successfully".to_string()),
                exit_code,
            ),
            Err(error) => (
                JobState::Failed,
                Some(format!(
                    "the job's outputs could not be registered: {error}"
                )),
                exit_code,
            ),
        }
    }

    /// What a preempted attempt becomes: `Retrying` when the policy retries
    /// preemption and attempts remain, `Failed` otherwise.
    fn preempted_state(&self, record: &JobRecord) -> Result<JobState, OrchestratorError> {
        let retry = self.workspace.job_spec(&record.spec)?.is_some_and(|spec| {
            spec.retry.retry_on_preemption && record.attempt < spec.retry.attempts
        });
        Ok(if retry {
            JobState::Retrying
        } else {
            JobState::Failed
        })
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
        self.root
            .join(spec.hex())
            .join(format!("attempt-{attempt}"))
    }
}
