// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements detached job submission for learning
// campaigns, for its clients. If your team needs expertise in scheduler-
// independent job control and reconciliation, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Submitting work that runs outside this process, and inspecting it: the
//! [`Jobs`] handle a [`crate::Splinter`] hands out.
//!
//! A submission is idempotent by the job's own identity, so resubmitting after
//! a crash names the same job and starts nothing twice. A job that runs
//! locally is started detached: its workers outlive the process that submitted
//! them, and a later process observes and reconciles it from the durable record
//! and the launch directory. The controller that does the work is
//! `splinter-orchestrator`'s [`JobController`]; this handle only supplies it
//! the configured launcher, the attempt root and the controller's name.

use serde::Serialize;

use splinter_core::clock::Clock;
use splinter_core::digest::Digest;
use splinter_core::job::JobSpec;
use splinter_orchestrator::jobs::{JobController, Submission};
use splinter_sandbox::job::LocalLauncher;
use splinter_store::jobs::JobRecord;
use splinter_store::workspace::Workspace;

use crate::Error;

/// What a submission did.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct JobSubmission {
    /// Whether this call started the job; `false` when the job was already
    /// submitted, in which case nothing was launched.
    pub launched: bool,
    /// The job's record after the call.
    pub job: JobRecord,
}

/// A job together with the specification it runs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct JobView {
    /// The job's current record.
    pub job: JobRecord,
    /// The specification the record names, when its document is stored.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spec: Option<JobSpec>,
}

/// The jobs of the state root a [`crate::Splinter`] works in.
///
/// Obtained from [`crate::Splinter::jobs`]; it borrows the handle's runtime and
/// starts each job's workers with the configured launcher.
pub struct Jobs<'a> {
    workspace: &'a Workspace,
    clock: &'a dyn Clock,
    launcher: LocalLauncher,
    root: std::path::PathBuf,
    owner: String,
    lease_seconds: u64,
}

impl<'a> Jobs<'a> {
    /// A handle over the runtime's state, launcher settings and controller
    /// name.
    pub(crate) fn new(
        workspace: &'a Workspace,
        clock: &'a dyn Clock,
        launcher: LocalLauncher,
        root: std::path::PathBuf,
        owner: String,
    ) -> Jobs<'a> {
        Jobs {
            workspace,
            clock,
            launcher,
            root,
            owner,
            lease_seconds: splinter_orchestrator::jobs::DEFAULT_JOB_LEASE_SECS,
        }
    }

    /// Submit `spec`, launching its first attempt unless the job already
    /// exists.
    pub fn submit(&self, spec: &JobSpec) -> Result<JobSubmission, Error> {
        self.submit_at(spec, splinter_core::clock::unix_now())
    }

    /// [`Self::submit`], judging the claim's expiry from `now_unix`.
    pub fn submit_at(&self, spec: &JobSpec, now_unix: u64) -> Result<JobSubmission, Error> {
        let (launched, job) = match self.controller().submit_at(spec, now_unix)? {
            Submission::Launched(job) => (true, job),
            Submission::Existing(job) => (false, job),
        };
        Ok(JobSubmission { launched, job })
    }

    /// Every submitted job's current record, in specification order.
    pub fn list(&self) -> Result<Vec<JobRecord>, Error> {
        Ok(self.workspace.jobs()?)
    }

    /// Bring every unfinished job up to date with its launcher: a job whose
    /// workers exited is completed or failed, and one that cannot be accounted
    /// for is lost. Returns the records that changed.
    pub fn reconcile(&self) -> Result<Vec<JobRecord>, Error> {
        self.controller().reconcile()
    }

    /// Start the next attempt of every job whose last attempt failed and whose
    /// policy allows another. Returns the records of the attempts launched.
    pub fn retry(&self) -> Result<Vec<JobRecord>, Error> {
        self.controller().retry_ready()
    }

    /// The job `id` names: its whole digest, `b3:<hex>`, or an unambiguous
    /// prefix of the hex.
    pub fn show(&self, id: &str) -> Result<JobView, Error> {
        let job = self.resolve(id)?;
        let spec = self.workspace.job_spec(&job.spec)?;
        Ok(JobView { job, spec })
    }

    /// Stop the job `id` names and record it cancelled. A job that already
    /// reached a terminal state is returned as it stands.
    pub fn cancel(&self, id: &str) -> Result<JobRecord, Error> {
        let job = self.resolve(id)?;
        self.controller().cancel(&job.spec)
    }

    /// The record `id` names, or the refusal that no job or more than one does.
    pub fn resolve(&self, id: &str) -> Result<JobRecord, Error> {
        let mut named: Vec<JobRecord> = self
            .workspace
            .jobs()?
            .into_iter()
            .filter(|job| names(&job.spec, id))
            .collect();
        match named.len() {
            0 => Err(Error::NotFound {
                what: "job",
                id: id.to_owned(),
            }),
            1 => Ok(named.remove(0)),
            matches => Err(Error::AmbiguousId {
                what: "job",
                id: id.to_owned(),
                matches,
            }),
        }
    }

    /// The controller for one call, over this handle's launcher and settings.
    fn controller(&self) -> JobController<'_> {
        JobController::new(
            self.workspace,
            &self.launcher,
            self.root.clone(),
            self.clock,
            &self.owner,
        )
        .with_lease(self.lease_seconds)
    }
}

/// Whether `id` names the job `spec` is: the whole digest, or a prefix of its
/// hex with or without the algorithm.
fn names(spec: &Digest, id: &str) -> bool {
    let text = spec.as_str();
    if text == id {
        return true;
    }
    let hex = text.rsplit(':').next().unwrap_or(text);
    let wanted = id.split_once(':').map_or(id, |(_, rest)| rest);
    !wanted.is_empty() && hex.starts_with(wanted)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest() -> Digest {
        JobSpec::single(vec!["/bin/true".into()]).digest()
    }

    #[test]
    fn an_id_names_a_job_by_its_whole_digest_or_a_hex_prefix() {
        let spec = digest();
        let text = spec.as_str().to_owned();
        let hex = text.rsplit(':').next().unwrap();
        assert!(names(&spec, &text));
        assert!(names(&spec, &text[..8]));
        assert!(names(&spec, &hex[..8]));
        assert!(!names(&spec, "nonsense"));
        assert!(!names(&spec, ""));
    }
}
