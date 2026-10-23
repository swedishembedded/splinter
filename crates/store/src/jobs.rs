// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements durable job records for learning campaigns,
// for its clients. If your team needs expertise in scheduler reconciliation or
// idempotent submission, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Job records: what was submitted to run elsewhere, and the last state a
//! controller observed for it.
//!
//! A submission is idempotent by the job's own identity: the specification is
//! a content-addressed document, and the pointer `job/<spec digest>` names the
//! current record. Submitting the same specification twice returns the record
//! that exists rather than creating a second attempt; two controllers racing to
//! submit it are separated by the pointer's compare-and-set, so exactly one of
//! them starts the workers.
//!
//! A record is immutable and the pointer moves to a new one, so the history of
//! observations stays on record. An update names the record it replaces: if
//! another controller moved the pointer first, the update is refused instead of
//! overwriting a state it never saw.

use serde::{Deserialize, Serialize};

use crate::error::StoreError;
use crate::workspace::Workspace;
use splinter_core::digest::{digest_of, Digest};
use splinter_core::job::{JobSpec, JobState};

const JOB_SPEC: &str = "job_spec";
const JOB_RECORD: &str = "job_record";

/// A controller's claim on a job, until `expires_unix`: while it is valid, no
/// other controller reconciles or retries the job. An expired lease is how a
/// controller that died stops owning its work.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JobLease {
    /// The controller that claimed the job.
    pub owner: String,
    /// When the claim stops being valid, in Unix seconds (inclusive).
    pub expires_unix: u64,
}

impl JobLease {
    /// Whether the claim still holds at `now_unix`.
    #[must_use]
    pub fn is_valid(&self, now_unix: u64) -> bool {
        now_unix <= self.expires_unix
    }
}

/// What a controller knows of one job.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JobRecord {
    /// The specification the job runs, by content address.
    pub spec: Digest,
    /// The attempt this record is about, counting from one.
    pub attempt: u32,
    /// Where the job stands.
    pub state: JobState,
    /// The launcher that carried it out ("local", a scheduler's kind).
    pub launcher: String,
    /// The directory the launch wrote, when one was started.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub launch: Option<String>,
    /// The process exit code, when the launcher saw it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    /// What happened, when the controller can say.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    /// When this record was written.
    pub updated_at: String,
    /// The controller's claim on the job, when one holds it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lease: Option<JobLease>,
}

impl JobRecord {
    /// The address this record is stored under and the pointer is moved to.
    pub fn address(&self) -> Digest {
        digest_of(self)
    }

    /// Whether the job may still change, so a restarted controller must
    /// reconcile it against its launcher.
    #[must_use]
    pub fn needs_reconciliation(&self) -> bool {
        self.state.needs_reconciliation()
    }
}

fn stored_address(text: &str) -> Result<Digest, StoreError> {
    Digest::parse(text).map_err(|e| StoreError::UndecodableObject {
        what: "job record".to_owned(),
        reason: e.to_string(),
    })
}

/// The pointer's name. A signal name cannot carry the colon of a digest's
/// algorithm prefix, so the algorithm and the hex are joined with a dash.
fn pointer_name(spec: &Digest) -> String {
    let text = spec.as_str();
    let (algorithm, hex) = text.split_once(':').unwrap_or(("raw", text));
    format!("job/{algorithm}-{hex}")
}

impl Workspace {
    /// Keep `spec` as a document; the same specification is stored once.
    pub fn put_job_spec(&self, spec: &JobSpec) -> Result<Digest, StoreError> {
        self.put_document(JOB_SPEC, spec)
    }

    /// The specification a record names, checked against its address.
    pub fn job_spec(&self, spec: &Digest) -> Result<Option<JobSpec>, StoreError> {
        self.get_document(JOB_SPEC, spec)
    }

    /// The current record of the job `spec` names, when it was submitted.
    pub fn job(&self, spec: &Digest) -> Result<Option<JobRecord>, StoreError> {
        let Some((_, address)) = self.pointer(&pointer_name(spec))? else {
            return Ok(None);
        };
        let address = stored_address(&address)?;
        self.get_document(JOB_RECORD, &address)
    }

    /// Submit `spec` once. Returns `None` when the job was already submitted,
    /// so a retried submission never starts a second attempt.
    ///
    /// The record is written before the caller launches any worker, and the
    /// pointer claims the job with a compare-and-set from "never set": of two
    /// controllers submitting the same specification, one gets `Some` and the
    /// other `None`.
    pub fn submit_job(
        &self,
        spec: &JobSpec,
        launcher: &str,
        at: &str,
        lease: Option<JobLease>,
    ) -> Result<Option<JobRecord>, StoreError> {
        let id = self.put_job_spec(spec)?;
        if self.job(&id)?.is_some() {
            return Ok(None);
        }
        let record = JobRecord {
            spec: id.clone(),
            attempt: 1,
            state: JobState::Pending,
            launcher: launcher.to_owned(),
            launch: None,
            exit_code: None,
            detail: None,
            updated_at: at.to_owned(),
            lease,
        };
        let address = self.put_document(JOB_RECORD, &record)?;
        match self.move_pointer(&pointer_name(&id), None, address.as_str(), at) {
            Ok(_) => Ok(Some(record)),
            // Another controller submitted it first; its record is the job.
            Err(StoreError::PointerConflict { .. }) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Replace `previous` with `next`, which must be the same specification and
    /// either the same attempt or the one after it (a retry). Refused when
    /// another controller observed the job first.
    pub fn update_job(
        &self,
        previous: &JobRecord,
        next: &JobRecord,
    ) -> Result<JobRecord, StoreError> {
        let continues =
            next.attempt == previous.attempt || next.attempt == previous.attempt.saturating_add(1);
        if previous.spec != next.spec || !continues {
            return Err(StoreError::Rejected {
                what: "job record",
                reason:
                    "an update must continue the same specification at the same or the next attempt"
                        .into(),
            });
        }
        let address = self.put_document(JOB_RECORD, next)?;
        self.move_pointer(
            &pointer_name(&next.spec),
            Some(previous.address().as_str()),
            address.as_str(),
            &next.updated_at,
        )?;
        Ok(next.clone())
    }

    /// Every job whose current record still needs reconciling, in specification
    /// order.
    pub fn jobs_needing_reconciliation(&self) -> Result<Vec<JobRecord>, StoreError> {
        let mut open = Vec::new();
        for (_, address) in self.pointers("job/")? {
            let address = stored_address(&address)?;
            let Some(record) = self.get_document::<JobRecord>(JOB_RECORD, &address)? else {
                continue;
            };
            if record.needs_reconciliation() {
                open.push(record);
            }
        }
        Ok(open)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StateRoot;

    fn workspace(name: &str) -> Workspace {
        let dir = std::env::temp_dir().join(format!("splinter-jobs-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        Workspace::at(&StateRoot::new(&dir))
    }

    fn spec() -> JobSpec {
        JobSpec::single(vec!["brain".into(), "train".into()])
    }

    #[test]
    fn submitting_the_same_specification_twice_keeps_one_job() {
        let workspace = workspace("once");
        let job = spec();
        let first = workspace
            .submit_job(&job, "local", "2026-01-01T00:00:00Z", None)
            .unwrap()
            .expect("first submission");
        assert_eq!(first.state, JobState::Pending);
        assert!(
            workspace
                .submit_job(&job, "local", "2026-01-01T00:00:01Z", None)
                .unwrap()
                .is_none(),
            "a retried submission must not create a second job"
        );
        assert_eq!(workspace.job(&job.digest()).unwrap().unwrap(), first);
    }

    #[test]
    fn an_update_must_name_the_record_it_replaces() {
        let workspace = workspace("cas");
        let job = spec();
        let first = workspace
            .submit_job(&job, "local", "2026-01-01T00:00:00Z", None)
            .unwrap()
            .unwrap();
        let mut running = first.clone();
        running.state = JobState::Running;
        running.updated_at = "2026-01-01T00:00:02Z".into();
        workspace.update_job(&first, &running).unwrap();
        // A controller that still holds the old record must not overwrite the new state.
        let mut stale = first.clone();
        stale.state = JobState::Admitted;
        stale.updated_at = "2026-01-01T00:00:03Z".into();
        assert!(matches!(
            workspace.update_job(&first, &stale),
            Err(StoreError::PointerConflict { .. })
        ));
        assert_eq!(
            workspace.job(&job.digest()).unwrap().unwrap().state,
            JobState::Running
        );
    }

    #[test]
    fn a_terminal_job_is_not_reconciled() {
        let workspace = workspace("terminal");
        let job = spec();
        let first = workspace
            .submit_job(&job, "local", "2026-01-01T00:00:00Z", None)
            .unwrap()
            .unwrap();
        assert_eq!(workspace.jobs_needing_reconciliation().unwrap().len(), 1);
        let mut done = first.clone();
        done.state = JobState::Completed;
        done.updated_at = "2026-01-01T00:00:05Z".into();
        workspace.update_job(&first, &done).unwrap();
        assert!(workspace.jobs_needing_reconciliation().unwrap().is_empty());
    }
}
