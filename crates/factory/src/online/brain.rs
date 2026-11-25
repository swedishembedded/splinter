// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements online reinforcement learning against a
// training service that serves each policy version it trains, for its
// clients. If your team needs expertise in streaming rollouts into a running
// policy-gradient job, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The [`Learner`] of an online run on brain's streaming policy-gradient job
//! ([`StreamJob`]).
//!
//! A group is sent as brain's training records (contract v2): each member's
//! call records ([`CallRecord`]) with the member's reward, the group's key as
//! `group` and `<key>/<index>` as `rollout`. The job's answers and events map
//! onto the seam one to one: accepted or already held, a full queue (or a job
//! not ready yet) with its allowance; applied, published, dropped as stale,
//! committed, and each step's policy tokens.
//!
//! **Committed before a restart.** A run resumed in a new epoch answers the
//! keys its checkpoints hold as duplicates and does not report them again;
//! a duplicate of a key this job did not accept in its own epoch is
//! therefore reported as committed. **A restart** (the job lost, or another
//! epoch) is a [`LearnerError::restarted`]: the controller resumes against
//! the run started again and sends everything not committed.

use std::collections::HashSet;
use std::path::Path;

use splinter_core::digest::Digest;
use splinter_core::factory::group::GroupKey;
use splinter_data::calls::CallRecord;
use splinter_data::{dataset_line, Objective};
use splinter_model::served_train::{ArtifactStore, Offered, StreamError, StreamEvent, StreamJob};

use super::seams::{
    Learner, LearnerError, LearnerEvent, LearnerState, LearnerTelemetry, OutgoingGroup,
    PublishedVersion, SendOutcome, StreamLearner,
};

/// An online run's learner on a streaming job.
pub struct BrainLearner {
    job: StreamJob,
    store: ArtifactStore,
    /// Keys this job accepted in its own epoch.
    accepted: HashSet<String>,
    /// Events found in answers, reported by the next poll.
    pending: Vec<LearnerEvent>,
    telemetry: LearnerTelemetry,
    /// Sums for the telemetry's means.
    staleness_sum: f64,
    queue_time_sum: f64,
    wait_sum: f64,
    measured_steps: u64,
}

fn error(e: StreamError) -> LearnerError {
    match e {
        StreamError::Restarted(why) => LearnerError::restarted(why),
        other => LearnerError::new(other.to_string()),
    }
}

fn keys(groups: Vec<String>) -> Result<Vec<GroupKey>, LearnerError> {
    groups
        .into_iter()
        .map(|g| GroupKey::new(g).map_err(|e| LearnerError::new(format!("the job named {e}"))))
        .collect()
}

/// `group` as brain's training records, one JSON line each.
///
/// # Errors
/// A member's records cannot be read.
pub fn records_of(group: &OutgoingGroup) -> Result<String, LearnerError> {
    let key = group.key.as_str();
    let mut body = String::new();
    for member in &group.members {
        let rollout = format!("{key}/{}", member.index);
        for line in member.records.lines().filter(|l| !l.trim().is_empty()) {
            let call: CallRecord = serde_json::from_str(line)
                .map_err(|e| LearnerError::new(format!("{rollout}: a call record: {e}")))?;
            let record = call.record(member.reward, key, &rollout);
            let line = dataset_line(Objective::PolicyGradient, &record)
                .map_err(|e| LearnerError::new(format!("{rollout}: {e}")))?;
            body.push_str(&line);
            body.push('\n');
        }
    }
    Ok(body)
}

impl BrainLearner {
    /// The learner of `job`, whose adapters are published to `store`.
    #[must_use]
    pub fn new(job: StreamJob, store: ArtifactStore) -> Self {
        Self {
            job,
            store,
            accepted: HashSet::new(),
            pending: Vec::new(),
            telemetry: LearnerTelemetry::default(),
            staleness_sum: 0.0,
            queue_time_sum: 0.0,
            wait_sum: 0.0,
            measured_steps: 0,
        }
    }

    fn step(&mut self, pg: &serde_json::Value) -> LearnerEvent {
        let tokens = pg["policy_tokens"].as_u64().unwrap_or(0);
        let t = &mut self.telemetry;
        t.steps += 1;
        t.trained_tokens += tokens;
        if let Some(dropped) = pg["dropped_stale"].as_u64() {
            t.dropped_stale = t.dropped_stale.max(dropped);
        }
        if let Some(max) = pg["staleness_max"].as_u64() {
            t.staleness_max = Some(t.staleness_max.unwrap_or(0).max(max));
        }
        if let (Some(mean), Some(queue), Some(wait)) = (
            pg["staleness_mean"].as_f64(),
            pg["time_in_queue_seconds"].as_f64(),
            pg["learner_wait_seconds"].as_f64(),
        ) {
            self.measured_steps += 1;
            self.staleness_sum += mean;
            self.queue_time_sum += queue;
            self.wait_sum += wait;
            #[allow(clippy::cast_precision_loss)]
            let n = self.measured_steps as f64;
            t.staleness_mean = Some(self.staleness_sum / n);
            t.time_in_queue_seconds = Some(self.queue_time_sum / n);
            t.learner_wait_seconds = Some(self.wait_sum);
        }
        LearnerEvent::Step { tokens }
    }
}

impl Learner for BrainLearner {
    fn state(&mut self) -> Result<LearnerState, LearnerError> {
        // The keys the run committed are learned from its duplicates.
        Ok(LearnerState {
            policy_version: self.job.version(),
            committed: Vec::new(),
        })
    }

    fn send(&mut self, group: &OutgoingGroup) -> Result<SendOutcome, LearnerError> {
        let body = records_of(group)?;
        let key = group.key.as_str();
        match self.job.offer(&body).map_err(error)? {
            Offered::Taken {
                accepted,
                duplicates,
                ..
            } => {
                if accepted.iter().any(|k| k == key) {
                    self.accepted.insert(key.to_string());
                    return Ok(SendOutcome::Accepted);
                }
                if !duplicates.iter().any(|k| k == key) {
                    return Err(LearnerError::new(format!(
                        "the job neither accepted nor held {key}"
                    )));
                }
                if !self.accepted.contains(key) {
                    // Held from before this job's epoch: in a checkpoint.
                    self.pending.push(LearnerEvent::Committed {
                        groups: vec![group.key.clone()],
                    });
                }
                Ok(SendOutcome::Duplicate)
            }
            Offered::Full { allowance, .. } => Ok(SendOutcome::Full {
                allowance_tokens: allowance,
            }),
            Offered::NotReady { .. } => Ok(SendOutcome::Full {
                allowance_tokens: 0,
            }),
        }
    }

    fn poll(&mut self) -> Result<Vec<LearnerEvent>, LearnerError> {
        let mut events = std::mem::take(&mut self.pending);
        for event in self.job.events().map_err(error)? {
            match event {
                StreamEvent::Applied { groups, .. } => events.push(LearnerEvent::Applied {
                    groups: keys(groups)?,
                }),
                StreamEvent::Published {
                    version,
                    digest,
                    step,
                    lineage,
                } => {
                    self.telemetry.published.push(PublishedVersion {
                        version,
                        digest,
                        step,
                        lineage,
                    });
                    events.push(LearnerEvent::Published { version });
                }
                StreamEvent::Dropped { groups } => events.push(LearnerEvent::Dropped {
                    groups: keys(groups)?,
                }),
                StreamEvent::Committed { step, groups } => {
                    self.telemetry.committed_step = Some(step);
                    events.push(LearnerEvent::Committed {
                        groups: keys(groups)?,
                    });
                }
                StreamEvent::Step { pg } => {
                    let step = self.step(&pg);
                    events.push(step);
                }
                StreamEvent::Done { status, error } => {
                    if status != "succeeded" {
                        return Err(LearnerError::new(format!(
                            "the job ended {status}: {}",
                            error.unwrap_or_default()
                        )));
                    }
                }
            }
        }
        Ok(events)
    }

    fn finish(&mut self) -> Result<(), LearnerError> {
        self.job.finish().map_err(error)
    }
}

impl StreamLearner for BrainLearner {
    fn telemetry(&self) -> LearnerTelemetry {
        LearnerTelemetry {
            tokenizer_digest: self.job.tokenizer_digest().map(str::to_owned),
            ..self.telemetry.clone()
        }
    }

    fn final_adapter(&mut self, to: &Path) -> Result<(), LearnerError> {
        let outputs = self.job.result().map_err(error)?;
        let hex = outputs["adapter_digest"].as_str().ok_or_else(|| {
            LearnerError::new(format!("the ended run names no adapter: {outputs}"))
        })?;
        let digest = Digest::from_content_hex(hex)
            .map_err(|e| LearnerError::new(format!("adapter digest {hex}: {e}")))?;
        self.store
            .fetch(&digest, to)
            .map_err(|e| LearnerError::new(e.to_string()))
    }

    fn served_as(&self) -> String {
        self.job.publish_adapter().to_string()
    }
}
