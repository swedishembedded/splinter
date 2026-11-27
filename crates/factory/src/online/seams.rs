// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements online reinforcement-learning controllers
// that never train a group twice or lose one they produced, for its clients.
// If your team needs expertise in asynchronous policy optimisation, you can
// procure our services by sending an email to info@swedishembedded.com.

//! What an online run is carried out with: the rollout executor, the
//! learner, the task selector and the durable ledger, each a seam the
//! controller drives and a test replaces.

use splinter_core::factory::group::{GroupKey, RolloutOutcome};
use splinter_store::factory::runstate::GroupEntry;

use super::control::Change;

/// One rollout to run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RolloutRequest {
    /// The group it belongs to.
    pub group: GroupKey,
    /// The task it attempts.
    pub task: String,
    /// Its index in the group.
    pub index: u32,
    /// The id of the one policy version every request of the rollout asks
    /// (a run that pins its episodes); `None` asks the policy the run
    /// serves, whichever version that is when each request is sent.
    pub policy: Option<String>,
}

/// A finished rollout: how it ended and what it recorded.
#[derive(Clone, Debug, PartialEq)]
pub struct FinishedRollout {
    /// How it ended.
    pub outcome: RolloutOutcome,
    /// Its training records, one JSON line per model request, without
    /// `reward` or `group` (the learner adapter writes those from the
    /// scored group).
    pub records: String,
}

/// A rollout that ended, one way or the other.
#[derive(Clone, Debug, PartialEq)]
pub struct RolloutEnd {
    /// The group.
    pub group: GroupKey,
    /// The rollout's index in it.
    pub index: u32,
    /// Its outcome, or why its infrastructure failed (a worker lost, a
    /// verifier that could not run): such a rollout is retried or
    /// abandoned, never scored.
    pub result: Result<FinishedRollout, String>,
}

/// The executor itself failed (not one rollout: that is a [`RolloutEnd`]).
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("rollout executor: {0}")]
pub struct ExecutorError(pub String);

/// What runs rollouts, many at once, each independently of the others.
pub trait RolloutExecutor {
    /// Start one rollout; it ends later, through [`Self::wait`].
    ///
    /// # Errors
    /// The executor cannot start rollouts at all.
    fn start(&mut self, request: RolloutRequest) -> Result<(), ExecutorError>;

    /// Block until some started rollout ends or the run clock reaches
    /// `until_ms`, and return every rollout that ended (none at the
    /// deadline). With nothing started, waits for the deadline.
    ///
    /// # Errors
    /// The executor failed.
    fn wait(&mut self, until_ms: u64) -> Result<Vec<RolloutEnd>, ExecutorError>;

    /// Stop every started rollout: the run is cancelled. Their rollouts stay
    /// running in the ledger and are started again on resume.
    fn cancel(&mut self);
}

/// One member of a group the learner is sent.
#[derive(Clone, Debug, PartialEq)]
pub struct Member {
    /// The rollout's index in the group.
    pub index: u32,
    /// Its verified reward.
    pub reward: f64,
    /// Its training records ([`FinishedRollout::records`]).
    pub records: String,
}

/// A complete, qualifying group, as the learner receives it: only its
/// scored, exact rollouts; its advantages are the learner's to compute.
#[derive(Clone, Debug, PartialEq)]
pub struct OutgoingGroup {
    /// Its key, which the learner deduplicates by.
    pub key: GroupKey,
    /// The task.
    pub task: String,
    /// Its members, by index.
    pub members: Vec<Member>,
}

/// How the learner answered a group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SendOutcome {
    /// Queued for training.
    Accepted,
    /// Already held or trained: nothing new.
    Duplicate,
    /// Its queue is full; it holds `allowance_tokens` at most. The same
    /// group is sent again later.
    Full {
        /// The learner's queue capacity in tokens.
        allowance_tokens: u64,
    },
}

/// What the learner reports as it trains.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LearnerEvent {
    /// An optimiser step trained `tokens` policy tokens.
    Step {
        /// The tokens it trained.
        tokens: u64,
    },
    /// The policy `version` is published: new requests are decoded by it.
    Published {
        /// The version.
        version: u64,
    },
    /// These groups' updates are applied, not yet in a durable checkpoint.
    Applied {
        /// The groups.
        groups: Vec<GroupKey>,
    },
    /// These groups were too far behind the learner's policy and dropped
    /// unapplied.
    Dropped {
        /// The groups.
        groups: Vec<GroupKey>,
    },
    /// These groups' updates are in a durable checkpoint: they are never
    /// sent again.
    Committed {
        /// The groups.
        groups: Vec<GroupKey>,
    },
}

/// Where the learner stands, read when a controller starts or resumes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LearnerState {
    /// Its published policy version.
    pub policy_version: u64,
    /// Every group in its durable checkpoints.
    pub committed: Vec<GroupKey>,
}

/// The learner failed (not a full queue: that is [`SendOutcome::Full`]).
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("learner: {message}")]
pub struct LearnerError {
    /// What failed.
    pub message: String,
    /// The learner was restarted under the run and lost what it had not
    /// committed: the run is resumed against it started anew, and sends
    /// again everything not committed.
    pub restarted: bool,
}

impl LearnerError {
    /// A failure of the learner.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            restarted: false,
        }
    }

    /// The learner restarted under the run.
    #[must_use]
    pub fn restarted(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            restarted: true,
        }
    }
}

/// What trains on groups as they are produced and publishes the policy.
pub trait Learner {
    /// Its published version and committed groups.
    ///
    /// # Errors
    /// The learner cannot be reached.
    fn state(&mut self) -> Result<LearnerState, LearnerError>;

    /// Hand it one group.
    ///
    /// # Errors
    /// The learner failed or refused the group's records.
    fn send(&mut self, group: &OutgoingGroup) -> Result<SendOutcome, LearnerError>;

    /// What happened since the last poll, in order; never blocks.
    ///
    /// # Errors
    /// The learner failed.
    fn poll(&mut self) -> Result<Vec<LearnerEvent>, LearnerError>;

    /// No more groups will come: train what is queued, even a partial
    /// step, and commit it.
    ///
    /// # Errors
    /// The learner failed.
    fn finish(&mut self) -> Result<(), LearnerError>;

    /// What it measured so far; a learner that measures nothing reports
    /// nothing.
    fn telemetry(&self) -> LearnerTelemetry {
        LearnerTelemetry::default()
    }

    /// The id its newest published version alone is served under, when it
    /// retains versions: what a pinned episode asks from its first request
    /// to its last.
    fn version_now(&self) -> Option<String> {
        None
    }
}

/// Which task each group attempts.
pub trait TaskSelector {
    /// The task of the run's `seq`-th group, or `None` when the run has
    /// asked for all its groups. Called with each seq once, in order (and
    /// again for the same seq on resume), so a selector whose answer
    /// depends only on `seq` resumes without state of its own.
    fn select(&mut self, seq: u64) -> Option<String>;

    /// A group of `task` was scored with these rewards (unscored rollouts
    /// left out): what an adaptive selector learns from.
    fn observe(&mut self, task: &str, rewards: &[f64]) {
        let _ = (task, rewards);
    }
}

/// The default selector: the tasks in turn, `groups` groups in all.
#[derive(Clone, Debug)]
pub struct RoundRobin {
    tasks: Vec<String>,
    groups: u64,
}

impl RoundRobin {
    /// `groups` groups over `tasks` in turn.
    ///
    /// # Errors
    /// No task is given.
    pub fn new(tasks: Vec<String>, groups: u64) -> Result<Self, super::SettingsError> {
        if tasks.is_empty() {
            return Err(super::SettingsError("a round robin needs a task".into()));
        }
        Ok(Self { tasks, groups })
    }
}

impl TaskSelector for RoundRobin {
    fn select(&mut self, seq: u64) -> Option<String> {
        let len = u64::try_from(self.tasks.len()).ok()?;
        let at = usize::try_from(seq % len).ok()?;
        (seq < self.groups).then(|| self.tasks[at].clone())
    }
}

/// The ledger failed.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("group ledger: {0}")]
pub struct LedgerError(pub String);

/// Where every group transition is made durable before it takes effect.
pub trait GroupLedger {
    /// Every group recorded, by seq.
    ///
    /// # Errors
    /// The ledger cannot be read.
    fn load(&mut self) -> Result<Vec<GroupEntry>, LedgerError>;

    /// Make `change` durable at `now_ms`.
    ///
    /// # Errors
    /// The ledger refused the change or failed.
    fn apply(&mut self, now_ms: u64, change: &Change) -> Result<(), LedgerError>;

    /// The records of the group's finished rollouts, by index.
    ///
    /// # Errors
    /// The ledger cannot be read.
    fn records(&mut self, key: &GroupKey) -> Result<Vec<(u32, String)>, LedgerError>;

    /// What the scored groups' rollouts came to per policy version, when
    /// the ledger keeps it.
    fn series(&self) -> Option<&super::health::VersionSeries> {
        None
    }

    /// The learner published `version`: a ledger that records the learner's
    /// version beside what it logs takes it from here.
    fn learner_version(&mut self, version: u64) {
        let _ = version;
    }
}

/// The run's clock in milliseconds: injected, so a test controls time.
pub trait RunClock {
    /// Milliseconds since an epoch fixed for the run.
    fn now_ms(&self) -> u64;
}

/// The wall clock: milliseconds since the Unix epoch.
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemRunClock;

impl RunClock for SystemRunClock {
    fn now_ms(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
    }
}

/// What a learner measured of its own side, for a run's report. A figure it
/// did not report is `None`, never zero.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
pub struct LearnerTelemetry {
    /// Optimiser steps it reported.
    pub steps: u64,
    /// Policy tokens its steps trained.
    pub trained_tokens: u64,
    /// How long it waited for groups, over every step, in seconds.
    pub learner_wait_seconds: Option<f64>,
    /// The mean over steps of each step's mean staleness, in versions.
    pub staleness_mean: Option<f64>,
    /// The largest staleness a step trained, in versions.
    pub staleness_max: Option<u64>,
    /// The mean time a group waited in its queue, in seconds.
    pub time_in_queue_seconds: Option<f64>,
    /// Groups it dropped as too old.
    pub dropped_stale: u64,
    /// Every version it published, in order.
    pub published: Vec<PublishedVersion>,
    /// The step of its last durable checkpoint.
    pub committed_step: Option<u64>,
    /// `sha256:<hex>` of the base's tokenizer, as the learner named it.
    pub tokenizer_digest: Option<String>,
    /// Each step's figures, in order (reported as the run's health).
    #[serde(skip)]
    pub step_health: Vec<super::health::StepHealth>,
    /// Steps whose update the trainer skipped (its guard judged them
    /// unsafe), with why, in order (reported as the run's health).
    #[serde(skip)]
    pub skipped_steps: Vec<String>,
}

/// One version a learner published: the serving-local version, the adapter's
/// digest, the step it holds and, when the server names it, the run-level
/// identity of its weights (which run, epoch and update; base and adapter
/// digests), so what served each version is attributable to exact code.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct PublishedVersion {
    /// The serving-local version.
    pub version: u64,
    /// The adapter's digest.
    pub digest: String,
    /// The step it holds.
    pub step: u64,
    /// Its run-level identity.
    pub lineage: Option<splinter_core::revision::PolicyLineage>,
    /// The id this version alone is served under while the learner retains
    /// it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version_id: Option<String>,
}

impl LearnerTelemetry {
    /// The newest published version that a durable checkpoint holds:
    /// `(version, digest)`.
    #[must_use]
    pub fn committed_version(&self) -> Option<(u64, &str)> {
        let step = self.committed_step?;
        self.published
            .iter()
            .filter(|p| p.step <= step)
            .max_by_key(|p| p.version)
            .map(|p| (p.version, p.digest.as_str()))
    }
}

/// A learner that reports its own side and hands back what it published.
pub trait StreamLearner: Learner {
    /// The adapter the ended run left (its final, committed version),
    /// copied to `to`.
    ///
    /// # Errors
    /// The run has not ended, left no adapter, or it cannot be copied.
    fn final_adapter(&mut self, to: &std::path::Path) -> Result<(), LearnerError>;

    /// The id every published version is served under.
    fn served_as(&self) -> String;
}

/// What a run's rollouts are asked to be.
#[derive(Clone, Debug)]
pub struct RolloutsRequest<'a> {
    /// The pinned task collection, one task folder per task.
    pub tasks: &'a std::path::Path,
    /// The id the learner serves the policy under.
    pub policy: &'a str,
    /// The sampling temperature (the whole distribution, nothing cut).
    pub temperature: f32,
    /// The sampler's top-K asked of every request.
    pub top_logprobs: Option<u32>,
    /// The most tokens a rollout's reply may sample (`None`: the context's room).
    pub max_output_tokens: Option<u64>,
    /// The seed every rollout's draws derive from.
    pub seed: u64,
    /// Where running rollouts keep their scratch files.
    pub scratch: &'a std::path::Path,
}

/// Runs an online run's rollouts: the agent runtime's side of an online
/// training stage.
pub trait OnlineRollouts: Sync {
    /// Its identity, part of every online stage's.
    fn identity(&self) -> String;

    /// Runs `body` with an executor of rollouts of `request`'s tasks and the
    /// tasks' ids as the executor names them; every rollout still running
    /// when `body` returns is stopped.
    ///
    /// # Errors
    /// The rollouts cannot be run (the tasks cannot be read, the policy
    /// cannot be reached, the sampling is refused).
    fn run(
        &self,
        request: &RolloutsRequest<'_>,
        body: &mut dyn FnMut(&mut dyn RolloutExecutor, &[String]),
    ) -> Result<(), String>;
}

/// An online policy-gradient run to start.
#[derive(Clone, Debug)]
pub struct OnlineTrain<'a> {
    /// The base model.
    pub base: &'a crate::recipe::ModelRef,
    /// The run's policy-gradient settings (`estimator`, `max_block`, ...),
    /// as the recipe wrote them.
    pub params: &'a serde_json::Map<String, serde_json::Value>,
    /// The run's durable identity: asked again under it, a restarted run
    /// continues from its last checkpoint.
    pub resume_key: &'a str,
    /// Steps between durable checkpoints.
    pub checkpoint_every: Option<u32>,
    /// The id every published version is served under.
    pub publish_adapter: &'a str,
    /// Steps between two published versions.
    pub publish_every: u32,
    /// Policy tokens a step trains.
    pub step_tokens: u64,
    /// Policy tokens the learner's queue holds at most.
    pub queue_tokens: u64,
    /// Versions a group may lag before the learner drops it.
    pub max_staleness: u64,
    /// Steps after which the run ends.
    pub max_steps: Option<u32>,
    /// The sampler's top-K a score-centred estimator centres over.
    pub top_k: Option<u32>,
    /// How many of the newest versions the run also serves under ids of
    /// their own (episode pinning, the watchdog's evaluations); `None`
    /// serves none.
    pub retain_versions: Option<u32>,
    /// The adapter the run continues, if it continues one.
    pub start: Option<&'a std::path::Path>,
    /// Where the run's job is recorded (the endpoint that placed it and its id), beside the run's state: a job a
    /// stopped controller left running is cancelled before the run is asked for again, since brain starts no other
    /// job of the run while one runs.
    pub job_place: &'a std::path::Path,
}
