// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that turn practice task folders
// into graded, replayable agent experience for its clients. If your team
// needs expertise in on-policy experience collection for agent training you
// can procure our services by sending an email to info@swedishembedded.com.

//! Practice task folders run as episodes: a policy attempts each task many
//! times, a verifier it cannot reach grades every attempt, and each attempt
//! is kept as experience.
//!
//! The pieces are the ones every solve uses. A directory of Harbor-format
//! task folders is imported into the task store ([`load_collection`], through
//! [`crate::harbor`]); a folder that is exam material is refused with
//! [`EpisodeError::Sealed`] and a malformed one with
//! [`EpisodeError::Malformed`], before anything runs. Each attempt is one
//! workspace episode of the solve stage: a fresh copy of the task's snapshot,
//! sven's agent in it under the policy, with its shell confined to the tree,
//! asked for the provider's policy trace (the prompt and sampled token ids
//! and each sampled token's log-probability, kept in the trajectory's step
//! metrics). Once the agent has finished, the task's verifiers grade the
//! patch it left in another fresh copy, with the hidden tests laid over it
//! only then, and the exposure audit fails an attempt that reached for them.
//!
//! An attempt's [`EpisodeRecord`] says how it ended ([`EpisodeOutcome`]):
//! graded pass or fail, exposed, the verifier failing or running out of
//! time, or the infrastructure failing. Only a graded or exposed attempt has
//! a reward; a verifier or infrastructure failure is never a reward of zero.
//! How the agent's own run ended (finished, out of time, out of budget,
//! stopped at its tool-call cap) is the record's `conclusion`.
//!
//! **Resumable.** An attempt is keyed by its content: the task's address and
//! its folder's digest, the policy, the configuration digest, the run's seed
//! and the attempt's index. Its record is written last, after the experience
//! and its verdicts are durable, and a pointer under that key names it. A
//! rerun skips every attempt whose record is there, except one the
//! infrastructure stopped, which is run again; an attempt killed before its
//! record was written is run again from the start (its experience, if one was
//! stored, stays immutable and unreferenced).

mod attempt;
mod collection;
mod executor;
mod grade;

use std::time::Duration;

use serde::{Deserialize, Serialize};
use splinter_agent::CancelToken;
use splinter_core::digest::{canonical_json, Digest};
use splinter_core::experience::ExperienceId;
use splinter_core::model_ref::ModelRef;
use splinter_core::prompt::CodingPrompt;
use splinter_model::Sampling;
use splinter_orchestrator::concurrency::fan_out_blocking;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

pub use collection::{load_collection, PracticeTask};
pub use executor::SvenEpisodes;

use crate::solving::harness_protocol;

/// The format tag of an [`EpisodeRecord`].
pub const RECORD_FORMAT: &str = "splinter.episode.v1";

/// What went wrong running episodes, as opposed to what an agent did.
#[derive(Debug, thiserror::Error)]
pub enum EpisodeError {
    /// A task folder is exam material: practice never reads it.
    #[error("{}: sealed exam material, never practised on: {reason}", .directory.display())]
    Sealed {
        /// The folder.
        directory: std::path::PathBuf,
        /// What seals it.
        reason: String,
    },
    /// A task folder is not a task that can be run.
    #[error("{}: not a runnable task: {reason}", .directory.display())]
    Malformed {
        /// The folder.
        directory: std::path::PathBuf,
        /// What is wrong with it.
        reason: String,
    },
    /// The settings cannot be run.
    #[error("{0}")]
    Settings(String),
    /// The run was cancelled.
    #[error("cancelled")]
    Cancelled,
    /// The store or the context failed.
    #[error(transparent)]
    Orchestrator(#[from] OrchestratorError),
}

impl From<EpisodeError> for OrchestratorError {
    fn from(error: EpisodeError) -> Self {
        match error {
            EpisodeError::Orchestrator(error) => error,
            EpisodeError::Cancelled => Self::Cancelled,
            other => Self::Refused(other.to_string()),
        }
    }
}

/// Writes every record of `report` to `path`, one JSON object per line,
/// atomically: a reader sees the old file or the whole new one.
///
/// # Errors
///
/// The file cannot be written.
pub fn write_records(path: &std::path::Path, report: &EpisodeReport) -> Result<(), EpisodeError> {
    let io =
        |e: std::io::Error| EpisodeError::Orchestrator(splinter_orchestrator::error::io(path)(e));
    let mut text = String::new();
    for record in &report.records {
        let line = serde_json::to_string(record)
            .map_err(|e| EpisodeError::Settings(format!("a record: {e}")))?;
        text.push_str(&line);
        text.push('\n');
    }
    let partial = path.with_extension("partial");
    std::fs::write(&partial, text).map_err(io)?;
    std::fs::rename(&partial, path).map_err(io)
}

/// How episodes are run. The seed has no default: a run is told its seed,
/// and records it.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EpisodeSettings {
    /// Attempts per task.
    pub attempts: u32,
    /// The seed every attempt's draws derive from.
    pub seed: u64,
    /// Attempts run at once.
    pub jobs: usize,
    /// How long one attempt's agent may run.
    #[serde(with = "seconds")]
    pub deadline: Duration,
    /// Tool calls one attempt may request; `None` is sven's own limit on
    /// tool rounds.
    pub max_tool_calls: Option<u32>,
    /// Output tokens one attempt may generate; `None` sets no budget.
    pub max_output_tokens: Option<u64>,
}

mod seconds {
    pub(super) fn serialize<S: serde::Serializer>(
        value: &std::time::Duration,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.serialize_f64(value.as_secs_f64())
    }
}

impl EpisodeSettings {
    /// `attempts` attempts per task drawn under `seed`, one at a time, each
    /// within the solve stage's deadline and the episode bounds every
    /// comparison runs under ([`harness_protocol`]).
    #[must_use]
    pub fn new(attempts: u32, seed: u64) -> Self {
        Self {
            attempts,
            seed,
            jobs: 1,
            deadline: crate::solving::DEFAULT_SOLVE_DEADLINE,
            max_tool_calls: harness_protocol().max_tool_calls,
            max_output_tokens: None,
        }
    }

    /// These settings with `jobs` attempts at once.
    #[must_use]
    pub fn with_jobs(mut self, jobs: usize) -> Self {
        self.jobs = jobs;
        self
    }
}

/// How the policy is asked to sample: temperature one with nothing cut
/// away, so the log-probabilities it reports are those of the distribution
/// it drew from (the behaviour probabilities a policy-gradient step needs).
/// Each attempt adds a seed of its own ([`exact_sampling`]).
pub const EXACT_SAMPLING: Sampling = Sampling {
    temperature: 1.0,
    top_k: 0,
    top_p: Some(1.0),
    seed: None,
    ..splinter_model::local::AGENT_SAMPLING
};

/// [`EXACT_SAMPLING`] drawing from `seed`.
#[must_use]
pub fn exact_sampling(seed: u64) -> Sampling {
    Sampling {
        seed: Some(seed),
        ..EXACT_SAMPLING
    }
}

/// The policy an episode ran under.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyIdentity {
    /// The model's identity, as its experiences name their solver.
    pub model: String,
    /// The adapter it served, by content, when it served one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter: Option<Digest>,
}

/// How an attempt ended once it was graded, or why it was not.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EpisodeOutcome {
    /// The verifier passed the agent's change. Reward 1.
    Passed,
    /// The verifier failed it (or there was no change to grade). Reward 0.
    Failed,
    /// The agent reached for what the verifier keeps (its tests, the state
    /// root): failed whatever its change does. Reward 0.
    Exposed,
    /// The verifier did not grade: it crashed, wrote no reward, or could not
    /// be run. No reward.
    VerifierError {
        /// What went wrong.
        detail: String,
    },
    /// The verifier ran out of time. No reward.
    VerifierTimeout,
    /// The episode could not run (the policy could not be reached, the
    /// task's environment could not be laid out). No reward and no
    /// experience; a rerun runs it again.
    Infrastructure {
        /// What went wrong.
        detail: String,
    },
}

impl EpisodeOutcome {
    /// The reward of an attempt that ended so; `None` when nothing graded
    /// the agent's work.
    #[must_use]
    pub fn reward(&self) -> Option<f64> {
        match self {
            Self::Passed => Some(1.0),
            Self::Failed | Self::Exposed => Some(0.0),
            Self::VerifierError { .. } | Self::VerifierTimeout | Self::Infrastructure { .. } => {
                None
            }
        }
    }
}

/// What an attempt used. A count the provider did not report is absent,
/// never zero.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EpisodeUsage {
    /// Model requests made.
    pub turns: u64,
    /// Tool calls the model requested.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<u32>,
    /// Prompt tokens over the attempt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u64>,
    /// Tokens the policy generated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u64>,
    /// How long the episode took, verification excluded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wall_seconds: Option<f64>,
}

/// The policy trace of an attempt, summarized: the calls are in the
/// experience's trajectory.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceSummary {
    /// Model calls in the trajectory.
    pub calls: usize,
    /// Of those, the ones with prompt ids, sampled ids and a log-probability
    /// for each sampled id.
    pub traced: usize,
    /// Whether every call is traced: the attempt's tokens can be trained on
    /// as sampled, without re-tokenizing.
    pub provable: bool,
}

/// One attempt, as recorded.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EpisodeRecord {
    /// Always [`RECORD_FORMAT`] for a record this build writes.
    pub format: String,
    /// The attempt's content key: what a rerun finds it by.
    pub key: Digest,
    /// The task's content address.
    pub task: Digest,
    /// The task's name.
    pub name: String,
    /// The digest of the task folder's files it was imported from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<Digest>,
    /// The attempt, from zero.
    pub attempt: u32,
    /// The seed of the attempt's draws.
    pub seed: u64,
    /// Whether the seed and the exact sampling ([`exact_sampling`]) were sent
    /// with every request of the attempt, whatever serves the policy. Every
    /// attempt run now is; a record written before requests carried their
    /// sampling says `false`: its policy sampled as its provider did, so its
    /// log-probabilities are of whatever distribution that was.
    pub seeded: bool,
    /// The policy.
    pub policy: PolicyIdentity,
    /// The digest of the configuration the attempt ran under.
    pub config: Digest,
    /// The experience; `None` when the episode could not run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub experience: Option<ExperienceId>,
    /// How the agent's run ended (`success`, `timeout`, `budget_exhausted`,
    /// ...), when it ran.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conclusion: Option<String>,
    /// How the attempt was graded.
    pub outcome: EpisodeOutcome,
    /// Its reward ([`EpisodeOutcome::reward`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reward: Option<f64>,
    /// What it used.
    pub usage: EpisodeUsage,
    /// Its policy trace, summarized.
    pub trace: TraceSummary,
}

/// What a run of episodes did.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EpisodeReport {
    /// The policy.
    pub policy: PolicyIdentity,
    /// The run's seed.
    pub seed: u64,
    /// The configuration digest every record carries.
    pub config: Digest,
    /// Every attempt asked for, by task (in the order given) and attempt.
    pub records: Vec<EpisodeRecord>,
    /// Attempts run now.
    pub ran: usize,
    /// Attempts a finished earlier run had recorded.
    pub reused: usize,
}

/// One run of episodes.
pub struct EpisodeRequest<'a> {
    /// The tasks.
    pub tasks: &'a [PracticeTask],
    /// The policy.
    pub policy: &'a ModelRef,
    /// How.
    pub settings: EpisodeSettings,
    /// Stops the run: no attempt starts after it fires.
    pub cancel: CancelToken,
}

/// What every attempt of a run is configured by, apart from its task, its
/// seed and its index: what its records' `config` digests.
#[derive(Serialize)]
struct Configuration<'a> {
    format: &'static str,
    #[serde(with = "seconds")]
    deadline: Duration,
    max_tool_calls: Option<u32>,
    max_output_tokens: Option<u64>,
    follow_ups: u32,
    max_repeated_results: Option<u32>,
    coding_prompt: &'a str,
    sampling: Sampling,
    policy_trace: bool,
}

/// The digest of the configuration `settings` runs attempts under.
fn configuration(settings: &EpisodeSettings, prompt: CodingPrompt) -> Result<Digest, EpisodeError> {
    let protocol = harness_protocol();
    let config = Configuration {
        format: RECORD_FORMAT,
        deadline: settings.deadline,
        max_tool_calls: settings.max_tool_calls,
        max_output_tokens: settings.max_output_tokens,
        follow_ups: protocol.follow_ups,
        max_repeated_results: protocol.max_repeated_results,
        coding_prompt: prompt.id(),
        // The seed is the attempt's, not the configuration's.
        sampling: EXACT_SAMPLING,
        policy_trace: true,
    };
    let bytes = canonical_json(&config)
        .map_err(|e| EpisodeError::Settings(format!("the configuration: {e}")))?;
    Ok(Digest::of(&bytes))
}

/// Runs `request.settings.attempts` attempts of every task under the
/// policy, skipping the ones an earlier run recorded, and returns every
/// attempt's record.
///
/// # Errors
///
/// [`EpisodeError::Settings`] for settings that cannot run,
/// [`EpisodeError::Cancelled`] when the run was cancelled, and the store's
/// errors. An attempt that could not run is a record, not an error.
pub fn run_episodes(
    ctx: &Context,
    request: &EpisodeRequest<'_>,
) -> Result<EpisodeReport, EpisodeError> {
    run_with_adapter(ctx, request, None)
}

/// [`run_episodes`], recording `adapter` as the adapter the policy serves.
pub(crate) fn run_with_adapter(
    ctx: &Context,
    request: &EpisodeRequest<'_>,
    adapter: Option<Digest>,
) -> Result<EpisodeReport, EpisodeError> {
    let settings = &request.settings;
    if settings.attempts == 0 || settings.jobs == 0 {
        return Err(EpisodeError::Settings(
            "attempts and jobs must each be at least one".into(),
        ));
    }
    let model = ctx.model(request.policy)?;
    let policy = PolicyIdentity {
        model: model.identity.clone(),
        adapter,
    };
    let config = configuration(settings, ctx.config().coding_prompt)?;
    let mut work = Vec::new();
    for task in request.tasks {
        for attempt in 0..settings.attempts {
            work.push(attempt::Planned {
                task,
                attempt,
                key: attempt::key(task, &policy, &config, settings.seed, attempt)?,
            });
        }
    }
    let run = attempt::Run {
        ctx,
        reference: request.policy,
        model: &model,
        policy: &policy,
        config: &config,
        settings,
        cancel: &request.cancel,
    };
    let done = fan_out_blocking(&work, settings.jobs, |planned| run.attempt(planned));
    let mut report = EpisodeReport {
        policy,
        seed: settings.seed,
        config,
        records: Vec::with_capacity(done.len()),
        ran: 0,
        reused: 0,
    };
    for result in done {
        let (record, reused) = match result? {
            attempt::Done::Ran(record) => (record, false),
            attempt::Done::Reused(record) => (record, true),
            attempt::Done::NotStarted => return Err(EpisodeError::Cancelled),
        };
        report.ran += usize::from(!reused);
        report.reused += usize::from(reused);
        report.records.push(record);
    }
    Ok(report)
}
