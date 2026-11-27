// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements online reinforcement learning from the exact
// tokens an agent's policy sampled, for its clients. If your team needs
// expertise in training agents on their own rollouts while they run, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Practice episodes as an online run's rollouts: the
//! [`RolloutExecutor`] of the factory's online controller.
//!
//! Each rollout is one episode ([`super`]): an isolated sven session in a
//! fresh copy of its task's tree, against the policy as served, asked for
//! the policy trace (and, when set, the sampler's top-K) on every request,
//! sampling at the run's temperature from the whole distribution, its
//! requests recorded exactly as sent in a log of its own and filed on its
//! trajectory. The task's verifiers grade it afterwards, outside the agent's
//! reach. Rollouts run concurrently, each on a thread of its own, and end in
//! whatever order they finish.
//!
//! How an episode ends becomes the rollout's end: a graded attempt (passed,
//! failed, exposed) is an outcome with its reward; one the examiner gave no
//! grade (a time limit on a task that does not grade timeouts, a spent
//! budget) an outcome with no reward; one whose verifier failed or ran out
//! of time, or whose infrastructure failed, an `Err` the controller retries
//! (never a reward of zero). The outcome's records are one per model call
//! ([`splinter_data::calls`]), written only when every call is exact.
//!
//! An episode is keyed by its content and its group (the run's seed, the
//! task, the policy, the configuration, the group key and the index), so a
//! controller that restarts a rollout it had seen finish is answered with
//! that episode, not a new one.
//!
//! Sampling is untruncated: a run whose sampler cuts the distribution
//! (top-k, top-p below 1, min-p above 0) is refused, because the policy's
//! log-probabilities are then not the probabilities the tokens were drawn
//! with, until the trainer replays the sampler's support.

use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Duration;

use splinter_agent::CancelToken;
use splinter_core::digest::Digest;
use splinter_core::factory::group::Behaviour;
use splinter_core::factory::truncation;
use splinter_core::harness::RecordedCalls;
use splinter_core::model_ref::ModelRef;
use splinter_data::calls::{project_calls, CallProjection, Sampler};
use splinter_factory::online::rollout::rollout_of;
use splinter_factory::online::seams::{
    ExecutorError, FinishedRollout, OnlineRollouts, RolloutEnd, RolloutExecutor, RolloutRequest,
    RolloutsRequest, RunClock, SystemRunClock,
};
use splinter_orchestrator::context::Context;

use super::attempt::{self, Capture, Done, Planned};
use super::{
    configuration, EpisodeError, EpisodeOutcome, EpisodeRecord, EpisodeSettings, PolicyIdentity,
    PracticeTask,
};

/// The sampling controls an online run asks for. Only the whole
/// distribution is accepted ([`Self::sampler`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OnlineSampling {
    /// The temperature, above zero, as every request is sent it.
    pub temperature: f32,
    /// The top-k cut; 0 is none.
    pub top_k: u32,
    /// The nucleus; 1 is none.
    pub top_p: f32,
    /// The min-p cut; 0 is none.
    pub min_p: f32,
}

impl OnlineSampling {
    /// The whole distribution at `temperature`.
    #[must_use]
    pub fn untruncated(temperature: f32) -> Self {
        Self {
            temperature,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
        }
    }

    /// The sampler the run's records declare.
    ///
    /// # Errors
    /// The temperature is not above zero, or the sampling cuts the
    /// distribution: refused by name until the trainer replays the
    /// sampler's support.
    pub fn sampler(&self) -> Result<Sampler, EpisodeError> {
        let cuts: Vec<String> = [
            (self.top_k != 0).then(|| format!("top_k {}", self.top_k)),
            (self.top_p < 1.0).then(|| format!("top_p {}", self.top_p)),
            (self.min_p > 0.0).then(|| format!("min_p {}", self.min_p)),
        ]
        .into_iter()
        .flatten()
        .collect();
        if !cuts.is_empty() || !(self.top_p <= 1.0 && self.min_p >= 0.0) {
            return Err(EpisodeError::Settings(format!(
                "an online run samples the whole distribution (top_k 0, top_p 1, min_p 0); \
                 this one cuts it ({}): the policy's log-probabilities would not be the \
                 probabilities its tokens were drawn with, and the trainer does not replay a \
                 sampler's support yet",
                cuts.join(", ")
            )));
        }
        // The temperature as the requests carry it on the wire (its
        // shortest decimal form), which is what the server reports back.
        let temperature: f64 = self.temperature.to_string().parse().map_err(|e| {
            EpisodeError::Settings(format!("the temperature {}: {e}", self.temperature))
        })?;
        Sampler::untruncated(temperature).map_err(EpisodeError::Settings)
    }
}

/// How an online run's rollouts run.
#[derive(Clone, Debug)]
pub struct OnlineSettings {
    /// The episode bounds and the run's seed (`attempts` and `jobs` are the
    /// controller's to decide and unused here; the temperature is the
    /// sampling's).
    pub episode: EpisodeSettings,
    /// How every request samples.
    pub sampling: OnlineSampling,
    /// The sampler's top-K asked of every request; `None` asks for none.
    pub top_logprobs: Option<u32>,
    /// Where running rollouts keep their request logs until they are filed
    /// on their trajectories: a private directory of the run's own.
    pub scratch: PathBuf,
}

/// What every rollout of a run shares.
struct Shared<'c> {
    ctx: &'c Context,
    /// Reaches the policy version a pinned rollout names.
    resolve: Option<&'c Resolve<'c>>,
    tasks: Vec<PracticeTask>,
    reference: Option<ModelRef>,
    model: splinter_agent::solve::Model,
    policy: PolicyIdentity,
    config: Digest,
    settings: EpisodeSettings,
    capture: Capture,
    sampler: Sampler,
    cancel: CancelToken,
}

/// How a policy served under an id is reached.
pub type Resolve<'c> = dyn Fn(&str) -> Result<OnlinePolicy, String> + Send + Sync + 'c;

/// The policy one rollout asks: the run's, or the version it is pinned to.
struct Asked {
    reference: Option<ModelRef>,
    model: splinter_agent::solve::Model,
    policy: PolicyIdentity,
}

impl Shared<'_> {
    /// The policy `request` asks.
    fn asked(&self, request: &RolloutRequest) -> Result<Option<Asked>, String> {
        let Some(id) = &request.policy else {
            return Ok(None);
        };
        let resolve = self
            .resolve
            .ok_or("this run cannot reach a pinned policy version: nothing resolves its id")?;
        let (reference, model) = match resolve(id)? {
            OnlinePolicy::Reference(reference) => {
                let model = self.ctx.model(&reference).map_err(|e| e.to_string())?;
                (Some(reference), model)
            }
            OnlinePolicy::Model(model) => (None, model),
        };
        Ok(Some(Asked {
            policy: PolicyIdentity {
                model: model.identity.clone(),
                adapter: None,
            },
            reference,
            model,
        }))
    }

    fn run<'a>(&'a self, asked: Option<&'a Asked>) -> attempt::Run<'a> {
        attempt::Run {
            ctx: self.ctx,
            reference: asked.map_or(self.reference.as_ref(), |a| a.reference.as_ref()),
            model: asked.map_or(&self.model, |a| &a.model),
            policy: asked.map_or(&self.policy, |a| &a.policy),
            config: &self.config,
            settings: &self.settings,
            cancel: &self.cancel,
            capture: Some(&self.capture),
        }
    }

    /// The end of the rollout `request`, run (or found recorded) now; `None`
    /// when the run was cancelled before it ended.
    fn rollout(&self, request: &RolloutRequest, task: &PracticeTask) -> Option<RolloutEnd> {
        let group = request.group.as_str();
        let end = |result| RolloutEnd {
            group: request.group.clone(),
            index: request.index,
            result,
        };
        let asked = match self.asked(request) {
            Ok(asked) => asked,
            Err(e) => return Some(end(Err(e))),
        };
        let key = match attempt::key(
            task,
            asked.as_ref().map_or(&self.policy, |a| &a.policy),
            &self.config,
            self.settings.seed,
            Some(group),
            request.index,
        ) {
            Ok(key) => key,
            Err(e) => return Some(end(Err(e.to_string()))),
        };
        let planned = Planned {
            task,
            attempt: request.index,
            group: Some(group),
            key,
        };
        match self.run(asked.as_ref()).attempt(&planned) {
            Ok(Done::Ran(record) | Done::Reused(record)) => Some(end(self.finished(&record))),
            Ok(Done::NotStarted) => None,
            Err(EpisodeError::Cancelled) if self.cancel.is_cancelled() => None,
            Err(e) => Some(end(Err(e.to_string()))),
        }
    }

    /// How a recorded episode ended, as a rollout's result. An episode a
    /// limit ended (its conclusion, or a server that refused its context) is
    /// a scored failure with its cause, never an infrastructure failure.
    fn finished(&self, record: &EpisodeRecord) -> Result<FinishedRollout, String> {
        let truncation = record
            .conclusion
            .as_deref()
            .and_then(truncation::of_conclusion)
            .or_else(|| match &record.outcome {
                EpisodeOutcome::Ungraded { detail, .. } => truncation::of_error(detail),
                _ => None,
            });
        let reward = match &record.outcome {
            EpisodeOutcome::Passed | EpisodeOutcome::Failed | EpisodeOutcome::Exposed => {
                record.outcome.reward()
            }
            // Graded by the examiner's rules as no grade: not a failure of
            // the infrastructure, and not a reward.
            EpisodeOutcome::Ungraded { .. } => None,
            EpisodeOutcome::VerifierError { detail } => {
                return Err(format!("the verifier failed: {detail}"))
            }
            EpisodeOutcome::VerifierTimeout => return Err("the verifier ran out of time".into()),
            EpisodeOutcome::Infrastructure { detail } => {
                return Err(format!("the episode could not run: {detail}"))
            }
        };
        let Some(id) = record.experience.as_ref() else {
            // A run the server refused for its context kept no trajectory:
            // its calls cannot be paired, so it is a failure that trains
            // nothing.
            return match truncation {
                Some(cause) => rollout_of(reward, Some(cause), None, &CallProjection::default()),
                None => Err("a graded episode kept no experience".into()),
            };
        };
        let experience = self
            .ctx
            .experiences()
            .get(id)
            .map_err(|e| format!("the episode's experience {id}: {e}"))?;
        let trajectory = &experience.trajectory;
        let calls = RecordedCalls::of(trajectory)
            .and_then(|recorded| recorded.calls())
            .unwrap_or_default();
        let projection = project_calls(trajectory, &calls, &self.sampler);
        rollout_of(
            reward,
            truncation,
            Some(Behaviour::of(trajectory)),
            &projection,
        )
    }
}

/// The online run's executor: rollouts on threads of `scope`, their ends on
/// one channel.
pub struct OnlineEpisodes<'s, 'c> {
    scope: &'s std::thread::Scope<'s, 'c>,
    shared: &'c Shared<'c>,
    clock: &'c dyn RunClock,
    sender: mpsc::Sender<RolloutEnd>,
    ends: mpsc::Receiver<RolloutEnd>,
    running: usize,
}

impl RolloutExecutor for OnlineEpisodes<'_, '_> {
    fn start(&mut self, request: RolloutRequest) -> Result<(), ExecutorError> {
        if self.shared.cancel.is_cancelled() {
            return Err(ExecutorError("the run is cancelled".into()));
        }
        let shared = self.shared;
        let task = shared
            .tasks
            .iter()
            .find(|t| t.task.to_string() == request.task)
            .ok_or_else(|| {
                ExecutorError(format!(
                    "{} is not a practice task of this run",
                    request.task
                ))
            })?;
        let sender = self.sender.clone();
        self.running += 1;
        self.scope.spawn(move || {
            if let Some(end) = shared.rollout(&request, task) {
                // The receiver outlives every rollout of the scope.
                let _ = sender.send(end);
            }
        });
        Ok(())
    }

    fn wait(&mut self, until_ms: u64) -> Result<Vec<RolloutEnd>, ExecutorError> {
        let left = Duration::from_millis(until_ms.saturating_sub(self.clock.now_ms()));
        let mut ended = Vec::new();
        if self.running == 0 {
            std::thread::sleep(left);
            return Ok(ended);
        }
        match self.ends.recv_timeout(left) {
            Ok(end) => ended.push(end),
            Err(mpsc::RecvTimeoutError::Timeout) => return Ok(ended),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Err(ExecutorError("the rollouts' channel closed".into()))
            }
        }
        ended.extend(self.ends.try_iter());
        self.running -= ended.len();
        Ok(ended)
    }

    fn cancel(&mut self) {
        self.shared.cancel.cancel();
    }
}

/// The policy an online run's rollouts ask.
#[derive(Clone)]
pub enum OnlinePolicy {
    /// A model the context loads by reference (and resamples per attempt).
    Reference(ModelRef),
    /// A model reached as it is: the version a learner serves under an id.
    Model(splinter_agent::solve::Model),
}

/// Runs `body` with an executor of rollouts of `tasks` under `policy`, as
/// `settings` say; every rollout still running when `body` returns is
/// cancelled and waited for.
///
/// # Errors
/// [`EpisodeError::Settings`] for a sampling that cuts the distribution or
/// a scratch directory that cannot be made; the context's errors.
pub fn run_online<R>(
    ctx: &Context,
    tasks: Vec<PracticeTask>,
    policy: &OnlinePolicy,
    settings: OnlineSettings,
    clock: &dyn RunClock,
    body: impl FnOnce(&mut OnlineEpisodes<'_, '_>) -> R,
) -> Result<R, EpisodeError> {
    run_online_resolving(ctx, tasks, policy, None, settings, clock, body)
}

/// [`run_online`], a rollout pinned to a policy version (its request names
/// the version's id) reaching it as `resolve` says.
///
/// # Errors
/// As [`run_online`].
pub fn run_online_resolving<R>(
    ctx: &Context,
    tasks: Vec<PracticeTask>,
    policy: &OnlinePolicy,
    resolve: Option<&Resolve<'_>>,
    settings: OnlineSettings,
    clock: &dyn RunClock,
    body: impl FnOnce(&mut OnlineEpisodes<'_, '_>) -> R,
) -> Result<R, EpisodeError> {
    let sampler = settings.sampling.sampler()?;
    std::fs::create_dir_all(&settings.scratch).map_err(|e| {
        EpisodeError::Settings(format!(
            "the scratch directory {}: {e}",
            settings.scratch.display()
        ))
    })?;
    let episode = EpisodeSettings {
        temperature: settings.sampling.temperature,
        ..settings.episode
    };
    let (reference, model) = match policy {
        OnlinePolicy::Reference(reference) => (Some(reference.clone()), ctx.model(reference)?),
        OnlinePolicy::Model(model) => (None, model.clone()),
    };
    let shared = Shared {
        ctx,
        resolve,
        tasks,
        reference,
        policy: PolicyIdentity {
            model: model.identity.clone(),
            adapter: None,
        },
        model,
        config: configuration(&episode, ctx.config().coding_prompt, settings.top_logprobs)?,
        settings: episode,
        capture: Capture {
            scratch: settings.scratch,
            top_logprobs: settings.top_logprobs,
        },
        sampler,
        cancel: CancelToken::new(),
    };
    let shared = &shared;
    Ok(std::thread::scope(|scope| {
        let (sender, ends) = mpsc::channel();
        let mut executor = OnlineEpisodes {
            scope,
            shared,
            clock,
            sender,
            ends,
            running: 0,
        };
        let out = body(&mut executor);
        executor.cancel();
        out
    }))
}

/// Practice episodes as the factory's online rollouts
/// ([`OnlineRollouts`]): an online stage's tasks (a pinned collection of
/// task folders) are imported and rolled out by [`run_online`] against the
/// policy the learner serves.
pub struct FactoryRollouts<'c> {
    ctx: &'c Context,
    episode: EpisodeSettings,
    policy: Box<PolicyOf<'c>>,
}

/// How the policy a learner serves under an id is reached.
type PolicyOf<'c> = Resolve<'c>;

impl<'c> FactoryRollouts<'c> {
    /// Rollouts in `ctx`, each bounded as `episode` says (its seed and
    /// temperature are each stage's), the served policy named `id` reached
    /// as `policy(id)` says.
    #[must_use]
    pub fn new(
        ctx: &'c Context,
        episode: EpisodeSettings,
        policy: impl Fn(&str) -> Result<OnlinePolicy, String> + Send + Sync + 'c,
    ) -> Self {
        Self {
            ctx,
            episode,
            policy: Box::new(policy),
        }
    }
}

impl OnlineRollouts for FactoryRollouts<'_> {
    fn identity(&self) -> String {
        format!(
            "sven practice episodes ({}; deadline {}s, {:?} tool calls, {:?} output tokens)",
            super::RECORD_FORMAT,
            self.episode.deadline.as_secs(),
            self.episode.max_tool_calls,
            self.episode.max_output_tokens
        )
    }

    fn run(
        &self,
        request: &RolloutsRequest<'_>,
        body: &mut dyn FnMut(&mut dyn RolloutExecutor, &[String]),
    ) -> Result<(), String> {
        let tasks = super::load_collection(self.ctx, request.tasks).map_err(|e| e.to_string())?;
        let ids: Vec<String> = tasks.iter().map(|t| t.task.to_string()).collect();
        let settings = OnlineSettings {
            episode: EpisodeSettings {
                seed: request.seed,
                max_output_tokens: request.max_output_tokens.or(self.episode.max_output_tokens),
                ..self.episode.clone()
            },
            sampling: OnlineSampling::untruncated(request.temperature),
            top_logprobs: request.top_logprobs,
            scratch: request.scratch.to_path_buf(),
        };
        let policy = (self.policy)(request.policy)?;
        run_online_resolving(
            self.ctx,
            tasks,
            &policy,
            Some(&*self.policy),
            settings,
            &SystemRunClock,
            |executor| {
                body(executor, &ids);
            },
        )
        .map_err(|e| e.to_string())
    }
}
