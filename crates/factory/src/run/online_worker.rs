// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements online reinforcement learning in which an
// agent practises with exactly the tools its task environment gives it, for
// its clients. If your team needs expertise in training agents on their own
// rollouts in real task environments, you can procure our services by
// sending an email to info@swedishembedded.com.

//! `train.online`'s rollouts on the execution worker: each rollout one
//! attempt of its task in the task's own environment, exactly as an
//! `attempt.collect` pass or an evaluation runs it, so the agent is given the
//! tools, configuration and sandbox its task declares (a task whose
//! environment gives its agent tools as MCP servers is run with them),
//! whatever the environment is.
//!
//! The agent samples every request at the run's temperature from the whole
//! distribution, with a seed of the rollout's own, and asks for the policy
//! trace (and, when set, the sampler's top-K); its own request log (`sven
//! --record-requests`) is filed on its trajectory, and its records are made
//! by the one projection every executor uses
//! ([`crate::online::rollout::rollout_of`]). The worker never grades: each
//! attempt's report is graded by the one classifier, a limit that ended it is
//! read by the one truncation classifier, and an attempt the infrastructure
//! decided is a failure the controller retries, never a reward.
//!
//! **Resumable.** Each rollout's attempts are written under
//! `<scratch>/<group>/<index>/<n>`. A rollout started again (a resumed
//! controller) first reads back what an earlier attempt finished
//! ([`ExecutionBackend::reconcile`]) and is answered with it; only a rollout
//! with no finished attempt runs again.
//!
//! **Cancellation.** A cancelled run (stopped, or rolled back) starts no new
//! attempt and kills the ones running, with everything they started; they
//! have no end, and a resumed run starts them again.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use splinter_core::digest::digest_of;
use splinter_core::factory::attempt::AttemptReport;
use splinter_core::factory::execution::{ModelBinding, SlotKey, BRAIN_DRIVER};
use splinter_core::factory::group::Behaviour;
use splinter_core::factory::trial::{grade, GradingPolicy, RunBudget, TimeoutPolicy};
use splinter_core::factory::truncation;
use splinter_core::harness::RecordedCalls;
use splinter_data::calls::{project_calls, CallProjection, Sampler};

use super::corpus::{file_calls, CALL_LOG};
use super::pass::{RolloutPass, TaskSelector};
use super::rollout::{pin_files, Arm, Running};
use super::{stage_error, Factory, RunError};
use crate::backend::{Batch, ExecutionBackend};
use crate::online::rollout::{rollout_of, Sessions};
use crate::online::seams::{
    ExecutorError, FinishedRollout, OnlineRollouts, RolloutEnd, RolloutExecutor, RolloutRequest,
    RolloutsRequest, RunClock, SystemRunClock,
};
use crate::plan::PlannedStage;
use crate::recipe::ModelRef;

/// The one agent adapter that records the policy trace an online run trains
/// on.
pub const TRACED_AGENT: &str = "sven";

/// The agent's flags an online run sets itself: how every request samples
/// and what is recorded of it. A recipe that sets one is refused.
const RUN_FLAGS: &[&str] = &[
    "--temperature",
    "--top-p",
    "--top-k",
    "--seed",
    "--logprobs",
    "--top-logprobs",
    "--sampler-support",
    "--record-requests",
];

/// How `train.online`'s rollouts run on the execution worker, in their
/// tasks' own environment: the agent and environment as an `attempt.collect`
/// stage names them.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OnlineAgent {
    /// The agent adapter: `sven`, the one that records the policy trace.
    pub agent: String,
    /// The environment file every rollout runs in (its sandbox, network and
    /// resources).
    pub environment: String,
    /// The agent's own arguments, besides the sampling the run sets.
    #[serde(default)]
    pub args: Vec<String>,
    /// Files the arguments refer to.
    #[serde(default)]
    pub files: Vec<String>,
    /// A system prompt file appended to the agent's own.
    #[serde(default)]
    pub system_prompt: Option<String>,
    /// Environment variables for the agent (`env:NAME` passes the worker's
    /// own).
    #[serde(default)]
    pub environment_variables: BTreeMap<String, String>,
    /// The per-rollout budget.
    #[serde(default)]
    pub budget: Option<RunBudget>,
    /// The tasks' timeout policy.
    #[serde(default)]
    pub timeout_policy: TimeoutPolicy,
}

impl OnlineAgent {
    /// Checks what can be checked without the run.
    ///
    /// # Errors
    /// Another agent than the traced one, or a flag the run sets itself.
    pub fn check(&self) -> Result<(), String> {
        if self.agent != TRACED_AGENT {
            return Err(format!(
                "agent.agent {:?}: an online run trains on the policy trace, which only the {TRACED_AGENT} \
                 adapter records",
                self.agent
            ));
        }
        if let Some(flag) = self
            .args
            .iter()
            .find(|a| RUN_FLAGS.iter().any(|f| a.split('=').next() == Some(*f)))
        {
            return Err(format!(
                "agent.args {flag}: an online run sets its sampling and what it records itself"
            ));
        }
        if self.environment.trim().is_empty() {
            return Err("agent.environment names no environment file".into());
        }
        Ok(())
    }

    /// The pass an arm of these rollouts runs: one attempt of a slot at a
    /// time, retried by the online controller, never by the pass.
    fn pass(&self, name: &str, seed: u64) -> RolloutPass {
        RolloutPass {
            name: name.to_string(),
            agent: self.agent.clone(),
            args: self.args.clone(),
            files: self.files.clone(),
            model: None,
            system_prompt: self.system_prompt.clone(),
            environment_variables: self.environment_variables.clone(),
            environment: self.environment.clone(),
            tasks: TaskSelector::default(),
            attempts: 1,
            concurrency: 1,
            budget: self.budget,
            timeout_policy: self.timeout_policy,
            retries: 0,
            seed,
            interactive: false,
            hint: None,
            harness: None,
            made_harness: None,
        }
    }
}

/// The agent's flags that make a rollout's every request sample as the run
/// asks and record what an online run trains on.
fn sampling_args(temperature: f32, seed: u64, top_logprobs: Option<u32>) -> Vec<String> {
    let mut args = vec![
        "--temperature".to_string(),
        temperature.to_string(),
        "--top-p".to_string(),
        "1".to_string(),
        "--seed".to_string(),
        seed.to_string(),
        "--logprobs".to_string(),
    ];
    if let Some(k) = top_logprobs {
        args.extend(["--top-logprobs".to_string(), k.to_string()]);
    }
    args
}

/// The seed of attempt `attempt` of rollout `index` of `group` in a run
/// seeded `seed`: a function of these alone.
fn rollout_seed(seed: u64, group: &str, index: u32, attempt: u32) -> u64 {
    digest_of(&(seed, group, index, attempt)).leading_u64()
}

/// How an attempt on the worker ended for the controller.
enum Ending {
    /// A rollout of the policy (or a failure to read what it left).
    Rollout(Result<FinishedRollout, String>),
    /// The infrastructure decided the attempt: it says nothing about the
    /// policy and is run again.
    Infrastructure(String),
}

/// The arm of `stage`'s rollouts: every task of its pinned collection, run
/// as `agent` says.
pub(crate) fn arm<'r>(
    factory: &'r Factory<'_>,
    stage: &PlannedStage,
    tasks: &str,
    agent: &OnlineAgent,
    seed: u64,
) -> Result<Arm<'r>, RunError> {
    agent.check().map_err(|e| stage_error(&stage.key, e))?;
    let collection = factory
        .collection(tasks)
        .ok_or_else(|| stage_error(&stage.key, format!("{tasks} pins no tasks")))?;
    Ok(Arm {
        name: stage.key.clone(),
        key: stage.key.clone(),
        pass: agent.pass(&stage.key, seed),
        tasks: collection.collection.tasks.iter().collect(),
        task_root: factory.view(tasks),
        model: None,
        champion: None,
        briefs: None,
        continue_from: None,
    })
}

/// The identity of an arm's rollouts on the worker: the worker's, and the
/// digest of everything the arm runs with (agent, environment, sandbox,
/// tasks).
pub(crate) fn identity(factory: &Factory<'_>, arm: &Arm<'_>) -> Result<String, RunError> {
    Ok(format!(
        "{} ({TRACED_AGENT} rollouts in their tasks' environments, {})",
        factory.services.backend.identity(),
        digest_of(&arm.parameters(factory)?)
    ))
}

/// `train.online`'s rollouts on the execution worker.
pub(crate) struct WorkerRollouts<'a, 'r> {
    running: Running<'a, 'r>,
    backend: &'a dyn ExecutionBackend,
    base: ModelBinding,
    identity: String,
}

impl<'a, 'r> WorkerRollouts<'a, 'r> {
    /// The rollouts of `arm`, whose agent reaches the policy over `base`'s
    /// connection, their work (pinned files, attempts) under `work`.
    pub(crate) fn new(
        factory: &'a Factory<'_>,
        arm: &'a Arm<'r>,
        base: &ModelRef,
        work: PathBuf,
    ) -> Result<Self, RunError> {
        let base = factory
            .resolved
            .connections
            .bind(base)
            .map_err(|e| stage_error(&arm.key, e))?;
        let running = Running::new(factory, arm, work)?;
        std::fs::create_dir_all(&running.work).map_err(super::io(&running.work))?;
        pin_files(factory, &running)?;
        Ok(Self {
            identity: identity(factory, arm)?,
            running,
            backend: factory.services.backend,
            base,
        })
    }
}

/// What every rollout of one `run` shares.
struct Shared<'w, 'a, 'r> {
    rollouts: &'w WorkerRollouts<'a, 'r>,
    binding: ModelBinding,
    sampler: Sampler,
    temperature: f32,
    top_logprobs: Option<u32>,
    seed: u64,
    scratch: PathBuf,
    policy: GradingPolicy,
    cancelled: AtomicBool,
    sessions: Sessions,
}

impl Shared<'_, '_, '_> {
    /// The end of `request`: an earlier attempt's, read back, or a new
    /// attempt's; `None` when the run was cancelled before it began.
    fn rollout(&self, request: &RolloutRequest) -> Option<RolloutEnd> {
        let end = |result| RolloutEnd {
            group: request.group.clone(),
            index: request.index,
            result,
        };
        let dir = self
            .scratch
            .join(request.group.as_str().replace([':', '/'], "-"))
            .join(request.index.to_string());
        let mut next = 0;
        for (n, out) in attempts(&dir) {
            next = n + 1;
            match self.rollouts.backend.reconcile(&out) {
                Ok(reports) => {
                    if let Some(report) = reports.into_iter().next() {
                        if let Ending::Rollout(done) = self.finished(&report, &out) {
                            return Some(end(done));
                        }
                    }
                }
                Err(e) => return Some(end(Err(e.to_string()))),
            }
        }
        if self.cancelled.load(Ordering::SeqCst) {
            return None;
        }
        match self.attempt(request, &dir.join(next.to_string()), next) {
            // A cancelled run's attempt was killed: no end, it runs again on resume.
            Err(_) if self.cancelled.load(Ordering::SeqCst) => None,
            result => Some(end(result)),
        }
    }

    /// One new attempt of `request` into `out`.
    fn attempt(
        &self,
        request: &RolloutRequest,
        out: &Path,
        attempt: u32,
    ) -> Result<FinishedRollout, String> {
        let running = &self.rollouts.running;
        let slot = SlotKey::new(
            running.arm.name.clone(),
            request.task.clone(),
            request.index,
        );
        let mut spec = running.spec(&slot, attempt);
        spec.seed = rollout_seed(self.seed, request.group.as_str(), request.index, attempt);
        spec.agent.args.extend(sampling_args(
            self.temperature,
            spec.seed,
            self.top_logprobs,
        ));
        // A pinned rollout asks the one version it started on.
        spec.model = Some(ModelBinding {
            model: request
                .policy
                .clone()
                .unwrap_or_else(|| self.binding.model.clone()),
            ..self.binding.clone()
        });
        let cancelled = || self.cancelled.load(Ordering::SeqCst);
        let reports = self
            .rollouts
            .backend
            .execute_cancellable(
                Batch {
                    specs: std::slice::from_ref(&spec),
                    tasks: &running.arm.task_root,
                    files: &running.work.join("files"),
                    out,
                    concurrency: 1,
                },
                &cancelled,
            )
            .map_err(|e| e.to_string())?;
        let report = reports
            .into_iter()
            .next()
            .ok_or("the execution worker reported no attempt")?;
        match self.finished(&report, out) {
            Ending::Rollout(done) => done,
            Ending::Infrastructure(why) => Err(why),
        }
    }

    /// How `report`'s attempt ended: graded by the one classifier, a limit
    /// that ended it read by the one truncation classifier (a scored
    /// failure even where the server's refusal reads as the model's error).
    fn finished(&self, report: &AttemptReport, out: &Path) -> Ending {
        let graded = grade(&report.evidence, &self.policy);
        let truncation = truncation::of_trial(&graded, &report.evidence);
        if truncation.is_none() && graded.status.is_infrastructure() {
            let cause = graded.cause.as_ref().map_or("", |c| c.message.as_str());
            return Ending::Infrastructure(format!("{:?}: {cause}", graded.status));
        }
        let reward = graded.passed.map(|passed| if passed { 1.0 } else { 0.0 });
        Ending::Rollout(
            self.projection(report, out)
                .and_then(|(projection, behaviour)| {
                    rollout_of(reward, truncation, behaviour, &projection, self.sessions)
                }),
        )
    }

    /// The projection of `report`'s calls (its trajectory with its own
    /// request log filed on it) and its agent's behaviour. A trial that left
    /// no trajectory projects to nothing (an inexact rollout, never trained).
    fn projection(
        &self,
        report: &AttemptReport,
        out: &Path,
    ) -> Result<(CallProjection, Option<Behaviour>), String> {
        let Some(trace) = &report.trace else {
            return Ok((CallProjection::default(), None));
        };
        let path = out.join(trace);
        let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut trajectory: atif::Trajectory =
            serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
        let log = out.join(&report.native_dir).join(CALL_LOG);
        if let Ok(requests) = std::fs::read(&log) {
            file_calls(&mut trajectory, &requests)
                .map_err(|e| format!("{}: {e}", log.display()))?;
        }
        let calls = RecordedCalls::of(&trajectory)
            .and_then(|recorded| recorded.calls())
            .unwrap_or_default();
        Ok((
            project_calls(&trajectory, &calls, &self.sampler),
            Some(Behaviour::of(&trajectory)),
        ))
    }
}

/// The attempts written under a rollout's directory, by number.
fn attempts(dir: &Path) -> Vec<(u32, PathBuf)> {
    let mut found: Vec<(u32, PathBuf)> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let n = e.file_name().to_str()?.parse::<u32>().ok()?;
            Some((n, e.path()))
        })
        .collect();
    found.sort();
    found
}

/// The executor of one `run`: rollouts on threads of `scope`, their ends on
/// one channel.
struct WorkerExecutor<'s, 'c, 'w, 'a, 'r> {
    scope: &'s std::thread::Scope<'s, 'c>,
    shared: &'c Shared<'w, 'a, 'r>,
    clock: &'c dyn RunClock,
    sender: mpsc::Sender<RolloutEnd>,
    ends: mpsc::Receiver<RolloutEnd>,
    running: usize,
}

impl RolloutExecutor for WorkerExecutor<'_, '_, '_, '_, '_> {
    fn start(&mut self, request: RolloutRequest) -> Result<(), ExecutorError> {
        if self.shared.cancelled.load(Ordering::SeqCst) {
            return Err(ExecutorError("the run is cancelled".into()));
        }
        if !self
            .shared
            .rollouts
            .running
            .arm
            .tasks
            .iter()
            .any(|t| t.id == request.task)
        {
            return Err(ExecutorError(format!(
                "{} is not a task of this run",
                request.task
            )));
        }
        let shared = self.shared;
        let sender = self.sender.clone();
        self.running += 1;
        self.scope.spawn(move || {
            if let Some(end) = shared.rollout(&request) {
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
        self.shared.cancelled.store(true, Ordering::SeqCst);
    }
}

impl OnlineRollouts for WorkerRollouts<'_, '_> {
    fn identity(&self) -> String {
        self.identity.clone()
    }

    fn run(
        &self,
        request: &RolloutsRequest<'_>,
        body: &mut dyn FnMut(&mut dyn RolloutExecutor, &[String]),
    ) -> Result<(), String> {
        let temperature: f64 = request
            .temperature
            .to_string()
            .parse()
            .map_err(|e| format!("the temperature {}: {e}", request.temperature))?;
        let shared = Shared {
            rollouts: self,
            // The learner's server is a brain server: the agent reaches it
            // with the driver that honours a request's seed and reads its
            // policy trace.
            binding: ModelBinding {
                model: request.policy.to_string(),
                driver: Some(BRAIN_DRIVER.to_string()),
                ..self.base.clone()
            },
            sampler: Sampler::untruncated(temperature)?,
            temperature: request.temperature,
            top_logprobs: request.top_logprobs,
            seed: request.seed,
            scratch: request.scratch.to_path_buf(),
            policy: GradingPolicy {
                timeout: self.running.arm.pass.timeout_policy,
                budget: self.running.arm.pass.budget,
            },
            cancelled: AtomicBool::new(false),
            sessions: request.sessions,
        };
        let ids: Vec<String> = self
            .running
            .arm
            .tasks
            .iter()
            .map(|t| t.id.clone())
            .collect();
        let shared = &shared;
        std::thread::scope(|scope| {
            let (sender, ends) = mpsc::channel();
            let mut executor = WorkerExecutor {
                scope,
                shared,
                clock: &SystemRunClock,
                sender,
                ends,
                running: 0,
            };
            body(&mut executor, &ids);
            executor.cancel();
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn agent() -> OnlineAgent {
        serde_json::from_value(
            serde_json::json!({"agent": "sven", "environment": "environments/x.yaml"}),
        )
        .unwrap()
    }

    #[test]
    fn only_the_traced_agent_runs_and_the_run_owns_its_sampling_flags() {
        assert!(agent().check().is_ok());
        let other = OnlineAgent {
            agent: "claude-code".into(),
            ..agent()
        };
        assert!(other.check().unwrap_err().contains("policy trace"));
        for flag in ["--temperature", "--seed=3", "--record-requests"] {
            let set = OnlineAgent {
                args: vec![flag.into(), "1".into()],
                ..agent()
            };
            assert!(
                set.check().unwrap_err().contains("sets its sampling"),
                "{flag}"
            );
        }
        let fine = OnlineAgent {
            args: vec!["--max-tool-rounds".into(), "40".into()],
            ..agent()
        };
        assert!(fine.check().is_ok());
    }

    #[test]
    fn every_request_samples_the_whole_distribution_with_the_rollouts_seed() {
        assert_eq!(
            sampling_args(0.7, 42, None),
            [
                "--temperature",
                "0.7",
                "--top-p",
                "1",
                "--seed",
                "42",
                "--logprobs"
            ]
        );
        assert_eq!(
            sampling_args(1.0, 1, Some(4))[6..],
            ["--logprobs", "--top-logprobs", "4"]
        );
        // seed: the derivation is checked for distinct inputs, any run seed serves.
        let seeds = [
            rollout_seed(9, "run:0", 0, 0),
            rollout_seed(9, "run:0", 1, 0),
            rollout_seed(9, "run:1", 0, 0),
            rollout_seed(9, "run:0", 0, 1),
        ];
        let distinct: std::collections::BTreeSet<_> = seeds.iter().collect();
        assert_eq!(
            distinct.len(),
            seeds.len(),
            "every rollout and attempt draws its own"
        );
    }

    #[test]
    fn a_rollouts_attempts_are_read_back_in_order() {
        let dir =
            std::env::temp_dir().join(format!("online-worker-attempts-{}", std::process::id()));
        for n in ["2", "0", "10", "notes"] {
            std::fs::create_dir_all(dir.join(n)).unwrap();
        }
        let found: Vec<u32> = attempts(&dir).into_iter().map(|(n, _)| n).collect();
        assert_eq!(found, [0, 2, 10]);
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(attempts(&dir).is_empty());
    }
}
