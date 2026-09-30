// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agent solvers whose every run is replayable
// evidence, for its clients. If your team needs expertise in agent
// environments or learning from agent experience, you can procure our
// services by sending an email to info@swedishembedded.com.

//! One task solved through sven, in exactly the environment it names.
//!
//! [`solve`] builds a sven engine for the environment - closed-book gets
//! no tools at all; a runtime environment gets exactly [`RUN_CODE`] - on a
//! caller-supplied model, runs one agent turn on the task's instruction
//! bounded by sven's own [`RunOptions`] (deadline, output-token budget,
//! cancel token), and hands back what an [`Experience`] records: the
//! trajectory sven exports, the final output, and how the run concluded.
//!
//! A task records the environment it is to be solved in (its snapshot pins
//! everything that determines behaviour), and [`solve`] refuses to solve it
//! in any other: a replay runs where the original ran, or not at all.
//!
//! Approval: `run_code` declares `ExecuteShell`, so sven asks before each
//! call. The solver approves a gate only when the call it gates is
//! `run_code` and refuses every other gate. That is safe because the
//! engine has no built-in tools, so nothing else can raise the gate; and
//! what a call may do is contained by its sandbox backend's limits (and,
//! for the container backend, its isolation), not by a person reading the
//! code. With nobody to ask, a question the agent puts is answered empty.

pub use crate::run_code::RUN_CODE;

use std::sync::Arc;
use std::time::Duration;

use splinter_sandbox::{ResolvedEnvironment, SandboxError};
use splinter_store::digest::Digest;
use splinter_store::experience::{Experience, ExperienceError, Provenance, Task};
use sven_sdk::model::ModelProvider;
use sven_sdk::tool::ToolCapability;
use sven_sdk::{
    atif, ApprovalPolicy, CallError, CancelToken, Engine, HumanGate, RunConclusion, RunOptions,
    Toolset, Usage,
};

use crate::run_code::RunCode;

/// The sven mode a solve runs: the conversational agent loop.
const SOLVER_MODE: &str = "agent";

/// The bounds of one solve, handed to sven as [`RunOptions`].
#[derive(Clone, Debug)]
pub struct SolveOptions {
    /// Wall-clock time the run may take ([`RunConclusion::Timeout`]).
    pub deadline: Duration,
    /// Output tokens the model may generate
    /// ([`RunConclusion::BudgetExhausted`]); `None` sets no budget.
    pub max_output_tokens: Option<u64>,
    /// Stops the run from outside ([`RunConclusion::Cancelled`]).
    pub cancel: Option<CancelToken>,
}

impl SolveOptions {
    /// A solve bounded by `deadline` alone.
    #[must_use]
    pub fn new(deadline: Duration) -> Self {
        Self {
            deadline,
            max_output_tokens: None,
            cancel: None,
        }
    }

    fn run_options(&self) -> RunOptions {
        let mut options = RunOptions::new().deadline(self.deadline);
        if let Some(tokens) = self.max_output_tokens {
            options = options.max_output_tokens(tokens);
        }
        if let Some(cancel) = &self.cancel {
            options = options.cancel(cancel.clone());
        }
        options
    }
}

/// What a solve produced: everything an [`Experience`] records about the
/// solver's side, and how the run ended.
#[derive(Clone, Debug)]
pub struct Solution {
    /// What the solver did, as sven exports it.
    pub trajectory: atif::Trajectory,
    /// The solver's final answer: its reply when the run concluded
    /// [`RunConclusion::Success`], `None` when it was stopped, whatever it
    /// had said by then.
    pub final_output: Option<String>,
    /// How the run ended.
    pub conclusion: RunConclusion,
    /// The tokens the provider reported.
    pub usage: Usage,
}

impl Solution {
    /// The experience of `task` this solution is, produced as `provenance`
    /// says.
    pub fn into_experience(
        self,
        task: Task,
        provenance: Provenance,
    ) -> Result<Experience, ExperienceError> {
        Experience::new(task, self.trajectory, self.final_output, provenance)
    }
}

/// Why a task could not be solved. A run stopped by a bound is a
/// [`Solution`] with that conclusion, not an error.
#[derive(Debug, thiserror::Error)]
pub enum SolveError {
    /// The environment the task records is not the one offered.
    #[error(
        "the task records environment {task_kind:?} ({task_snapshot:?}), \
         not the offered {offered_kind:?} ({offered_snapshot:?})"
    )]
    EnvironmentMismatch {
        /// The kind the task records.
        task_kind: String,
        /// The snapshot the task records.
        task_snapshot: Option<Digest>,
        /// The offered environment's kind.
        offered_kind: String,
        /// The offered environment's snapshot.
        offered_snapshot: Option<Digest>,
    },
    /// The offered environment's record could not be computed.
    #[error("cannot record the environment: {0}")]
    Environment(#[from] SandboxError),
    /// sven could not build or run the agent.
    #[error("the solver could not run: {0}")]
    Engine(#[from] CallError),
}

/// A model and the identity experiences record it by (the string an
/// experience's provenance names as its solver).
#[derive(Clone)]
pub struct Model {
    /// The model.
    pub provider: Arc<dyn ModelProvider>,
    /// Its identity.
    pub identity: String,
}

impl Model {
    /// `provider`, known as `identity`.
    #[must_use]
    pub fn new(provider: Arc<dyn ModelProvider>, identity: impl Into<String>) -> Self {
        Self {
            provider,
            identity: identity.into(),
        }
    }
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model")
            .field("identity", &self.identity)
            .finish_non_exhaustive()
    }
}

/// Solves `task` in `environment` on `model`, within `options`.
pub async fn solve(
    task: &Task,
    environment: &ResolvedEnvironment,
    model: Arc<dyn ModelProvider>,
    options: SolveOptions,
) -> Result<Solution, SolveError> {
    solve_prompted(task, &task.instruction, environment, model, options).await
}

/// Solves `task` as [`solve`] does, prompting the solver with `prompt`
/// instead of the bare instruction: the instruction with teacher-only
/// material the task carries beside it, such as a critique of an earlier
/// attempt. The trajectory records the prompt the solver was sent; a view
/// replaces that turn with what the student may see, so the experience's
/// instruction stays the task's own.
pub async fn solve_prompted(
    task: &Task,
    prompt: &str,
    environment: &ResolvedEnvironment,
    model: Arc<dyn ModelProvider>,
    options: SolveOptions,
) -> Result<Solution, SolveError> {
    let offered = environment.record()?;
    if offered != task.environment {
        return Err(SolveError::EnvironmentMismatch {
            task_kind: task.environment.kind.clone(),
            task_snapshot: task.environment.snapshot.clone(),
            offered_kind: offered.kind,
            offered_snapshot: offered.snapshot,
        });
    }
    let engine = engine(environment, model)?;
    let mut agent = engine.agent(SOLVER_MODE);
    let outcome = agent.send_with(prompt, options.run_options()).await?;
    let final_output = (outcome.conclusion == RunConclusion::Success).then_some(outcome.reply);
    Ok(Solution {
        trajectory: agent.trajectory(),
        final_output,
        conclusion: outcome.conclusion,
        usage: outcome.usage,
    })
}

/// The engine for `environment`: no built-in tools, and only the
/// environment's own tool.
fn engine(
    environment: &ResolvedEnvironment,
    model: Arc<dyn ModelProvider>,
) -> Result<Engine, CallError> {
    let builder = Engine::builder()
        .model_provider(model)
        .toolset(Toolset::none());
    match environment {
        ResolvedEnvironment::ClosedBook => builder.approvals(ApprovalPolicy::Deny).build(),
        ResolvedEnvironment::Runtime(runtime) => builder
            .tool(Arc::new(RunCode::new(runtime.clone())))
            .approvals(ApprovalPolicy::ask(approve_run_code_only))
            .build(),
    }
}

/// Approves a `run_code` call and nothing else, by the call it gates;
/// see the module documentation for why that is safe.
fn approve_run_code_only(gate: HumanGate) {
    // A dropped receiver means the turn already ended; nothing waits for
    // the answer.
    match gate {
        HumanGate::Approval {
            capability,
            call,
            reply_tx,
            ..
        } => {
            let is_run_code = call.is_some_and(|call| call.name == RUN_CODE);
            let _ = reply_tx.send(is_run_code && capability == ToolCapability::ExecuteShell);
        }
        HumanGate::Question { reply_tx, .. } => {
            let _ = reply_tx.send(String::new());
        }
    }
}
