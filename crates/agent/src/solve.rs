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
//! cancel token) and the model's stream idle limit, and hands back what an
//! [`Experience`] records: the trajectory sven exports, the final output,
//! and how the run concluded.
//!
//! A task records the environment it is to be solved in (its snapshot pins
//! everything that determines behaviour), and [`solve`] refuses to solve it
//! in any other: a replay runs where the original ran, or not at all.
//!
//! Every request the solver's model receives runs under [`SYSTEM_PROMPT`],
//! Splinter's own short system prompt, in place of sven's coding-agent
//! one: what the policy answers under is what its training records show.
//!
//! Approval: the engine runs under sven's default automatic approval. It
//! has no built-in tools, so the only call a solver can make is
//! `run_code`, and what that call may do is contained by its sandbox
//! backend's limits (and, for the container backend, its isolation), not
//! by a person reading the code. With nobody to ask, sven answers a
//! question the agent puts with its no-user answer.

pub use crate::run_code::RUN_CODE;
pub use splinter_lab::SYSTEM_PROMPT;

use std::sync::Arc;
use std::time::Duration;

use splinter_record::digest::Digest;
use splinter_record::experience::{Experience, ExperienceError, Provenance, Task};
use splinter_sandbox::{ResolvedEnvironment, SandboxError};
use sven_sdk::config::Config;
use sven_sdk::model::ModelProvider;
use sven_sdk::{atif, CallError, CancelToken, Engine, RunConclusion, RunOptions, Toolset, Usage};

use crate::run_code::RunCode;
use crate::system_prompt::UnderSystemPrompt;

/// The sven mode a solve runs: the conversational agent loop.
pub(crate) const SOLVER_MODE: &str = "agent";

/// The bounds of one solve, handed to sven as its [`RunOptions`] and, for
/// the stream idle limit, its engine configuration.
#[derive(Clone, Debug)]
pub struct SolveOptions {
    /// Wall-clock time the run may take ([`RunConclusion::Timeout`]).
    pub deadline: Duration,
    /// Output tokens the model may generate
    /// ([`RunConclusion::BudgetExhausted`]); `None` sets no budget.
    pub max_output_tokens: Option<u64>,
    /// Stops the run from outside ([`RunConclusion::Cancelled`]).
    pub cancel: Option<CancelToken>,
    /// The longest silence between two chunks of the model's stream before
    /// sven declares it stale and fails the run: the solving model's own
    /// [`Model::stream_idle`]. `None` keeps sven's default. A limit past
    /// the deadline is the deadline: no silence outlasts the run.
    pub stream_idle: Option<Duration>,
}

impl SolveOptions {
    /// A solve bounded by `deadline` alone.
    #[must_use]
    pub fn new(deadline: Duration) -> Self {
        Self {
            deadline,
            max_output_tokens: None,
            cancel: None,
            stream_idle: None,
        }
    }

    /// The run bounds sven enforces for one run.
    #[must_use]
    pub fn run_options(&self) -> RunOptions {
        let mut options = RunOptions::new().deadline(self.deadline);
        if let Some(tokens) = self.max_output_tokens {
            options = options.max_output_tokens(tokens);
        }
        if let Some(cancel) = &self.cancel {
            options = options.cancel(cancel.clone());
        }
        options
    }

    /// The sven configuration an engine running these bounds is built with:
    /// sven's default, with the stream idle limit when one is set, in whole
    /// seconds rounded up (sven's unit, where zero means its default) and
    /// capped at the deadline.
    #[must_use]
    pub fn engine_config(&self) -> Config {
        let mut config = Config::default();
        if let Some(limit) = self.stream_idle {
            let limit = limit.min(self.deadline);
            let secs = limit.as_secs() + u64::from(limit.subsec_nanos() > 0);
            config.agent.stream_idle_timeout_secs = Some(secs.max(1));
        }
        config
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
    /// The longest silence it may keep between two chunks of a streamed
    /// reply; `None` keeps sven's default, the guard against a remote wire
    /// gone stale. A local model is silent through its prefill for as long
    /// as the prompt takes, so it carries a longer limit.
    pub stream_idle: Option<Duration>,
}

impl Model {
    /// `provider`, known as `identity`.
    #[must_use]
    pub fn new(provider: Arc<dyn ModelProvider>, identity: impl Into<String>) -> Self {
        Self {
            provider,
            identity: identity.into(),
            stream_idle: None,
        }
    }

    /// This model, allowed `limit` of silence between two stream chunks.
    #[must_use]
    pub fn with_stream_idle(mut self, limit: Duration) -> Self {
        self.stream_idle = Some(limit);
        self
    }
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model")
            .field("identity", &self.identity)
            .field("stream_idle", &self.stream_idle)
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

/// Introduces the material an open-book prompt shows before the
/// instruction.
pub const MATERIAL_HEADING: &str = "Reference material:";

/// The prompt of an open-book solve - a teacher's, shown a task's grounding
/// material, or an `ask` shown a source: [`MATERIAL_HEADING`], each piece of
/// `material` after a blank line, then a blank line and `instruction`, the
/// way a student view renders the privileged items it keeps. With no
/// material it is the instruction alone.
#[must_use]
pub fn open_book_prompt(instruction: &str, material: &[String]) -> String {
    if material.is_empty() {
        return instruction.to_string();
    }
    let mut prompt = String::from(MATERIAL_HEADING);
    for piece in material {
        prompt.push_str("\n\n");
        prompt.push_str(piece.trim_end());
    }
    prompt.push_str("\n\n");
    prompt.push_str(instruction);
    prompt
}

/// Solves `task` as [`solve`] does, prompting the solver with `prompt`
/// instead of the bare instruction: the instruction with teacher-only
/// material beside it, such as a critique of an earlier attempt or, for a
/// teacher's solve, the task's grounding material
/// ([`open_book_prompt`]). The trajectory records the prompt the solver was sent; a view
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
    let engine = engine(environment, model, options.engine_config())?;
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

/// The engine for `environment` under `config`: no built-in tools, and
/// only the environment's own tool.
pub(crate) fn engine(
    environment: &ResolvedEnvironment,
    model: Arc<dyn ModelProvider>,
    config: Config,
) -> Result<Engine, CallError> {
    let builder = Engine::builder()
        .config(config)
        .model_provider(UnderSystemPrompt::wrap(model))
        .toolset(Toolset::none());
    match environment {
        ResolvedEnvironment::ClosedBook => builder.build(),
        ResolvedEnvironment::Runtime(runtime) => builder
            .tool(Arc::new(RunCode::new(runtime.clone())))
            .build(),
    }
}
