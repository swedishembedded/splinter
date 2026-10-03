// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that acquire a capability
// from a document or a tool and prove it with evidence. If your team needs
// expertise in agent infrastructure or small-model training loops, you can
// procure our services by sending an email to info@swedishembedded.com.

//! A pipeline: stages run in order over one shared state, and the engine
//! records the run.
//!
//! A [`Stage`] does one thing to the state and says what it did. Everything
//! about *running* stages is the engine's, so no stage has to remember it:
//!
//! * a cancel requested from any process stops the run before the next stage;
//! * a spent budget stops it before the next stage that honors one, saying
//!   which;
//! * a stage that does not apply to the state is skipped and not recorded;
//! * a stage that finds it cannot go on says so ([`Flow::Stop`]) and the run
//!   ends there, its own record kept;
//! * every stage that ran is recorded in the run, with how long it took, and
//!   with why it failed when it did.
//!
//! What the stages are and in which order is the pipeline's definition and
//! belongs to the crate that defines the pipeline.

use std::time::Instant;

use crate::context::Context;
use crate::error::OrchestratorError;
use crate::runs::Recorder;

/// Whether a pipeline goes on after a stage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Flow {
    /// On to the next stage.
    Continue,
    /// The run ends here, for the reason given: a gate that closed, not a
    /// failure.
    Stop(String),
}

/// What a stage reports when it ends.
#[derive(Clone, Debug)]
pub struct StageEnd {
    /// What the stage did, as the run records it; `None` when it did nothing
    /// worth a record, and so has none.
    pub summary: Option<serde_json::Value>,
    /// Whether the pipeline goes on.
    pub flow: Flow,
}

impl StageEnd {
    /// A stage that did its work, recorded as `summary`, and the pipeline
    /// goes on.
    #[must_use]
    pub fn done(summary: serde_json::Value) -> Self {
        Self {
            summary: Some(summary),
            flow: Flow::Continue,
        }
    }

    /// A stage that did its work, recorded as `summary`, and the pipeline
    /// ends here for `reason`.
    #[must_use]
    pub fn stop(summary: serde_json::Value, reason: impl Into<String>) -> Self {
        Self {
            summary: Some(summary),
            flow: Flow::Stop(reason.into()),
        }
    }

    /// A stage that found nothing to record, and the pipeline ends here for
    /// `reason`.
    #[must_use]
    pub fn halt(reason: impl Into<String>) -> Self {
        Self {
            summary: None,
            flow: Flow::Stop(reason.into()),
        }
    }
}

/// One step of a pipeline over the state `S`.
pub trait Stage<S> {
    /// The stage's name, as the run records it.
    fn name(&self) -> &'static str;

    /// Whether the stage applies to `state`; a stage that does not is
    /// skipped and not recorded.
    fn enabled(&self, _state: &S) -> bool {
        true
    }

    /// Whether a spent budget stops the pipeline before this stage. A stage
    /// that only reads what earlier ones left, or that must finish what was
    /// started, says no.
    fn honors_budget(&self) -> bool {
        true
    }

    /// Does the stage's work on `state`.
    fn run(
        &self,
        ctx: &Context,
        run: &mut Recorder<'_>,
        state: &mut S,
    ) -> Result<StageEnd, OrchestratorError>;
}

/// A stage's work as a function over the state.
pub type StageFn<S> =
    fn(&Context, &mut Recorder<'_>, &mut S) -> Result<StageEnd, OrchestratorError>;

/// A [`Stage`] made of functions: a pipeline defined as data.
pub struct FnStage<S> {
    name: &'static str,
    enabled: fn(&S) -> bool,
    honors_budget: bool,
    run: StageFn<S>,
}

impl<S> FnStage<S> {
    /// A stage named `name` that does `run`, applies to every state and
    /// honors the budget.
    #[must_use]
    pub fn new(name: &'static str, run: StageFn<S>) -> Self {
        Self {
            name,
            enabled: |_| true,
            honors_budget: true,
            run,
        }
    }

    /// This stage applying only where `enabled` says so.
    #[must_use]
    pub fn when(mut self, enabled: fn(&S) -> bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// This stage running whatever the budget.
    #[must_use]
    pub fn ignoring_budget(mut self) -> Self {
        self.honors_budget = false;
        self
    }
}

impl<S> Stage<S> for FnStage<S> {
    fn name(&self) -> &'static str {
        self.name
    }

    fn enabled(&self, state: &S) -> bool {
        (self.enabled)(state)
    }

    fn honors_budget(&self) -> bool {
        self.honors_budget
    }

    fn run(
        &self,
        ctx: &Context,
        run: &mut Recorder<'_>,
        state: &mut S,
    ) -> Result<StageEnd, OrchestratorError> {
        (self.run)(ctx, run, state)
    }
}

/// Stages, in order.
pub struct Pipeline<'a, S> {
    stages: Vec<Box<dyn Stage<S> + 'a>>,
}

impl<S> Default for Pipeline<'_, S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, S> Pipeline<'a, S> {
    /// A pipeline with no stage yet.
    #[must_use]
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// This pipeline followed by `stage`.
    #[must_use]
    pub fn then(mut self, stage: impl Stage<S> + 'a) -> Self {
        self.stages.push(Box::new(stage));
        self
    }

    /// The stages' names, in order.
    #[must_use]
    pub fn names(&self) -> Vec<&'static str> {
        self.stages.iter().map(|stage| stage.name()).collect()
    }

    /// Runs the stages over `state`, recording each in `run`. Ends with
    /// `Ok(None)` when every stage that applied ran, `Ok(Some(reason))` when
    /// the pipeline stopped (a stage said so, or `deadline` passed), and the
    /// error of a stage that failed or of a cancel.
    pub fn run(
        &self,
        ctx: &Context,
        run: &mut Recorder<'_>,
        state: &mut S,
        deadline: Option<Instant>,
    ) -> Result<Option<String>, OrchestratorError> {
        for stage in &self.stages {
            run.check_cancelled()?;
            if !stage.enabled(state) {
                continue;
            }
            if stage.honors_budget() && deadline.is_some_and(|d| Instant::now() >= d) {
                return Ok(Some(format!(
                    "the budget was spent before the {} stage",
                    stage.name()
                )));
            }
            let started = Instant::now();
            let ended = match stage.run(ctx, run, state) {
                Ok(ended) => ended,
                Err(e) => {
                    // The stage's error is the one worth reporting; a
                    // failure to record it would only hide it.
                    let _ = run.stage_measured(
                        stage.name(),
                        &serde_json::Value::Null,
                        Some(started.elapsed()),
                        Some(e.to_string()),
                    );
                    return Err(e);
                }
            };
            if let Some(summary) = &ended.summary {
                run.stage_measured(stage.name(), summary, Some(started.elapsed()), None)?;
            }
            if let Flow::Stop(reason) = ended.flow {
                return Ok(Some(reason));
            }
        }
        Ok(None)
    }
}
