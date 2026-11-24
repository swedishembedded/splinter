// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements closed-book measurement of what a model
// answers, for its clients. If your team needs expertise in measuring what a
// model has learned and kept, you can procure our services by sending an
// email to info@swedishembedded.com.

//! `eval --suite FILE`: one model graded closed-book on the tasks of a file.
//!
//! The file is in the anchor suite's format ([`anchor::read_tasks`]): every
//! task is closed-book and graded against its reference by code, so no judge
//! is needed. The model decodes greedily, as a release gate's arms do, and
//! is any model reference (`policy:<alias>` for what an alias serves). Every
//! item is kept with its answer and verdict, so the run can be regraded
//! without asking the model again; the evaluation is a recorded run.
//!
//! A factory run's candidates are evaluated by its `evaluation` stages, and
//! a release is decided by its gate; this measures one model on tasks the
//! caller chose, and decides nothing.

use std::path::PathBuf;

use serde::Serialize;
use splinter_agent::CancelToken;
use splinter_core::model_ref::ModelRef;
use splinter_eval::gate::SuiteSummary;
use splinter_eval::paired::accuracy;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_orchestrator::runs::{record, Recorded};

use crate::graded::GradedItem;
use crate::release::anchor;
use crate::release::probe::{self, Suite};

/// One `eval --suite`.
#[derive(Clone, Debug, Serialize)]
pub struct SuiteEval {
    /// The model graded.
    pub model: ModelRef,
    /// The file of anchor-format tasks.
    pub suite: PathBuf,
}

/// What `eval --suite` reports.
#[derive(Clone, Debug, Serialize)]
pub struct SuiteGraded {
    /// The model graded, as it was named.
    pub model: String,
    /// The suite.
    pub suite: SuiteSummary,
    /// Tasks a verifier decided.
    pub graded: usize,
    /// Of those, the fraction right; `None` when none was decided.
    pub accuracy: Option<f64>,
    /// Every task of the suite, in file order, with the answer given and how
    /// it was decided.
    pub items: Vec<GradedItem>,
}

/// Grades `request.model` on `request.suite`; see the module documentation.
///
/// # Errors
/// [`OrchestratorError::Refused`] for a file that is missing, holds a bad
/// line or holds no task, before the model is asked; the model's and the
/// store's own errors.
pub fn grade(
    ctx: &Context,
    request: &SuiteEval,
    cancel: &CancelToken,
) -> Result<SuiteGraded, OrchestratorError> {
    let tasks = anchor::read_tasks(&request.suite)?;
    let suite = Suite::of_tasks(request.suite.display().to_string(), tasks);
    let model = probe::greedy(ctx, &request.model)?;
    let probes = probe::grade(ctx, &model, &suite, cancel)?;
    let outcomes: Vec<Option<bool>> = probes.iter().map(|p| p.verdict).collect();
    let (accuracy, graded) = accuracy(&outcomes);
    let items = suite
        .tasks
        .iter()
        .zip(&probes)
        .map(|(task, probe)| GradedItem::keep(ctx, task, probe))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(SuiteGraded {
        model: request.model.to_string(),
        suite: suite.summary(),
        graded,
        accuracy,
        items,
    })
}

/// [`grade`] as a recorded `eval` run.
///
/// # Errors
/// As [`grade`]; the run store's own errors.
pub fn evaluate(
    ctx: &Context,
    request: &SuiteEval,
) -> Result<Recorded<SuiteGraded>, OrchestratorError> {
    record(ctx, "eval", request, |run| {
        grade(ctx, request, &run.cancel_token())
    })
}
