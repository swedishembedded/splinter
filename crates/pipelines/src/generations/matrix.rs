// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements generations of a code-editing agent that
// keep what earlier generations learned, for its clients. If your team needs
// expertise in measuring retention across successive fine-tunes of an
// agent, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The full retention matrix of a coding lineage: `A[k][i]`, generation `k`
//! on task `i`, every cell measured.
//!
//! The matrix the gates fill ([`crate::release::continual`]) holds the cells
//! a release's own gate measured: its exam, and the exams of its ancestors.
//! Here every generation, the base before any training included, is measured
//! on every protected task ([`super::panel`]), so the cells the gates never
//! look at - an earlier generation on a later exam - are there too, and
//! forward transfer and the gain over the base have something to rest on.
//! A task is learned at the generation that trained on the repository its
//! family belongs to; its unit of evidence is its family within that
//! generation's exam.

use splinter_eval::continual::{
    ContinualMetrics, Interval, MatrixError, ResultsMatrix, TaskColumn,
};
use splinter_model::stats::bootstrap_interval;

use serde::Serialize;
use splinter_agent::CancelToken;
use splinter_core::digest::Digest;
use splinter_core::release::ReleaseId;
use splinter_eval::gate::AgenticConfig;

use super::panel::{measure, Panel, PanelRow};
use crate::release::arm;
use crate::release::probe::Arm;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

const INTERVAL_LEVEL: f64 = 0.95;
const INTERVAL_RESAMPLES: usize = 4000;

/// The matrix of `rows` over the tasks of `panel`: row `i` is generation `i`
/// (row 0 the base), a task absent from a row is an unmeasured cell, and the
/// base's score on a task is the one forward transfer is measured against.
pub fn matrix_of(panel: &Panel, rows: &[PanelRow]) -> Result<ResultsMatrix, OrchestratorError> {
    for (i, row) in rows.iter().enumerate() {
        if row.generation != i {
            return Err(OrchestratorError::Refused(format!(
                "row {i} of the matrix is generation {}: the rows are the generations, in order",
                row.generation
            )));
        }
    }
    let base = rows.first();
    let verdict = |row: &PanelRow, column: usize, task: &str| -> Option<f64> {
        row.cells
            .iter()
            .find(|c| c.column == column && c.task.to_string() == task)
            .and_then(|c| c.verdict)
            .map(f64::from)
    };
    let tasks: Vec<(usize, TaskColumn)> = panel
        .tasks()
        .map(|(column, task)| {
            let id = task.task.id.to_string();
            let family = task.concepts.first().cloned().unwrap_or_else(|| id.clone());
            let learned_at = panel.columns[column].learned_at;
            (
                column,
                TaskColumn {
                    base: base.and_then(|b| verdict(b, column, &id)),
                    id,
                    // The same file in another generation's snapshot is
                    // another population of tasks: a unit of its own.
                    unit: format!("{family}@{learned_at}"),
                    learned_at,
                },
            )
        })
        .collect();
    let mut matrix = ResultsMatrix::new(
        rows.len().max(1),
        tasks.iter().map(|(_, t)| t.clone()).collect(),
    )
    .map_err(matrix_error)?;
    for (i, row) in rows.iter().enumerate() {
        for (t, (column, task)) in tasks.iter().enumerate() {
            if let Some(v) = verdict(row, *column, &task.id) {
                matrix.set(i, t, v).map_err(matrix_error)?;
            }
        }
    }
    Ok(matrix)
}

fn matrix_error(e: MatrixError) -> OrchestratorError {
    OrchestratorError::Refused(format!("the retention matrix: {e}"))
}

/// The metrics of `matrix`, intervals over families resampled from `seed`.
#[must_use]
pub fn metrics(matrix: &ResultsMatrix, seed: u64) -> ContinualMetrics {
    ContinualMetrics::of(matrix).with_intervals(|values| {
        bootstrap_interval(values, INTERVAL_RESAMPLES, INTERVAL_LEVEL, seed).map(|i| Interval {
            level: INTERVAL_LEVEL,
            low: i.low,
            high: i.high,
        })
    })
}

/// Every generation of a lineage measured on its whole panel, and the matrix
/// and metrics they make.
#[derive(Clone, Debug, Serialize)]
pub struct RetentionMatrix {
    /// The releases, oldest first: generation `k` is entry `k - 1`, and
    /// generation 0 is the base.
    pub releases: Vec<ReleaseId>,
    /// The panel's columns, by name.
    pub columns: Vec<String>,
    /// Each generation's verdict on every protected task.
    pub rows: Vec<PanelRow>,
    /// `A[k][i]`.
    pub matrix: ResultsMatrix,
    /// The metrics over it.
    pub metrics: ContinualMetrics,
}

/// The base and every release `alias` descends from, measured on the exam of
/// each of them under `work`: the whole matrix, the cells the gates never
/// look at (an earlier generation on a later exam) included. A measurement
/// is an agent's attempt at every protected task per generation, which is
/// what a full matrix costs; none is cached across calls.
pub fn retention_matrix(
    ctx: &Context,
    alias: &str,
    work: AgenticConfig,
    cancel: &CancelToken,
) -> Result<RetentionMatrix, OrchestratorError> {
    let store = ctx.releases();
    let head = store
        .alias(alias)?
        .ok_or_else(|| OrchestratorError::Refused(format!("alias {alias} points at no release")))?;
    let mut lineage = store.lineage(&head)?;
    lineage.reverse();
    let panel = Panel::of_lineage(ctx, alias)?;
    if panel.columns.is_empty() {
        return Err(OrchestratorError::Refused(format!(
            "no release of {alias}'s lineage has an exam to measure on: `practice` reserves one"
        )));
    }
    let mut rows = Vec::with_capacity(lineage.len() + 1);
    let base = arm(ctx.config(), None);
    let model = ctx.model(&base)?;
    rows.push(measure(
        ctx,
        &Arm::named(&base, &model),
        (0, None),
        &panel,
        work,
        cancel,
    )?);
    for (k, release) in lineage.iter().enumerate() {
        let reference = arm(ctx.config(), Some(release.adapter()?));
        let model = ctx.model(&reference)?;
        rows.push(measure(
            ctx,
            &Arm::named(&reference, &model),
            (k + 1, Some(release.id.clone())),
            &panel,
            work,
            cancel,
        )?);
    }
    let matrix = matrix_of(&panel, &rows)?;
    let seed = {
        let digest = Digest::of(head.to_string().as_bytes());
        u64::from_str_radix(&digest.hex()[..16], 16).unwrap_or_default()
    };
    Ok(RetentionMatrix {
        releases: lineage.iter().map(|r| r.id.clone()).collect(),
        columns: panel.columns.iter().map(|c| c.name.clone()).collect(),
        metrics: metrics(&matrix, seed),
        rows,
        matrix,
    })
}
