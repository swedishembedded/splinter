// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements generations of a code-editing agent that
// keep what earlier generations learned, for its clients. If your team needs
// expertise in measuring retention across successive fine-tunes of an
// agent, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Generations of a coding policy: gains that persist.
//!
//! `practice` trains one generation on one repository and puts it through the
//! release gate. A continual run is several of them, and what it must show is
//! that each generation kept what the ones before it held. This module is
//! that, on the coding path the facts path ([`crate::absorb`]) already has:
//!
//! * [`panel`] - the protected tasks (the exam of every generation so far,
//!   families never trained on) and a model measured on all of them;
//! * [`matrix`] - the full retention matrix over generations and protected
//!   tasks, the base before any training as generation 0, with the metrics
//!   of [`splinter_eval::continual`];
//! * [`run`] - one generation: the frozen incumbent measured on the panel, a
//!   candidate trained on the new admitted experience (with a replay policy
//!   chosen from [`run::ReplayChoice`]), both measured on the panel and the
//!   new exam at equal inference settings, rejection on retention and
//!   regression limits, release through the gate, and a ledger entry with
//!   everything the decision rested on.
//!
//! The release gate stays the only thing that moves an alias; this module
//! decides whether a candidate may go to it, and records why.

pub mod matrix;
pub mod panel;
pub mod replay_weights;
pub mod run;

pub use matrix::{matrix_of, metrics, retention_matrix, RetentionMatrix};
pub use panel::{measure, Cell, Panel, PanelColumn, PanelRow};
pub use run::{
    assess, read_ledger, run_generation, Assessment, FamilyChange, GenerationRecord,
    GenerationReport, GenerationRequest, Limits, Outcome, ReplayChoice,
};

/// An experiment's ledger beside the full matrix of the lineage it made.
#[derive(Clone, Debug, serde::Serialize)]
pub struct SeriesReport {
    /// The experiment.
    pub experiment: String,
    /// Generations the ledger holds that were accepted.
    pub accepted: usize,
    /// Every generation, accepted or not, with what it was decided on.
    pub ledger: Vec<GenerationRecord>,
    /// The matrix of the lineage `alias` points at, when it could be made.
    pub matrix: Option<RetentionMatrix>,
    /// Why there is none.
    pub matrix_unavailable: Option<String>,
}

/// The ledger of `experiment` and the retention matrix of the lineage
/// `alias` points at, measured under `work`.
pub fn series(
    ctx: &splinter_orchestrator::context::Context,
    experiment: &str,
    alias: &str,
    work: splinter_eval::gate::AgenticConfig,
    cancel: &splinter_agent::CancelToken,
) -> Result<SeriesReport, splinter_orchestrator::error::OrchestratorError> {
    let ledger = read_ledger(ctx, experiment)?;
    let accepted = ledger
        .iter()
        .filter(|g| matches!(g.outcome, Outcome::Accepted { .. }))
        .count();
    let (matrix, matrix_unavailable) = match retention_matrix(ctx, alias, work, cancel) {
        Ok(matrix) => (Some(matrix), None),
        Err(e) if e.is_refusal() => (None, Some(e.to_string())),
        Err(e) => return Err(e),
    };
    Ok(SeriesReport {
        experiment: experiment.to_string(),
        accepted,
        ledger,
        matrix,
        matrix_unavailable,
    })
}
