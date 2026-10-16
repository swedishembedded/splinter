// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measurement of what a continually updated
// model keeps and what it loses, for its clients. If your team needs
// expertise in continual-learning evaluation for fine-tuned models, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The continual-learning metrics of the releases an alias descends from,
//! read from the gate reports the releases carry.
//!
//! Generation `k` is the `k`-th release of the lineage, oldest first, and
//! task `k` is that release's own held-out suite (a family of its own).
//! What the gate measured fills the matrix:
//!
//! * the improvement check scored release `k` on its own suite (`A[k][k]`)
//!   and its champion on the same suite (`A[k-1][k]`: what the model before
//!   knew of it, the cell forward transfer reads);
//! * the retention check scored release `k` on each earlier release's suite
//!   (`A[k][j]`, `j < k`).
//!
//! The untouched base model is measured only by the first release, whose
//! champion is the base; every later suite has no base score and so no
//! forward transfer or learning gain. That absence is reported, never
//! replaced by a zero. A release gated on claims or on a predictive model's
//! metrics has no graded-answer report and leaves its row unmeasured.

use serde::Serialize;
use splinter_core::release::ReleaseId;
use splinter_eval::continual::{
    ContinualMetrics, Interval, MatrixError, ResultsMatrix, TaskColumn,
};
use splinter_eval::gate::GateReport;
use splinter_model::stats::bootstrap_interval;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// The share of resampled means an interval over units holds.
const INTERVAL_LEVEL: f64 = 0.95;
/// Resamples an interval over units is made from.
const INTERVAL_RESAMPLES: usize = 4000;

/// What one release's gate measured, by release.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GenerationEvidence {
    /// The release.
    pub release: Option<ReleaseId>,
    /// The release's accuracy on its own suite; `None` when not measured.
    pub own: Option<f64>,
    /// The champion's accuracy on that suite.
    pub champion_on_own: Option<f64>,
    /// The release's accuracy on each earlier release's suite.
    pub retained: Vec<(ReleaseId, Option<f64>)>,
}

impl GenerationEvidence {
    /// What `report` measured for `release`.
    #[must_use]
    pub fn of(release: ReleaseId, report: &GateReport) -> Self {
        let improvement = report.improvement.measured.as_ref();
        Self {
            release: Some(release),
            own: improvement.and_then(|i| i.comparison.candidate_accuracy),
            champion_on_own: improvement.and_then(|i| i.comparison.baseline_accuracy),
            retained: report
                .retention
                .measured
                .iter()
                .flat_map(|r| &r.suites)
                .map(|s| (s.release.clone(), s.comparison.candidate_accuracy))
                .collect(),
        }
    }
}

/// The matrix `evidence` (oldest generation first) fills.
pub fn matrix_of(evidence: &[GenerationEvidence]) -> Result<ResultsMatrix, MatrixError> {
    let columns = evidence
        .iter()
        .enumerate()
        .map(|(k, e)| TaskColumn {
            id: e
                .release
                .as_ref()
                .map_or_else(|| k.to_string(), ToString::to_string),
            unit: format!("suite-{k}"),
            learned_at: k,
            // Only the first release's champion is the untouched base.
            base: if k == 0 { e.champion_on_own } else { None },
        })
        .collect();
    let mut matrix = ResultsMatrix::new(evidence.len(), columns)?;
    for (k, e) in evidence.iter().enumerate() {
        if let Some(v) = e.own {
            matrix.set(k, k, v)?;
        }
        if let (Some(v), true) = (e.champion_on_own, k > 0) {
            matrix.set(k - 1, k, v)?;
        }
        for (suite, v) in &e.retained {
            let earlier = evidence[..k]
                .iter()
                .position(|g| g.release.as_ref() == Some(suite));
            if let (Some(j), Some(v)) = (earlier, v) {
                matrix.set(k, j, *v)?;
            }
        }
    }
    Ok(matrix)
}

/// What `status` and the JSON report say of continual learning.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ContinualStatus {
    /// Measured over the lineage.
    Measured {
        /// The releases, oldest first: generation `k` is entry `k`.
        releases: Vec<ReleaseId>,
        /// The metrics.
        metrics: Box<ContinualMetrics>,
    },
    /// Nothing to measure yet.
    NotMeasured {
        /// Why.
        reason: String,
    },
}

/// A seed derived from the lineage, so that the same releases give the same
/// intervals (FNV-1a over the newest release's id).
fn lineage_seed(newest: &ReleaseId) -> u64 {
    newest
        .to_string()
        .bytes()
        .fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
        })
}

/// The releases `alias` descends from, oldest first, and the matrix their
/// gate reports fill; `Err` carries why there is no lineage to read.
struct Lineage {
    head: ReleaseId,
    releases: Vec<ReleaseId>,
    matrix: ResultsMatrix,
}

fn read(ctx: &Context, alias: &str) -> Result<Result<Lineage, String>, OrchestratorError> {
    let store = ctx.releases();
    let Some(head) = store.alias(alias)? else {
        return Ok(Err(format!("alias {alias} points at no release")));
    };
    let mut lineage = store.lineage(&head)?;
    lineage.reverse();
    let evidence: Vec<GenerationEvidence> = lineage
        .iter()
        .map(|r| match r.manifest.gate.llm() {
            Some(report) => GenerationEvidence::of(r.id.clone(), report),
            None => GenerationEvidence {
                release: Some(r.id.clone()),
                ..GenerationEvidence::default()
            },
        })
        .collect();
    let matrix = matrix_of(&evidence).map_err(|e| {
        OrchestratorError::Refused(format!(
            "the gate reports of {alias}'s lineage disagree: {e}"
        ))
    })?;
    Ok(Ok(Lineage {
        head,
        releases: lineage.into_iter().map(|r| r.id).collect(),
        matrix,
    }))
}

/// The metrics of `matrix`, with intervals over units resampled from a seed
/// derived from `head`.
fn metrics_of(matrix: &ResultsMatrix, head: &ReleaseId) -> ContinualMetrics {
    let seed = lineage_seed(head);
    ContinualMetrics::of(matrix).with_intervals(|values| {
        bootstrap_interval(values, INTERVAL_RESAMPLES, INTERVAL_LEVEL, seed).map(|i| Interval {
            level: INTERVAL_LEVEL,
            low: i.low,
            high: i.high,
        })
    })
}

/// The metrics of the lineage `alias` points at.
pub fn continual(ctx: &Context, alias: &str) -> Result<ContinualStatus, OrchestratorError> {
    let lineage = match read(ctx, alias)? {
        Ok(lineage) if lineage.releases.len() >= 2 => lineage,
        Ok(_) => {
            return Ok(ContinualStatus::NotMeasured {
                reason: "fewer than two releases: nothing was learned after the first".into(),
            })
        }
        Err(reason) => return Ok(ContinualStatus::NotMeasured { reason }),
    };
    let metrics = metrics_of(&lineage.matrix, &lineage.head);
    Ok(ContinualStatus::Measured {
        releases: lineage.releases,
        metrics: Box::new(metrics),
    })
}

/// The results matrix of a lineage and the metrics read from it: what
/// `eval matrix` prints, every cell the gates measured and the rest absent.
#[derive(Clone, Debug, Serialize)]
pub struct MatrixReport {
    /// The alias whose lineage it is.
    pub alias: String,
    /// The releases, oldest first: generation `k` is entry `k`, and so is
    /// task `k` (that release's own held-out suite).
    pub releases: Vec<ReleaseId>,
    /// `A[k][i]`.
    pub matrix: ResultsMatrix,
    /// The metrics over it.
    pub metrics: ContinualMetrics,
}

/// The matrix of the lineage `alias` points at; refused when it points at
/// no release.
pub fn matrix(ctx: &Context, alias: &str) -> Result<MatrixReport, OrchestratorError> {
    let lineage = read(ctx, alias)?.map_err(OrchestratorError::Refused)?;
    Ok(MatrixReport {
        alias: alias.to_string(),
        metrics: metrics_of(&lineage.matrix, &lineage.head),
        releases: lineage.releases,
        matrix: lineage.matrix,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use splinter_core::digest::Digest;

    fn id(n: u8) -> ReleaseId {
        ReleaseId(Digest::of(&[n]))
    }

    fn evidence(n: u8, own: f64, champion: f64, retained: &[(u8, f64)]) -> GenerationEvidence {
        GenerationEvidence {
            release: Some(id(n)),
            own: Some(own),
            champion_on_own: Some(champion),
            retained: retained.iter().map(|(j, v)| (id(*j), Some(*v))).collect(),
        }
    }

    #[test]
    fn gate_numbers_fill_the_cells_they_measured_and_no_others() {
        let matrix = matrix_of(&[
            evidence(0, 0.6, 0.4, &[]),
            evidence(1, 0.7, 0.2, &[(0, 0.5)]),
        ])
        .unwrap();
        let m = ContinualMetrics::of(&matrix);
        let value = |m: &splinter_eval::continual::Measure| match m {
            splinter_eval::continual::Measure::Measured { value, .. } => *value,
            other => panic!("{other:?}"),
        };
        assert!((value(&m.acc) - 0.6).abs() < 1e-12); // (0.5 + 0.7) / 2
        assert!((value(&m.bwt) + 0.1).abs() < 1e-12); // 0.5 - 0.6
                                                      // Release 0's champion was the base (0.4): its gain is 0.2. Release 1
                                                      // has no base score, so forward transfer is not measured.
        assert!((value(&m.learning_gain) - 0.2).abs() < 1e-12);
        assert!(matches!(
            m.forward_transfer,
            splinter_eval::continual::Measure::NotMeasured { .. }
        ));
    }
}
