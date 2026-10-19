// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements generations of a code-editing agent that
// keep what earlier generations learned, for its clients. If your team needs
// expertise in measuring retention across successive fine-tunes of an
// agent, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The panel of protected tasks, and every generation measured on it.
//!
//! A [`Panel`] is the held-out exams of the generations so far, one
//! [`PanelColumn`] per generation: the families `practice` reserved before
//! it made a single rollout and never trained on, in that run or any later
//! one. Measuring a model on the panel ([`measure`]) attempts every task as
//! an agent under the same bounds and seeds for every model, graded by the
//! workspace verifier and the exposure audit, and keeps each task's verdict
//! as a [`Cell`]: right, wrong, or absent when the harness could not carry
//! the attempt out or no majority formed. Nothing a measurement produces is
//! stored as an experience, so none of it can reach training data.

use serde::{Deserialize, Serialize};
use splinter_agent::CancelToken;
use splinter_core::digest::Digest;
use splinter_core::experience::Task;
use splinter_core::release::ReleaseId;
use splinter_eval::gate::AgenticConfig;

use crate::release::exams;
use crate::release::probe::{grade_on, Arm, Suite};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// The tasks one generation's exam holds.
#[derive(Clone, Debug)]
pub struct PanelColumn {
    /// What the column is: the exam of generation `learned_at`.
    pub name: String,
    /// The generation that trained on the repository this exam belongs to,
    /// counted from 1 (generation 0 is the base before any training).
    pub learned_at: usize,
    /// The tasks, each a change to a repository.
    pub tasks: Vec<Task>,
}

/// The protected tasks, by generation.
#[derive(Clone, Debug, Default)]
pub struct Panel {
    /// One column per generation that reserved an exam, in order.
    pub columns: Vec<PanelColumn>,
}

impl Panel {
    /// The panel of the releases `alias` descends from, oldest first: the
    /// exam reserved against each release's datasets. A release no
    /// `practice` reserved an exam for has no column.
    pub fn of_lineage(ctx: &Context, alias: &str) -> Result<Self, OrchestratorError> {
        let store = ctx.releases();
        let Some(head) = store.alias(alias)? else {
            return Ok(Self::default());
        };
        let mut lineage = store.lineage(&head)?;
        lineage.reverse();
        let mut panel = Self::default();
        // A task is in the column of the first release whose datasets name its
        // exam: a rebuild's datasets include its ancestors', whose exams are
        // not protected twice.
        let mut seen: std::collections::BTreeSet<Digest> = std::collections::BTreeSet::new();
        for (k, release) in lineage.iter().enumerate() {
            let split = exams::split(ctx, &release.manifest.datasets)?;
            let mut tasks = Vec::new();
            for exam in split.exams {
                let Some(set) = exam.tasks else { continue };
                for entry in ctx.tasks().get_set(&set)?.members {
                    let task = ctx.tasks().get(&entry.task)?;
                    if seen.insert(task.task.id.clone()) {
                        tasks.push(task);
                    }
                }
            }
            if !tasks.is_empty() {
                panel.columns.push(PanelColumn {
                    name: format!("exam of generation {} ({})", k + 1, release.id),
                    learned_at: k + 1,
                    tasks,
                });
            }
        }
        Ok(panel)
    }

    /// This panel with `column` after the others.
    #[must_use]
    pub fn with(mut self, column: PanelColumn) -> Self {
        self.columns.push(column);
        self
    }

    /// Every task of the panel, with the index of its column.
    pub fn tasks(&self) -> impl Iterator<Item = (usize, &Task)> {
        self.columns
            .iter()
            .enumerate()
            .flat_map(|(c, column)| column.tasks.iter().map(move |t| (c, t)))
    }
}

/// How one model fared on one task.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cell {
    /// The task.
    pub task: Digest,
    /// Its family: the file it asks to change.
    pub family: String,
    /// The column of the panel it is in.
    pub column: usize,
    /// Right or wrong by a strict majority of the graded attempts; `None`
    /// when no attempt was graded or none formed a majority.
    pub verdict: Option<bool>,
}

/// One model measured on the whole panel.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PanelRow {
    /// The generation the model is: 0 is the base.
    pub generation: usize,
    /// What it was loaded from.
    pub model: String,
    /// The release it is, when it is one.
    pub release: Option<ReleaseId>,
    /// Its verdict on every task of the panel, in the panel's order.
    pub cells: Vec<Cell>,
}

impl PanelRow {
    /// The pass rate over the cells of `column` that have a verdict, and
    /// how many; `None` rate with none.
    #[must_use]
    pub fn rate(&self, column: usize) -> (Option<f64>, usize) {
        rate_of(self.cells.iter().filter(|c| c.column == column))
    }

    /// The pass rate of each family over its cells that have a verdict,
    /// with how many; a family with none is absent.
    #[must_use]
    pub fn by_family(&self) -> std::collections::BTreeMap<String, (f64, usize)> {
        let mut families: std::collections::BTreeMap<&str, Vec<&Cell>> =
            std::collections::BTreeMap::new();
        for cell in &self.cells {
            families.entry(&cell.family).or_default().push(cell);
        }
        families
            .into_iter()
            .filter_map(|(family, cells)| {
                let (rate, n) = rate_of(cells.into_iter());
                rate.map(|r| (family.to_string(), (r, n)))
            })
            .collect()
    }
}

fn rate_of<'a>(cells: impl Iterator<Item = &'a Cell>) -> (Option<f64>, usize) {
    let verdicts: Vec<bool> = cells.filter_map(|c| c.verdict).collect();
    if verdicts.is_empty() {
        return (None, 0);
    }
    let passed = verdicts.iter().filter(|v| **v).count();
    (Some(passed as f64 / verdicts.len() as f64), verdicts.len())
}

/// `arm` measured on `panel` as generation `generation`, every task
/// attempted under `work`.
pub fn measure(
    ctx: &Context,
    arm: &Arm<'_>,
    who: (usize, Option<ReleaseId>),
    panel: &Panel,
    work: AgenticConfig,
    cancel: &CancelToken,
) -> Result<PanelRow, OrchestratorError> {
    let (generation, release) = who;
    let mut cells = Vec::new();
    for (c, column) in panel.columns.iter().enumerate() {
        let suite = Suite::workspace(column.name.clone(), column.tasks.clone()).with_work(work);
        let probes = grade_on(ctx, arm, &suite, cancel)?;
        for (task, probe) in suite.tasks.iter().zip(probes) {
            cells.push(Cell {
                task: task.task.id.clone(),
                family: task
                    .concepts
                    .first()
                    .cloned()
                    .unwrap_or_else(|| task.task.id.to_string()),
                column: c,
                verdict: probe.verdict,
            });
        }
    }
    Ok(PanelRow {
        generation,
        model: arm
            .reference
            .map_or_else(|| arm.model.identity.clone(), ToString::to_string),
        release,
        cells,
    })
}
