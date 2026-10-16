// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measurement of what a continually updated
// model keeps and what it loses, for its clients. If your team needs
// expertise in continual-learning evaluation for fine-tuned models, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Continual-learning metrics over a results matrix. Pure: no model, no files.
//!
//! `A[k][i]` is the performance on task `i` after learning through
//! generation `k` (a pass rate in `[0, 1]`). A cell nobody measured is
//! absent, never `0`: every metric below skips it and says how many tasks
//! and units it rests on, and a metric with nothing to rest on is
//! [`Measure::NotMeasured`]. Task `i` was learned at generation
//! [`TaskColumn::learned_at`]; the last generation is `K - 1`. Each task
//! also carries the score of the untouched base model on it, when that was
//! measured.
//!
//! The unit of evidence is the family (the task's [`TaskColumn::unit`]), not
//! the item: a metric is the mean over units of the unit's mean over its
//! tasks, and its interval resamples units.
//!
//! * **ACC**: the average final performance, `mean_i A[K-1][i]`.
//! * **BWT** (backward transfer): `mean_i (A[K-1][i] - A[g_i][i])` over tasks
//!   learned before the last generation, `g_i = learned_at`. Negative when
//!   learning later things cost earlier ones.
//! * **Forgetting**: `mean_i (max_{g_i <= k < K-1} A[k][i] - A[K-1][i])` over
//!   tasks learned before the last generation. The maximum is taken over
//!   the generations from the one that learned the task up to the one before
//!   the last, so a task that peaked after its learning step counts the full
//!   fall from the peak; it is never negative.
//! * **Forward transfer**: `mean_i (A[g_i - 1][i] - base[i])` over tasks
//!   learned after generation 0: what the model knew of a task before
//!   learning it, against the untouched base model (not a control trained
//!   fresh).
//! * **Learning gain**: `mean_i (A[g_i][i] - base[i])`, what learning a task
//!   added over the base at the step that learned it; the **gain curve** is
//!   the same over everything learned through generation `k`, scored at
//!   generation `k`, for each `k`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Why a matrix could not be built.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum MatrixError {
    /// A task was learned at a generation the matrix does not have.
    #[error("task {task} was learned at generation {learned_at}, but there are {generations}")]
    LearnedAfterEnd {
        /// The task.
        task: String,
        /// Its generation.
        learned_at: usize,
        /// Generations in the matrix.
        generations: usize,
    },
    /// A cell lies outside the matrix.
    #[error("cell ({generation}, {task}) lies outside the matrix")]
    OutOfRange {
        /// The generation index given.
        generation: usize,
        /// The task index given.
        task: usize,
    },
    /// A score is not a pass rate.
    #[error("{what} is {value}, not a pass rate in [0, 1]")]
    NotARate {
        /// What was scored.
        what: String,
        /// The value.
        value: f64,
    },
}

/// One task of the matrix.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TaskColumn {
    /// What the task is.
    pub id: String,
    /// The family the task belongs to: the unit of evidence.
    pub unit: String,
    /// The generation that learned it.
    pub learned_at: usize,
    /// The untouched base model's score on it; `None` when not measured.
    pub base: Option<f64>,
}

/// `A[k][i]` with unmeasured cells absent.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResultsMatrix {
    generations: usize,
    tasks: Vec<TaskColumn>,
    cells: Vec<Vec<Option<f64>>>,
}

fn rate(what: impl Into<String>, value: f64) -> Result<f64, MatrixError> {
    if (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err(MatrixError::NotARate {
            what: what.into(),
            value,
        })
    }
}

impl ResultsMatrix {
    /// An empty matrix of `generations` rows over `tasks`.
    pub fn new(generations: usize, tasks: Vec<TaskColumn>) -> Result<Self, MatrixError> {
        for task in &tasks {
            if task.learned_at >= generations {
                return Err(MatrixError::LearnedAfterEnd {
                    task: task.id.clone(),
                    learned_at: task.learned_at,
                    generations,
                });
            }
            if let Some(base) = task.base {
                rate(format!("the base score on {}", task.id), base)?;
            }
        }
        let cells = vec![vec![None; tasks.len()]; generations];
        Ok(Self {
            generations,
            tasks,
            cells,
        })
    }

    /// Records the score on task `task` after generation `generation`.
    pub fn set(&mut self, generation: usize, task: usize, value: f64) -> Result<(), MatrixError> {
        let Some(column) = self
            .tasks
            .get(task)
            .filter(|_| generation < self.generations)
        else {
            return Err(MatrixError::OutOfRange { generation, task });
        };
        let value = rate(
            format!("the score on {} at generation {generation}", column.id),
            value,
        )?;
        self.cells[generation][task] = Some(value);
        Ok(())
    }

    /// Makes a cell unmeasured again.
    pub fn clear(&mut self, generation: usize, task: usize) {
        if let Some(cell) = self.cells.get_mut(generation).and_then(|r| r.get_mut(task)) {
            *cell = None;
        }
    }

    /// The score on task `task` after generation `generation`; `None` when
    /// it was not measured (or the cell lies outside the matrix).
    #[must_use]
    pub fn cell(&self, generation: usize, task: usize) -> Option<f64> {
        self.cells.get(generation)?.get(task).copied().flatten()
    }

    /// Generations (rows).
    #[must_use]
    pub fn generations(&self) -> usize {
        self.generations
    }

    /// The tasks (columns), in order.
    #[must_use]
    pub fn tasks(&self) -> &[TaskColumn] {
        &self.tasks
    }
}

/// A percentile interval over units.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Interval {
    /// The share of resampled values it holds.
    pub level: f64,
    /// Its lower end.
    pub low: f64,
    /// Its upper end.
    pub high: f64,
}

/// One metric: a number and what it rests on, or a statement that it was
/// not measured.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Measure {
    /// Measured over at least one task.
    Measured {
        /// The mean over units of the unit's mean.
        value: f64,
        /// Tasks it rests on.
        tasks: usize,
        /// Units (families) it rests on: the n that an interval is about.
        units: usize,
        /// Each unit's value, so that an interval can be made from them.
        unit_values: Vec<f64>,
        /// The interval over units; `None` where too few units to make one.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        interval: Option<Interval>,
    },
    /// Nothing the metric needs was measured.
    NotMeasured {
        /// What was missing.
        reason: String,
    },
}

impl Measure {
    fn of(per_task: Vec<(&str, f64)>, missing: &str) -> Self {
        if per_task.is_empty() {
            return Self::NotMeasured {
                reason: missing.into(),
            };
        }
        let mut by_unit: BTreeMap<&str, (f64, usize)> = BTreeMap::new();
        for (unit, v) in &per_task {
            let e = by_unit.entry(unit).or_default();
            e.0 += v;
            e.1 += 1;
        }
        let unit_values: Vec<f64> = by_unit.values().map(|(s, n)| s / *n as f64).collect();
        Self::Measured {
            value: unit_values.iter().sum::<f64>() / unit_values.len() as f64,
            tasks: per_task.len(),
            units: unit_values.len(),
            unit_values,
            interval: None,
        }
    }

    fn with_interval(self, interval: &dyn Fn(&[f64]) -> Option<Interval>) -> Self {
        match self {
            Self::Measured {
                value,
                tasks,
                units,
                unit_values,
                ..
            } => {
                let interval = interval(&unit_values);
                Self::Measured {
                    value,
                    tasks,
                    units,
                    unit_values,
                    interval,
                }
            }
            not @ Self::NotMeasured { .. } => not,
        }
    }
}

/// The learning-gain curve's value after one generation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CurvePoint {
    /// The generation.
    pub generation: usize,
    /// The mean gain over the base on everything learned through it.
    pub gain: Measure,
}

/// Every metric of one matrix.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContinualMetrics {
    /// Generations in the matrix.
    pub generations: usize,
    /// Average final performance.
    pub acc: Measure,
    /// Backward transfer.
    pub bwt: Measure,
    /// Forgetting, from the best score since learning.
    pub forgetting: Measure,
    /// Forward transfer against the untouched base.
    pub forward_transfer: Measure,
    /// Gain over the base at the step that learned each task.
    pub learning_gain: Measure,
    /// The gain over the base after each generation.
    pub gain_curve: Vec<CurvePoint>,
}

impl ContinualMetrics {
    /// The metrics of `m`; see the module documentation for each.
    #[must_use]
    pub fn of(m: &ResultsMatrix) -> Self {
        let last = m.generations.saturating_sub(1);
        let tasks = || m.tasks.iter().enumerate();
        let acc = Measure::of(
            tasks()
                .filter_map(|(i, t)| Some((t.unit.as_str(), m.cell(last, i)?)))
                .collect(),
            "no task has a final score",
        );
        let bwt = Measure::of(
            tasks()
                .filter(|(_, t)| t.learned_at < last)
                .filter_map(|(i, t)| {
                    Some((t.unit.as_str(), m.cell(last, i)? - m.cell(t.learned_at, i)?))
                })
                .collect(),
            "no task learned before the last generation has both its score after learning and its final score",
        );
        let forgetting = Measure::of(
            tasks()
                .filter(|(_, t)| t.learned_at < last)
                .filter_map(|(i, t)| {
                    let peak = (t.learned_at..last)
                        .filter_map(|k| m.cell(k, i))
                        .max_by(f64::total_cmp)?;
                    Some((t.unit.as_str(), peak - m.cell(last, i)?))
                })
                .collect(),
            "no task learned before the last generation has a score since learning and a final score",
        );
        let forward_transfer = Measure::of(
            tasks()
                .filter(|(_, t)| t.learned_at > 0)
                .filter_map(|(i, t)| {
                    Some((t.unit.as_str(), m.cell(t.learned_at - 1, i)? - t.base?))
                })
                .collect(),
            "no task learned after generation 0 has a score before learning and a base score",
        );
        let learning_gain = Measure::of(
            tasks()
                .filter_map(|(i, t)| Some((t.unit.as_str(), m.cell(t.learned_at, i)? - t.base?)))
                .collect(),
            "no task has both a score after learning and a base score",
        );
        let gain_curve = (0..m.generations)
            .map(|k| CurvePoint {
                generation: k,
                gain: Measure::of(
                    tasks()
                        .filter(|(_, t)| t.learned_at <= k)
                        .filter_map(|(i, t)| Some((t.unit.as_str(), m.cell(k, i)? - t.base?)))
                        .collect(),
                    "no task learned by this generation has both a score and a base score",
                ),
            })
            .collect();
        Self {
            generations: m.generations,
            acc,
            bwt,
            forgetting,
            forward_transfer,
            learning_gain,
            gain_curve,
        }
    }

    /// These metrics with an interval over units made by `interval` for each
    /// measured one (including the curve's).
    #[must_use]
    pub fn with_intervals(self, interval: impl Fn(&[f64]) -> Option<Interval>) -> Self {
        let f: &dyn Fn(&[f64]) -> Option<Interval> = &interval;
        Self {
            generations: self.generations,
            acc: self.acc.with_interval(f),
            bwt: self.bwt.with_interval(f),
            forgetting: self.forgetting.with_interval(f),
            forward_transfer: self.forward_transfer.with_interval(f),
            learning_gain: self.learning_gain.with_interval(f),
            gain_curve: self
                .gain_curve
                .into_iter()
                .map(|p| CurvePoint {
                    generation: p.generation,
                    gain: p.gain.with_interval(f),
                })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests;
