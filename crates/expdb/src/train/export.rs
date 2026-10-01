// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Samples with their references resolved to text, and their export.

use std::io::Write;

use serde::Serialize;

use super::plan::{Sample, SampleBody, TrainingPlan};
use crate::error::{Error, Result};
use crate::id::ContentId;
use crate::manifest::Snapshot;

/// One rollout of a group, as text.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MaterializedRollout {
    /// The attempt's whole path.
    pub response: String,
    /// Its reward.
    pub reward: f64,
    /// Its reward relative to the group.
    pub advantage: f64,
}

/// A sample with its references resolved to text.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Materialized {
    /// Imitate `target` given `context`.
    Sft {
        /// What came before.
        context: String,
        /// What to produce.
        target: String,
        /// How much the sample counts.
        weight: f64,
    },
    /// Prefer `chosen` over `rejected` given `context`.
    Preference {
        /// The shared context.
        context: String,
        /// The better decision.
        chosen: String,
        /// The worse decision.
        rejected: String,
        /// How much better, in reward.
        reward_gap: f64,
    },
    /// A group of rollouts of one task.
    Group {
        /// The task.
        context: String,
        /// The rollouts.
        rollouts: Vec<MaterializedRollout>,
    },
    /// One step of an environment.
    Transition {
        /// What came before.
        context: String,
        /// What was done.
        action: String,
        /// The world before.
        state: ContentId,
        /// The world after.
        next_state: ContentId,
        /// The policy's log-probability of the action, if recorded.
        old_logprob: Option<f64>,
        /// The policy's value estimate, if recorded.
        value_estimate: Option<f64>,
        /// The reward, if one was given.
        reward: Option<f64>,
        /// Whether the episode ended here.
        done: bool,
    },
    /// A score for one step.
    StepLabel {
        /// What came before.
        context: String,
        /// The step scored.
        step: String,
        /// The score.
        score: f64,
        /// How sure the evaluator is.
        confidence: f64,
    },
}

impl Snapshot {
    /// Resolves a sample's references to text.
    pub fn materialize(&self, sample: &Sample) -> Result<Materialized> {
        Ok(match &sample.body {
            SampleBody::Sft {
                context,
                target,
                weight,
            } => Materialized::Sft {
                context: self.render(context)?,
                target: self.render(target)?,
                weight: *weight,
            },
            SampleBody::Preference {
                context,
                chosen,
                rejected,
                reward_gap,
            } => Materialized::Preference {
                context: self.render(context)?,
                chosen: self.render(chosen)?,
                rejected: self.render(rejected)?,
                reward_gap: *reward_gap,
            },
            SampleBody::Group { context, rollouts } => Materialized::Group {
                context: self.render(context)?,
                rollouts: rollouts
                    .iter()
                    .map(|r| {
                        Ok(MaterializedRollout {
                            response: self.render(&r.response)?,
                            reward: r.reward,
                            advantage: r.advantage,
                        })
                    })
                    .collect::<Result<_>>()?,
            },
            SampleBody::Transition {
                context,
                action,
                state,
                next_state,
                old_logprob,
                value_estimate,
                reward,
                done,
            } => Materialized::Transition {
                context: self.render(context)?,
                action: self.render(action)?,
                state: *state,
                next_state: *next_state,
                old_logprob: *old_logprob,
                value_estimate: *value_estimate,
                reward: *reward,
                done: *done,
            },
            SampleBody::StepLabel {
                context,
                step,
                score,
                confidence,
            } => Materialized::StepLabel {
                context: self.render(context)?,
                step: self.render(step)?,
                score: *score,
                confidence: *confidence,
            },
        })
    }
}

impl TrainingPlan {
    /// Writes the plan as JSON lines: a projection for tools that want a file.
    /// The plan and its snapshot remain the dataset.
    pub fn export_jsonl(&self, snapshot: &Snapshot, out: &mut impl Write) -> Result<()> {
        for sample in &self.samples {
            let line = serde_json::to_string(&snapshot.materialize(sample)?).map_err(|source| {
                Error::Encode {
                    what: "exported sample",
                    source,
                }
            })?;
            writeln!(out, "{line}").map_err(Error::io("the export"))?;
        }
        Ok(())
    }
}
