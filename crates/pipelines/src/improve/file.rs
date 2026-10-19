// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the choice between retraining, a better
// harness, more search or no change at all, on measured cost-adjusted gain,
// for its clients. If your team needs expertise in deciding how an agent
// should be improved before spending a training budget, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The file an improvement experiment is written in.
//!
//! ```toml
//! capability = "repair failing tests in the date library"
//! tasks = "<task set id>"
//! incumbent = "policy:default"
//! attempts = 4                     # per task, for the incumbent and the arms
//!
//! [protocol]                       # the incumbent's episode protocol
//! follow_ups = 2
//! max_tool_calls = 60
//! max_repeated_results = 3
//!
//! [pricing]                        # seconds
//! value_per_success = 120
//! expected_uses = 5000
//! min_units = 4
//!
//! [[intervention]]
//! kind = "no-change"
//! [[intervention]]
//! kind = "samples"
//! samples = 8
//! [[intervention]]
//! kind = "protocol"
//! label = "more follow-ups"
//! follow_ups = 5
//! [[intervention]]
//! kind = "candidate"
//! label = "teacher demonstrations + sft"
//! candidate = "candidate-0193..."     # a kept candidate: its adapter, its cost from its run
//! [[intervention]]
//! kind = "candidate"
//! label = "an adapter from elsewhere"
//! model = "local:qwen3-8b+demo-sft"
//! one_time_seconds = 5400
//! [[intervention]]
//! kind = "unavailable"
//! label = "search"
//! reason = "the search facility has not landed"
//! ```

use serde::Deserialize;
use splinter_core::model_ref::ModelRef;
use splinter_eval::intervention::{Pricing, DEFAULT_CONFIDENCE_Z};
use splinter_store::tasks::TaskSetId;

use super::{training_seconds, Evaluation, ImprovementExperiment, InterventionSpec, ProtocolSpec};
use crate::curriculum::frontier::PassAtK;
use crate::release::arm;
use crate::train::load_candidate;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// The fewest units an arm must be measured on, when the file names none.
pub const DEFAULT_MIN_UNITS: usize = 4;

/// An experiment as the file states it.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentFile {
    capability: String,
    tasks: String,
    #[serde(default = "policy_default")]
    incumbent: String,
    attempts: usize,
    temperature: Option<f32>,
    top_k: Option<u32>,
    protocol: Option<ProtocolFile>,
    pricing: PricingFile,
    #[serde(rename = "intervention")]
    interventions: Vec<InterventionFile>,
}

fn policy_default() -> String {
    splinter_core::model_ref::POLICY_DEFAULT.into()
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProtocolFile {
    follow_ups: u32,
    max_tool_calls: Option<u32>,
    max_repeated_results: Option<u32>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PricingFile {
    value_per_success: f64,
    expected_uses: f64,
    confidence_z: Option<f64>,
    min_units: Option<usize>,
    budget: Option<f64>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum InterventionFile {
    NoChange,
    Samples {
        samples: usize,
    },
    Protocol {
        label: String,
        follow_ups: u32,
        max_tool_calls: Option<u32>,
        max_repeated_results: Option<u32>,
    },
    Candidate {
        label: String,
        /// A candidate `practice`, `learn` or `train` kept: its adapter is
        /// the model and the run that made it gives the cost.
        candidate: Option<String>,
        /// Else the model, and what training it cost.
        model: Option<String>,
        one_time_seconds: Option<f64>,
    },
    Unavailable {
        label: String,
        reason: String,
    },
}

/// The experiment `text` states, before its task set is resolved.
pub fn parse_experiment(text: &str) -> Result<ExperimentFile, OrchestratorError> {
    toml::from_str(text)
        .map_err(|e| OrchestratorError::Refused(format!("the improvement experiment: {e}")))
}

fn model(text: &str) -> Result<ModelRef, OrchestratorError> {
    text.parse()
        .map_err(|e| OrchestratorError::Refused(format!("model {text:?}: {e}")))
}

impl ExperimentFile {
    /// The id of the task set the file names, as written; the caller
    /// resolves it against the store.
    #[must_use]
    pub fn tasks(&self) -> &str {
        &self.tasks
    }

    /// The experiment, over the task set `tasks` that [`Self::tasks`] named.
    pub fn into_experiment(
        self,
        ctx: &Context,
        tasks: TaskSetId,
    ) -> Result<ImprovementExperiment, OrchestratorError> {
        let interventions = self
            .interventions
            .into_iter()
            .map(|i| {
                Ok(match i {
                    InterventionFile::NoChange => InterventionSpec::NoChange,
                    InterventionFile::Samples { samples } => {
                        InterventionSpec::MoreSamples { samples }
                    }
                    InterventionFile::Protocol {
                        label,
                        follow_ups,
                        max_tool_calls,
                        max_repeated_results,
                    } => InterventionSpec::Protocol {
                        label,
                        protocol: ProtocolSpec {
                            follow_ups,
                            max_tool_calls,
                            max_repeated_results,
                        },
                    },
                    InterventionFile::Candidate {
                        label,
                        candidate,
                        model: reference,
                        one_time_seconds,
                    } => match (candidate, reference, one_time_seconds) {
                        (Some(id), None, given) => {
                            let kept = load_candidate(ctx, &id)?;
                            let seconds = match given {
                                Some(seconds) => seconds,
                                None => {
                                    training_seconds(ctx, &kept.candidate)?.ok_or_else(|| {
                                        OrchestratorError::Refused(format!(
                                            "no run recorded what candidate {} cost to make: give \
                                         one_time_seconds",
                                            kept.candidate
                                        ))
                                    })?
                                }
                            };
                            InterventionSpec::Candidate {
                                label,
                                model: arm(ctx.config(), Some(&kept.adapter)),
                                one_time_seconds: seconds,
                            }
                        }
                        (None, Some(reference), Some(one_time_seconds)) => {
                            InterventionSpec::Candidate {
                                label,
                                model: model(&reference)?,
                                one_time_seconds,
                            }
                        }
                        _ => {
                            return Err(OrchestratorError::Refused(format!(
                                "intervention {label:?}: name a candidate, or a model with its \
                                 one_time_seconds"
                            )))
                        }
                    },
                    InterventionFile::Unavailable { label, reason } => {
                        InterventionSpec::Unavailable { label, reason }
                    }
                })
            })
            .collect::<Result<Vec<_>, OrchestratorError>>()?;
        let experiment = ImprovementExperiment {
            target_capability: self.capability,
            target_tasks: tasks,
            incumbent: model(&self.incumbent)?,
            evaluation: Evaluation {
                pass_at_k: PassAtK::new(self.attempts, self.temperature, self.top_k),
                protocol: self
                    .protocol
                    .map_or_else(ProtocolSpec::default, |p| ProtocolSpec {
                        follow_ups: p.follow_ups,
                        max_tool_calls: p.max_tool_calls,
                        max_repeated_results: p.max_repeated_results,
                    }),
            },
            interventions,
            pricing: Pricing {
                value_per_success: self.pricing.value_per_success,
                expected_uses: self.pricing.expected_uses,
                confidence_z: self.pricing.confidence_z.unwrap_or(DEFAULT_CONFIDENCE_Z),
                min_units: self.pricing.min_units.unwrap_or(DEFAULT_MIN_UNITS),
                budget: self.pricing.budget,
            },
        };
        experiment.validate()?;
        Ok(experiment)
    }
}
