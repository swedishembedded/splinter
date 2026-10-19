// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the choice between retraining, a better
// harness, more search or no change at all, on measured cost-adjusted gain,
// for its clients. If your team needs expertise in deciding how an agent
// should be improved before spending a training budget, you can procure our
// services by sending an email to info@swedishembedded.com.

//! `improve`: which way of improving an agent on a target capability is
//! worth its cost, decided from what each was measured to do.
//!
//! An [`ImprovementExperiment`] names the capability (a set of tasks whose
//! families are the units of evidence), the incumbent, the interventions to
//! consider, how each is evaluated and what a success is worth. Each
//! intervention is measured against the incumbent on the same tasks by an
//! [`ArmRunner`] ([`SolveArms`] measures by solving and verifying); the rule
//! in [`splinter_eval::intervention`] chooses by lower-bound gain net of cost,
//! or chooses to change nothing. This pipeline only decides: it does not
//! apply the choice, train, or move an alias.
//!
//! An intervention whose mechanism is not in the tree (a search policy until
//! sven's search facility lands, retrieval, a decision model) is declared
//! [`InterventionSpec::Unavailable`] with its reason and is reported as not
//! measured, never as a gain of zero.

mod arms;
mod file;

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use splinter_agent::work::WorkProtocol;
use splinter_core::model_ref::ModelRef;
use splinter_eval::intervention::{decide, ArmResult, Pricing, Verdict};
use splinter_store::experiences::SetId;
use splinter_store::tasks::TaskSetId;

use crate::curriculum::frontier::PassAtK;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_orchestrator::runs::{record, Recorded};

pub use arms::SolveArms;
pub use file::{parse_experiment, ExperimentFile};

/// A model reference as the text a command line names it by.
fn model_text<S: serde::Serializer>(model: &ModelRef, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.collect_str(model)
}

/// How an episode is run: the harness an arm may change.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtocolSpec {
    /// Follow-up turns after red checks.
    pub follow_ups: u32,
    /// Tool calls in one attempt; `None` sets no cap.
    pub max_tool_calls: Option<u32>,
    /// Repeated identical answers before an attempt is stopped; `None` sets
    /// no bound.
    pub max_repeated_results: Option<u32>,
}

impl Default for ProtocolSpec {
    fn default() -> Self {
        crate::solving::harness_protocol().into()
    }
}

impl From<WorkProtocol> for ProtocolSpec {
    fn from(work: WorkProtocol) -> Self {
        Self {
            follow_ups: work.follow_ups,
            max_tool_calls: work.max_tool_calls,
            max_repeated_results: work.max_repeated_results,
        }
    }
}

impl From<ProtocolSpec> for WorkProtocol {
    fn from(spec: ProtocolSpec) -> Self {
        Self {
            follow_ups: spec.follow_ups,
            max_tool_calls: spec.max_tool_calls,
            max_repeated_results: spec.max_repeated_results,
        }
    }
}

/// One way of improving the agent on the target capability.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InterventionSpec {
    /// Leave the incumbent as it is.
    NoChange,
    /// Ask for more samples of every task and keep the verified one.
    MoreSamples {
        /// Attempts per task.
        samples: usize,
    },
    /// The incumbent under another episode protocol: a different harness.
    Protocol {
        /// What the experiment calls it.
        label: String,
        /// The protocol.
        protocol: ProtocolSpec,
    },
    /// A model trained by an intervention - a teacher's demonstrations and
    /// supervised training, preference training, rehearsal - and the seconds
    /// it took to train.
    Candidate {
        /// What the experiment calls it.
        label: String,
        /// The model: `local:<base>+<adapter>` or a policy alias.
        #[serde(serialize_with = "model_text")]
        model: ModelRef,
        /// Seconds spent training it, from the run that trained it.
        one_time_seconds: f64,
    },
    /// An intervention whose mechanism is not available yet.
    Unavailable {
        /// What the experiment calls it.
        label: String,
        /// Why it cannot be measured.
        reason: String,
    },
}

impl InterventionSpec {
    /// The label the arm is reported under.
    #[must_use]
    pub fn label(&self) -> String {
        match self {
            Self::NoChange => "no-change".into(),
            Self::MoreSamples { samples } => format!("samples-{samples}"),
            Self::Protocol { label, .. }
            | Self::Candidate { label, .. }
            | Self::Unavailable { label, .. } => label.clone(),
        }
    }
}

/// How the arms and the incumbent are evaluated.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Evaluation {
    /// Attempts per task and how they are sampled: the same for the
    /// incumbent and for every arm that does not change them.
    pub pass_at_k: PassAtK,
    /// The incumbent's episode protocol.
    pub protocol: ProtocolSpec,
}

/// What an improvement experiment asks.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ImprovementExperiment {
    /// The capability to improve, as the people reading the report name it.
    pub target_capability: String,
    /// The tasks that stand for it; their families are the units.
    pub target_tasks: TaskSetId,
    /// The model to improve on.
    #[serde(serialize_with = "model_text")]
    pub incumbent: ModelRef,
    /// How everything is measured.
    pub evaluation: Evaluation,
    /// The interventions to consider.
    pub interventions: Vec<InterventionSpec>,
    /// What a success is worth and how sure an arm must be.
    pub pricing: Pricing,
}

impl ImprovementExperiment {
    /// Refuses an experiment that cannot be answered.
    pub fn validate(&self) -> Result<(), OrchestratorError> {
        self.evaluation.pass_at_k.validate()?;
        if self.interventions.is_empty() {
            return Err(OrchestratorError::Refused(
                "an improvement experiment names at least one intervention".into(),
            ));
        }
        let mut seen = BTreeSet::new();
        for intervention in &self.interventions {
            let label = intervention.label();
            if !seen.insert(label.clone()) {
                return Err(OrchestratorError::Refused(format!(
                    "two interventions are called {label:?}"
                )));
            }
            if let InterventionSpec::MoreSamples { samples } = intervention {
                if *samples < 2 {
                    return Err(OrchestratorError::Refused(format!(
                        "an arm of {samples} sample(s) is the incumbent"
                    )));
                }
            }
        }
        let pricing = &self.pricing;
        let positive = |v: f64| v.is_finite() && v > 0.0;
        if !(positive(pricing.value_per_success)
            && positive(pricing.expected_uses)
            && positive(pricing.confidence_z))
        {
            return Err(OrchestratorError::Refused(
                "the pricing's value of a success, expected uses and confidence must be positive \
                 numbers"
                    .into(),
            ));
        }
        Ok(())
    }
}

/// What it cost, in seconds, to make candidate `id`: the stages of the run
/// that trained it, up to and including the training - the tasks, the
/// teacher's demonstrations, the datasets and the training itself, not the
/// release gate. `None` when no recorded run trained a candidate of that id.
pub fn training_seconds(ctx: &Context, id: &str) -> Result<Option<f64>, OrchestratorError> {
    for run in splinter_store::runs::list_runs(ctx.workspace())? {
        let Some(train) = run
            .stages
            .iter()
            .position(|s| s.stage == "train" && s.summary["candidate"] == id)
        else {
            continue;
        };
        let timed: Option<Vec<u64>> = run.stages[..=train].iter().map(|s| s.duration_ms).collect();
        return Ok(timed.map(|ms| ms.iter().sum::<u64>() as f64 / 1000.0));
    }
    Ok(None)
}

/// What measuring one arm gave.
#[derive(Clone, Debug, PartialEq)]
pub struct Measured {
    /// Per family: the arm's pass rate minus the incumbent's.
    pub deltas: Vec<f64>,
    /// Seconds spent once to have the arm.
    pub one_time_cost: f64,
    /// Seconds the arm adds to each use.
    pub per_use_cost: f64,
    /// The arm's graded attempts, when it made any.
    pub experience_set: Option<SetId>,
    /// What a reader must know to read the number.
    pub note: Option<String>,
}

/// Measures an intervention against the incumbent on the experiment's
/// target tasks. A mechanism, not a decision: [`improve`] decides.
pub trait ArmRunner {
    /// Measures `spec`. An error is a failure to measure, not a result.
    fn measure(
        &mut self,
        experiment: &ImprovementExperiment,
        spec: &InterventionSpec,
    ) -> Result<Measured, OrchestratorError>;
}

/// One intervention as the report keeps it.
#[derive(Clone, Debug, Serialize)]
pub struct ArmEvidence {
    /// The intervention's label.
    pub label: String,
    /// What was measured, when something was.
    pub measured: Option<ArmResult>,
    /// The arm's attempts, when it made any.
    pub experience_set: Option<SetId>,
    /// What a reader must know to read it, or why it was not measured.
    pub note: Option<String>,
}

/// What `improve` reports.
#[derive(Clone, Debug, Serialize)]
pub struct ImprovementReport {
    /// The capability.
    pub target_capability: String,
    /// The tasks that stand for it.
    pub target_tasks: TaskSetId,
    /// The incumbent.
    pub incumbent: String,
    /// The pricing the decision was made under.
    pub pricing: Pricing,
    /// Each intervention, in the order asked.
    pub arms: Vec<ArmEvidence>,
    /// The decision and every measured arm's score.
    pub verdict: Verdict,
}

/// Measures every intervention of `experiment` with `runner` and chooses;
/// see the module documentation.
pub fn improve(
    ctx: &Context,
    experiment: &ImprovementExperiment,
    runner: &mut dyn ArmRunner,
) -> Result<Recorded<ImprovementReport>, OrchestratorError> {
    experiment.validate()?;
    record(ctx, "improve", experiment, |_run| {
        let mut arms = Vec::with_capacity(experiment.interventions.len());
        let mut results = Vec::new();
        for spec in &experiment.interventions {
            let label = spec.label();
            if let InterventionSpec::Unavailable { reason, .. } = spec {
                arms.push(ArmEvidence {
                    label,
                    measured: None,
                    experience_set: None,
                    note: Some(format!("not measured: {reason}")),
                });
                continue;
            }
            let measured = runner.measure(experiment, spec)?;
            let result = ArmResult {
                label: label.clone(),
                deltas: measured.deltas,
                one_time_cost: measured.one_time_cost,
                per_use_cost: measured.per_use_cost,
            };
            results.push(result.clone());
            arms.push(ArmEvidence {
                label,
                measured: Some(result),
                experience_set: measured.experience_set,
                note: measured.note,
            });
        }
        Ok(ImprovementReport {
            target_capability: experiment.target_capability.clone(),
            target_tasks: experiment.target_tasks.clone(),
            incumbent: experiment.incumbent.to_string(),
            pricing: experiment.pricing,
            arms,
            verdict: decide(&results, &experiment.pricing),
        })
    })
}

/// What one fixed strategy (always apply the same intervention) comes to
/// over the experiments it was measured in.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FixedStrategy {
    /// Experiments it was measured and eligible in.
    pub experiments: usize,
    /// Its net value over them, in seconds.
    pub net: f64,
    /// What the per-experiment choice came to over the same experiments.
    pub selected_net_on_these: f64,
}

/// The per-experiment choice against every fixed strategy, cost-adjusted.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FixedComparison {
    /// What choosing in each experiment came to over all of them: the net
    /// value of the arm chosen, or nothing for no change.
    pub selected_net: f64,
    /// Each intervention applied everywhere it was measured; one that was
    /// not measured in an experiment cannot be applied there and is compared
    /// only where it was.
    pub strategies: std::collections::BTreeMap<String, FixedStrategy>,
    /// Whether the choice is worth at least as much as every fixed strategy
    /// over the experiments the strategy was measured in.
    pub beats_every_fixed_strategy: bool,
}

/// The choices `reports` made against always applying one intervention.
#[must_use]
pub fn fixed_comparison(reports: &[&ImprovementReport]) -> FixedComparison {
    let chosen = |report: &ImprovementReport| match &report.verdict.decision {
        splinter_eval::intervention::Decision::Apply { net_value, .. } => *net_value,
        splinter_eval::intervention::Decision::NoChange { .. } => 0.0,
    };
    let mut strategies: std::collections::BTreeMap<String, FixedStrategy> =
        std::collections::BTreeMap::new();
    for report in reports {
        for score in &report.verdict.scores {
            let Some(net) = score.net_value else { continue };
            let entry = strategies
                .entry(score.label.clone())
                .or_insert(FixedStrategy {
                    experiments: 0,
                    net: 0.0,
                    selected_net_on_these: 0.0,
                });
            entry.experiments += 1;
            entry.net += net;
            entry.selected_net_on_these += chosen(report);
        }
    }
    FixedComparison {
        selected_net: reports.iter().map(|r| chosen(r)).sum(),
        beats_every_fixed_strategy: strategies
            .values()
            .all(|s| s.selected_net_on_these >= s.net),
        strategies,
    }
}
