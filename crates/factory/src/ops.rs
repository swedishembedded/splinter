// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements data and learning pipelines built from typed,
// reusable operations instead of one-off scripts, for its clients. If your
// team needs expertise in reproducible machine-learning data factories, you
// can procure our services by sending an email to info@swedishembedded.com.

//! The factory's built-in operations: their contracts and typed parameters.
//!
//! | operation | inputs | output |
//! |---|---|---|
//! | `tasks.pin` | `collection` (files) | `tasks` |
//! | `attempt.collect` | `tasks`, `unsolved_by`, `solved_by` (attempts, optional, many), `briefs` (optional) | `attempts` |
//! | `attempt.import` | `tasks`, `jobs` (files, many) | `attempts` |
//! | `dataset.import` | `records` (files) | `dataset` |
//! | `dataset.project` | `attempts` (many) | `dataset` |
//! | `dataset.extend` | `base`, `candidates` (datasets) | `dataset` |
//! | `model.train` | `dataset` | `model` |
//! | `evaluation` | `candidate` (model) | `evaluation` |
//! | `promotion` | `evaluations` (many) | `release` |
//! | `agent` | `input` (files, many) | `output` ([`crate::agent_op`]) |
//! | `attempts.failures` | `tasks`, `attempts` (many) | `failures` ([`crate::failures_op`]) |
//! | `harness` | `input` (files, many, optional) | `output` ([`crate::harness_op`]) |
//!
//! Each operation's parameters are one struct that refuses unknown fields;
//! the descriptor's parameter schema is generated from that struct, so the
//! contract and the check never disagree. A field that names a model names
//! a role of the recipe's `models:`.

use std::collections::BTreeMap;

use schemars::{schema_for, JsonSchema};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use splinter_core::factory::execution::Isolation;
use splinter_core::factory::operation::{
    ArtifactKind, OperationDescriptor, Port, Reproducibility, OPERATION_SCHEMA,
};
use splinter_core::factory::trial::{RunBudget, TimeoutPolicy};

use crate::registry::{OperationSpec, Registry};

/// `tasks.pin`: the tasks of a collection, pinned into the run.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PinParams {
    /// Only the tasks of this split.
    #[serde(default)]
    pub split: Option<String>,
}

/// What a hinted pass shows the model beside the task.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum Hint {
    /// The task's reference solution (`solution/solve.sh`).
    Solution,
    /// The task's brief in the stage's `briefs` input: knowledge, a plan of
    /// subgoals, a teacher's notes ([`crate::hint`]).
    Brief,
}

fn one() -> u32 {
    1
}

fn two() -> u32 {
    2
}

/// `attempt.collect`: agents attempt tasks, every attempt graded.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AttemptParams {
    /// The agent adapter.
    pub agent: String,
    /// The agent's own arguments, passed as they are.
    #[serde(default)]
    pub args: Vec<String>,
    /// Files the arguments refer to, pinned into the run.
    #[serde(default)]
    pub files: Vec<String>,
    /// The model role, unless the agent brings its own model.
    #[serde(default)]
    pub model: Option<String>,
    /// A system prompt file appended to the agent's own.
    #[serde(default)]
    pub system_prompt: Option<String>,
    /// Environment variables for the agent: a literal, or `env:NAME`.
    #[serde(default)]
    pub environment_variables: BTreeMap<String, String>,
    /// The environment file.
    pub environment: String,
    /// Only these task ids.
    #[serde(default)]
    pub tasks: Vec<String>,
    /// Attempts per task.
    #[serde(default = "one")]
    pub attempts: u32,
    /// Attempts run at once.
    #[serde(default = "one")]
    pub concurrency: u32,
    /// The per-trial budget.
    #[serde(default)]
    pub budget: Option<RunBudget>,
    /// The timeout policy of its tasks.
    #[serde(default)]
    pub timeout_policy: TimeoutPolicy,
    /// Extra physical attempts a slot may take when the infrastructure
    /// decided its attempt.
    #[serde(default = "two")]
    pub retries: u32,
    /// The seed attempt seeds are drawn from.
    pub seed: u64,
    /// Played by people (`splinter play`), never run unattended.
    #[serde(default)]
    pub interactive: bool,
    /// A hinted pass ([`crate::hint`]).
    #[serde(default)]
    pub hint: Option<Hint>,
    /// A sven harness every attempt runs in place of the agent's own
    /// conversation, given the task's instruction as its `task` input.
    #[serde(default)]
    pub harness: Option<String>,
}

/// `attempt.import`: the trials of Harbor jobs run elsewhere, as attempts
/// of the pinned tasks, running nothing ([`crate::run::import`]).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ImportParams {
    /// The model the jobs' trials ran on, recorded on each trial as given.
    /// Not a role: nothing is served, and Harbor records only the agent's
    /// own model name, which may be a gateway's alias. Without it, Harbor's
    /// `provider/name`.
    #[serde(default)]
    pub model_label: Option<String>,
    /// Only these task ids.
    #[serde(default)]
    pub tasks: Vec<String>,
}

/// `dataset.import`: training records made elsewhere, as a dataset
/// ([`crate::run::dataset_import`]).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ImportDatasetParams {
    /// The model role the records are for.
    pub target: String,
    /// The longest record, in the target's tokens, when bounded: a longer
    /// one is excluded and listed.
    #[serde(default)]
    pub max_block: Option<u64>,
    /// The records keep the target's reasoning (it is trained to reason
    /// before it answers); off: no-think mode, and a record that carries
    /// reasoning is refused.
    #[serde(default)]
    pub thinking: bool,
}

/// `dataset.extend`: new records admitted into an existing dataset only as
/// new solutions of their task ([`crate::run::dataset_extend`]).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExtendDatasetParams {
    /// The model role the dataset is for.
    pub target: String,
    /// The most records a task has in the result, the base's included.
    #[serde(default = "four")]
    pub per_task: usize,
    /// Argument names left out when two solutions' tool calls are compared:
    /// the free text a call carries beside what it does (a stated intent).
    #[serde(default)]
    pub ignore_call_arguments: Vec<String>,
    /// The share, in percent, of the tasks new to the base that are held out
    /// (as whole tasks). A task the base already has stays on its side.
    #[serde(default = "ten")]
    pub holdout_percent: u8,
    /// The seed the held-out draw of a new task is made by; keep the base's.
    pub holdout_seed: u64,
}

fn four() -> usize {
    4
}
fn ten() -> u8 {
    10
}

fn sft() -> String {
    "sft".into()
}
fn verified_success() -> String {
    "verified-success".into()
}
fn native_sft() -> String {
    "native-sft".into()
}
fn by_task() -> String {
    "task".into()
}

/// How validation records are held out.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ValidationSplit {
    /// Group records by this before splitting (`task`, `task_family`).
    #[serde(default = "by_task")]
    pub group_by: String,
    /// The fraction of groups held out, in percent.
    pub percent: u32,
}

/// `dataset.project`: training records projected from attempts.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ProjectParams {
    /// The objective.
    #[serde(default = "sft")]
    pub objective: String,
    /// The admission rule.
    #[serde(default = "verified_success")]
    pub admission: String,
    /// The projection.
    #[serde(default = "native_sft")]
    pub projection: String,
    /// The model role the records are for.
    pub target: String,
    /// The longest record, in the target's tokens, when bounded.
    #[serde(default)]
    pub max_block: Option<u64>,
    /// Train the target to reason before it answers (off: no-think mode).
    #[serde(default)]
    pub thinking: bool,
    /// How validation records are held out.
    #[serde(default)]
    pub validation: Option<ValidationSplit>,
    /// The split's seed.
    pub seed: u64,
}

fn brain() -> String {
    "brain".into()
}

/// `model.train`: one candidate trained from a base.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TrainParams {
    /// The training engine.
    #[serde(default = "brain")]
    pub engine: String,
    /// The base model's role.
    pub base: String,
    /// The method (`lora`, `full`).
    pub method: String,
    /// The trainer's own parameters, validated by the trainer.
    #[serde(default)]
    pub params: Value,
    /// The training seed.
    pub seed: u64,
}

/// `evaluation`: a candidate against a baseline under an evaluation plan.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EvaluationParams {
    /// The evaluation plan file.
    pub plan: String,
    /// The agent every arm runs with.
    pub agent: String,
    /// The agent's arguments, identical for every arm.
    #[serde(default)]
    pub args: Vec<String>,
    /// Files the arguments refer to, pinned into the run and given to every
    /// arm's agent as an attempt stage's are.
    #[serde(default)]
    pub files: Vec<String>,
    /// A system prompt file appended to the agent's own, the same for every
    /// arm: a candidate trained on records that carry it is served with it.
    #[serde(default)]
    pub system_prompt: Option<String>,
    /// The per-trial budget of every arm.
    #[serde(default)]
    pub budget: Option<RunBudget>,
    /// The baseline model's role.
    pub baseline: String,
    /// The comparisons the plan's significance level is divided among (the
    /// candidates of one experiment); the plan's own setting, else one.
    #[serde(default)]
    pub comparisons: Option<u32>,
}

/// `promotion`: what the run decides about its candidates.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PromotionParams {
    /// Promote the best improved candidate.
    #[serde(default)]
    pub promote: bool,
}

/// The operation names.
pub mod names {
    /// `tasks.pin`.
    pub const PIN: &str = "tasks.pin";
    /// `attempt.collect`.
    pub const ATTEMPTS: &str = "attempt.collect";
    /// `attempt.import`.
    pub const IMPORT: &str = "attempt.import";
    /// `dataset.import`.
    pub const IMPORT_DATASET: &str = "dataset.import";
    /// `dataset.project`.
    pub const PROJECT: &str = "dataset.project";
    /// `dataset.extend`.
    pub const EXTEND_DATASET: &str = "dataset.extend";
    /// `model.train`.
    pub const TRAIN: &str = "model.train";
    /// `evaluation`.
    pub const EVALUATION: &str = "evaluation";
    /// `promotion`.
    pub const PROMOTION: &str = "promotion";
}

/// `params` as `P`, or why not.
///
/// # Errors
/// An unknown, missing or ill-typed parameter.
pub fn params<P: DeserializeOwned>(params: &Value) -> Result<P, String> {
    serde_json::from_value(params.clone()).map_err(|e| e.to_string())
}

fn port(name: &str, kind: ArtifactKind) -> Port {
    Port {
        name: name.into(),
        kind,
        many: false,
        optional: false,
    }
}

/// A built-in operation: its contract, and its parameters' type.
struct Builtin<P> {
    name: &'static str,
    summary: &'static str,
    inputs: Vec<Port>,
    output: Port,
    reproducibility: Reproducibility,
    isolation: Isolation,
    roles: fn(&P) -> Vec<String>,
}

impl<P: JsonSchema + DeserializeOwned + Send + Sync> OperationSpec for Builtin<P> {
    fn descriptor(&self) -> OperationDescriptor {
        OperationDescriptor {
            schema: OPERATION_SCHEMA.into(),
            name: self.name.into(),
            revision: 1,
            summary: self.summary.into(),
            inputs: self.inputs.clone(),
            outputs: vec![self.output.clone()],
            parameters: serde_json::to_value(schema_for!(P)).unwrap_or(Value::Null),
            reproducibility: self.reproducibility,
            isolation: self.isolation,
        }
    }

    fn check(&self, value: &Value) -> Result<(), String> {
        params::<P>(value).map(|_| ())
    }

    fn model_roles(&self, value: &Value) -> Vec<String> {
        params::<P>(value)
            .map(|p| (self.roles)(&p))
            .unwrap_or_default()
    }
}

/// Every built-in operation.
#[must_use]
pub fn registry() -> Registry {
    Registry::default()
        .with(crate::agent_op::Agent)
        .with(crate::failures_op::Failures)
        .with(crate::harness_op::HarnessOp)
        .with(crate::knowledge_ops::Ingest)
        .with(crate::knowledge_ops::Admit)
        .with(crate::task_ops::Generate)
        .with(crate::task_ops::Answer)
        .with(crate::env_ops::Build)
        .with(Builtin::<PinParams> {
            name: names::PIN,
            summary: "pin a collection's tasks (one split) into the run",
            inputs: vec![port("collection", ArtifactKind::Files)],
            output: port("tasks", ArtifactKind::Tasks),
            reproducibility: Reproducibility::Deterministic,
            isolation: Isolation::Any,
            roles: |_| Vec::new(),
        })
        .with(Builtin::<AttemptParams> {
            name: names::ATTEMPTS,
            summary: "agents attempt tasks; every attempt graded by the one classifier",
            inputs: vec![
                port("tasks", ArtifactKind::Tasks),
                Port {
                    many: true,
                    optional: true,
                    ..port("unsolved_by", ArtifactKind::Attempts)
                },
                Port {
                    many: true,
                    optional: true,
                    ..port("solved_by", ArtifactKind::Attempts)
                },
                Port {
                    optional: true,
                    ..port("briefs", ArtifactKind::Files)
                },
                Port {
                    optional: true,
                    ..port("continue_from", ArtifactKind::Attempts)
                },
            ],
            output: port("attempts", ArtifactKind::Attempts),
            reproducibility: Reproducibility::Nondeterministic,
            isolation: Isolation::Any,
            roles: |p| p.model.iter().cloned().collect(),
        })
        .with(Builtin::<ImportParams> {
            name: names::IMPORT,
            summary: "the trials of Harbor jobs run elsewhere, matched to the pinned tasks and \
                      graded by the one classifier; runs nothing",
            inputs: vec![
                port("tasks", ArtifactKind::Tasks),
                Port {
                    many: true,
                    ..port("jobs", ArtifactKind::Files)
                },
            ],
            output: port("attempts", ArtifactKind::Attempts),
            reproducibility: Reproducibility::Deterministic,
            isolation: Isolation::Any,
            roles: |_| Vec::new(),
        })
        .with(Builtin::<ImportDatasetParams> {
            name: names::IMPORT_DATASET,
            summary: "training records made elsewhere (a directory of train.jsonl and \
                      validation.jsonl), carried over unchanged as a dataset",
            inputs: vec![port("records", ArtifactKind::Files)],
            output: port("dataset", ArtifactKind::Dataset),
            reproducibility: Reproducibility::Deterministic,
            isolation: Isolation::Any,
            roles: |p| vec![p.target.clone()],
        })
        .with(Builtin::<ExtendDatasetParams> {
            name: names::EXTEND_DATASET,
            summary: "new verified records added to an existing dataset only when they are a \
                      new solution of their task, a task's side never changing",
            inputs: vec![
                port("base", ArtifactKind::Dataset),
                port("candidates", ArtifactKind::Dataset),
            ],
            output: port("dataset", ArtifactKind::Dataset),
            reproducibility: Reproducibility::Deterministic,
            isolation: Isolation::Any,
            roles: |p| vec![p.target.clone()],
        })
        .with(Builtin::<ProjectParams> {
            name: names::PROJECT,
            summary: "project attempts into training records",
            inputs: vec![Port {
                many: true,
                ..port("attempts", ArtifactKind::Attempts)
            }],
            output: port("dataset", ArtifactKind::Dataset),
            reproducibility: Reproducibility::Seeded,
            isolation: Isolation::Any,
            roles: |p| vec![p.target.clone()],
        })
        .with(Builtin::<TrainParams> {
            name: names::TRAIN,
            summary: "train one candidate from a base through the training engine",
            inputs: vec![port("dataset", ArtifactKind::Dataset)],
            output: port("model", ArtifactKind::Model),
            reproducibility: Reproducibility::Seeded,
            isolation: Isolation::Any,
            roles: |p| vec![p.base.clone()],
        })
        .with(Builtin::<EvaluationParams> {
            name: names::EVALUATION,
            summary: "evaluate a candidate against a baseline under an evaluation plan",
            inputs: vec![
                port("candidate", ArtifactKind::Model),
                Port {
                    optional: true,
                    ..port("tasks", ArtifactKind::Tasks)
                },
            ],
            output: port("evaluation", ArtifactKind::Evaluation),
            reproducibility: Reproducibility::Nondeterministic,
            isolation: Isolation::Any,
            roles: |p| vec![p.baseline.clone()],
        })
        .with(Builtin::<PromotionParams> {
            name: names::PROMOTION,
            summary: "record every decision and promote the best improved candidate",
            inputs: vec![Port {
                many: true,
                ..port("evaluations", ArtifactKind::Evaluation)
            }],
            output: port("release", ArtifactKind::Release),
            reproducibility: Reproducibility::Deterministic,
            isolation: Isolation::Any,
            roles: |_| Vec::new(),
        })
}
