// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements declarative learning experiments that take
// a model from agent rollouts to an evaluated candidate, for its clients. If
// your team needs expertise in model fine-tuning pipelines, you can procure
// our services by sending an email to info@swedishembedded.com.

//! A recipe: the human-edited YAML file that says what an experiment does.
//!
//! ```yaml
//! schema: splinter.factory/v1
//! name: vim-mastery
//! tasks: { collection: collections/vim-mastery.yaml, split: train }
//! rollouts:
//!   - { name: local, agent: sven, model: "brain:Qwen/Qwen3-4B",
//!       environment: environments/vim/environment.yaml, attempts: 4 }
//!   - { name: remote, agent: sven, model: "hosted:deepseek/deepseek-v4.1-flash",
//!       environment: environments/vim/environment.yaml,
//!       tasks: { unsolved_by: [local] }, attempts: 4 }
//! dataset: { name: sft, sources: [local, remote], target: "brain:Qwen/Qwen3-4B" }
//! training:
//!   base: "brain:Qwen/Qwen3-4B"
//!   variants: [{ name: lora-r16, method: lora, params: { rank: 16 } }]
//! eval: { plan: eval/vim-generalization.yaml, agent: sven,
//!         baseline: "brain:Qwen/Qwen3-4B", candidates: [lora-r16] }
//! ```
//!
//! Every path is relative to the recipe's repository root (the directory the
//! recipe is run from, normally the task repository). Loading checks the
//! shape (unknown keys are refused, so a typo is never silently ignored);
//! [`Recipe::validate`] checks the references between sections.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use splinter_core::factory::trial::{RunBudget, TimeoutPolicy};

/// The schema every recipe names.
pub const RECIPE_SCHEMA: &str = "splinter.factory/v1";

/// A model named the way a recipe names one: `connection:model`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModelRef {
    /// The connection in the connections file (`brain`, `hosted`).
    pub connection: String,
    /// The model id the connection's endpoint serves.
    pub model: String,
}

impl FromStr for ModelRef {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text.split_once(':') {
            Some((connection, model))
                if !connection.is_empty()
                    && !model.is_empty()
                    && connection
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') =>
            {
                Ok(Self {
                    connection: connection.into(),
                    model: model.into(),
                })
            }
            _ => Err(format!(
                "{text:?} is not a model reference of the form connection:model"
            )),
        }
    }
}

impl fmt::Display for ModelRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.connection, self.model)
    }
}

impl Serialize for ModelRef {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for ModelRef {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        String::deserialize(deserializer)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}

/// Which tasks a section takes.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub struct TaskSelector {
    /// Only the tasks with no verified success in any of these passes,
    /// frozen when the pass starts.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unsolved_by: Vec<String>,
    /// Only these task ids.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ids: Vec<String>,
}

fn all_tasks<'de, D: Deserializer<'de>>(deserializer: D) -> Result<TaskSelector, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Form {
        Word(String),
        Selector(TaskSelector),
    }
    match Form::deserialize(deserializer)? {
        Form::Word(word) if word == "all" => Ok(TaskSelector::default()),
        Form::Word(word) => Err(serde::de::Error::custom(format!(
            "tasks: {word:?} is neither `all` nor a selector (unsolved_by, ids)"
        ))),
        Form::Selector(selector) => Ok(selector),
    }
}

/// The collection a recipe runs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TasksSection {
    /// The collection file.
    pub collection: String,
    /// Only the tasks of this split.
    #[serde(default)]
    pub split: Option<String>,
}

/// One environment variable a recipe gives an agent: a literal, or
/// `env:NAME`, the worker's own variable `NAME`.
pub type EnvSetting = String;

/// One rollout pass.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RolloutPass {
    /// The pass's name, unique in the recipe.
    pub name: String,
    /// The agent adapter (`sven`, `claude-code`, `codex`).
    pub agent: String,
    /// The agent's own arguments, passed as they are.
    #[serde(default)]
    pub args: Vec<String>,
    /// Files the arguments refer to, pinned into the run.
    #[serde(default)]
    pub files: Vec<String>,
    /// The model, unless the agent brings its own.
    #[serde(default)]
    pub model: Option<ModelRef>,
    /// A system prompt file appended to the agent's own.
    #[serde(default)]
    pub system_prompt: Option<String>,
    /// Environment variables for the agent.
    #[serde(default)]
    pub environment_variables: BTreeMap<String, EnvSetting>,
    /// The environment file.
    pub environment: String,
    /// Which tasks.
    #[serde(default, deserialize_with = "all_tasks")]
    pub tasks: TaskSelector,
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
    /// Extra physical attempts a slot may take when the infrastructure (a
    /// sandbox, the model endpoint, the verifier) decided its attempt.
    #[serde(default = "two")]
    pub retries: u32,
    /// The pass's seed; attempt seeds are drawn from it.
    pub seed: u64,
    /// An interactive pass: nothing runs it unattended; its trials are the
    /// sessions people play with `splinter play`, each graded like any
    /// trial and marked guided.
    #[serde(default)]
    pub interactive: bool,
    /// A hinted pass: each task is worked from a copy whose instruction
    /// carries the hint ([`crate::hint`]); its verified successes are
    /// learned under the task's own instruction.
    #[serde(default)]
    pub hint: Option<Hint>,
}

/// What a hinted pass shows the model beside the task.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Hint {
    /// The task's reference solution (`solution/solve.sh`).
    Solution,
}

fn one() -> u32 {
    1
}

fn two() -> u32 {
    2
}

/// The dataset section.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetSection {
    /// The dataset's name.
    pub name: String,
    /// The rollout passes it reads.
    pub sources: Vec<String>,
    /// The objective.
    #[serde(default = "sft")]
    pub objective: String,
    /// The admission rule.
    #[serde(default = "verified_success")]
    pub admission: String,
    /// The projection.
    #[serde(default = "native_sft")]
    pub projection: String,
    /// The model the records are for.
    pub target: ModelRef,
    /// The longest record, in the target's tokens, when bounded.
    #[serde(default)]
    pub max_block: Option<u64>,
    /// Train the target to reason before it answers. Off (the default), the
    /// target is trained for no-think mode: the reasoning a teacher's replies
    /// carry is removed from every turn, and the trainer is told the same.
    #[serde(default)]
    pub thinking: bool,
    /// How validation records are held out.
    #[serde(default)]
    pub validation: Option<ValidationSplit>,
    /// The split's seed.
    pub seed: u64,
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

/// How validation records are held out.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValidationSplit {
    /// Group records by this before splitting (`task`, `task_family`).
    #[serde(default = "by_task")]
    pub group_by: String,
    /// The fraction of groups held out, in percent.
    pub percent: u32,
}

fn by_task() -> String {
    "task".into()
}

/// One training variant.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingVariant {
    /// The variant's name, unique in the recipe.
    pub name: String,
    /// The method (`lora`, `full`).
    pub method: String,
    /// The trainer's own parameters, passed as they are and validated by
    /// the trainer.
    #[serde(default)]
    pub params: serde_json::Value,
    /// The training seed.
    pub seed: u64,
}

/// The training section.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingSection {
    /// The training engine.
    #[serde(default = "brain")]
    pub engine: String,
    /// The base model.
    pub base: ModelRef,
    /// The dataset to train on (default: the recipe's dataset).
    #[serde(default)]
    pub dataset: Option<String>,
    /// The variants, each a candidate.
    pub variants: Vec<TrainingVariant>,
}

fn brain() -> String {
    "brain".into()
}

/// The evaluation section.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvalSection {
    /// The evaluation plan file.
    pub plan: String,
    /// The agent every arm runs with.
    pub agent: String,
    /// The agent's arguments, identical for every arm.
    #[serde(default)]
    pub args: Vec<String>,
    /// The baseline model.
    pub baseline: ModelRef,
    /// The candidates, by training variant name.
    pub candidates: Vec<String>,
}

/// The release section.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReleaseSection {
    /// Promote the best improved candidate.
    #[serde(default)]
    pub promote: bool,
}

/// A whole recipe.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Recipe {
    /// Always [`RECIPE_SCHEMA`].
    pub schema: String,
    /// The experiment's name.
    pub name: String,
    /// The tasks.
    pub tasks: TasksSection,
    /// The rollout passes, in order.
    #[serde(default)]
    pub rollouts: Vec<RolloutPass>,
    /// The dataset.
    #[serde(default)]
    pub dataset: Option<DatasetSection>,
    /// Training.
    #[serde(default)]
    pub training: Option<TrainingSection>,
    /// Evaluation.
    #[serde(default)]
    pub eval: Option<EvalSection>,
    /// Release.
    #[serde(default)]
    pub release: Option<ReleaseSection>,
}

/// Why a recipe is refused; every problem found, not only the first.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("the recipe is invalid:\n  {}", .0.join("\n  "))]
pub struct RecipeError(pub Vec<String>);

impl Recipe {
    /// Parse a recipe from YAML.
    ///
    /// # Errors
    /// Not YAML, the wrong shape, an unknown key, or the wrong schema.
    pub fn parse(text: &str) -> Result<Self, RecipeError> {
        let recipe: Self =
            serde_yaml::from_str(text).map_err(|e| RecipeError(vec![e.to_string()]))?;
        if recipe.schema != RECIPE_SCHEMA {
            return Err(RecipeError(vec![format!(
                "schema is {:?}; this Splinter reads {RECIPE_SCHEMA}",
                recipe.schema
            )]));
        }
        Ok(recipe)
    }

    /// Check the references between sections: unique names, passes that
    /// only depend on earlier passes, sources and candidates that exist.
    ///
    /// # Errors
    /// Every problem found.
    pub fn validate(&self) -> Result<(), RecipeError> {
        let mut problems = Vec::new();
        let mut passes = BTreeSet::new();
        for pass in &self.rollouts {
            if !plain_name(&pass.name) {
                problems.push(format!(
                    "rollout name {:?} must be letters, digits, - or _",
                    pass.name
                ));
            }
            for upstream in &pass.tasks.unsolved_by {
                if !passes.contains(upstream.as_str()) {
                    problems.push(format!(
                        "rollout {} selects tasks unsolved by {upstream}, which is not an earlier pass",
                        pass.name
                    ));
                }
            }
            if !pass.tasks.unsolved_by.is_empty() && !pass.tasks.ids.is_empty() {
                problems.push(format!(
                    "rollout {} gives both unsolved_by and ids",
                    pass.name
                ));
            }
            if pass.attempts == 0 || pass.concurrency == 0 {
                problems.push(format!(
                    "rollout {} needs at least one attempt and a concurrency of 1",
                    pass.name
                ));
            }
            if pass.agent.trim().is_empty() {
                problems.push(format!("rollout {} names no agent", pass.name));
            }
            if pass.hint.is_some() && pass.interactive {
                problems.push(format!(
                    "rollout {}: a hinted pass is never interactive; a person guiding the agent is the hint",
                    pass.name
                ));
            }
            if !passes.insert(pass.name.as_str()) {
                problems.push(format!("rollout {} is named twice", pass.name));
            }
        }
        if let Some(dataset) = &self.dataset {
            if dataset.sources.is_empty() {
                problems.push(format!("dataset {} has no source", dataset.name));
            }
            for source in &dataset.sources {
                if !passes.contains(source.as_str()) {
                    problems.push(format!(
                        "dataset {} reads {source}, which is not a rollout pass",
                        dataset.name
                    ));
                }
            }
            if let Some(split) = &dataset.validation {
                if split.percent >= 100 {
                    problems.push(format!(
                        "dataset {} holds out {}% for validation",
                        dataset.name, split.percent
                    ));
                }
            }
        }
        let mut variants = BTreeSet::new();
        if let Some(training) = &self.training {
            match (&training.dataset, &self.dataset) {
                (Some(name), Some(dataset)) if *name != dataset.name => {
                    problems.push(format!(
                        "training reads dataset {name}, which the recipe does not build"
                    ));
                }
                (_, None) => problems.push("training needs the recipe's dataset section".into()),
                _ => {}
            }
            if training.variants.is_empty() {
                problems.push("training has no variant".into());
            }
            for variant in &training.variants {
                if !plain_name(&variant.name) {
                    problems.push(format!(
                        "variant name {:?} must be letters, digits, - or _",
                        variant.name
                    ));
                }
                if !variants.insert(variant.name.as_str()) {
                    problems.push(format!("variant {} is named twice", variant.name));
                }
            }
        }
        if let Some(eval) = &self.eval {
            if eval.candidates.is_empty() {
                problems.push("eval has no candidate".into());
            }
            for candidate in &eval.candidates {
                if !variants.contains(candidate.as_str()) {
                    problems.push(format!(
                        "eval candidate {candidate} is not a training variant"
                    ));
                }
            }
        }
        if self.release.as_ref().is_some_and(|r| r.promote) && self.eval.is_none() {
            problems.push("release promotes only on an evaluation, and the recipe has none".into());
        }
        if problems.is_empty() {
            Ok(())
        } else {
            Err(RecipeError(problems))
        }
    }

    /// Every file the recipe names, relative to its root, in a fixed order.
    #[must_use]
    pub fn referenced_files(&self) -> Vec<String> {
        let mut files = BTreeSet::new();
        files.insert(self.tasks.collection.clone());
        for pass in &self.rollouts {
            files.insert(pass.environment.clone());
            files.extend(pass.files.iter().cloned());
            files.extend(pass.system_prompt.iter().cloned());
        }
        if let Some(eval) = &self.eval {
            files.insert(eval.plan.clone());
        }
        files.into_iter().collect()
    }
}

/// Whether `name` is letters, digits, `-` or `_`, and not empty.
#[must_use]
pub fn plain_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

/// `text`, a recipe, with its evaluation plan replaced by `plan`: how a
/// run's candidates are evaluated under another plan without retraining
/// them (the changed recipe is run as a fork of the run).
///
/// # Errors
/// The text is not a recipe with an `eval` section.
pub fn with_eval_plan(text: &str, plan: &str) -> Result<String, RecipeError> {
    let mut recipe: serde_yaml::Value =
        serde_yaml::from_str(text).map_err(|e| RecipeError(vec![e.to_string()]))?;
    let section = recipe
        .get_mut("eval")
        .filter(|s| s.is_mapping())
        .ok_or_else(|| RecipeError(vec!["the recipe evaluates nothing".into()]))?;
    section["plan"] = serde_yaml::Value::String(plan.to_owned());
    serde_yaml::to_string(&recipe).map_err(|e| RecipeError(vec![e.to_string()]))
}
