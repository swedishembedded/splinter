// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements experiment recipes that are refused before
// they spend anything when a stage cannot do what it says, for its clients.
// If your team needs expertise in machine-learning experiment orchestration,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! Why a stage's parameters are refused, as rules over the parameters
//! alone: what [`crate::resolve`] reports beside what reading the
//! repository's files finds.

use std::collections::BTreeMap;

use crate::champion;
use crate::collection::LoadedCollection;
use crate::eval_plan::EvalPlan;
use crate::graph::CompiledPlan;
use crate::ops::{self, names, AttemptParams, ProjectParams};
use crate::plan::PlannedStage;
use crate::registry::Registry;

/// Why an evaluation under `plan` (at `path`) has no one source of tasks:
/// its plan's collection, the stage's pinned tasks, or its examiner's.
pub(crate) fn task_source_problem(plan: &EvalPlan, path: &str, given: bool) -> Option<String> {
    match (&plan.collection, &plan.authority, given) {
        (Some(_), _, true) => Some(format!(
            "{path} names its collection, and the stage gives tasks too"
        )),
        (None, None, false) => Some(format!(
            "{path} names no collection: bind the stage's tasks"
        )),
        (None, Some(_), true) => Some(format!("{path} is protected: its examiner holds its tasks")),
        _ => None,
    }
}

/// Why an attempt stage cannot continue failed attempts: only sven
/// continues a conversation, unhinted (the instruction is the conversation's
/// first turn) and unplayed, as its own agent and not a harness.
pub(crate) fn fork_refusals(params: &AttemptParams) -> Vec<&'static str> {
    let mut out = Vec::new();
    if params.agent != "sven" {
        out.push("continues failed attempts, which only the sven agent does");
    }
    if params.hint.is_some() {
        out.push("continues failed attempts, whose instruction is already given: it takes no hint");
    }
    if params.harness.is_some() {
        out.push("continues failed attempts as the agent's own conversation, never a harness's");
    }
    if params.interactive {
        out.push("continues failed attempts, which is never interactive");
    }
    out
}

/// Why an attempt stage cannot run its harness: only sven runs one, told
/// its model; a harness is never played and carries its own instructions.
pub(crate) fn harness_refusals(params: &AttemptParams) -> Vec<&'static str> {
    let mut out = Vec::new();
    if params.agent != "sven" {
        out.push("runs a harness, which only the sven agent runs");
    }
    if params.model.is_none() {
        out.push("runs a harness, which is told its model: name a model role");
    }
    if params.interactive {
        out.push("runs a harness, which is never interactive");
    }
    if params.system_prompt.is_some() {
        out.push("runs a harness, which carries its own instructions: it takes no system_prompt");
    }
    out
}

/// Why the evaluation plans' authorities cannot examine: an authority that
/// is an operator the repository does not have.
pub(crate) fn authority_problems(
    plans: &BTreeMap<String, (EvalPlan, Option<LoadedCollection>)>,
    registry: &Registry,
) -> Vec<String> {
    plans
        .iter()
        .filter_map(|(path, (plan, _))| {
            let name = plan.authority.as_ref()?.operator.as_ref()?;
            registry
                .get(name)
                .and_then(|o| o.executor())
                .is_none()
                .then(|| {
                    format!("{path}: its authority is the operator {name}, which the repository does not have")
                })
        })
        .collect()
}

/// What is wrong with a `dataset.project` stage: its sources, its split,
/// and its objective's parameters and sources.
pub(crate) fn projection_problems(
    stage: &PlannedStage,
    params: &ProjectParams,
    compiled: &CompiledPlan,
) -> Vec<String> {
    let mut out = Vec::new();
    if stage.sources("attempts").is_empty() {
        out.push("projects no attempts".to_owned());
    }
    if params.validation.as_ref().is_some_and(|v| v.percent >= 100) {
        out.push("holds out every record for validation".to_owned());
    }
    if !ops::OBJECTIVES.contains(&params.objective.as_str()) {
        out.push(format!(
            "objective {} is not one of {:?}",
            params.objective,
            ops::OBJECTIVES
        ));
    }
    if params.preference.is_some() && params.objective != "preference" {
        out.push("gives preference options to another objective".to_owned());
    }
    if params.objective != "sft" {
        // The views pair and weigh attempts as they were made: a
        // hinted or continued one would teach what the hint showed.
        let defaults = ops::params::<ProjectParams>(&serde_json::json!({
            "target": params.target, "seed": params.seed
        }))
        .ok();
        if defaults
            .is_some_and(|d| d.projection != params.projection || d.admission != params.admission)
            || params.max_block.is_some()
        {
            out.push(format!(
                    "projection, admission and max_block are an SFT dataset's; a {} dataset is projected by its own view",
                    params.objective
                ),
            );
        }
        for source in stage.sources("attempts") {
            let attempts = compiled
                .stages
                .iter()
                .find(|s| s.id == source && s.operation == names::ATTEMPTS);
            let shown = attempts.is_some_and(|s| {
                s.inputs.contains_key("continue_from")
                    || ops::params::<AttemptParams>(&s.params)
                        .is_ok_and(|p| p.hint.is_some() || p.harness.is_some())
            });
            if shown {
                out.push(format!("{source} is hinted, continued or runs a harness; a {} dataset is projected only from attempts made as the target will be asked", params.objective),
                );
            }
        }
    }
    if params.objective == "sft" && params.projection == "native-sft" {
        for source in stage.sources("attempts") {
            let harnessed = compiled
                .stages
                .iter()
                .find(|s| s.id == source && s.operation == names::ATTEMPTS)
                .and_then(|s| ops::params::<AttemptParams>(&s.params).ok())
                .is_some_and(|p| p.harness.is_some());
            if harnessed {
                out.push(format!("{source} runs a harness, which records no model requests to project natively; use projection: portable-sft"),
                );
            }
        }
    }
    if compiled
        .models
        .get(&params.target)
        .is_some_and(champion::is_champion)
    {
        out.push("a champion is served, never a dataset's target: name the base model".to_owned());
    }
    out
}

/// A `model.train` stage given a dataset of an objective it does not train.
pub(crate) fn trained_objective_problems(
    stage: &PlannedStage,
    compiled: &CompiledPlan,
) -> Vec<String> {
    let mut out = Vec::new();
    // The trainer fine-tunes on chat records; a dataset of another
    // objective needs the trainer of its regime.
    for source in stage.sources("dataset") {
        let objective = compiled
            .stages
            .iter()
            .find(|s| s.id == source && s.operation == names::PROJECT)
            .and_then(|s| ops::params::<ProjectParams>(&s.params).ok())
            .map(|p| p.objective)
            .filter(|o| o != "sft");
        if let Some(objective) = objective {
            out.push(format!(
                "{source} is a {objective} dataset; model.train trains SFT datasets"
            ));
        }
    }
    out
}
