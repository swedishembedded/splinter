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

use std::collections::{BTreeMap, BTreeSet};

use crate::champion;
use crate::collection::LoadedCollection;
use crate::eval_plan::EvalPlan;
use crate::plan::Plan;

use crate::graph::CompiledPlan;
use crate::ops::{self, names, AttemptParams, EvaluationParams, ProjectParams, PromotionParams};
use crate::plan::PlannedStage;
use crate::registry::Registry;
use splinter_core::factory::task::TaskRef;

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
    let answers = !stage.sources("answers").is_empty();
    if stage.sources("attempts").is_empty() && !answers {
        out.push("projects no attempts and no answers".to_owned());
    }
    if answers && params.objective == "sft" && !stage.sources("attempts").is_empty() {
        out.push(
            "an SFT dataset projects attempts or answers, not both: an attempt is learned action by action, an answer as its final reply"
                .to_owned(),
        );
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
    if let Some(reward) = &params.reward {
        if params.objective != "policy-gradient" {
            out.push("gives reward options to another objective than policy-gradient".to_owned());
        }
        for (name, percent) in &reward.penalties_percent {
            if name != "loop" && name != "repeat_after_error" {
                out.push(format!(
                    "penalises {name}, which is not a behaviour metric (loop, repeat_after_error)"
                ));
            }
            if *percent == 0 || *percent > 300 {
                out.push(format!(
                    "the penalty of {name} is {percent} percent, not in 1..=300"
                ));
            }
        }
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
        {
            out.push(format!(
                    "projection and admission are an SFT dataset's; a {} dataset is projected by its own view",
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
    if params.per_task.is_some() || params.prefer.is_some() {
        if params.objective != "sft" {
            out.push(format!(
                "per_task and prefer are an SFT dataset's; a {} dataset is projected by its own view",
                params.objective
            ));
        } else if params.per_task.is_none() {
            out.push("prefer chooses the trials a per_task limit keeps; name per_task".to_owned());
        } else if params.per_task == Some(0) {
            out.push("per_task: 0 keeps no trial; name at least 1".to_owned());
        }
    }
    if stage.source("tasks").is_some() && (params.objective != "sft" || answers) {
        out.push(
            "tasks filters the attempts of an SFT dataset; another objective or answers take none"
                .to_owned(),
        );
    }
    if params.control.is_some() && params.objective != "sft" {
        out.push(format!(
            "control is an SFT dataset's; a {} dataset is projected by its own view",
            params.objective
        ));
    }
    if params.sessions != ops::Sessions::PerCall && params.objective != "policy-gradient" {
        out.push(format!(
            "sessions: merged is a policy-gradient dataset's; a {} dataset is projected by its own view",
            params.objective
        ));
    }
    if params.supervise != ops::Supervise::All {
        if params.objective != "sft" {
            out.push(format!(
                "supervise is an SFT dataset's; a {} dataset is projected by its own view",
                params.objective
            ));
        } else if params.projection != "native-sft" {
            out.push(
                "supervise: productive needs projection: native-sft: only the requests the model was sent show which results were errors"
                    .to_owned(),
            );
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

/// A `model.train` stage given a dataset no LoRA regime trains: a
/// retrieval dataset's contrastive triples train an embedder.
pub(crate) fn trained_objective_problems(
    stage: &PlannedStage,
    compiled: &CompiledPlan,
) -> Vec<String> {
    let mut out = Vec::new();
    // `start` names a champion served from the base the candidate trains on.
    if let Ok(params) = ops::params::<crate::ops::TrainParams>(&stage.params) {
        if let Some(role) = &params.start {
            let start = compiled.models.get(role);
            let base = compiled.models.get(&params.base);
            if !start.is_some_and(crate::champion::is_champion) {
                out.push(format!("start {role} is not a champion model role"));
            } else if base.is_some_and(crate::champion::is_champion) {
                out.push(
                    "a candidate continues a champion's adapter on a base model: name the base"
                        .into(),
                );
            }
        }
    }
    for source in stage.sources("dataset") {
        let retrieval = compiled
            .stages
            .iter()
            .find(|s| s.id == source && s.operation == names::PROJECT)
            .and_then(|s| ops::params::<ProjectParams>(&s.params).ok())
            .is_some_and(|p| p.objective == "retrieval");
        if retrieval {
            out.push(format!(
                "{source} is a retrieval dataset; a retrieval dataset trains an embedder, not a model.train LoRA"
            ));
        }
    }
    out
}

/// What is wrong with a `promotion` stage: a promotion with no evaluation,
/// or without one of every purpose it requires.
pub(crate) fn promotion_problems(
    stage: &PlannedStage,
    params: &PromotionParams,
    compiled: &CompiledPlan,
) -> Vec<String> {
    let mut out = Vec::new();
    if !params.promote {
        if params.requires.is_some() {
            out.push("requires is a promotion's, and this one promotes nothing".to_owned());
        }
        return out;
    }
    let evaluations = stage.sources("evaluations");
    if evaluations.is_empty() {
        out.push("promotes only on an evaluation, and it is given none".to_owned());
        return out;
    }
    if params.divergence_window < 2 {
        out.push("divergence_window is at least 2: a trend needs two generations".to_owned());
    }
    if !(0.0..=1.0).contains(&params.retention_margin) {
        out.push("retention_margin is a share of the tasks, between 0 and 1".to_owned());
    }
    let purposes: Vec<ops::Purpose> = evaluations
        .iter()
        .filter_map(|source| compiled.stages.iter().find(|s| s.id == *source))
        .filter_map(|s| ops::params::<EvaluationParams>(&s.params).ok())
        .map(|p| p.purpose)
        .collect();
    for purpose in params.required() {
        if !purposes.contains(&purpose) {
            out.push(format!(
                "no {} evaluation: a promoted candidate must have one; add it, or narrow `requires`",
                purpose.name()
            ));
        }
    }
    out
}

/// What is wrong with a `harness.search` stage's parameters.
pub(crate) fn search_problems(
    stage: &PlannedStage,
    params: &ops::HarnessSearchParams,
) -> Vec<String> {
    let mut out = Vec::new();
    if stage.source("tasks").is_none() {
        out.push("searches on no tasks: bind `tasks` to a tasks.pin stage".to_owned());
    }
    if params.agent != "sven" {
        out.push("searches harnesses, which only the sven agent runs".to_owned());
    }
    if params.candidates == 0 || params.attempts == 0 || params.concurrency == 0 {
        out.push("needs at least one candidate, one attempt and a concurrency of 1".to_owned());
    }
    if !(params.alpha > 0.0 && params.alpha < 1.0) {
        out.push("alpha is a level between 0 and 1".to_owned());
    }
    out
}

/// The tasks a dataset stage would learn from that are examination material:
/// on the `eval` split, or the very task (by content) an evaluation plan of
/// the recipe examines under any name. Trials of them could otherwise reach
/// training, and a candidate would be examined on what it was taught.
pub(crate) fn examination_material(
    plan: &Plan,
    dataset: &PlannedStage,
    collections: &BTreeMap<String, LoadedCollection>,
    eval_plans: &BTreeMap<String, (EvalPlan, Option<LoadedCollection>)>,
) -> Vec<String> {
    let examined: Vec<Examined<'_>> = eval_plans
        .values()
        .filter_map(|(plan, loaded)| loaded.as_ref().map(|l| (plan, l)))
        .flat_map(|(plan, loaded)| {
            loaded
                .collection
                .tasks
                .iter()
                .filter(move |t| plan.split.is_none() || t.split == plan.split)
                .map(|task| Examined {
                    task,
                    instruction: instruction_of(&loaded.root.join(&task.id)),
                })
        })
        .collect();
    // The collections whose trials reach the dataset: through its attempt
    // stages, and those stages' tasks.
    let pins: BTreeSet<&str> = dataset
        .sources("attempts")
        .into_iter()
        .filter_map(|a| plan.stage(a))
        .filter_map(|a| a.source("tasks"))
        .collect();
    let mut problems = Vec::new();
    for training in pins.iter().filter_map(|p| collections.get(*p)) {
        problems.extend(examined_tasks(training, &examined, &dataset.key));
    }
    problems
}

/// An examined task and what it asks.
struct Examined<'a> {
    task: &'a TaskRef,
    instruction: String,
}

/// What the task folder `dir` asks: its `instruction.md`, empty when it has
/// none (a task with nothing to ask repeats nothing).
fn instruction_of(dir: &std::path::Path) -> String {
    std::fs::read_to_string(dir.join("instruction.md")).unwrap_or_default()
}

fn examined_tasks(
    training: &LoadedCollection,
    examined: &[Examined<'_>],
    dataset: &str,
) -> Vec<String> {
    let mut problems = Vec::new();
    for task in &training.collection.tasks {
        if task.split.as_deref() == Some("eval") {
            problems.push(format!(
                "{} is examination material (split eval) and cannot reach dataset {dataset}: select the train split",
                task.id
            ));
        } else if let Some(same) = examined.iter().find(|e| e.task.digest == task.digest) {
            problems.push(format!(
                "{} is the same task as {}, which the eval plan examines; it cannot reach dataset {dataset}",
                task.id, same.task.id
            ));
        } else {
            // A copy with a byte or a word changed has another digest: the
            // same task by the rule the release's leakage gate holds
            // instructions to (an exact repeat once normalised, or most word
            // shingles shared).
            let instruction = instruction_of(&training.root.join(&task.id));
            let near = examined.iter().find(|e| {
                !instruction.trim().is_empty()
                    && splinter_knowledge::tasks::dedup::repeat_of(
                        &instruction,
                        &e.instruction,
                        splinter_knowledge::tasks::DEFAULT_SHINGLE_WORDS,
                        splinter_knowledge::tasks::DEFAULT_MAX_OVERLAP,
                    )
                    .is_some()
            });
            if let Some(copy) = near {
                problems.push(format!(
                    "{} asks what {} asks, which the eval plan examines (a near copy); it cannot reach dataset {dataset}",
                    task.id, copy.task.id
                ));
            }
        }
    }
    problems
}

/// What is wrong with a `task.mutate` stage: a revision not named by its
/// commit (the run's identity is the tree's), no git repository there, or a
/// share held out that leaves nothing to train on.
pub(crate) fn mutate_problems(
    params: &ops::MutateParams,
    repository: &std::path::Path,
) -> Vec<String> {
    let mut out = Vec::new();
    let commit =
        params.revision.len() == 40 && params.revision.chars().all(|c| c.is_ascii_hexdigit());
    if !commit {
        out.push(format!(
            "revision {:?}: name the revision by its commit, so the run is the tree's",
            params.revision
        ));
    }
    let at = repository.join(&params.repository);
    if !at.join(".git").exists() && !at.join("HEAD").is_file() {
        out.push(format!("{} is not a git repository", at.display()));
    }
    if params.eval_percent >= 100 {
        out.push("eval_percent holds every family out, leaving nothing to train on".to_owned());
    }
    out
}

/// What is wrong with a `model.train` stage's budget before any record is
/// counted: `epochs` beside `steps`, or epochs that are not positive.
pub(crate) fn budget_problems(stage: &PlannedStage) -> Vec<String> {
    ops::params::<ops::TrainParams>(&stage.params)
        .map(|p| crate::budget::problems(&p.params))
        .unwrap_or_default()
}

/// A `model.train` stage whose `replay` cannot be rehearsed: replay
/// rehearses chat records, so neither the dataset trained nor a replayed
/// one may be of a view objective, and a share is a share.
pub(crate) fn replay_problems(stage: &PlannedStage, compiled: &CompiledPlan) -> Vec<String> {
    let mut out = Vec::new();
    let replayed = stage.sources("replay");
    let share = ops::params::<ops::TrainParams>(&stage.params)
        .ok()
        .and_then(|p| p.replay_share);
    if let Some(share) = share {
        if replayed.is_empty() {
            out.push("replay_share is given and nothing is replayed".to_owned());
        } else if !(share > 0.0 && share < 1.0) {
            out.push("replay_share is a share of the draws, between 0 and 1".to_owned());
        }
    }
    if replayed.is_empty() {
        return out;
    }
    let objective = |source: &str| {
        compiled
            .stages
            .iter()
            .find(|s| s.id == source && s.operation == names::PROJECT)
            .and_then(|s| ops::params::<ProjectParams>(&s.params).ok())
            .map(|p| p.objective)
            .filter(|o| o != "sft")
    };
    for source in stage.sources("dataset").into_iter().chain(replayed) {
        if let Some(objective) = objective(source) {
            out.push(format!(
                "{source} is a {objective} dataset; replay rehearses chat records, beside a chat dataset"
            ));
        }
    }
    out
}
