// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible experiments whose every input
// is pinned before work starts, for its clients. If your team needs
// expertise in reproducible machine-learning runs, you can procure our
// services by sending an email to info@swedishembedded.com.

//! A recipe compiled, with every file it names read, checked and pinned.
//!
//! [`resolve`] compiles the recipe against the built-in operations
//! ([`crate::ops`]), then reads what each stage's parameters name: a
//! `tasks.pin` stage's collection, an `attempt.collect` stage's environment,
//! agent files and prompt, an `attempt.import` stage's Harbor job
//! directories (pinned by their tree digest), an `evaluation` stage's plan
//! with its own environment and collection; and binds every model role to a connection
//! or a champion. Every problem is reported at once, so `splinter validate`
//! shows them all. The [`Lock`] records the digest of each input: a resumed
//! run compares the repository against it and refuses to continue on inputs
//! that changed under it.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use splinter_core::digest::{digest_of, Digest};
use splinter_core::factory::task::{plain_relative, TaskCollection, TaskRef};

use crate::champion::{self, Champion};
use crate::collection::{self, LoadedCollection};
use crate::connections::{Connection, Connections};
use crate::environment::Environment;
use crate::eval_plan::EvalPlan;
use crate::graph::{self, CompiledPlan};
use crate::ops::{
    self, names, AttemptParams, EvaluationParams, Hint, ImportDatasetParams, ImportParams,
    PinParams, ProjectParams, PromotionParams, TrainParams,
};
use crate::plan::{Plan, PlannedStage};
use crate::recipe::{ModelRef, RecipeError};
use crate::registry::Registry;
use crate::tree::tree_digest;

/// The schema of a [`Lock`].
pub const LOCK_SCHEMA: &str = "splinter.lock/v2";

/// What a run was started on.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Lock {
    /// Always [`LOCK_SCHEMA`].
    pub schema: String,
    /// The recipe's digest.
    pub recipe: Digest,
    /// Every file the recipe names, and every imported Harbor job directory
    /// (by its tree digest), by repository-relative path, and every
    /// champion, by reference.
    pub files: BTreeMap<String, Digest>,
    /// The pinned tasks of every `tasks.pin` stage.
    pub tasks: Digest,
    /// The connections the recipe's models are reached through, as
    /// configured (names of credential variables only); a connection no
    /// model uses is not part of it.
    pub connections: Digest,
}

/// A resolved recipe.
#[derive(Clone, Debug)]
pub struct Resolved {
    /// The repository root every path is relative to.
    pub repository: PathBuf,
    /// The recipe's text, as read (its digest is the lock's).
    pub recipe_text: String,
    /// The compiled recipe.
    pub compiled: CompiledPlan,
    /// The tasks each `tasks.pin` stage selects, by stage.
    pub collections: BTreeMap<String, LoadedCollection>,
    /// Environments by path.
    pub environments: BTreeMap<String, Environment>,
    /// Evaluation plans, with their own collection, by path.
    pub eval_plans: BTreeMap<String, (EvalPlan, Option<LoadedCollection>)>,
    /// The champions the recipe's models name, by reference.
    pub champions: BTreeMap<String, Champion>,
    /// The connections.
    pub connections: Connections,
    /// The lock.
    pub lock: Lock,
    /// The graph of stages.
    pub plan: Plan,
    /// The operations the recipe was compiled against; the run computes
    /// with theirs.
    pub registry: Arc<Registry>,
}

impl Resolved {
    /// The recipe's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.compiled.name
    }

    /// The model the recipe binds to `role`.
    #[must_use]
    pub fn model(&self, role: &str) -> Option<&ModelRef> {
        self.compiled.models.get(role)
    }
}

/// Reads repository files, recording each one's digest and every problem.
struct Reader<'r> {
    repository: &'r Path,
    files: BTreeMap<String, Digest>,
    problems: Vec<String>,
}

impl Reader<'_> {
    fn read(&mut self, relative: &str) -> Option<String> {
        if !plain_relative(relative) {
            self.problems.push(format!(
                "{relative:?} is not a plain path relative to the repository"
            ));
            return None;
        }
        match std::fs::read_to_string(self.repository.join(relative)) {
            Ok(text) => {
                self.files
                    .insert(relative.to_owned(), Digest::of(text.as_bytes()));
                Some(text)
            }
            Err(e) => {
                self.problems.push(format!("{relative}: {e}"));
                None
            }
        }
    }

    /// Pin the Harbor job directory `relative` by the digest of its whole
    /// tree: a trial added to or changed in it is a changed input.
    fn harbor_job(&mut self, relative: &str) {
        if !plain_relative(relative) {
            self.problems.push(format!(
                "{relative:?} is not a plain path relative to the repository"
            ));
            return;
        }
        let path = self.repository.join(relative);
        if !path.is_dir() {
            self.problems
                .push(format!("{relative} is not a Harbor job directory"));
            return;
        }
        match tree_digest(&path) {
            Ok(digest) => {
                self.files.insert(relative.to_owned(), digest);
            }
            Err(e) => self.problems.push(e.to_string()),
        }
    }

    fn environment(&mut self, path: &str, environments: &mut BTreeMap<String, Environment>) {
        if environments.contains_key(path) {
            return;
        }
        if let Some(text) = self.read(path) {
            match Environment::parse(&text) {
                Ok(env) => {
                    environments.insert(path.to_owned(), env);
                }
                Err(RecipeError(found)) => self
                    .problems
                    .extend(found.into_iter().map(|p| format!("{path}: {p}"))),
            }
        }
    }

    fn collection(&mut self, path: &str) -> Option<LoadedCollection> {
        self.read(path)?;
        match collection::load(self.repository, path) {
            Ok(loaded) => Some(loaded),
            Err(e) => {
                self.problems.push(e.to_string());
                None
            }
        }
    }
}

/// What resolving collects besides the plan.
#[derive(Default)]
struct Found {
    environments: BTreeMap<String, Environment>,
    eval_plans: BTreeMap<String, (EvalPlan, Option<LoadedCollection>)>,
    collections: BTreeMap<String, LoadedCollection>,
}

/// Compile and check `recipe_text` with everything it names under
/// `repository`.
///
/// # Errors
/// Every problem found: the recipe, a missing or invalid file, a model with
/// no connection, an environment a pass cannot run in.
pub fn resolve(
    repository: &Path,
    recipe_text: &str,
    connections: Connections,
) -> Result<Resolved, RecipeError> {
    resolve_with(repository, recipe_text, connections, ops::registry())
}

/// [`resolve`] against the operations of `registry` (the built-in ones and
/// any other the run is given).
///
/// # Errors
/// As [`resolve`].
pub fn resolve_with(
    repository: &Path,
    recipe_text: &str,
    connections: Connections,
    registry: Registry,
) -> Result<Resolved, RecipeError> {
    let load = |path: &str| -> Result<String, String> {
        if !plain_relative(path) {
            return Err(format!(
                "{path:?} is not a plain path relative to the repository"
            ));
        }
        std::fs::read_to_string(repository.join(path)).map_err(|e| e.to_string())
    };
    // What compiled is checked further, so every problem is reported at once.
    let (compiled, problems) = graph::compile_partly(recipe_text, &registry, &load)?;
    let mut reader = Reader {
        repository,
        files: BTreeMap::new(),
        problems,
    };
    let champions = bind_models(&compiled, &connections, repository, &mut reader);
    let mut found = Found::default();
    let mut plan = Plan {
        stages: compiled
            .stages
            .iter()
            .map(|s| PlannedStage {
                key: s.id.clone(),
                operation: s.operation.clone(),
                revision: s.revision,
                depends_on: s.depends_on.clone(),
                work: String::new(),
                inputs: s.inputs.clone(),
                params: s.params.clone(),
                foreach: s.foreach.clone(),
                concurrency: s.concurrency,
            })
            .collect(),
    };
    for stage in &plan.stages {
        check(stage, &compiled, &mut reader, &mut found);
        if let Some(operation) = registry.get(&stage.operation) {
            let mut named = BTreeMap::new();
            for path in operation.files(&stage.params) {
                if let Some(text) = reader.read(&path) {
                    named.insert(path, text);
                }
            }
            let problems = operation.check_files(&stage.params, &named);
            reader.problems.extend(
                problems
                    .into_iter()
                    .map(|p| format!("stage {}: {p}", stage.key)),
            );
            if operation.executor().is_some() {
                pin_inputs(stage, &compiled, &mut reader);
            }
        }
    }
    for stage in &mut plan.stages {
        stage.work = crate::describe::work(stage, &compiled, &found.collections);
    }
    for stage in plan.of_operation(names::PROJECT) {
        reader.problems.extend(examination_material(
            &plan,
            stage,
            &found.collections,
            &found.eval_plans,
        ));
    }
    let Reader {
        files, problems, ..
    } = reader;
    if !problems.is_empty() {
        return Err(RecipeError(problems));
    }
    let lock = Lock {
        schema: LOCK_SCHEMA.into(),
        recipe: Digest::of(recipe_text.as_bytes()),
        files,
        tasks: digest_of(
            &found
                .collections
                .iter()
                .map(|(k, v)| (k.clone(), &v.collection))
                .collect::<BTreeMap<_, _>>(),
        ),
        connections: digest_of(&used_connections(&connections, &compiled, &champions)),
    };
    Ok(Resolved {
        repository: repository.to_path_buf(),
        recipe_text: recipe_text.to_owned(),
        compiled,
        collections: found.collections,
        environments: found.environments,
        eval_plans: found.eval_plans,
        champions,
        connections,
        lock,
        plan,
        registry: Arc::new(registry),
    })
}

/// Why an attempt stage cannot run its harness: only sven runs one, told
/// its model; a harness is never played and carries its own instructions.
fn harness_refusals(params: &AttemptParams) -> Vec<&'static str> {
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

/// The recipe inputs a computed stage reads (bound to a port or fanned out
/// over), each pinned in the lock by the digest of what the run copies.
fn pin_inputs(stage: &PlannedStage, compiled: &CompiledPlan, reader: &mut Reader<'_>) {
    let named = stage.inputs.values().flatten().chain(stage.foreach.iter());
    for source in named {
        let graph::Source::Input(name) = source else {
            continue;
        };
        let Some(spec) = compiled.inputs.get(name) else {
            continue;
        };
        if reader.files.contains_key(&spec.path) {
            continue;
        }
        if !plain_relative(&spec.path) {
            reader.problems.push(format!(
                "input {name}: {:?} is not a plain path relative to the repository",
                spec.path
            ));
            continue;
        }
        match crate::tree::entry_digest(&reader.repository.join(&spec.path)) {
            Ok(digest) => {
                reader.files.insert(spec.path.clone(), digest);
            }
            Err(e) => reader.problems.push(format!("input {name}: {e}")),
        }
    }
}

/// Every model role bound: a champion loaded (and its base's connection
/// checked), any other model's connection checked.
fn bind_models(
    compiled: &CompiledPlan,
    connections: &Connections,
    repository: &Path,
    reader: &mut Reader<'_>,
) -> BTreeMap<String, Champion> {
    let mut champions = BTreeMap::new();
    for (role, model) in &compiled.models {
        if champion::is_champion(model) {
            match champion::load(repository, model) {
                Ok(found) => {
                    if let Err(e) = connections.bind(&found.record.base) {
                        reader.problems.push(format!("model {role} ({model}): {e}"));
                    }
                    reader.files.insert(model.to_string(), found.digest.clone());
                    champions.insert(model.to_string(), found);
                }
                Err(e) => reader.problems.push(format!("model {role}: {e}")),
            }
        } else if let Err(e) = connections.bind(model) {
            reader.problems.push(format!("model {role}: {e}"));
        }
    }
    champions
}

/// Every check of one stage that reading its files allows.
fn check(
    stage: &PlannedStage,
    compiled: &CompiledPlan,
    reader: &mut Reader<'_>,
    found: &mut Found,
) {
    let id = stage.key.clone();
    let problem = |reader: &mut Reader<'_>, text: &str| {
        reader.problems.push(format!("stage {id}: {text}"));
    };
    match stage.operation.as_str() {
        names::PIN => {
            let Ok(params) = stage.params::<PinParams>() else {
                return;
            };
            let Some(input) = stage.input("collection") else {
                problem(
                    reader,
                    "a collection is read from the recipe's `inputs:` (a collection file)",
                );
                return;
            };
            let Some(path) = compiled.inputs.get(input).map(|i| i.path.clone()) else {
                return;
            };
            let Some(mut loaded) = reader.collection(&path) else {
                return;
            };
            loaded
                .collection
                .tasks
                .retain(|t| params.split.is_none() || t.split == params.split);
            if loaded.collection.tasks.is_empty() {
                problem(
                    reader,
                    &format!(
                        "{path} has no task of split {}",
                        params.split.as_deref().unwrap_or("(any)")
                    ),
                );
            }
            found.collections.insert(stage.key.clone(), loaded);
        }
        names::ATTEMPTS => {
            let Ok(params) = stage.params::<AttemptParams>() else {
                return;
            };
            reader.environment(&params.environment, &mut found.environments);
            for file in params.files.iter().chain(params.system_prompt.iter()) {
                reader.read(file);
            }
            for (name, value) in &params.environment_variables {
                if !value.starts_with("env:")
                    && splinter_core::factory::execution::credential_like(name)
                {
                    problem(
                        reader,
                        &format!("{name} looks like a credential; write env:{name} to pass the worker's own variable"),
                    );
                }
            }
            if params.attempts == 0 || params.concurrency == 0 {
                problem(reader, "needs at least one attempt and a concurrency of 1");
            }
            if params.agent.trim().is_empty() {
                problem(reader, "names no agent");
            }
            if params.hint.is_some() && params.interactive {
                problem(
                    reader,
                    "a hinted pass is never interactive; a person guiding the agent is the hint",
                );
            }
            match (params.hint, stage.source("briefs").is_some()) {
                (Some(Hint::Brief), false) => problem(
                    reader,
                    "hint: brief shows each task its brief, and the stage is given none: bind `briefs`",
                ),
                (Some(Hint::Solution) | None, true) => problem(
                    reader,
                    "is given briefs, which only a pass with hint: brief shows",
                ),
                _ => {}
            }
            if let Some(harness) = &params.harness {
                for refusal in harness_refusals(&params) {
                    problem(reader, refusal);
                }
                if let Some(text) = reader.read(harness) {
                    if let Err(why) = splinter_agent::attempt_harness::check(&text) {
                        problem(reader, &format!("{harness}: {why}"));
                    }
                }
            }
            if !params.tasks.is_empty() && !stage.sources("unsolved_by").is_empty() {
                problem(reader, "gives both task ids and unsolved_by");
            }
            if let Some(pin) = stage.source("tasks").and_then(|s| found.collections.get(s)) {
                let missing: Vec<String> = params
                    .tasks
                    .iter()
                    .filter(|t| pin.collection.task(t).is_none())
                    .cloned()
                    .collect();
                for task in missing {
                    problem(
                        reader,
                        &format!("names task {task}, which its tasks do not hold"),
                    );
                }
            }
        }
        names::IMPORT => {
            let Ok(params) = stage.params::<ImportParams>() else {
                return;
            };
            let jobs = stage.inputs_of("jobs");
            if jobs.is_empty() {
                problem(
                    reader,
                    "imports no Harbor job: bind `jobs` to the recipe's inputs (job directories)",
                );
            }
            for input in jobs {
                if let Some(path) = compiled.inputs.get(input).map(|i| i.path.clone()) {
                    reader.harbor_job(&path);
                }
            }
            if let Some(pin) = stage.source("tasks").and_then(|s| found.collections.get(s)) {
                for task in params
                    .tasks
                    .iter()
                    .filter(|t| pin.collection.task(t).is_none())
                {
                    problem(
                        reader,
                        &format!("names task {task}, which its tasks do not hold"),
                    );
                }
            }
        }
        names::IMPORT_DATASET => {
            if stage.input("records").is_none() {
                problem(
                    reader,
                    "imports no records: bind `records` to a recipe input (a directory of train.jsonl and validation.jsonl)",
                );
            }
            pin_inputs(stage, compiled, reader);
            if let Ok(params) = stage.params::<ImportDatasetParams>() {
                if compiled
                    .models
                    .get(&params.target)
                    .is_some_and(champion::is_champion)
                {
                    problem(
                        reader,
                        "a champion is served, never a dataset's target: name the base model",
                    );
                }
            }
        }
        names::PROJECT => {
            let Ok(params) = stage.params::<ProjectParams>() else {
                return;
            };
            if stage.sources("attempts").is_empty() {
                problem(reader, "projects no attempts");
            }
            if params.validation.as_ref().is_some_and(|v| v.percent >= 100) {
                problem(reader, "holds out every record for validation");
            }
            if params.projection == "native-sft" {
                for source in stage.sources("attempts") {
                    let harnessed = compiled
                        .stages
                        .iter()
                        .find(|s| s.id == source && s.operation == names::ATTEMPTS)
                        .and_then(|s| ops::params::<AttemptParams>(&s.params).ok())
                        .is_some_and(|p| p.harness.is_some());
                    if harnessed {
                        problem(
                            reader,
                            &format!("{source} runs a harness, which records no model requests to project natively; use projection: portable-sft"),
                        );
                    }
                }
            }
            if compiled
                .models
                .get(&params.target)
                .is_some_and(champion::is_champion)
            {
                problem(
                    reader,
                    "a champion is served, never a dataset's target: name the base model",
                );
            }
        }
        names::TRAIN => {
            let Ok(params) = stage.params::<TrainParams>() else {
                return;
            };
            if compiled
                .models
                .get(&params.base)
                .is_some_and(champion::is_champion)
            {
                problem(
                    reader,
                    "a champion is served, never trained from: a variant trains from a base model",
                );
            }
        }
        names::EVALUATION => {
            let Ok(params) = stage.params::<EvaluationParams>() else {
                return;
            };
            for file in params.files.iter().chain(params.system_prompt.iter()) {
                reader.read(file);
            }
            if found.eval_plans.contains_key(&params.plan) {
                return;
            }
            let Some(text) = reader.read(&params.plan) else {
                return;
            };
            match EvalPlan::parse(&text) {
                Ok(plan) => {
                    if let Some(env) = &plan.environment {
                        reader.environment(env, &mut found.environments);
                    }
                    let tasks = plan
                        .collection
                        .as_ref()
                        .and_then(|file| reader.collection(file));
                    found.eval_plans.insert(params.plan.clone(), (plan, tasks));
                }
                Err(RecipeError(problems)) => reader.problems.extend(
                    problems
                        .into_iter()
                        .map(|p| format!("{}: {p}", params.plan)),
                ),
            }
        }
        names::PROMOTION
            if stage.params::<PromotionParams>().is_ok_and(|p| p.promote)
                && stage.sources("evaluations").is_empty() =>
        {
            problem(
                reader,
                "promotes only on an evaluation, and it is given none",
            );
        }
        _ => {}
    }
}

/// The connections the recipe's models are reached through (a champion's
/// through its base): what a run's lock pins. Another connection of the
/// machine or the repository, added or changed, changes no run.
fn used_connections<'c>(
    connections: &'c Connections,
    compiled: &CompiledPlan,
    champions: &BTreeMap<String, Champion>,
) -> BTreeMap<&'c str, &'c Connection> {
    let names: BTreeSet<&str> = compiled
        .models
        .values()
        .map(|m| match champions.get(&m.to_string()) {
            Some(champion) => champion.record.base.connection.as_str(),
            None => m.connection.as_str(),
        })
        .collect();
    connections
        .connections
        .iter()
        .filter(|(name, _)| names.contains(name.as_str()))
        .map(|(name, c)| (name.as_str(), c))
        .collect()
}

/// The tasks a dataset stage would learn from that are examination material:
/// on the `eval` split, or the very task (by content) an evaluation plan of
/// the recipe examines under any name. Trials of them could otherwise reach
/// training, and a candidate would be examined on what it was taught.
fn examination_material(
    plan: &Plan,
    dataset: &PlannedStage,
    collections: &BTreeMap<String, LoadedCollection>,
    eval_plans: &BTreeMap<String, (EvalPlan, Option<LoadedCollection>)>,
) -> Vec<String> {
    let examined: Vec<&TaskRef> = eval_plans
        .values()
        .filter_map(|(plan, loaded)| loaded.as_ref().map(|l| (plan, l)))
        .flat_map(|(plan, loaded)| {
            loaded
                .collection
                .tasks
                .iter()
                .filter(move |t| plan.split.is_none() || t.split == plan.split)
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
    for training in pins
        .iter()
        .filter_map(|p| collections.get(*p))
        .map(|l| &l.collection)
    {
        problems.extend(examined_tasks(training, &examined, &dataset.key));
    }
    problems
}

fn examined_tasks(training: &TaskCollection, examined: &[&TaskRef], dataset: &str) -> Vec<String> {
    let mut problems = Vec::new();
    for task in &training.tasks {
        if task.split.as_deref() == Some("eval") {
            problems.push(format!(
                "{} is examination material (split eval) and cannot reach dataset {dataset}: select the train split",
                task.id
            ));
        } else if let Some(same) = examined.iter().find(|e| e.digest == task.digest) {
            problems.push(format!(
                "{} is the same task as {}, which the eval plan examines; it cannot reach dataset {dataset}",
                task.id, same.id
            ));
        }
    }
    problems
}

impl Lock {
    /// What differs between this lock and `now`, one line each; empty when
    /// the inputs are unchanged.
    #[must_use]
    pub fn changes(&self, now: &Self) -> Vec<String> {
        let mut out = Vec::new();
        if self.recipe != now.recipe {
            out.push("the recipe changed".into());
        }
        if self.tasks != now.tasks {
            out.push("the task collection changed".into());
        }
        if self.connections != now.connections {
            out.push("the connections changed".into());
        }
        for (path, digest) in &self.files {
            match now.files.get(path) {
                Some(d) if d == digest => {}
                Some(_) => out.push(format!("{path} changed")),
                None => out.push(format!("{path} is no longer named")),
            }
        }
        for path in now.files.keys().filter(|p| !self.files.contains_key(*p)) {
            out.push(format!("{path} is newly named"));
        }
        out
    }
}
