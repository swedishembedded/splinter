// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model-driven task generation that turns
// any source into verifiable training tasks, for its clients. If your team
// needs expertise in synthetic task generation or verifier design, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The tasks stage: stored sources become a task set.
//!
//! Every text part of every source is shown to the generator model a
//! window of [`SECTIONS_PER_REQUEST`] sections at a time, once per model
//! kind asked for, and what code admits is stored; the `denoise` kind needs
//! no model and yields one task per text part. The set records which
//! generator produced each task, from which prompt.

use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

use serde::Serialize;
use splinter_knowledge::denoise::{Denoise, GENERATOR as DENOISE_GENERATOR};
use splinter_knowledge::tasks::{
    Catalogue, GenerateError, GenerationPolicy, ModelTaskGenerator, SourceText, TaskKind,
    DEFAULT_REQUEST_DEADLINE,
};
use splinter_lab::denoise::KIND as DENOISE_KIND;
use splinter_store::digest::Digest;
use splinter_store::experience::Task;
use splinter_store::source::SourceId;
use splinter_store::tasks::{TaskEntry, TaskSet, TaskSetId};
use sven_sdk::CancelToken;

use crate::context::Context;
use crate::error::CampaignError;
use crate::ids;
use crate::model_ref::ModelRef;

/// Sections one generation request shows the model: enough context for
/// tasks that combine sections, few enough that the prompt stays small.
pub const SECTIONS_PER_REQUEST: usize = 8;

/// The seed denoise tasks are generated with, so the same part always
/// yields the same task.
pub const DENOISE_SEED: u64 = 0;

/// The task kinds `learn` generates when none are named.
pub const DEFAULT_LEARN_KINDS: &[&str] = &["recall"];

/// The kinds a command may name: the built-in catalogue's, and `denoise`.
#[must_use]
pub fn known_kinds() -> Vec<String> {
    let mut names: Vec<String> = Catalogue::builtin()
        .kinds()
        .map(|k| k.name.clone())
        .collect();
    names.push(DENOISE_KIND.to_string());
    names.sort();
    names
}

/// `names` checked against [`known_kinds`], each once, in the order given.
pub fn check_kinds(names: &[String]) -> Result<Vec<String>, CampaignError> {
    if names.is_empty() {
        return Err(CampaignError::Refused("name at least one task kind".into()));
    }
    let known = known_kinds();
    let mut seen = BTreeSet::new();
    let mut kinds = Vec::new();
    for name in names {
        if !known.contains(name) {
            return Err(CampaignError::Refused(format!(
                "unknown task kind {name:?}; the kinds are {}",
                known.join(", ")
            )));
        }
        if seen.insert(name.clone()) {
            kinds.push(name.clone());
        }
    }
    Ok(kinds)
}

/// One generation request's bounds and inputs.
pub struct Generation<'a> {
    /// The sources to generate from.
    pub sources: &'a [SourceId],
    /// The kinds, checked by [`check_kinds`].
    pub kinds: &'a [String],
    /// The generator model.
    pub generator: &'a ModelRef,
    /// What the learner is after, added to every model kind's brief.
    pub goal: Option<&'a str>,
    /// No request starts after this, and none runs past it.
    pub deadline: Option<Instant>,
    /// Stops generation.
    pub cancel: CancelToken,
}

/// Admitted and rejected proposals of one kind.
#[derive(Clone, Debug, Default, Serialize)]
pub struct KindTally {
    /// Tasks admitted.
    pub admitted: usize,
    /// Proposals rejected.
    pub rejected: usize,
}

/// What the tasks stage reports.
#[derive(Clone, Debug, Serialize)]
pub struct TasksGenerated {
    /// The task set (`solve <task_set>`).
    pub task_set: TaskSetId,
    /// Tasks in it.
    pub tasks: usize,
    /// Text parts generated from.
    pub parts: usize,
    /// Admitted and rejected, by kind.
    pub per_kind: BTreeMap<String, KindTally>,
    /// Rejections, by reason.
    pub rejected: BTreeMap<String, usize>,
    /// Why generation stopped before every part was covered, if it did.
    pub stopped: Option<String>,
}

/// Generates tasks of `request.kinds` from `request.sources` and stores
/// them as a task set.
pub fn generate(ctx: &Context, request: &Generation<'_>) -> Result<TasksGenerated, CampaignError> {
    let catalogue = Catalogue::builtin();
    let model_kinds: Vec<TaskKind> = request
        .kinds
        .iter()
        .filter_map(|name| catalogue.get(name))
        .map(|kind| with_goal(kind, request.goal))
        .collect();
    let denoise = request.kinds.iter().any(|k| k == DENOISE_KIND);
    let generator = if model_kinds.is_empty() {
        None
    } else {
        Some(model_generator(ctx, request, &model_kinds)?)
    };
    let mut batch = Batch::default();
    let mut stopped = None;
    'sources: for source_id in request.sources {
        let source = ctx.sources().get_source(source_id)?;
        for part in source
            .parts
            .iter()
            .filter(|p| p.media_type.starts_with("text/"))
        {
            if request.cancel.is_cancelled() {
                return Err(CampaignError::Cancelled);
            }
            if request.deadline.is_some_and(|d| Instant::now() >= d) {
                stopped = Some("the budget was spent before every part was covered".into());
                break 'sources;
            }
            batch.parts += 1;
            if denoise {
                let content = ctx.sources().read_blob(&part.content)?;
                match Denoise::new(DENOISE_SEED).generate(&source, &part.name, &content) {
                    Ok(task) => batch.admit(ctx, DENOISE_KIND, task, DENOISE_GENERATOR, None)?,
                    Err(e) => batch.reject(DENOISE_KIND, &e.to_string()),
                }
            }
            if let Some(generator) = &generator {
                let text = SourceText::load(&ctx.sources(), source_id, &part.name)?;
                generate_part(ctx, generator, &text, &model_kinds, &mut batch)?;
            }
        }
    }
    let set = TaskSet {
        name: set_name(request),
        members: batch.entries,
    };
    let tasks = set.members.len();
    Ok(TasksGenerated {
        task_set: ctx.tasks().put_set(&set)?,
        tasks,
        parts: batch.parts,
        per_kind: batch.per_kind,
        rejected: batch.rejected,
        stopped,
    })
}

/// `kind` with the learner's goal added to its brief.
fn with_goal(kind: &TaskKind, goal: Option<&str>) -> TaskKind {
    let mut kind = kind.clone();
    if let Some(goal) = goal.filter(|g| !g.trim().is_empty()) {
        kind.brief = format!(
            "{}\n\nThe learner's goal: {goal}. Prefer tasks that serve it.",
            kind.brief
        );
    }
    kind
}

/// The generator for the model kinds, offered the runtimes they run code
/// in; refused when one of those runtimes is not available here.
fn model_generator(
    ctx: &Context,
    request: &Generation<'_>,
    kinds: &[TaskKind],
) -> Result<ModelTaskGenerator, CampaignError> {
    let model = ctx.model(request.generator)?;
    let mut runtimes = Vec::new();
    for name in kinds.iter().filter_map(|k| k.runtime.as_deref()) {
        let runtime = ctx.environments().runtime(name).map_err(|e| {
            CampaignError::Refused(format!(
                "task kinds that run {name} need it available here: {e}"
            ))
        })?;
        runtimes.push(runtime);
    }
    let policy = GenerationPolicy {
        deadline: remaining(request.deadline, DEFAULT_REQUEST_DEADLINE),
        ..GenerationPolicy::default()
    };
    Ok(ModelTaskGenerator::new(model, ctx.sources())
        .with_runtimes(runtimes)
        .with_policy(policy)
        .with_cancel(request.cancel.clone()))
}

/// One text part, a window of sections at a time.
fn generate_part(
    ctx: &Context,
    generator: &ModelTaskGenerator,
    text: &SourceText,
    kinds: &[TaskKind],
    batch: &mut Batch,
) -> Result<(), CampaignError> {
    let kind_refs: Vec<&TaskKind> = kinds.iter().collect();
    let positions: Vec<usize> = (0..text.sections().len()).collect();
    for window in positions.chunks(SECTIONS_PER_REQUEST) {
        let shown = text.clone().select(window)?;
        let report = match ctx.block_on(generator.generate(&shown, &kind_refs)) {
            Ok(report) => report,
            Err(GenerateError::NoSections) => return Ok(()),
            Err(e) => return Err(e.into()),
        };
        for (kind, counts) in &report.per_kind {
            batch.per_kind.entry(kind.clone()).or_default().rejected += counts.rejected;
        }
        for (reason, count) in &report.rejected {
            let name = serde_json::to_value(reason)
                .ok()
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap_or_else(|| format!("{reason:?}"));
            *batch.rejected.entry(name).or_default() += count;
        }
        for generated in report.admitted {
            let kind = generated.task.task.kind.clone();
            batch.admit(
                ctx,
                &kind,
                generated.task,
                &generated.generator,
                Some(generated.prompt),
            )?;
        }
    }
    Ok(())
}

/// The tasks admitted so far, and the tallies.
#[derive(Default)]
struct Batch {
    entries: Vec<TaskEntry>,
    seen: BTreeSet<Digest>,
    parts: usize,
    per_kind: BTreeMap<String, KindTally>,
    rejected: BTreeMap<String, usize>,
}

impl Batch {
    /// Stores `task` and adds it to the set, once.
    fn admit(
        &mut self,
        ctx: &Context,
        kind: &str,
        task: Task,
        generator: &str,
        prompt: Option<Digest>,
    ) -> Result<(), CampaignError> {
        let id = ctx.tasks().put(&task)?;
        if self.seen.insert(id.clone()) {
            self.per_kind.entry(kind.to_string()).or_default().admitted += 1;
            self.entries.push(TaskEntry {
                task: id,
                generator: Some(generator.to_string()),
                prompt,
            });
        }
        Ok(())
    }

    fn reject(&mut self, kind: &str, reason: &str) {
        self.per_kind.entry(kind.to_string()).or_default().rejected += 1;
        *self.rejected.entry(reason.to_string()).or_default() += 1;
    }
}

/// What a set is called: its kinds, its sources, and the goal.
fn set_name(request: &Generation<'_>) -> String {
    let mut name = format!(
        "{} tasks from {} source(s) by {}",
        request.kinds.join("+"),
        request.sources.len(),
        request.generator
    );
    if let Some(goal) = request.goal {
        name.push_str(&format!(" for: {goal}"));
    }
    name
}

/// What is left of `default` before `deadline`.
pub(crate) fn remaining(deadline: Option<Instant>, default: Duration) -> Duration {
    deadline.map_or(default, |d| {
        default.min(d.saturating_duration_since(Instant::now()))
    })
}

/// One task set, as `tasks list` shows it.
#[derive(Clone, Debug, Serialize)]
pub struct TaskSetSummary {
    /// Its id.
    pub id: TaskSetId,
    /// Its name.
    pub name: String,
    /// Tasks in it.
    pub tasks: usize,
}

/// What `tasks list` reports.
#[derive(Clone, Debug, Serialize)]
pub struct TaskSetList {
    /// Every stored task set, in id order.
    pub task_sets: Vec<TaskSetSummary>,
}

/// Every stored task set.
pub fn list(ctx: &Context) -> Result<TaskSetList, CampaignError> {
    let store = ctx.tasks();
    let task_sets = store
        .list_sets()?
        .into_iter()
        .map(|id| {
            let set = store.get_set(&id)?;
            Ok(TaskSetSummary {
                tasks: set.members.len(),
                name: set.name,
                id,
            })
        })
        .collect::<Result<_, CampaignError>>()?;
    Ok(TaskSetList { task_sets })
}

/// One task of a set, as `tasks show` lists it.
#[derive(Clone, Debug, Serialize)]
pub struct TaskLine {
    /// The task's id.
    pub task: Digest,
    /// Its kind.
    pub kind: String,
    /// What the student is asked.
    pub instruction: String,
    /// Who generated it.
    pub generator: Option<String>,
}

/// What `tasks show` reports: a set with its tasks, or one task whole.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "shows", rename_all = "snake_case")]
pub enum TaskShow {
    /// A task set.
    Set {
        /// Its id.
        id: TaskSetId,
        /// Its name.
        name: String,
        /// Its tasks, in order.
        tasks: Vec<TaskLine>,
    },
    /// One task, privileged material included.
    Task {
        /// The task.
        task: Box<Task>,
    },
}

/// The task set or task `id` names.
pub fn show(ctx: &Context, id: &str) -> Result<TaskShow, CampaignError> {
    let store = ctx.tasks();
    match resolve_set(ctx, id) {
        Ok(set_id) => {
            let set = store.get_set(&set_id)?;
            let tasks = set
                .members
                .into_iter()
                .map(|entry| {
                    let task = store.get(&entry.task)?;
                    Ok(TaskLine {
                        task: entry.task,
                        kind: task.task.kind,
                        instruction: task.instruction,
                        generator: entry.generator,
                    })
                })
                .collect::<Result<_, CampaignError>>()?;
            Ok(TaskShow::Set {
                id: set_id,
                name: set.name,
                tasks,
            })
        }
        Err(CampaignError::NotFound { .. }) => {
            let task_id = ids::resolve("task set or task", id, store.list()?)?;
            Ok(TaskShow::Task {
                task: Box::new(store.get(&task_id)?),
            })
        }
        Err(e) => Err(e),
    }
}

/// The stored task set `id` (or a unique prefix of it) names.
pub fn resolve_set(ctx: &Context, id: &str) -> Result<TaskSetId, CampaignError> {
    let stored = ctx.tasks().list_sets()?.into_iter().map(|s| s.0);
    Ok(TaskSetId(ids::resolve("task set", id, stored)?))
}
