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
//! no model and yields one task per text part. Single sections - those of
//! concepts queued for new tasks - are shown to the generator model one at
//! a time. The set records which generator produced each task, from which
//! prompt, and what its question is about.
//!
//! Across the whole set, two tasks that ask the same question of the same
//! subject with answers that disagree are both left out
//! ([`splinter_knowledge::tasks::dedup::contradictions`]): two sources may
//! well disagree, and a model trained on both answers learns neither.

use std::sync::Arc;

use splinter_agent::proposer::SvenProposer;
use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

use serde::Serialize;
use splinter_agent::CancelToken;
use splinter_core::digest::Digest;
use splinter_core::experience::{PrivilegedKind, Task};
use splinter_core::kinds::DENOISE as DENOISE_KIND;
use splinter_core::source::SourceId;
use splinter_knowledge::advice::{advice_sections, judgment_sections};
use splinter_knowledge::concepts::SectionRef;
use splinter_knowledge::denoise::{Denoise, GENERATOR as DENOISE_GENERATOR};
use splinter_knowledge::tasks::dedup::{contradictions, Asked};
use splinter_knowledge::tasks::{
    Catalogue, Focus, GenerateError, GenerationPolicy, ModelTaskGenerator, Rejection, SourceText,
    TaskKind, DEFAULT_REQUEST_DEADLINE,
};
use splinter_store::tasks::{TaskEntry, TaskSet, TaskSetId};

use splinter_core::model_ref::ModelRef;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_orchestrator::ids;

/// Sections one generation request shows the model: enough context for
/// tasks that combine sections, few enough that a small model keeps its
/// answers tied to what it was shown instead of to the reply example.
pub const SECTIONS_PER_REQUEST: usize = 3;

/// The most windows of [`SECTIONS_PER_REQUEST`] sections a text part is shown
/// through for the kinds that read every section. A longer part is sampled at
/// evenly spaced windows so that its end is represented as well as its start,
/// and a run over more text than its budget covers still touches every part
/// it reaches.
pub const MAX_WINDOWS_PER_PART: usize = 4;

/// How many proposals of a kind are refused before its admission rate is
/// judged.
pub const MIN_PROPOSALS_BEFORE_GIVING_UP: usize = 24;

/// The share of a kind's proposals that must be admitted for it to be asked
/// for again once [`MIN_PROPOSALS_BEFORE_GIVING_UP`] have been made.
pub const MIN_ADMISSION_RATE: f64 = 0.1;

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
pub fn check_kinds(names: &[String]) -> Result<Vec<String>, OrchestratorError> {
    if names.is_empty() {
        return Err(OrchestratorError::Refused(
            "name at least one task kind".into(),
        ));
    }
    let known = known_kinds();
    let mut seen = BTreeSet::new();
    let mut kinds = Vec::new();
    for name in names {
        if !known.contains(name) {
            return Err(OrchestratorError::Refused(format!(
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
    /// Single sections to generate from beside them, one request each: the
    /// sections of concepts queued for new tasks. Only the model kinds
    /// generate from a section.
    pub sections: &'a [SectionRef],
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

/// One proposal turned away, and exactly why.
#[derive(Clone, Debug, Serialize)]
pub struct RejectionNote {
    /// The kind it was proposed as.
    pub kind: String,
    /// The reason, as [`TasksGenerated::rejected`] counts it.
    pub reason: String,
    /// What failed: for a malformed reply, the reply itself.
    pub detail: String,
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
    /// Every rejection of a model's proposal, with what failed.
    pub rejections: Vec<RejectionNote>,
    /// Single sections generated from, beside the sources' parts.
    pub sections: usize,
    /// Why generation stopped before every part was covered, if it did.
    pub stopped: Option<String>,
    /// The kinds generation gave up on and why: asked for until their
    /// proposals were refused too often to be worth another request.
    pub dropped: BTreeMap<String, String>,
}

/// Generates tasks of `request.kinds` from `request.sources` and stores
/// them as a task set.
pub fn generate(
    ctx: &Context,
    request: &Generation<'_>,
) -> Result<TasksGenerated, OrchestratorError> {
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
    let mut batch = Batch {
        deadline: request.deadline,
        ..Batch::default()
    };
    let mut stopped = None;
    'sources: for source_id in request.sources {
        let source = ctx.sources().get_source(source_id)?;
        let mut parts: Vec<_> = source
            .parts
            .iter()
            .filter(|p| p.media_type.starts_with("text/"))
            .collect();
        // Not in name order: a run that stops early has covered a spread of
        // the parts, not the first files.
        parts.sort_by_cached_key(|p| Digest::of(p.name.as_bytes()));
        for part in parts {
            if request.cancel.is_cancelled() {
                return Err(OrchestratorError::Cancelled);
            }
            if request.deadline.is_some_and(|d| Instant::now() >= d) {
                stopped = Some("the budget was spent before every part was covered".into());
                break 'sources;
            }
            batch.parts += 1;
            if denoise {
                let content = ctx.sources().read_blob(&part.content)?;
                match Denoise::new(DENOISE_SEED).generate(&source, &part.name, &content) {
                    Ok(task) => batch.admit(
                        ctx,
                        DENOISE_KIND,
                        task,
                        Provenance {
                            generator: DENOISE_GENERATOR,
                            prompt: None,
                            subject: None,
                        },
                    )?,
                    Err(e) => batch.reject(DENOISE_KIND, &e.to_string()),
                }
            }
            if let Some(generator) = &generator {
                let text = SourceText::load(&ctx.sources(), source_id, &part.name)?;
                generate_part(ctx, generator, &text, &model_kinds, &mut batch)?;
                if batch.expired {
                    stopped = Some("the budget was spent inside a part".into());
                    break 'sources;
                }
            }
        }
    }
    let mut sections = 0;
    if stopped.is_none() && generator.is_none() && !request.sections.is_empty() {
        *batch
            .rejected
            .entry("no model kind was asked for to generate from a queued section".into())
            .or_default() += request.sections.len();
    }
    if stopped.is_none() {
        if let Some(generator) = &generator {
            for section in request.sections {
                if request.cancel.is_cancelled() {
                    return Err(OrchestratorError::Cancelled);
                }
                if request.deadline.is_some_and(|d| Instant::now() >= d) {
                    stopped = Some("the budget was spent before every section was covered".into());
                    break;
                }
                let text = SourceText::load(&ctx.sources(), &section.source, &section.part)?;
                match text.select(&[section.section]) {
                    Ok(shown) => {
                        sections += 1;
                        generate_part(ctx, generator, &shown, &model_kinds, &mut batch)?;
                    }
                    Err(e) => *batch.rejected.entry(format!("{section}: {e}")).or_default() += 1,
                }
            }
        }
    }
    batch.drop_contradictions(&catalogue);
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
        rejections: batch.rejections,
        sections,
        stopped,
        dropped: batch.dropped,
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
) -> Result<ModelTaskGenerator, OrchestratorError> {
    let model = ctx.model(request.generator)?;
    let mut runtimes = Vec::new();
    for name in kinds.iter().filter_map(|k| k.runtime.as_deref()) {
        let runtime = ctx.environments().runtime(name).map_err(|e| {
            OrchestratorError::Refused(format!(
                "task kinds that run {name} need it available here: {e}"
            ))
        })?;
        runtimes.push(runtime);
    }
    let policy = GenerationPolicy {
        deadline: remaining(request.deadline, DEFAULT_REQUEST_DEADLINE),
        ..GenerationPolicy::default()
    };
    let proposer = SvenProposer::new(model).with_cancel(request.cancel.clone());
    Ok(ModelTaskGenerator::new(Arc::new(proposer), ctx.sources())
        .with_runtimes(runtimes)
        .with_policy(policy))
}

/// One text part, a window of sections at a time.
fn generate_part(
    ctx: &Context,
    generator: &ModelTaskGenerator,
    text: &SourceText,
    kinds: &[TaskKind],
    batch: &mut Batch,
) -> Result<(), OrchestratorError> {
    // A kind with a focus is shown only the sections it is about, by
    // itself; the rest are shown every section together.
    let (focused, general): (Vec<&TaskKind>, Vec<&TaskKind>) =
        kinds.iter().partition(|kind| kind.focus.is_some());
    if !general.is_empty() {
        let positions: Vec<usize> = (0..text.sections().len()).collect();
        let pass = Pass {
            text,
            positions: &positions,
            kinds: &general,
            cap: Some(MAX_WINDOWS_PER_PART),
        };
        run_windows(ctx, generator, &pass, batch)?;
    }
    for kind in focused {
        let positions = match kind.focus {
            Some(Focus::Advice) => advice_sections(text),
            Some(Focus::Judgment) => judgment_sections(text),
            None => continue,
        };
        let pass = Pass {
            text,
            positions: &positions,
            kinds: &[kind],
            cap: None,
        };
        run_windows(ctx, generator, &pass, batch)?;
    }
    Ok(())
}

/// At most `cap` of `windows`, evenly spaced from the first to the last; all
/// of them when there are no more than `cap` (or no cap).
fn spread<T: Copy>(windows: Vec<T>, cap: Option<usize>) -> Vec<T> {
    match cap {
        Some(cap) if cap >= 2 && windows.len() > cap => (0..cap)
            .map(|i| windows[i * (windows.len() - 1) / (cap - 1)])
            .collect(),
        _ => windows,
    }
}

/// One pass over a text part: which of its sections, for which kinds, and
/// how many requests it may make.
#[derive(Clone, Copy)]
struct Pass<'a> {
    text: &'a SourceText,
    /// The sections put to the generator.
    positions: &'a [usize],
    kinds: &'a [&'a TaskKind],
    /// The most windows asked; `None` asks for every one.
    cap: Option<usize>,
}

/// The sections of the pass, a window of [`SECTIONS_PER_REQUEST`] at a time,
/// put to the generator for its kinds.
fn run_windows(
    ctx: &Context,
    generator: &ModelTaskGenerator,
    pass: &Pass<'_>,
    batch: &mut Batch,
) -> Result<(), OrchestratorError> {
    let Pass {
        text,
        positions,
        kinds,
        cap,
    } = *pass;
    let windows = spread(positions.chunks(SECTIONS_PER_REQUEST).collect(), cap);
    for window in windows {
        if batch.deadline.is_some_and(|d| Instant::now() >= d) {
            batch.expired = true;
            return Ok(());
        }
        let active = batch.still_worth_asking(kinds);
        if active.is_empty() {
            return Ok(());
        }
        let shown = text.clone().select(window)?;
        let report = match ctx.block_on(generator.generate(&shown, &active)) {
            Ok(report) => report,
            Err(GenerateError::NoSections) => return Ok(()),
            Err(e) => return Err(e.into()),
        };
        for (kind, counts) in &report.per_kind {
            batch.per_kind.entry(kind.clone()).or_default().rejected += counts.rejected;
        }
        for (reason, count) in &report.rejected {
            *batch.rejected.entry(reason_name(reason)).or_default() += count;
        }
        batch
            .rejections
            .extend(report.rejections.into_iter().map(|r| RejectionNote {
                reason: reason_name(&r.reason),
                kind: r.kind,
                detail: r.detail,
            }));
        for generated in report.admitted {
            let kind = generated.task.task.kind.clone();
            batch.admit(
                ctx,
                &kind,
                generated.task,
                Provenance {
                    generator: &generated.generator,
                    prompt: Some(generated.prompt),
                    subject: generated.subject,
                },
            )?;
        }
    }
    Ok(())
}

/// How an admitted task came to be, as its set entry records it.
struct Provenance<'a> {
    generator: &'a str,
    prompt: Option<Digest>,
    subject: Option<String>,
}

/// A question of the set as the contradiction rule compares it.
struct Question {
    task: Digest,
    kind: String,
    instruction: String,
    subject: Option<String>,
    reference: String,
}

/// The tasks admitted so far, and the tallies.
#[derive(Default)]
struct Batch {
    entries: Vec<TaskEntry>,
    seen: BTreeSet<Digest>,
    /// The admitted questions whose answers can contradict one another.
    questions: Vec<Question>,
    parts: usize,
    per_kind: BTreeMap<String, KindTally>,
    rejected: BTreeMap<String, usize>,
    rejections: Vec<RejectionNote>,
    /// No window is put to the generator after this.
    deadline: Option<Instant>,
    /// Whether the deadline passed with windows still to cover.
    expired: bool,
    /// The kinds given up on, and why.
    dropped: BTreeMap<String, String>,
}

impl Batch {
    /// `kinds` without those the sources have shown they cannot satisfy: a
    /// kind with at least [`MIN_PROPOSALS_BEFORE_GIVING_UP`] proposals of which
    /// fewer than [`MIN_ADMISSION_RATE`] were admitted is dropped, and why is
    /// recorded. Questions about letters, for one, rarely name a subject, and
    /// asking a kind that needs one for every window spends the budget on
    /// proposals that are all refused.
    fn still_worth_asking<'k>(&mut self, kinds: &[&'k TaskKind]) -> Vec<&'k TaskKind> {
        let mut active = Vec::with_capacity(kinds.len());
        for &kind in kinds {
            if self.dropped.contains_key(&kind.name) {
                continue;
            }
            let tally = self.per_kind.get(&kind.name);
            let (admitted, rejected) = tally.map_or((0, 0), |t| (t.admitted, t.rejected));
            let proposed = admitted + rejected;
            if proposed >= MIN_PROPOSALS_BEFORE_GIVING_UP
                && (admitted as f64) < MIN_ADMISSION_RATE * proposed as f64
            {
                self.dropped.insert(
                    kind.name.clone(),
                    format!("admitted {admitted} of {proposed} proposals, too few to keep asking"),
                );
            } else {
                active.push(kind);
            }
        }
        active
    }

    /// Stores `task` and adds it to the set, once.
    fn admit(
        &mut self,
        ctx: &Context,
        kind: &str,
        task: Task,
        provenance: Provenance<'_>,
    ) -> Result<(), OrchestratorError> {
        let id = ctx.tasks().put(&task)?;
        if self.seen.insert(id.clone()) {
            self.per_kind.entry(kind.to_string()).or_default().admitted += 1;
            let reference = task
                .privileged
                .iter()
                .find(|p| p.kind == PrivilegedKind::Reference)
                .map(|p| p.content.clone());
            if let Some(reference) = reference {
                self.questions.push(Question {
                    task: id.clone(),
                    kind: kind.to_string(),
                    instruction: task.instruction,
                    subject: provenance.subject.clone(),
                    reference,
                });
            }
            self.entries.push(TaskEntry {
                task: id,
                generator: Some(provenance.generator.to_string()),
                prompt: provenance.prompt,
                variant_of: None,
                subject: provenance.subject,
            });
        }
        Ok(())
    }

    fn reject(&mut self, kind: &str, reason: &str) {
        self.per_kind.entry(kind.to_string()).or_default().rejected += 1;
        *self.rejected.entry(reason.to_string()).or_default() += 1;
    }

    /// Leaves out every task that asks the same question of the same
    /// subject as another and answers it differently, of the kinds whose
    /// answer is one exact fact ([`TaskKind::exact_answer`]).
    fn drop_contradictions(&mut self, catalogue: &Catalogue) {
        let policy = GenerationPolicy::default();
        let questions: Vec<&Question> = self
            .questions
            .iter()
            .filter(|q| catalogue.get(&q.kind).is_some_and(TaskKind::exact_answer))
            .collect();
        let asked: Vec<Asked<'_>> = questions
            .iter()
            .map(|q| Asked {
                instruction: &q.instruction,
                subject: q.subject.as_deref(),
                reference: &q.reference,
            })
            .collect();
        let mut dropped: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (i, j) in contradictions(&asked, policy.shingle_words, policy.max_overlap) {
            dropped.entry(i).or_default().push(j);
            dropped.entry(j).or_default().push(i);
        }
        if dropped.is_empty() {
            return;
        }
        let reason = reason_name(&Rejection::Contradiction);
        let mut gone = BTreeSet::new();
        for (at, others) in &dropped {
            let question = questions[*at];
            let answers: Vec<String> = others
                .iter()
                .map(|&o| format!("{:?}", questions[o].reference))
                .collect();
            let tally = self.per_kind.entry(question.kind.clone()).or_default();
            tally.admitted -= 1;
            tally.rejected += 1;
            *self.rejected.entry(reason.clone()).or_default() += 1;
            self.rejections.push(RejectionNote {
                kind: question.kind.clone(),
                reason: reason.clone(),
                detail: format!(
                    "{:?} answers {:?}; another task of the set asks it of {:?} and answers {}",
                    question.instruction,
                    question.reference,
                    question.subject.as_deref().unwrap_or_default(),
                    answers.join(", ")
                ),
            });
            gone.insert(question.task.clone());
        }
        self.entries.retain(|entry| !gone.contains(&entry.task));
        self.questions.retain(|q| !gone.contains(&q.task));
    }
}

/// A rejection reason as the report names it: its serialized form.
pub(crate) fn reason_name(reason: &Rejection) -> String {
    serde_json::to_value(reason)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_else(|| format!("{reason:?}"))
}

/// What a set is called: its kinds, its sources and sections, and the goal.
fn set_name(request: &Generation<'_>) -> String {
    let sections = if request.sections.is_empty() {
        String::new()
    } else {
        format!(" and {} queued section(s)", request.sections.len())
    };
    let mut name = format!(
        "{} tasks from {} source(s){sections} by {}",
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
pub fn list(ctx: &Context) -> Result<TaskSetList, OrchestratorError> {
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
        .collect::<Result<_, OrchestratorError>>()?;
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
pub fn show(ctx: &Context, id: &str) -> Result<TaskShow, OrchestratorError> {
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
                .collect::<Result<_, OrchestratorError>>()?;
            Ok(TaskShow::Set {
                id: set_id,
                name: set.name,
                tasks,
            })
        }
        Err(OrchestratorError::NotFound { .. }) => {
            let task_id = ids::resolve("task set or task", id, store.list()?)?;
            Ok(TaskShow::Task {
                task: Box::new(store.get(&task_id)?),
            })
        }
        Err(e) => Err(e),
    }
}

/// The stored task set `id` (or a unique prefix of it) names.
pub fn resolve_set(ctx: &Context, id: &str) -> Result<TaskSetId, OrchestratorError> {
    let stored = ctx.tasks().list_sets()?.into_iter().map(|s| s.0);
    Ok(TaskSetId(ids::resolve("task set", id, stored)?))
}
