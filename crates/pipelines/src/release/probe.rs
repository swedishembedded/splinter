// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements closed-book evaluation suites graded by
// verifiers the model under test cannot reach, for its clients. If your
// team needs expertise in model evaluation, you can procure our services
// by sending an email to info@swedishembedded.com.

//! Suites of tasks a model is probed on, and the grading of one model on
//! one suite.
//!
//! A held-out suite is the tasks of the records training held out: the
//! datasets are concatenated in order and split by the holdout rule
//! (`splinter_data::holdout`), exactly as training split them, and each
//! held-out record's task is found in the task store (or, for a record of a
//! revision, in the experience it was projected from). A probe is
//! closed-book: the model sees the instruction alone. A task that is not
//! closed-book, or that cannot be found, is excluded and counted.
//!
//! The variants suite is the same fact asked in other words: the stored
//! variants ([`crate::variants`]) of the tasks the datasets' trained-on
//! records were projected from. It is built from what training trained on,
//! so only the facts the candidate learned are measured; a variant of a
//! held-out record's task measures nothing it learned.
//!
//! A model is probed decoding greedily where its sampling can be set here
//! ([`greedy`]), so a verdict is the weights', not one draw's.
//!
//! A suite built for a candidate that works in a workspace holds tasks that
//! are changes to a repository ([`Suite::workspace`]); they are attempted by
//! the worker in their own copy of the snapshot and graded by the workspace
//! verifier and the exposure audit (`crate::release::agentic`), and a
//! suite built from a dataset's records (the held-out records, the variants
//! of trained tasks) never holds one: a record's task is a task the policy
//! solved, which is not an exam.
//!
//! A probe runs under the one system prompt every solve runs under and
//! every training record shows (`splinter_core::prompt::SYSTEM_PROMPT`). Grading is
//! by the task kind's own verifiers, without a judge, and the store's
//! decision rule over their verdicts: `Some(true)` right, `Some(false)`
//! wrong, `None` when no verifier decided. Each [`Probe`] keeps the answer
//! beside its verdict, so two models - or one model on two serving paths -
//! can be compared on what they said, not only on how it was graded.
//! Probe answers are not stored as experiences, so nothing a probe
//! produces can reach a training set.

use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;
use splinter_agent::solve::{solve, solve_prompted, Model, SolveOptions};
use splinter_agent::CancelToken;
use splinter_core::dataset::DatasetId;
use splinter_core::digest::Digest;
use splinter_core::experience::{Environment, Experience, ExperienceId, Provenance, Task};
use splinter_data::holdout::holdout_split_records;
use splinter_eval::gate::{AgenticConfig, SuiteSummary, Withheld};
use splinter_eval::paired::PairedOutcome;
use splinter_eval::verifiers::Strongest;
use splinter_model::local::GREEDY_SAMPLING;
use splinter_model::Sampling;
use splinter_sandbox::ResolvedEnvironment;
use splinter_store::decision::decide;

use crate::release::agentic;
use crate::solving::DEFAULT_SOLVE_DEADLINE;
use crate::variants::stored_variants;
use crate::verify::{verifiers_for, Judge, Use};
use splinter_core::model_ref::ModelRef;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::{io, OrchestratorError};

/// Why a task was left out of a suite: it is not solved closed-book.
pub const NOT_CLOSED_BOOK: &str = "not_closed_book";
/// Why a task was left out of a workspace suite: it is not a change to a
/// repository.
pub const NOT_WORKSPACE: &str = "not_workspace";
/// Why a record was left out of a suite: its task cannot be found.
pub const TASK_UNKNOWN: &str = "task_unknown";

/// Tasks a model is probed on.
#[derive(Clone, Debug)]
pub struct Suite {
    /// What the suite is: `held-out`, `retention:<release>`, `anchor`, or
    /// a file.
    pub name: String,
    /// The tasks, each once, in the order they were found.
    pub tasks: Vec<Task>,
    /// Records or tasks left out, by reason.
    pub excluded: BTreeMap<String, usize>,
    /// How its tasks that work in a workspace are attempted: the same for
    /// every model probed on it.
    pub work: AgenticConfig,
}

impl Suite {
    /// What this suite was.
    #[must_use]
    pub fn summary(&self) -> SuiteSummary {
        SuiteSummary {
            name: self.name.clone(),
            tasks: self.tasks.len(),
            excluded: self.excluded.clone(),
        }
    }

    /// A suite of `tasks`, excluding (and counting) those not closed-book
    /// and repeats.
    #[must_use]
    pub fn of_tasks(name: impl Into<String>, tasks: Vec<Task>) -> Self {
        let mut suite = Self::empty(name);
        for task in tasks {
            suite.add(task);
        }
        suite
    }

    /// A suite of `tasks` that are changes to a repository, excluding (and
    /// counting) those that are not, and repeats.
    #[must_use]
    pub fn workspace(name: impl Into<String>, tasks: Vec<Task>) -> Self {
        let mut suite = Self::empty(name);
        for task in tasks {
            suite.add_workspace(task);
        }
        suite
    }

    fn empty(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            tasks: Vec::new(),
            excluded: BTreeMap::new(),
            work: AgenticConfig::default(),
        }
    }

    /// This suite, its tasks that work in a workspace attempted as `work`
    /// says.
    #[must_use]
    pub fn with_work(mut self, work: AgenticConfig) -> Self {
        self.work = work;
        self
    }

    /// This suite and `other`'s tasks, and what each left out. A task of
    /// `other` is one it admitted: it is not asked again of this suite's
    /// rule.
    pub(crate) fn absorb(&mut self, other: Suite) {
        for task in other.tasks {
            self.push_unique(task);
        }
        for (reason, count) in other.excluded {
            *self.excluded.entry(reason).or_default() += count;
        }
    }

    /// How many of its tasks work in a workspace.
    #[must_use]
    pub fn workspace_tasks(&self) -> usize {
        self.tasks
            .iter()
            .filter(|t| agentic::is_workspace(t))
            .count()
    }

    pub(crate) fn add(&mut self, task: Task) {
        if task.environment.kind != Environment::CLOSED_BOOK {
            *self.excluded.entry(NOT_CLOSED_BOOK.into()).or_default() += 1;
        } else {
            self.push_unique(task);
        }
    }

    /// Adds `task` if it is a change to a repository, else counts it as left
    /// out.
    pub(crate) fn add_workspace(&mut self, task: Task) {
        if agentic::is_workspace(&task) {
            self.push_unique(task);
        } else {
            *self.excluded.entry(NOT_WORKSPACE.into()).or_default() += 1;
        }
    }

    fn push_unique(&mut self, task: Task) {
        if !self.tasks.iter().any(|t| t.task.id == task.task.id) {
            self.tasks.push(task);
        }
    }
}

/// The records of `datasets`, concatenated in order, split as training
/// splits them: `(trained on, held out)`. A record is a non-blank line.
pub(crate) fn split_records(
    ctx: &Context,
    datasets: &[DatasetId],
) -> Result<(Vec<String>, Vec<String>), OrchestratorError> {
    let mut records = Vec::new();
    for id in datasets {
        let stored = ctx.datasets().get(id)?;
        let text = std::fs::read_to_string(&stored.path).map_err(io(&stored.path))?;
        records.extend(
            text.lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(str::to_string),
        );
    }
    Ok(match holdout_split_records(&records) {
        Some((train, held_out)) => (
            train.into_iter().cloned().collect(),
            held_out.into_iter().cloned().collect(),
        ),
        None => (records, Vec::new()),
    })
}

/// The held-out suite of `datasets`, named `name`.
pub fn held_out(
    ctx: &Context,
    name: impl Into<String>,
    datasets: &[DatasetId],
) -> Result<Suite, OrchestratorError> {
    let (_, held_out) = split_records(ctx, datasets)?;
    let mut suite = Suite::of_tasks(name, Vec::new());
    for record in &held_out {
        match record_task(ctx, record)? {
            Some(task) => suite.add(task),
            None => *suite.excluded.entry(TASK_UNKNOWN.into()).or_default() += 1,
        }
    }
    Ok(suite)
}

/// The variants suite of `datasets`, named `name`: the stored variants of
/// every task a trained-on record was projected from. A record's task is
/// known by the address it was generated under, the task of a retry
/// without its critiques.
pub fn trained_variants(
    ctx: &Context,
    name: impl Into<String>,
    datasets: &[DatasetId],
) -> Result<Suite, OrchestratorError> {
    let (trained_on, _) = split_records(ctx, datasets)?;
    let mut trained = BTreeSet::new();
    for line in &trained_on {
        if let Some(task) = trained_task(ctx, line)? {
            trained.insert(task);
        }
    }
    let mut suite = Suite::of_tasks(name, Vec::new());
    for (variant, original) in stored_variants(ctx)? {
        if trained.contains(&original) {
            suite.add(ctx.tasks().get(&variant)?);
        }
    }
    Ok(suite)
}

/// The tasks of the records `datasets` were trained on, each once.
pub fn trained_tasks(
    ctx: &Context,
    datasets: &[DatasetId],
) -> Result<Vec<Task>, OrchestratorError> {
    let (trained_on, _) = split_records(ctx, datasets)?;
    let mut tasks: Vec<Task> = Vec::new();
    for line in &trained_on {
        if let Some(task) = record_task(ctx, line)? {
            if !tasks.iter().any(|t| t.task.id == task.task.id) {
                tasks.push(task);
            }
        }
    }
    Ok(tasks)
}

/// The address of the task the record `line` was trained from: its task
/// as generated when that can be found, else the one its metadata names.
fn trained_task(ctx: &Context, line: &str) -> Result<Option<Digest>, OrchestratorError> {
    if let Some(task) = record_task(ctx, line)? {
        return Ok(Some(task.without_critiques()?.task.id));
    }
    Ok(serde_json::from_str::<RecordLine>(line)
        .ok()
        .and_then(|record| record.metadata.task))
}

/// Where a record came from, as its metadata says.
#[derive(Deserialize)]
struct RecordLine {
    metadata: RecordOrigin,
}

#[derive(Deserialize)]
struct RecordOrigin {
    #[serde(default)]
    task: Option<Digest>,
    #[serde(default)]
    experiences: Vec<ExperienceId>,
}

/// The task the dataset record `line` was projected from, when it can be
/// found.
fn record_task(ctx: &Context, line: &str) -> Result<Option<Task>, OrchestratorError> {
    let Ok(record) = serde_json::from_str::<RecordLine>(line) else {
        return Ok(None);
    };
    if let Some(task) = record.metadata.task {
        if ctx.tasks().contains(&task)? {
            return Ok(Some(ctx.tasks().get(&task)?));
        }
    }
    let experiences = ctx.experiences();
    for id in &record.metadata.experiences {
        if experiences.contains(id)? {
            return Ok(Some(experiences.get(id)?.to_task()));
        }
    }
    Ok(None)
}

/// The model `reference` names, decoding greedily
/// ([`GREEDY_SAMPLING`]); as it samples where its sampling cannot be set
/// here (a model reached over an API, or handed in rather than loaded).
pub fn greedy(ctx: &Context, reference: &ModelRef) -> Result<Model, OrchestratorError> {
    let greedy = Sampling {
        thinking: ctx.config().thinking,
        ..GREEDY_SAMPLING
    };
    match ctx.resampled(reference, greedy)? {
        Some(model) => Ok(model),
        None => ctx.model(reference),
    }
}

/// What a model answered to one task of a suite, and how that was graded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Probe {
    /// Its final answer; `None` when its run ended without one.
    pub answer: Option<String>,
    /// The verdict: `Some(true)` right, `Some(false)` wrong, `None` when
    /// no verifier decided.
    pub verdict: Option<bool>,
    /// The attempts at a task that works in a workspace which were set aside
    /// instead of graded; nothing for any other task.
    pub withheld: Withheld,
}

impl Probe {
    /// An answer graded `verdict`, with no attempt set aside.
    #[must_use]
    pub fn graded(answer: Option<String>, verdict: Option<bool>) -> Self {
        Self {
            answer,
            verdict,
            withheld: Withheld::default(),
        }
    }
}

/// A model to be probed: the model itself, and the reference it is loaded
/// from when it was loaded for this probe, so that each attempt at a task
/// that works in a workspace can draw from a seed of its own.
#[derive(Clone, Copy)]
pub struct Arm<'a> {
    /// The model.
    pub model: &'a Model,
    /// What it was loaded from; `None` for a model handed in (a scripted
    /// one, or one reached over an API), which samples as it does.
    pub reference: Option<&'a ModelRef>,
}

impl<'a> Arm<'a> {
    /// A model handed in.
    #[must_use]
    pub fn handed(model: &'a Model) -> Self {
        Self {
            model,
            reference: None,
        }
    }

    /// `model`, which `reference` names.
    #[must_use]
    pub fn named(reference: &'a ModelRef, model: &'a Model) -> Self {
        Self {
            model,
            reference: Some(reference),
        }
    }
}

/// `model`'s answer and outcome on every task of `suite`, in order; see
/// the module documentation.
pub fn grade(
    ctx: &Context,
    model: &Model,
    suite: &Suite,
    cancel: &CancelToken,
) -> Result<Vec<Probe>, OrchestratorError> {
    grade_on(ctx, &Arm::handed(model), suite, cancel)
}

/// `arm`'s answer and outcome on every task of `suite`, in order. A task
/// that works in a workspace is attempted by the worker and graded by its
/// verifiers (`crate::release::agentic`); every other is asked closed-book
/// as [`grade`] always asked it.
pub fn grade_on(
    ctx: &Context,
    arm: &Arm<'_>,
    suite: &Suite,
    cancel: &CancelToken,
) -> Result<Vec<Probe>, OrchestratorError> {
    if suite.workspace_tasks() == 0 {
        return grade_closed_book(ctx, arm.model, suite, cancel);
    }
    let (work, closed): (Vec<&Task>, Vec<&Task>) = suite
        .tasks
        .iter()
        .partition(|task| agentic::is_workspace(task));
    let closed_suite = Suite {
        tasks: closed.into_iter().cloned().collect(),
        ..Suite::empty(suite.name.clone())
    };
    let mut asked = if closed_suite.tasks.is_empty() {
        Vec::new()
    } else {
        grade_closed_book(ctx, arm.model, &closed_suite, cancel)?
    }
    .into_iter();
    let mut attempted = agentic::grade(ctx, arm, &work, &suite.work, cancel)?.into_iter();
    // Back in the suite's order.
    Ok(suite
        .tasks
        .iter()
        .filter_map(|task| {
            if agentic::is_workspace(task) {
                attempted.next()
            } else {
                asked.next()
            }
        })
        .collect())
}

/// `model`'s closed-book answer and outcome on every task of `suite`.
fn grade_closed_book(
    ctx: &Context,
    model: &Model,
    suite: &Suite,
    cancel: &CancelToken,
) -> Result<Vec<Probe>, OrchestratorError> {
    // A judge that is refused is refused before the model is asked anything.
    Judge::active(ctx, Use::Comparison)?;
    let answered = answer_all(ctx, model, suite, cancel)?;
    judge_all(ctx, suite, answered)
}

/// What a model answered to every task of a suite, not yet decided.
pub struct Answered(Vec<Experience>);

/// `model`'s answer to every task of `suite`, in order, as [`grade`] asks
/// them; [`judge_all`] decides them. The two are apart so that a model
/// served by another process can be stopped between them: the judge is a
/// model of its own that needs the device the server holds.
pub fn answer_all(
    ctx: &Context,
    model: &Model,
    suite: &Suite,
    cancel: &CancelToken,
) -> Result<Answered, OrchestratorError> {
    // Every task is answered before any is judged: a device holds one base
    // at a time, and a judge that is another model than the one asked would
    // otherwise be loaded and the asked model reloaded for every task.
    suite
        .tasks
        .iter()
        .map(|task| answer(ctx, model, task, cancel))
        .collect::<Result<Vec<_>, _>>()
        .map(Answered)
}

/// The outcome of every answer of `answered`, to the tasks of `suite`, as
/// [`grade`] decides it. The judge the command names grades the kinds a
/// judge grades; the judged verdict is the one a task of such a kind is
/// decided by, and two models are compared on it.
pub fn judge_all(
    ctx: &Context,
    suite: &Suite,
    answered: Answered,
) -> Result<Vec<Probe>, OrchestratorError> {
    let judge = Judge::active(ctx, Use::Comparison)?;
    suite
        .tasks
        .iter()
        .zip(answered.0)
        .map(|(task, experience)| verdict_of(ctx, task, experience, judge.as_ref()))
        .collect()
}

/// `model`'s closed-book answer to `task`, as an experience that is never
/// stored.
pub(crate) fn answer(
    ctx: &Context,
    model: &Model,
    task: &Task,
    cancel: &CancelToken,
) -> Result<Experience, OrchestratorError> {
    answer_prompted(ctx, model, task, None, cancel)
}

/// [`answer`], asked `prompt` in place of the task's instruction (the
/// instruction with material shown before it, say); the experience records
/// the task as it is.
pub(crate) fn answer_prompted(
    ctx: &Context,
    model: &Model,
    task: &Task,
    prompt: Option<&str>,
    cancel: &CancelToken,
) -> Result<Experience, OrchestratorError> {
    if cancel.is_cancelled() {
        return Err(OrchestratorError::Cancelled);
    }
    let mut options = SolveOptions::new(DEFAULT_SOLVE_DEADLINE);
    options.cancel = Some(cancel.clone());
    let options = model.solving(options);
    let environment = ResolvedEnvironment::ClosedBook;
    let solution = match prompt {
        None => ctx.block_on(solve(task, &environment, model.provider.clone(), options))?,
        Some(prompt) => ctx.block_on(solve_prompted(
            task,
            prompt,
            &environment,
            model.provider.clone(),
            options,
        ))?,
    };
    if cancel.is_cancelled() {
        return Err(OrchestratorError::Cancelled);
    }
    Ok(solution.into_experience(
        task.clone(),
        Provenance::new(model.identity.clone(), ctx.clock()),
    )?)
}

/// `experience`, `task`'s answer, decided by the task's verifiers.
fn verdict_of(
    ctx: &Context,
    task: &Task,
    experience: Experience,
    judge: Option<&Judge>,
) -> Result<Probe, OrchestratorError> {
    let verifiers: Strongest = verifiers_for(ctx, task, &[], judge)?;
    let verification = verifiers.run(task, &experience)?;
    Ok(Probe::graded(
        experience.final_output,
        decide(&verification.annotations).map(|d| d.passed),
    ))
}

/// `candidate` and `baseline` outcomes on `suite`'s tasks, paired by task.
#[must_use]
pub fn pair(suite: &Suite, candidate: &[Probe], baseline: &[Probe]) -> Vec<PairedOutcome> {
    suite
        .tasks
        .iter()
        .zip(candidate.iter().zip(baseline))
        .map(|(task, (c, b))| PairedOutcome {
            item: task.task.id.to_string(),
            candidate: c.verdict,
            baseline: b.verdict,
            cluster: None,
        })
        .collect()
}

/// `outcomes` with the cluster of each task of `suite` set: the family of
/// source text it is grounded in ([`crate::grouping::task_clusters`]), or for
/// a change to a repository the file its fault was seeded in, so a
/// significance test counts a family once.
pub fn clustered(
    ctx: &Context,
    suite: &Suite,
    mut outcomes: Vec<PairedOutcome>,
) -> Result<Vec<PairedOutcome>, OrchestratorError> {
    let clusters = crate::grouping::task_clusters(ctx, &suite.tasks)?;
    for ((outcome, cluster), task) in outcomes.iter_mut().zip(clusters).zip(&suite.tasks) {
        // A change to a repository is grounded in no source text: its family
        // is the file its fault was seeded in, which it declares as its first
        // concept.
        outcome.cluster = cluster.or_else(|| {
            agentic::is_workspace(task)
                .then(|| task.concepts.first().cloned())
                .flatten()
        });
    }
    Ok(outcomes)
}
