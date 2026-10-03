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
use splinter_agent::solve::{solve, Model, SolveOptions};
use splinter_agent::CancelToken;
use splinter_core::digest::Digest;
use splinter_core::experience::{Environment, Experience, ExperienceId, Provenance, Task};
use splinter_data::holdout::holdout_split_records;
use splinter_data::DatasetId;
use splinter_eval::gate::SuiteSummary;
use splinter_eval::paired::PairedOutcome;
use splinter_eval::verifiers::Strongest;
use splinter_model::local::GREEDY_SAMPLING;
use splinter_sandbox::ResolvedEnvironment;
use splinter_store::decision::decide;

use crate::context::Context;
use crate::error::{io, CampaignError};
use crate::model_ref::ModelRef;
use crate::solving::DEFAULT_SOLVE_DEADLINE;
use crate::variants::stored_variants;
use crate::verify::verifiers_for;

/// Why a task was left out of a suite: it is not solved closed-book.
pub const NOT_CLOSED_BOOK: &str = "not_closed_book";
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
        let mut suite = Self {
            name: name.into(),
            tasks: Vec::new(),
            excluded: BTreeMap::new(),
        };
        for task in tasks {
            suite.add(task);
        }
        suite
    }

    /// This suite and `other`'s tasks, and what each left out.
    pub(crate) fn absorb(&mut self, other: Suite) {
        for task in other.tasks {
            self.add(task);
        }
        for (reason, count) in other.excluded {
            *self.excluded.entry(reason).or_default() += count;
        }
    }

    pub(crate) fn add(&mut self, task: Task) {
        if task.environment.kind != Environment::CLOSED_BOOK {
            *self.excluded.entry(NOT_CLOSED_BOOK.into()).or_default() += 1;
        } else if !self.tasks.iter().any(|t| t.task.id == task.task.id) {
            self.tasks.push(task);
        }
    }
}

/// The records of `datasets`, concatenated in order, split as training
/// splits them: `(trained on, held out)`. A record is a non-blank line.
pub(crate) fn split_records(
    ctx: &Context,
    datasets: &[DatasetId],
) -> Result<(Vec<String>, Vec<String>), CampaignError> {
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
) -> Result<Suite, CampaignError> {
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
) -> Result<Suite, CampaignError> {
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

/// The address of the task the record `line` was trained from: its task
/// as generated when that can be found, else the one its metadata names.
fn trained_task(ctx: &Context, line: &str) -> Result<Option<Digest>, CampaignError> {
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
fn record_task(ctx: &Context, line: &str) -> Result<Option<Task>, CampaignError> {
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
pub fn greedy(ctx: &Context, reference: &ModelRef) -> Result<Model, CampaignError> {
    match ctx.resampled(reference, GREEDY_SAMPLING)? {
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
}

/// `model`'s answer and outcome on every task of `suite`, in order; see
/// the module documentation.
pub fn grade(
    ctx: &Context,
    model: &Model,
    suite: &Suite,
    cancel: &CancelToken,
) -> Result<Vec<Probe>, CampaignError> {
    suite
        .tasks
        .iter()
        .map(|task| grade_one(ctx, model, task, cancel))
        .collect()
}

/// `model`'s closed-book answer to `task`, as an experience that is never
/// stored.
pub(crate) fn answer(
    ctx: &Context,
    model: &Model,
    task: &Task,
    cancel: &CancelToken,
) -> Result<Experience, CampaignError> {
    if cancel.is_cancelled() {
        return Err(CampaignError::Cancelled);
    }
    let mut options = SolveOptions::new(DEFAULT_SOLVE_DEADLINE);
    options.cancel = Some(cancel.clone());
    options.stream_idle = model.stream_idle;
    let solution = ctx.block_on(solve(
        task,
        &ResolvedEnvironment::ClosedBook,
        model.provider.clone(),
        options,
    ))?;
    if cancel.is_cancelled() {
        return Err(CampaignError::Cancelled);
    }
    Ok(solution.into_experience(
        task.clone(),
        Provenance::new(model.identity.clone(), ctx.clock()),
    )?)
}

fn grade_one(
    ctx: &Context,
    model: &Model,
    task: &Task,
    cancel: &CancelToken,
) -> Result<Probe, CampaignError> {
    let experience = answer(ctx, model, task, cancel)?;
    let verifiers: Strongest = verifiers_for(ctx, task, &[], None)?;
    let verification = verifiers.run(task, &experience)?;
    Ok(Probe {
        verdict: decide(&verification.annotations).map(|d| d.passed),
        answer: experience.final_output,
    })
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
        })
        .collect()
}
