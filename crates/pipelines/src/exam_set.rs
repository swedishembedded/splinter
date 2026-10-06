// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements frozen held-out examinations of what a model
// learned from a person's writing, for its clients. If your team needs
// expertise in building an exam that cannot drift between the models it
// compares, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The exam set: tasks written from the reserved families alone
//! ([`crate::reserve`]), within a budget, then frozen.
//!
//! The tasks are generated once, before any candidate exists to be compared
//! on them, and chosen so that no family supplies more than its share when
//! the budget is smaller than what generation admitted. The set is a stored
//! task set, which cannot change, and a manifest beside it under the state
//! root, pinned in a ledger ([`splinter_data::frozen::Ledger`]): the same name
//! with other content is refused, so every model is examined on the same
//! tasks.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use splinter_agent::CancelToken;
use splinter_core::digest::Digest;
use splinter_core::experience::Task;
use splinter_core::source::{CapturedSource, Origin, PartContent, SourceId};
use splinter_data::frozen::Ledger;
use splinter_store::tasks::{TaskEntry, TaskSet, TaskSetId};

use crate::grouping::task_clusters;
use crate::reserve::{reserve, Reservation, ReserveRequest, ReservedFamily};
use crate::sources::{add, SourceTarget};
use crate::tasks::{generate, Generation};
use splinter_core::model_ref::ModelRef;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::{io, OrchestratorError};

/// The tasks of each family of the final test when none is named: fifty
/// families of eight make about four hundred tasks, enough for four chances in
/// five of seeing a gain of ten points at the discordance and the clustering
/// the plan assumes ([`crate::powered::plan`]).
pub const DEFAULT_TASKS_PER_FAMILY: usize = 8;

/// The tasks of each family of the dev suite when none is named.
pub const DEFAULT_DEV_TASKS_PER_FAMILY: usize = 4;

/// The directory under the state root the exam sets are kept in.
const DIRECTORY: &str = "exams";

/// The manifest's file name.
const MANIFEST: &str = "exam.json";

/// What an exam set is made from.
pub struct ExamSetRequest<'a> {
    /// The sources the exam is written from: reserved parts alone.
    pub sources: &'a [SourceId],
    /// The families reserved, whether or not a task is written from each.
    pub families: &'a [ReservedFamily],
    /// Whether this is the final test or the dev suite.
    pub role: Role,
    /// The kinds of task; the ones a judge can grade against a reference.
    pub kinds: &'a [String],
    /// The generator model.
    pub generator: &'a ModelRef,
    /// What the learner is after.
    pub goal: Option<&'a str>,
    /// Who wrote the sources, when they are one person's.
    pub author: Option<&'a str>,
    /// The most tasks of one family the exam holds.
    pub tasks_per_family: usize,
    /// Stops generation.
    pub cancel: CancelToken,
}

/// What an exam set is for.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// The final test: opened once, after one checkpoint is chosen.
    #[default]
    Final,
    /// The dev suite a checkpoint is chosen on.
    Dev,
}

/// The comparison an exam is decided on, fixed before any model is put to it.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Primary {
    /// The arm said to be better: the candidate asked under the prompt it is
    /// deployed with.
    pub first: String,
    /// The arm it is compared with: the base told the same, asked the same.
    pub second: String,
    /// What is compared: the share of tasks the judge says give what the
    /// reference says, on the greedy answer, with a family-clustered interval.
    pub endpoint: String,
}

impl Default for Primary {
    fn default() -> Self {
        Self {
            first: crate::powered::PERSONA.into(),
            second: crate::powered::PROMPTED.into(),
            endpoint: "share of tasks judged right on the greedy answer".into(),
        }
    }
}

/// One task of the exam and the family it is written from.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExamTask {
    /// The task's address.
    pub task: Digest,
    /// Its family's name.
    pub family: String,
}

/// A frozen exam: its manifest as it is written beside the task set.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ExamSet {
    /// The manifest's own address, which names the exam.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub id: String,
    /// The stored task set holding the tasks.
    pub task_set: TaskSetId,
    /// The sources the tasks were written from: the reserved parts alone.
    pub sources: Vec<SourceId>,
    /// The families reserved, whether or not tasks were written from each.
    pub families: Vec<ReservedFamily>,
    /// The tasks, each with its family.
    pub tasks: Vec<ExamTask>,
    /// The kinds asked for.
    pub kinds: Vec<String>,
    /// The most tasks of one family it was held to.
    pub tasks_per_family: usize,
    /// What it is for.
    #[serde(default)]
    pub role: Role,
    /// The primary comparison, fixed with the exam.
    #[serde(default)]
    pub primary: Primary,
    /// What the exam is made of, so that the file alone carries it to
    /// another state root: [`ExamSet::load`] installs it where it is missing.
    #[serde(default)]
    pub bundle: Bundle,
}

/// The tasks and source text of an exam, carried by its manifest.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Bundle {
    /// The tasks, in the manifest's order.
    pub tasks: Vec<Task>,
    /// The sources the tasks were written from, with their text.
    pub sources: Vec<BundledSource>,
}

/// A source with the text of its parts.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct BundledSource {
    /// Its address, which installing it must reproduce.
    pub id: SourceId,
    /// Where it came from.
    pub origin: Origin,
    /// Its text parts: `(name, media type, text)`.
    pub parts: Vec<(String, String, String)>,
}

impl ExamSet {
    /// Families that have at least one task.
    #[must_use]
    pub fn families_examined(&self) -> usize {
        self.tasks
            .iter()
            .map(|t| t.family.as_str())
            .collect::<std::collections::BTreeSet<_>>()
            .len()
    }

    /// The directory this exam is kept in under `ctx`'s state root.
    fn directory(ctx: &Context, id: &str) -> PathBuf {
        ctx.root().path().join(DIRECTORY).join(id)
    }

    /// The manifest's file.
    #[must_use]
    pub fn file(&self, ctx: &Context) -> PathBuf {
        Self::directory(ctx, &self.id).join(MANIFEST)
    }

    /// The exam named `name`: a manifest file, or an exam's id (or unique
    /// prefix of it) under the state root. Refused when the manifest is not
    /// what was frozen.
    pub fn load(ctx: &Context, name: &str) -> Result<Self, OrchestratorError> {
        let path = if Path::new(name).is_file() {
            PathBuf::from(name)
        } else {
            Self::resolve(ctx, name)?
        };
        let bytes = std::fs::read(&path).map_err(io(&path))?;
        Ledger::beside(&path)
            .check_file(&path)
            .map_err(|e| OrchestratorError::Refused(e.to_string()))?;
        let mut set: ExamSet =
            serde_json::from_slice(&bytes).map_err(|source| OrchestratorError::Json {
                what: path.display().to_string(),
                source,
            })?;
        set.id = Digest::of(&bytes).hex().chars().take(16).collect();
        set.install(ctx)?;
        Ok(set)
    }

    /// Puts what the exam is made of into `ctx`'s stores where it is not
    /// already: the tasks, and the sources with the text of their parts. The
    /// same content has the same address, so an exam made in another state
    /// root is the same exam here.
    fn install(&self, ctx: &Context) -> Result<(), OrchestratorError> {
        let tasks = ctx.tasks();
        for task in &self.bundle.tasks {
            tasks.put(task)?;
        }
        let sources = ctx.sources();
        for source in &self.bundle.sources {
            if sources.contains(&source.id)? {
                continue;
            }
            let parts = source
                .parts
                .iter()
                .map(|(name, media_type, text)| PartContent {
                    name: name.clone(),
                    media_type: media_type.clone(),
                    bytes: text.clone().into_bytes(),
                })
                .collect();
            let captured = CapturedSource::new(source.origin.clone(), parts, ctx.clock())
                .map_err(|e| OrchestratorError::Refused(format!("the exam's sources: {e}")))?;
            if captured.source().id != source.id {
                return Err(OrchestratorError::Refused(format!(
                    "the exam's source {} does not install as itself: it holds a part that is not \
                     text",
                    source.id
                )));
            }
            sources.put_source(&captured)?;
        }
        Ok(())
    }

    fn resolve(ctx: &Context, prefix: &str) -> Result<PathBuf, OrchestratorError> {
        let root = ctx.root().path().join(DIRECTORY);
        let found: Vec<PathBuf> = std::fs::read_dir(&root)
            .map_err(io(&root))?
            .filter_map(Result::ok)
            .filter(|entry| entry.file_name().to_string_lossy().starts_with(prefix))
            .map(|entry| entry.path().join(MANIFEST))
            .collect();
        match found.as_slice() {
            [one] => Ok(one.clone()),
            [] => Err(OrchestratorError::Refused(format!(
                "no exam set {prefix:?} under {}",
                root.display()
            ))),
            _ => Err(OrchestratorError::Refused(format!(
                "{prefix:?} names more than one exam set"
            ))),
        }
    }
}

/// Generates the exam from `request.sources`, holds it to its tasks per
/// family, and freezes it.
///
/// # Errors
/// Refused when no task was admitted from the reserved text.
pub fn create(ctx: &Context, request: &ExamSetRequest<'_>) -> Result<ExamSet, OrchestratorError> {
    let generated = generate(
        ctx,
        &Generation {
            sources: request.sources,
            sections: &[],
            kinds: request.kinds,
            generator: request.generator,
            goal: request.goal,
            author: request.author,
            deadline: None,
            cancel: request.cancel.clone(),
        },
    )?;
    let store = ctx.tasks();
    let set = store.get_set(&generated.task_set)?;
    let tasks = set
        .members
        .iter()
        .map(|entry| store.get(&entry.task))
        .collect::<Result<Vec<_>, _>>()?;
    let clusters = task_clusters(ctx, &tasks)?;
    let reserved: std::collections::BTreeSet<&str> =
        request.families.iter().map(|f| f.family.as_str()).collect();
    // By family, in an order that does not follow generation: a stable hash
    // of the task's address.
    let mut by_family: BTreeMap<String, Vec<&TaskEntry>> = BTreeMap::new();
    for (entry, cluster) in set.members.iter().zip(clusters) {
        if let Some(family) = cluster.filter(|c| reserved.contains(c.as_str())) {
            by_family.entry(family).or_default().push(entry);
        }
    }
    for entries in by_family.values_mut() {
        entries.sort_by_cached_key(|e| Digest::of(e.task.as_str().as_bytes()).to_string());
    }
    // At most `tasks_per_family` of each family, so that one prolific family
    // does not make the exam.
    let mut chosen: Vec<(String, &TaskEntry)> = by_family
        .iter()
        .flat_map(|(family, entries)| {
            entries
                .iter()
                .take(request.tasks_per_family)
                .map(move |entry| (family.clone(), *entry))
        })
        .collect();
    if chosen.is_empty() {
        return Err(OrchestratorError::Refused(format!(
            "no task was admitted from the reserved text, so there is no exam to freeze \
             (proposals rejected, by reason: {:?})",
            generated.rejected
        )));
    }
    chosen.sort_by(|a, b| (&a.0, a.1.task.as_str()).cmp(&(&b.0, b.1.task.as_str())));
    let exam_tasks: Vec<ExamTask> = chosen
        .iter()
        .map(|(family, entry)| ExamTask {
            task: entry.task.clone(),
            family: family.clone(),
        })
        .collect();
    let task_set = store.put_set(&TaskSet {
        name: "exam".into(),
        members: chosen.iter().map(|(_, entry)| (*entry).clone()).collect(),
    })?;
    let bundle = Bundle {
        tasks: chosen
            .iter()
            .map(|(_, entry)| store.get(&entry.task))
            .collect::<Result<_, _>>()?,
        sources: request
            .sources
            .iter()
            .map(|id| bundled(ctx, id))
            .collect::<Result<_, _>>()?,
    };
    let mut exam = ExamSet {
        id: String::new(),
        task_set,
        sources: request.sources.to_vec(),
        families: request.families.to_vec(),
        tasks: exam_tasks,
        kinds: request.kinds.to_vec(),
        tasks_per_family: request.tasks_per_family,
        role: request.role,
        primary: Primary::default(),
        bundle,
    };
    let bytes = serde_json::to_vec_pretty(&exam).map_err(|source| OrchestratorError::Json {
        what: "the exam manifest".into(),
        source,
    })?;
    exam.id = Digest::of(&bytes).hex().chars().take(16).collect();
    let directory = ExamSet::directory(ctx, &exam.id);
    std::fs::create_dir_all(&directory).map_err(io(&directory))?;
    let file = directory.join(MANIFEST);
    let ledger = Ledger::beside(&file);
    ledger
        .check(MANIFEST, &bytes)
        .map_err(|e| OrchestratorError::Refused(e.to_string()))?;
    std::fs::write(&file, &bytes).map_err(io(&file))?;
    ledger
        .pin_file(&file)
        .map_err(|e| OrchestratorError::Refused(e.to_string()))?;
    Ok(exam)
}

/// An exam to be made from sources, apart from any `learn`.
pub struct NewExam<'a> {
    /// The sources, as a command line names them.
    pub sources: &'a [SourceTarget],
    /// Families to reserve for the final test.
    pub families: usize,
    /// Tasks of each.
    pub tasks_per_family: usize,
    /// Families to reserve for the dev suite.
    pub dev_families: usize,
    /// Tasks of each.
    pub dev_tasks_per_family: usize,
    /// Varies the choice of families.
    pub seed: u64,
    /// The kinds of task.
    pub kinds: &'a [String],
    /// The generator model.
    pub generator: &'a ModelRef,
    /// What the learner is after.
    pub goal: Option<&'a str>,
    /// Who wrote the sources, when they are one person's.
    pub author: Option<&'a str>,
    /// Candidates (by id) whose training, and that of the releases they
    /// continue, any reserved family must be clear of: the exam is for
    /// measuring them.
    pub not_trained_by: &'a [String],
    /// Stops generation.
    pub cancel: CancelToken,
}

/// What [`build`] made.
#[derive(Clone, Debug, Serialize)]
pub struct ExamBuilt {
    /// The families reserved and the sources divided by them.
    pub reservation: Reservation,
    /// The frozen final test.
    pub exam: ExamSet,
    /// The frozen dev suite, when families were reserved for one.
    pub dev: Option<ExamSet>,
}

/// Captures `request.sources`, reserves the exam's families among them
/// (clear of what the named candidates were trained on), writes the exam from
/// the reserved text and freezes it.
pub fn build(ctx: &Context, request: &NewExam<'_>) -> Result<ExamBuilt, OrchestratorError> {
    let mut sources = Vec::new();
    for target in request.sources {
        sources.push(add(ctx, target)?.source.id);
    }
    let mut touched_by = Vec::new();
    for candidate in request.not_trained_by {
        touched_by.extend(crate::powered::datasets_trained_on(
            ctx,
            &crate::train::load_candidate(ctx, candidate)?,
        )?);
    }
    let reservation = reserve(
        ctx,
        &ReserveRequest {
            sources: &sources,
            families: request.families,
            dev_families: request.dev_families,
            seed: request.seed,
            touched_by: &touched_by,
        },
    )?;
    let made = |role: Role, sources: &[SourceId], families: &[ReservedFamily], per: usize| {
        create(
            ctx,
            &ExamSetRequest {
                sources,
                families,
                role,
                kinds: request.kinds,
                generator: request.generator,
                goal: request.goal,
                author: request.author,
                tasks_per_family: per,
                cancel: request.cancel.clone(),
            },
        )
    };
    let exam = made(
        Role::Final,
        &reservation.exam,
        &reservation.families,
        request.tasks_per_family,
    )?;
    let dev = (!reservation.dev.is_empty())
        .then(|| {
            made(
                Role::Dev,
                &reservation.dev,
                &reservation.dev_families,
                request.dev_tasks_per_family,
            )
        })
        .transpose()?;
    Ok(ExamBuilt {
        reservation,
        exam,
        dev,
    })
}

/// `id` with the text of its parts.
fn bundled(ctx: &Context, id: &SourceId) -> Result<BundledSource, OrchestratorError> {
    let store = ctx.sources();
    let source = store.get_source(id)?;
    let mut parts = Vec::with_capacity(source.parts.len());
    for part in &source.parts {
        let text = String::from_utf8(store.read_blob(&part.content)?).map_err(|_| {
            OrchestratorError::Refused(format!(
                "part {} of source {id} is not text, so the exam cannot carry it",
                part.name
            ))
        })?;
        parts.push((part.name.clone(), part.media_type.clone(), text));
    }
    Ok(BundledSource {
        id: id.clone(),
        origin: source.origin,
        parts,
    })
}
