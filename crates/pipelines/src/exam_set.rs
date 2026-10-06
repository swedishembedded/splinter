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
use splinter_core::source::SourceId;
use splinter_data::frozen::Ledger;
use splinter_store::tasks::{TaskEntry, TaskSet, TaskSetId};

use crate::grouping::task_clusters;
use crate::reserve::{Reservation, ReservedFamily};
use crate::tasks::{generate, Generation};
use splinter_core::model_ref::ModelRef;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::{io, OrchestratorError};

/// The tasks an exam holds when no budget is named: over about thirty
/// families, enough discordant tasks to see a gain of a fifth.
pub const DEFAULT_EXAM_TASKS: usize = 240;

/// The directory under the state root the exam sets are kept in.
const DIRECTORY: &str = "exams";

/// The manifest's file name.
const MANIFEST: &str = "exam.json";

/// What an exam set is made from.
pub struct ExamSetRequest<'a> {
    /// The reservation the exam is written from.
    pub reservation: &'a Reservation,
    /// The kinds of task; the ones a judge can grade against a reference.
    pub kinds: &'a [String],
    /// The generator model.
    pub generator: &'a ModelRef,
    /// What the learner is after.
    pub goal: Option<&'a str>,
    /// Who wrote the sources, when they are one person's.
    pub author: Option<&'a str>,
    /// The most tasks the exam holds.
    pub max_tasks: usize,
    /// Stops generation.
    pub cancel: CancelToken,
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
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
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
    /// The budget it was held to.
    pub max_tasks: usize,
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
        Ok(set)
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

/// Generates the exam from `request.reservation.exam`, holds it to its
/// budget, and freezes it.
///
/// # Errors
/// Refused when no task was admitted from the reserved text.
pub fn create(ctx: &Context, request: &ExamSetRequest<'_>) -> Result<ExamSet, OrchestratorError> {
    let generated = generate(
        ctx,
        &Generation {
            sources: &request.reservation.exam,
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
    let reserved: std::collections::BTreeSet<&str> = request
        .reservation
        .families
        .iter()
        .map(|f| f.family.as_str())
        .collect();
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
    // Round-robin over the families, so that a short budget is spread and
    // one prolific family does not make the exam.
    let mut chosen: Vec<(String, &TaskEntry)> = Vec::new();
    let mut round = 0;
    while chosen.len() < request.max_tasks {
        let before = chosen.len();
        for (family, entries) in &by_family {
            if let Some(entry) = entries.get(round) {
                if chosen.len() < request.max_tasks {
                    chosen.push((family.clone(), entry));
                }
            }
        }
        if chosen.len() == before {
            break;
        }
        round += 1;
    }
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
    let mut exam = ExamSet {
        id: String::new(),
        task_set,
        sources: request.reservation.exam.clone(),
        families: request.reservation.families.clone(),
        tasks: exam_tasks,
        kinds: request.kinds.to_vec(),
        max_tasks: request.max_tasks,
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
