// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements leakage-free held-out examinations of
// code-editing agents, for its clients. If your team needs expertise in
// measuring a release on the families of a repository it was never trained
// on, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The exams `practice` reserved, kept so that a release can be measured on
//! them.
//!
//! `practice` reserves the families of its exam before any rollout and keeps
//! the tasks of those families apart. What it trains on is a dataset; the
//! exam reserved against that dataset is recorded under the dataset's name
//! ([`record`]), so the gate can find, for a candidate and for every release
//! it descends from, the families that model was never trained on
//! ([`split`]). The tasks are those of the exam's task set; a task of a
//! family the candidate trained on after all - a family an earlier run
//! trained and a later run reserved - is no exam for it and is left out,
//! counted as leaked ([`suite`]).

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use splinter_core::dataset::DatasetId;
use splinter_core::digest::Digest;
use splinter_eval::gate::AgenticConfig;
use splinter_store::tasks::TaskSetId;

use crate::release::leakage::LEAKED;
use crate::release::probe::Suite;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::{io, OrchestratorError};

/// The class of the stored record of an exam.
const EXAM: &str = "practice_exam";

/// The prefix of the pointer that names the exam reserved against a dataset.
const EXAM_POINTER: &str = "practice-exam-";

/// The exam reserved against one dataset.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExamRecord {
    /// The dataset the exam was reserved against.
    pub dataset: DatasetId,
    /// Every family reserved, this run's and earlier runs'.
    pub families: BTreeSet<String>,
    /// The tasks of those families this run made, set apart and never
    /// solved; `None` when it made none.
    pub tasks: Option<TaskSetId>,
}

fn pointer_of(dataset: &DatasetId) -> String {
    format!("{EXAM_POINTER}{}", dataset.0.hex())
}

/// Records `exam` as the one reserved against its dataset, replacing an
/// earlier record of the same dataset: the later run knew more of the
/// families.
pub fn record(ctx: &Context, exam: &ExamRecord) -> Result<(), OrchestratorError> {
    let name = pointer_of(&exam.dataset);
    let digest = ctx.workspace().put_document(EXAM, exam)?;
    let current = ctx.workspace().pointer(&name)?.map(|(_, value)| value);
    if current.as_deref() != Some(digest.as_str()) {
        ctx.workspace().move_pointer(
            &name,
            current.as_deref(),
            digest.as_str(),
            &ctx.clock().utc_now(),
        )?;
    }
    Ok(())
}

/// The exam reserved against `dataset`; `None` for a dataset no `practice`
/// reserved one against (a closed-book dataset, say).
pub fn of_dataset(
    ctx: &Context,
    dataset: &DatasetId,
) -> Result<Option<ExamRecord>, OrchestratorError> {
    ctx.workspace().refresh()?;
    let Some((_, value)) = ctx.workspace().pointer(&pointer_of(dataset))? else {
        return Ok(None);
    };
    let digest = Digest::parse(&value).map_err(|e| {
        OrchestratorError::Refused(format!(
            "the pointer to the exam of dataset {dataset} is corrupt: {e}"
        ))
    })?;
    Ok(ctx.workspace().get_document(EXAM, &digest)?)
}

/// Datasets divided by whether an exam was reserved against them.
pub(crate) struct Split {
    /// The datasets no exam was reserved against: their held-out records are
    /// the closed-book suites' business.
    pub(crate) plain: Vec<DatasetId>,
    /// The exams reserved against the others.
    pub(crate) exams: Vec<ExamRecord>,
}

/// `datasets` divided into those with an exam and those without.
pub(crate) fn split(ctx: &Context, datasets: &[DatasetId]) -> Result<Split, OrchestratorError> {
    let mut divided = Split {
        plain: Vec::new(),
        exams: Vec::new(),
    };
    for dataset in datasets {
        match of_dataset(ctx, dataset)? {
            Some(exam) => divided.exams.push(exam),
            None => divided.plain.push(dataset.clone()),
        }
    }
    Ok(divided)
}

/// The families of every record of `datasets` and of the files of records
/// `mixed_in` (replayed or rehearsed, by artifact digest): what a model
/// trained on them has seen, held-out records included.
pub(crate) fn trained_families(
    ctx: &Context,
    datasets: &[DatasetId],
    mixed_in: &[Option<&Digest>],
) -> Result<BTreeSet<String>, OrchestratorError> {
    #[derive(Deserialize)]
    struct Line {
        #[serde(default)]
        metadata: Option<Metadata>,
    }
    #[derive(Deserialize)]
    struct Metadata {
        #[serde(default)]
        group: Option<String>,
    }
    let mut paths = Vec::new();
    for id in datasets {
        paths.push(ctx.datasets().get(id)?.path);
    }
    for digest in mixed_in.iter().copied().flatten() {
        paths.push(ctx.artifacts().path(digest)?);
    }
    let mut families = BTreeSet::new();
    for path in paths {
        let text = std::fs::read_to_string(&path).map_err(io(&path))?;
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            // A record that is not a line of this shape has no family.
            if let Ok(Line {
                metadata: Some(Metadata { group: Some(group) }),
            }) = serde_json::from_str(line)
            {
                families.insert(group);
            }
        }
    }
    Ok(families)
}

/// The suite of the tasks of `exams`, named `name` and probed under `work`,
/// without the tasks of a family in `trained`, counted as leaked. A task of
/// no family cannot be shown to be outside what was trained on and is left
/// out the same way.
pub(crate) fn suite(
    ctx: &Context,
    name: &str,
    exams: &[ExamRecord],
    trained: &BTreeSet<String>,
    work: AgenticConfig,
) -> Result<Suite, OrchestratorError> {
    let mut tasks = Vec::new();
    for exam in exams {
        let Some(set) = &exam.tasks else { continue };
        for entry in ctx.tasks().get_set(set)?.members {
            tasks.push(ctx.tasks().get(&entry.task)?);
        }
    }
    let mut suite = Suite::workspace(name, Vec::new()).with_work(work);
    for task in tasks {
        let leaked = task
            .concepts
            .first()
            .is_none_or(|family| trained.contains(family));
        if leaked {
            *suite.excluded.entry(LEAKED.into()).or_default() += 1;
        } else {
            suite.add_workspace(task);
        }
    }
    Ok(suite)
}
