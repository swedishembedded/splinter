// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements interchange of agent benchmark tasks between
// task formats for its clients. If your team needs expertise in importing
// containerised benchmark tasks into reproducible, verifiable task sets, you
// can procure our services by sending an email to info@swedishembedded.com.

//! Harbor interoperability: tasks in, tasks out, and trials read back as
//! experiences.
//!
//! **Import** turns the Harbor tasks under some directories into a task set
//! (see [`splinter_knowledge::tasks::workspace`] for what is honoured and
//! what is refused). Each imported task is recorded with where it came from -
//! the checkout's commit and whether it was dirty, and a digest of the
//! task directory - as a document, never as part of the task, so the task's
//! content address does not depend on where it was read. The external task
//! repository is a versioned source read at a revision, not a database.
//!
//! **Export** writes a task set as Harbor tasks.
//!
//! **Trial ingestion** reads the directories Harbor leaves for each trial
//! (`result.json` or `results.json`, `agent/trajectory.json` in ATIF) into
//! experiences of the imported tasks. The verdict is Harbor's: the verifier
//! passed it when its `reward` is exactly 1, failed it otherwise, and a trial
//! with no reward or an exception has no verdict at all. A trial without an
//! ATIF trajectory is refused, never reconstructed. Do not `verify` an
//! ingested set: its attempts carry no patch to grade.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use splinter_core::annotation::{Annotation, AnnotationBody, Outcome, Producer, Strength};
use splinter_core::digest::Digest;
use splinter_core::experience::{Experience, Provenance};
use splinter_knowledge::tasks::workspace::{discover, Discovered, HarborTask, Oracle};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_store::experiences::{ExperienceSet, SetId};
use splinter_store::tasks::TaskSetId;

use crate::practice::{held_tasks, keep_tasks};

/// The document class an imported task's origin is stored under.
const ORIGIN: &str = "harbor_task";

/// The producer of a Harbor verdict.
pub(crate) const VERIFIER_PRODUCER: &str = "harbor/verifier";

/// Where an imported task was read from.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Source {
    /// The task directory as given.
    pub directory: PathBuf,
    /// The commit of the checkout the directory is in, when it is in one.
    pub commit: Option<String>,
    /// Whether that checkout had uncommitted changes.
    pub dirty: Option<bool>,
    /// A digest over the task directory's files, so a changed task is a
    /// different source whatever the commit says.
    pub digest: Digest,
}

/// A Harbor task and the task it became.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Origin {
    /// The Harbor task's name.
    pub name: String,
    /// The splinter task's address.
    pub task: Digest,
    /// Where it was read from.
    pub source: Source,
}

/// What an export wrote.
#[derive(Clone, Debug, Serialize)]
pub struct Exports {
    /// The tasks written.
    pub written: Vec<Exported>,
}

/// A task not imported, and why.
#[derive(Clone, Debug, Serialize)]
pub struct Refused {
    /// The task directory.
    pub directory: PathBuf,
    /// Why.
    pub reason: String,
}

/// What an import did.
#[derive(Clone, Debug, Serialize)]
pub struct Imported {
    /// The set of the tasks imported; `None` when none was.
    pub task_set: Option<TaskSetId>,
    /// The tasks imported.
    pub imported: Vec<Origin>,
    /// The tasks refused.
    pub refused: Vec<Refused>,
}

fn io_error(path: &Path, source: std::io::Error) -> OrchestratorError {
    splinter_orchestrator::error::io(path)(source)
}

/// Every file under `dir`, sorted by path.
fn files(dir: &Path) -> Result<Vec<PathBuf>, OrchestratorError> {
    let mut found = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(next) = pending.pop() {
        for entry in std::fs::read_dir(&next).map_err(|e| io_error(&next, e))? {
            let entry = entry.map_err(|e| io_error(&next, e))?;
            let path = entry.path();
            if entry.file_type().map_err(|e| io_error(&path, e))?.is_dir() {
                pending.push(path);
            } else {
                found.push(path);
            }
        }
    }
    found.sort();
    Ok(found)
}

/// The digest of `dir`'s files: each relative path and its bytes.
fn directory_digest(dir: &Path) -> Result<Digest, OrchestratorError> {
    let mut bytes = Vec::new();
    for file in files(dir)? {
        let relative = file
            .strip_prefix(dir)
            .unwrap_or(&file)
            .to_string_lossy()
            .into_owned();
        let content = std::fs::read(&file).map_err(|e| io_error(&file, e))?;
        bytes.extend_from_slice(relative.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(Digest::of(&content).to_string().as_bytes());
        bytes.push(b'\n');
    }
    Ok(Digest::of(&bytes))
}

/// The commit and dirtiness of the git checkout `dir` is in, when it is in one.
fn checkout(dir: &Path) -> (Option<String>, Option<bool>) {
    let git = |args: &[&str]| {
        splinter_knowledge::capture::git::read_only(dir)
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    };
    let commit = git(&["rev-parse", "HEAD"]);
    let dirty = commit
        .as_ref()
        .and_then(|_| git(&["status", "--porcelain", "--untracked-files=no"]))
        .map(|status| !status.is_empty());
    (commit, dirty)
}

/// Lists the Harbor tasks under `root` and whether each can be imported.
/// Nothing is written.
///
/// # Errors
///
/// [`OrchestratorError`] when `root` cannot be read.
pub fn discover_harbor(root: &Path) -> Result<Discovery, OrchestratorError> {
    let (commit, dirty) = checkout(root);
    Ok(Discovery {
        root: root.to_path_buf(),
        commit,
        dirty,
        tasks: discover(root).map_err(|e| OrchestratorError::Refused(e.to_string()))?,
    })
}

/// The Harbor tasks under a directory, at the revision of the checkout the
/// directory is in.
#[derive(Clone, Debug, Serialize)]
pub struct Discovery {
    /// The directory searched.
    pub root: PathBuf,
    /// The commit of the checkout it is in, when it is in one.
    pub commit: Option<String>,
    /// Whether that checkout had uncommitted changes.
    pub dirty: Option<bool>,
    /// The tasks found, with whether each can be imported.
    pub tasks: Vec<Discovered>,
}

/// Imports the Harbor tasks in `directories` (each a task directory) as one
/// task set.
///
/// # Errors
///
/// [`OrchestratorError`] when the workspace backend is unavailable or the
/// store cannot be written. A task that cannot be imported is in
/// [`Imported::refused`], not an error.
pub fn import_harbor(
    ctx: &Context,
    directories: &[PathBuf],
    oracle: Oracle,
) -> Result<Imported, OrchestratorError> {
    let backend = ctx.environments().workspace()?;
    let generator = splinter_knowledge::tasks::workspace::WorkspaceGenerator::new(
        backend.sandbox,
        backend.host,
        ctx.artifacts(),
        backend.scratch.join("harbor"),
    );
    let held = held_tasks(ctx)?;
    let mut admitted = Vec::new();
    let mut origins = Vec::new();
    let mut refused = Vec::new();
    for directory in directories {
        let outcome = HarborTask::read(directory).and_then(|task| {
            generator
                .import_harbor(&task, oracle, &held, ctx.clock())
                .map(|generation| (task, generation))
        });
        match outcome {
            Ok((task, generation)) => match generation.admitted.into_iter().next() {
                Some(made) => {
                    let (commit, dirty) = checkout(directory);
                    origins.push(Origin {
                        name: task.name,
                        task: made.task.task.id.clone(),
                        source: Source {
                            directory: directory.clone(),
                            commit,
                            dirty,
                            digest: directory_digest(directory)?,
                        },
                    });
                    admitted.push(made);
                }
                None => refused.push(Refused {
                    directory: directory.clone(),
                    reason: format!("not admitted: {:?}", generation.refused),
                }),
            },
            Err(error) => refused.push(Refused {
                directory: directory.clone(),
                reason: error.to_string(),
            }),
        }
    }
    let task_set = if admitted.is_empty() {
        None
    } else {
        for origin in &origins {
            ctx.workspace().put_document(ORIGIN, origin)?;
        }
        Some(keep_tasks(ctx, "harbor", &admitted)?)
    };
    Ok(Imported {
        task_set,
        imported: origins,
        refused,
    })
}

/// What an export wrote.
#[derive(Clone, Debug, Serialize)]
pub struct Exported {
    /// The task's address.
    pub task: Digest,
    /// The Harbor task directory written.
    pub directory: PathBuf,
}

/// Writes every workspace task of `set` under `out` as a Harbor task, in a
/// directory named for the task's address, with a Dockerfile `FROM image`.
///
/// # Errors
///
/// [`OrchestratorError`] when a task is not a workspace task or cannot be
/// written.
pub fn export_harbor(
    ctx: &Context,
    set: &TaskSetId,
    out: &Path,
    image: &str,
) -> Result<Exports, OrchestratorError> {
    let backend = ctx.environments().workspace()?;
    let generator = splinter_knowledge::tasks::workspace::WorkspaceGenerator::new(
        backend.sandbox,
        backend.host,
        ctx.artifacts(),
        backend.scratch.join("harbor"),
    );
    let tasks = ctx.tasks();
    let mut written = Vec::new();
    for entry in tasks.get_set(set)?.members {
        let task = tasks.get(&entry.task)?;
        let directory = out.join(entry.task.hex());
        generator
            .export_harbor(&task, &directory, image)
            .map_err(|e| OrchestratorError::Refused(e.to_string()))?;
        written.push(Exported {
            task: entry.task.clone(),
            directory,
        });
    }
    Ok(Exports { written })
}

/// A trial not ingested, and why.
#[derive(Clone, Debug, Serialize)]
pub struct TrialRefused {
    /// The trial directory.
    pub directory: PathBuf,
    /// Why.
    pub reason: String,
}

/// What an ingestion did.
#[derive(Clone, Debug, Serialize)]
pub struct Ingested {
    /// The set of the experiences ingested; `None` when none was.
    pub experience_set: Option<SetId>,
    /// Trials ingested, by the outcome Harbor's verifier gave them.
    pub verdicts: BTreeMap<String, usize>,
    /// The trials refused.
    pub refused: Vec<TrialRefused>,
}

fn text<'a>(value: &'a Value, path: &[&str]) -> Option<&'a str> {
    path.iter()
        .try_fold(value, |at, key| at.get(key))
        .and_then(Value::as_str)
}

/// The verdict Harbor's verifier gave a trial: `reward` exactly 1 passes, any
/// other reward fails (partial credit is not success), and a missing
/// verifier result or an exception is no verdict.
fn harbor_outcome(result: &Value) -> (Outcome, Value) {
    let rewards = result.pointer("/verifier_result/rewards").cloned();
    let excepted = result.get("exception_info").is_some_and(|e| !e.is_null());
    let outcome = match (&rewards, excepted) {
        (Some(Value::Object(map)), false) => match map.get("reward").and_then(Value::as_f64) {
            Some(reward) if (reward - 1.0).abs() < f64::EPSILON => Outcome::Pass,
            Some(_) => Outcome::Fail,
            None => Outcome::Abstain,
        },
        _ => Outcome::Abstain,
    };
    (outcome, rewards.unwrap_or(Value::Null))
}

fn read_result(trial: &Path) -> Result<Value, String> {
    for name in ["result.json", "results.json"] {
        let path = trial.join(name);
        if path.is_file() {
            let text =
                std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            return serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()));
        }
    }
    Err("the trial has no result.json".into())
}

/// Reads the Harbor trials in `trials` as experiences of the tasks of `set`.
///
/// # Errors
///
/// [`OrchestratorError`] when the store cannot be read or written. A trial
/// that cannot be read, names a task outside `set` or has no ATIF
/// trajectory is in [`Ingested::refused`].
pub fn ingest_harbor_trials(
    ctx: &Context,
    set: &TaskSetId,
    trials: &[PathBuf],
) -> Result<Ingested, OrchestratorError> {
    let tasks = ctx.tasks();
    let in_set: Vec<Digest> = tasks
        .get_set(set)?
        .members
        .into_iter()
        .map(|e| e.task)
        .collect();
    let mut by_name: BTreeMap<String, Digest> = BTreeMap::new();
    for id in ctx.workspace().document_ids(ORIGIN)? {
        if let Some(origin) = ctx.workspace().get_document::<Origin>(ORIGIN, &id)? {
            if in_set.contains(&origin.task) {
                by_name.insert(origin.name, origin.task);
            }
        }
    }
    let store = ctx.experiences();
    let mut members = Vec::new();
    let mut verdicts: BTreeMap<String, usize> = BTreeMap::new();
    let mut refused = Vec::new();
    let mut attempts: BTreeMap<Digest, u32> = BTreeMap::new();
    for trial in trials {
        let reason = |why: String| TrialRefused {
            directory: trial.clone(),
            reason: why,
        };
        let result = match read_result(trial) {
            Ok(result) => result,
            Err(why) => {
                refused.push(reason(why));
                continue;
            }
        };
        let Some(name) = text(&result, &["task_name"]) else {
            refused.push(reason("the result names no task".into()));
            continue;
        };
        let Some(task_id) = by_name.get(name) else {
            refused.push(reason(format!(
                "task {name:?} is not an imported task of this set"
            )));
            continue;
        };
        let trajectory = match std::fs::read_to_string(trial.join("agent/trajectory.json"))
            .map_err(|e| e.to_string())
            .and_then(|t| serde_json::from_str::<atif::Trajectory>(&t).map_err(|e| e.to_string()))
        {
            Ok(trajectory) => trajectory,
            Err(why) => {
                refused.push(reason(format!(
                    "no usable agent/trajectory.json (ATIF): {why}"
                )));
                continue;
            }
        };
        let task = tasks.get(task_id)?;
        let agent = text(&result, &["agent_info", "name"]).unwrap_or("unknown");
        let version = text(&result, &["agent_info", "version"]).unwrap_or("unknown");
        let model = text(&result, &["agent_info", "model_info", "name"]).unwrap_or("unknown");
        // Trials of one task by one agent are attempts at it, numbered in the
        // order given, so two trials that left the same trajectory stay two
        // experiences, each with the verdict of its own trial.
        let attempt = attempts.entry(task_id.clone()).or_insert(0u32);
        let mut provenance =
            Provenance::new(format!("harbor/{agent}@{version}/{model}"), ctx.clock());
        provenance.attempt = Some(*attempt);
        *attempt += 1;
        let experience = match Experience::new(task, trajectory, None, provenance) {
            Ok(experience) => experience,
            Err(error) => {
                refused.push(reason(error.to_string()));
                continue;
            }
        };
        let id = store.put(&experience)?;
        let (outcome, rewards) = harbor_outcome(&result);
        *verdicts
            .entry(format!("{outcome:?}").to_lowercase())
            .or_default() += 1;
        store.annotate(&Annotation {
            experience: id.clone(),
            producer: Producer {
                name: VERIFIER_PRODUCER.into(),
                version: "1".into(),
            },
            body: AnnotationBody::Verdict {
                outcome,
                strength: Strength::Executable,
                evidence: serde_json::json!({
                    "rewards": rewards,
                    "trial": text(&result, &["trial_name"]),
                    "task_checksum": text(&result, &["task_checksum"]),
                    "exception": result.get("exception_info").filter(|e| !e.is_null()),
                }),
            },
        })?;
        if !members.contains(&id) {
            members.push(id);
        }
    }
    let experience_set = if members.is_empty() {
        None
    } else {
        Some(store.put_set(&ExperienceSet {
            name: format!("harbor trials of {set}"),
            members,
        })?)
    };
    Ok(Ingested {
        experience_set,
        verdicts,
        refused,
    })
}
