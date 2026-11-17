// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that turn practice task folders
// into graded, replayable agent experience for its clients. If your team
// needs expertise in on-policy experience collection for agent training you
// can procure our services by sending an email to info@swedishembedded.com.

//! A directory of practice task folders, as tasks in the store.

use std::path::{Path, PathBuf};

use serde::Serialize;
use splinter_core::digest::Digest;
use splinter_knowledge::tasks::workspace::{discover, HarborTask, Oracle};
use splinter_orchestrator::context::Context;

use super::EpisodeError;
use crate::harbor::{folder_digest, import_harbor, imports};

/// A practice task: a task folder and the task it was imported as.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PracticeTask {
    /// The task's name (`[task] name`, else the folder's).
    pub name: String,
    /// The task's content address.
    pub task: Digest,
    /// The digest of the folder's files when it was imported.
    pub source: Digest,
    /// The folder.
    pub directory: PathBuf,
}

/// The task folders under `root` (or `root` itself when it is one) as
/// tasks, by name: each imported once, and found again by its files'
/// digest on a later load.
///
/// Every folder is read and held to the bar before any is imported, so a
/// collection with one bad folder imports nothing.
///
/// # Errors
///
/// [`EpisodeError::Sealed`] when a folder is exam material,
/// [`EpisodeError::Malformed`] when one cannot be read or imported (no
/// verifier, no instruction, an environment the host cannot honour, a check
/// that already passes on the starting files), and the store's errors.
pub fn load_collection(ctx: &Context, root: &Path) -> Result<Vec<PracticeTask>, EpisodeError> {
    let malformed = |directory: &Path, reason: String| EpisodeError::Malformed {
        directory: directory.to_path_buf(),
        reason,
    };
    let found = discover(root).map_err(|e| malformed(root, e.to_string()))?;
    if found.is_empty() {
        return Err(malformed(
            root,
            "no task folder (a task.toml) is under it".into(),
        ));
    }
    for folder in &found {
        if let Some(why) = &folder.refusal {
            return Err(malformed(&folder.dir, why.clone()));
        }
        let task =
            HarborTask::read(&folder.dir).map_err(|e| malformed(&folder.dir, e.to_string()))?;
        if let Some(reason) = task
            .sealed()
            .map_err(|e| malformed(&folder.dir, e.to_string()))?
        {
            return Err(EpisodeError::Sealed {
                directory: folder.dir.clone(),
                reason,
            });
        }
    }
    let known = imports(ctx)?;
    let mut tasks = Vec::with_capacity(found.len());
    let mut new = Vec::new();
    for folder in &found {
        match known.get(&folder_digest(&folder.dir)?).cloned() {
            Some(origin) => tasks.push(PracticeTask {
                name: origin.name,
                task: origin.task,
                source: origin.source.digest,
                directory: folder.dir.clone(),
            }),
            None => new.push(folder.dir.clone()),
        }
    }
    if !new.is_empty() {
        let made = import_harbor(ctx, &new, Oracle::Try)?;
        if let Some(refused) = made.refused.first() {
            return Err(malformed(&refused.directory, refused.reason.clone()));
        }
        tasks.extend(made.imported.into_iter().map(|origin| PracticeTask {
            name: origin.name,
            task: origin.task,
            directory: origin.source.directory,
            source: origin.source.digest,
        }));
    }
    tasks.sort_by(|a, b| (&a.name, &a.directory).cmp(&(&b.name, &b.directory)));
    Ok(tasks)
}
