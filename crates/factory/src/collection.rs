// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pinned, verifiable task collections for
// agent training and evaluation for its clients. If your team needs
// expertise in benchmark task design, you can procure our services by
// sending an email to info@swedishembedded.com.

//! A task collection file of the task repository, and the Harbor task
//! directories it names.
//!
//! ```yaml
//! schema: splinter.collection/v1
//! name: vim-mastery
//! root: tasks/vim
//! include: [train, eval]
//! ```
//!
//! Every directory under `root` holding a `task.toml` is a task (Harbor's
//! format); its id is its path relative to `root`, its split the first
//! segment of that path, its family the `family` (or `bench_family`)
//! metadata key, and its digest the digest of its whole directory. `include`
//! keeps only the ids under the given prefixes.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use splinter_core::factory::task::{TaskCollection, TaskRef};

use crate::recipe::RecipeError;
use crate::tree::{tree_digest, TreeError};

/// The schema every collection file names.
pub const COLLECTION_SCHEMA: &str = "splinter.collection/v1";

/// A collection file.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CollectionFile {
    /// Always [`COLLECTION_SCHEMA`].
    pub schema: String,
    /// The collection's name.
    pub name: String,
    /// The directory holding the tasks, relative to the repository root.
    pub root: String,
    /// Only tasks whose id starts with one of these path prefixes.
    #[serde(default)]
    pub include: Vec<String>,
    /// The command (an argv, run from the repository root) that writes the
    /// task directories under `root`: a task repository's own compiler of
    /// its task sources. `splinter generate` runs it; a run reads only what
    /// it wrote.
    #[serde(default)]
    pub generate: Option<Vec<String>>,
}

/// Why a collection cannot be loaded.
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    /// The file is wrong.
    #[error(transparent)]
    Recipe(#[from] RecipeError),
    /// A task directory cannot be read.
    #[error(transparent)]
    Tree(#[from] TreeError),
    /// A `task.toml` is unreadable.
    #[error("{path}: {message}")]
    Task {
        /// The file.
        path: PathBuf,
        /// What is wrong.
        message: String,
    },
    /// The collection is refused.
    #[error(transparent)]
    Collection(#[from] splinter_core::factory::task::CollectionError),
}

/// A loaded collection: the pinned tasks and where their directories are.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadedCollection {
    /// The pinned tasks.
    pub collection: TaskCollection,
    /// The directory the task ids are relative to.
    pub root: PathBuf,
}

impl LoadedCollection {
    /// The tasks of `split`, or every task.
    #[must_use]
    pub fn split(&self, split: Option<&str>) -> Vec<&TaskRef> {
        self.collection
            .tasks
            .iter()
            .filter(|t| split.is_none() || t.split.as_deref() == split)
            .collect()
    }
}

/// Load the collection file at `file` (repository-relative `relative`) under
/// the repository `root`.
///
/// # Errors
/// The file or a task is unreadable, or the collection is empty.
pub fn load(repository: &Path, relative: &str) -> Result<LoadedCollection, LoadError> {
    let path = repository.join(relative);
    let text = std::fs::read_to_string(&path).map_err(|e| LoadError::Task {
        path: path.clone(),
        message: e.to_string(),
    })?;
    let file: CollectionFile =
        serde_yaml::from_str(&text).map_err(|e| RecipeError(vec![format!("{relative}: {e}")]))?;
    if file.schema != COLLECTION_SCHEMA {
        return Err(RecipeError(vec![format!(
            "{relative}: schema is {:?}; expected {COLLECTION_SCHEMA}",
            file.schema
        )])
        .into());
    }
    if !splinter_core::factory::task::plain_relative(&file.root) {
        return Err(RecipeError(vec![format!(
            "{relative}: root {:?} is not a plain relative path",
            file.root
        )])
        .into());
    }
    let root = repository.join(&file.root);
    if !root.is_dir() {
        let hint = match &file.generate {
            Some(_) => format!("; `splinter generate {relative}` writes it"),
            None => String::new(),
        };
        return Err(RecipeError(vec![format!(
            "{relative}: its root {} does not exist{hint}",
            file.root
        )])
        .into());
    }
    scan(&root, file.name, &file.include)
}

/// The tasks below `root` (every directory holding a `task.toml`), as the
/// collection `name`; with `include`, only the tasks at or below those
/// prefixes. A collection a stage produced is read this way, its root the
/// product's view.
///
/// # Errors
/// A task cannot be read, or two share an id.
pub fn scan(root: &Path, name: String, include: &[String]) -> Result<LoadedCollection, LoadError> {
    let mut found = Vec::new();
    find_tasks(root, root, &mut found)?;
    let mut tasks = Vec::new();
    for dir in found {
        let id = dir
            .strip_prefix(root)
            .map_err(|e| LoadError::Task {
                path: dir.clone(),
                message: e.to_string(),
            })?
            .to_string_lossy()
            .replace('\\', "/");
        let wanted = include.is_empty()
            || include
                .iter()
                .any(|prefix| id == *prefix || id.starts_with(&format!("{prefix}/")));
        if wanted {
            tasks.push(task_ref(&dir, id)?);
        }
    }
    Ok(LoadedCollection {
        collection: TaskCollection::new(name, tasks)?,
        root: root.to_path_buf(),
    })
}

fn find_tasks(root: &Path, dir: &Path, found: &mut Vec<PathBuf>) -> Result<(), LoadError> {
    if dir.join("task.toml").is_file() && dir != root {
        found.push(dir.to_path_buf());
        return Ok(());
    }
    let read = |e: std::io::Error| LoadError::Task {
        path: dir.to_path_buf(),
        message: e.to_string(),
    };
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(read)?
        .map(|e| e.map(|e| e.path()))
        .collect::<Result<_, _>>()
        .map_err(read)?;
    entries.sort();
    for entry in entries {
        let name = entry
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        if entry.is_dir() && !entry.is_symlink() && !name.starts_with('.') {
            find_tasks(root, &entry, found)?;
        }
    }
    Ok(())
}

fn task_ref(dir: &Path, id: String) -> Result<TaskRef, LoadError> {
    let path = dir.join("task.toml");
    let fail = |message: String| LoadError::Task {
        path: path.clone(),
        message,
    };
    let text = std::fs::read_to_string(&path).map_err(|e| fail(e.to_string()))?;
    let toml: toml::Value = text
        .parse()
        .map_err(|e: toml::de::Error| fail(e.to_string()))?;
    let name = toml
        .get("task")
        .and_then(|t| t.get("name"))
        .and_then(toml::Value::as_str)
        .map_or_else(|| id.clone(), str::to_owned);
    let metadata = toml.get("metadata");
    let family = ["family", "bench_family"]
        .iter()
        .find_map(|key| {
            metadata
                .and_then(|m| m.get(*key))
                .and_then(toml::Value::as_str)
        })
        .map(str::to_owned);
    let split = id.split_once('/').map(|(first, _)| first.to_owned());
    Ok(TaskRef {
        digest: tree_digest(dir)?,
        id,
        name,
        family,
        split,
    })
}

/// Run the collection file's `generate` command from the repository root.
///
/// # Errors
/// The file is unreadable or names no command, or the command failed.
pub fn generate(repository: &Path, relative: &str) -> Result<(), LoadError> {
    let path = repository.join(relative);
    let text = std::fs::read_to_string(&path).map_err(|e| LoadError::Task {
        path: path.clone(),
        message: e.to_string(),
    })?;
    let file: CollectionFile =
        serde_yaml::from_str(&text).map_err(|e| RecipeError(vec![format!("{relative}: {e}")]))?;
    let Some((program, args)) = file.generate.as_deref().and_then(<[String]>::split_first) else {
        return Err(RecipeError(vec![format!("{relative} names no generate command")]).into());
    };
    let status = std::process::Command::new(program)
        .args(args)
        .current_dir(repository)
        .status()
        .map_err(|e| LoadError::Task {
            path: path.clone(),
            message: format!("{program}: {e}"),
        })?;
    if status.success() {
        Ok(())
    } else {
        Err(LoadError::Task {
            path,
            message: format!("{program} exited with {status}"),
        })
    }
}
