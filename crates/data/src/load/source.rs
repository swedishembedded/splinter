// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training pipelines that read any dataset
// the same way, for its clients. If your team needs expertise in loading,
// sampling and versioning datasets for agent training, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Where a dataset's split files come from: a directory, a dataset product,
//! or a repository of the Hugging Face Hub.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use splinter_hub::{Hub, RepoId, RepoPath};

use super::{canonical_split, Dataset, DatasetDict, Provenance, Record};

/// The largest file a hub dataset's download may be.
const FILE_LIMIT: u64 = 8 << 30;

/// Why a dataset could not be loaded.
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    /// The reference names nothing that can be read.
    #[error("{0:?} is neither a directory of split files, a dataset product nor a hub repository (owner/name)")]
    NoSuchDataset(String),
    /// A hub dataset was asked for without a client or a cache.
    #[error("{0} is a hub dataset: give the options a hub client and a cache directory")]
    NeedsHub(String),
    /// The configuration is not a directory of the dataset.
    #[error("the dataset has no configuration {0:?}")]
    NoConfig(String),
    /// A file or the hub could not be read.
    #[error("{path}: {message}")]
    Read {
        /// What was being read.
        path: String,
        /// Why not.
        message: String,
    },
    /// A line is not a record of the type asked for.
    #[error("{path} line {line}: {message}")]
    Record {
        /// The split file.
        path: String,
        /// Its line, from 1.
        line: usize,
        /// Why the line is not a record.
        message: String,
    },
    /// The dataset holds no split file.
    #[error("{0} holds no split file (SPLIT.jsonl at its root, in data/, or in the configuration's directory)")]
    Empty(String),
}

/// A reference, parsed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Reference {
    /// A directory of `SPLIT.jsonl` files.
    Directory(PathBuf),
    /// A dataset product a factory run wrote (a directory with `dataset.json`).
    Product(PathBuf),
    /// A repository of the Hub, at a branch, a tag or a commit.
    Hub {
        /// `owner/name`.
        repo: String,
        /// What was asked for after `@`, if anything.
        revision: Option<String>,
    },
}

impl Reference {
    /// `text` as a reference: `hf:owner/name[@rev]` or `owner/name[@rev]`
    /// when no such path exists, else a path.
    ///
    /// # Errors
    /// It names nothing that can be read.
    pub fn parse(text: &str) -> Result<Self, LoadError> {
        let hub = |rest: &str| {
            let (repo, revision) = match rest.split_once('@') {
                Some((repo, rev)) => (repo, Some(rev.to_owned())),
                None => (rest, None),
            };
            Self::Hub {
                repo: repo.to_owned(),
                revision,
            }
        };
        if let Some(rest) = text.strip_prefix("hf:") {
            return Ok(hub(rest));
        }
        let path = Path::new(text);
        if path.join("dataset.json").is_file() {
            return Ok(Self::Product(path.to_owned()));
        }
        if path.is_dir() {
            return Ok(Self::Directory(path.to_owned()));
        }
        let repo = text.split('@').next().unwrap_or(text);
        if repo.matches('/').count() == 1 && !repo.starts_with(['.', '/', '~']) {
            return Ok(hub(text));
        }
        Err(LoadError::NoSuchDataset(text.to_owned()))
    }
}

/// What loading needs besides the reference.
#[derive(Debug, Default)]
pub struct LoadOptions<'a> {
    /// The revision of a hub dataset (a branch, a tag or a commit); it
    /// overrides the one after `@` in the reference.
    pub revision: Option<&'a str>,
    /// The hub client, for a hub dataset.
    pub hub: Option<&'a Hub>,
    /// Where hub datasets are kept, one directory per repository and commit.
    pub cache: Option<&'a Path>,
}

/// The splits of the dataset `reference` names, read as records of type `R`.
/// `config` selects a subdirectory of the dataset, as a Hub configuration
/// does.
///
/// # Errors
/// The dataset cannot be found or read, or a line is not an `R`.
pub fn load_dataset<R: Record>(
    reference: &str,
    config: Option<&str>,
    options: &LoadOptions<'_>,
) -> Result<DatasetDict<R>, LoadError> {
    let (files, revision) = match Reference::parse(reference)? {
        Reference::Directory(root) => (split_files(&root, config)?, None),
        Reference::Product(root) => (product_files(&root)?, None),
        Reference::Hub { repo, revision } => {
            let asked = options.revision.map(str::to_owned).or(revision);
            let (root, commit) = fetch(&repo, asked.as_deref(), options)?;
            (split_files(&root, config)?, Some(commit))
        }
    };
    if files.is_empty() {
        return Err(LoadError::Empty(reference.to_owned()));
    }
    let mut splits = BTreeMap::new();
    for (name, path) in files {
        let provenance = Provenance {
            reference: reference.to_owned(),
            config: config.map(str::to_owned),
            revision: revision.clone(),
            file: path.display().to_string(),
        };
        splits.insert(name.clone(), Dataset::new(name, provenance, read(&path)?));
    }
    Ok(DatasetDict::new(splits))
}

/// The records of one split file, each with its line number (from 1; blank
/// lines are skipped and still count).
///
/// # Errors
/// The file cannot be read, or a line is not an `R`.
pub fn read_records<R: Record>(path: &Path) -> Result<Vec<(usize, R)>, LoadError> {
    let text = std::fs::read_to_string(path).map_err(|e| LoadError::Read {
        path: path.display().to_string(),
        message: e.to_string(),
    })?;
    text.lines()
        .enumerate()
        .filter(|(_, line)| !line.is_empty())
        .map(|(index, line)| {
            R::parse(line)
                .map(|record| (index + 1, record))
                .map_err(|message| LoadError::Record {
                    path: path.display().to_string(),
                    line: index + 1,
                    message,
                })
        })
        .collect()
}

fn read<R: Record>(path: &Path) -> Result<Vec<R>, LoadError> {
    Ok(read_records(path)?.into_iter().map(|(_, r)| r).collect())
}

/// The split files of a directory: `SPLIT.jsonl` at the root or in `data/`,
/// or, for a configuration, in its own directory (also under `data/`). The
/// nearer directory wins when both have a split.
fn split_files(root: &Path, config: Option<&str>) -> Result<BTreeMap<String, PathBuf>, LoadError> {
    let places: Vec<PathBuf> = match config {
        Some(config) => vec![root.join(config), root.join("data").join(config)],
        None => vec![root.to_owned(), root.join("data")],
    };
    if let Some(config) = config {
        if !places.iter().any(|p| p.is_dir()) {
            return Err(LoadError::NoConfig(config.to_owned()));
        }
    }
    let mut files = BTreeMap::new();
    for place in places {
        // A place that is not a directory holds no split; one that is, and
        // cannot be read, is an error, never an empty dataset.
        if !place.is_dir() {
            continue;
        }
        let unreadable = |e: std::io::Error| LoadError::Read {
            path: place.display().to_string(),
            message: e.to_string(),
        };
        let mut found: Vec<PathBuf> = Vec::new();
        for entry in std::fs::read_dir(&place).map_err(unreadable)? {
            let path = entry.map_err(unreadable)?.path();
            if path.is_file() && path.extension().is_some_and(|x| x == "jsonl") {
                found.push(path);
            }
        }
        found.sort();
        for path in found {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                files
                    .entry(canonical_split(stem).to_owned())
                    .or_insert(path);
            }
        }
    }
    Ok(files)
}

/// The split files a dataset product lists.
fn product_files(root: &Path) -> Result<BTreeMap<String, PathBuf>, LoadError> {
    let path = root.join("dataset.json");
    let read_err = |message: String| LoadError::Read {
        path: path.display().to_string(),
        message,
    };
    let text = std::fs::read_to_string(&path).map_err(|e| read_err(e.to_string()))?;
    let product: splinter_core::factory::product::DatasetProduct =
        serde_json::from_str(&text).map_err(|e| read_err(e.to_string()))?;
    let mut files = BTreeMap::new();
    files.insert("train".to_owned(), root.join(&product.train.path));
    let validation = root.join(&product.validation.path);
    if validation.is_file() {
        files.insert("validation".to_owned(), validation);
    }
    Ok(files)
}

/// A hub dataset's `.jsonl` files at one commit, in the cache.
fn fetch(
    repo: &str,
    revision: Option<&str>,
    options: &LoadOptions<'_>,
) -> Result<(PathBuf, String), LoadError> {
    let (Some(hub), Some(cache)) = (options.hub, options.cache) else {
        return Err(LoadError::NeedsHub(repo.to_owned()));
    };
    let hub_err = |message: String| LoadError::Read {
        path: repo.to_owned(),
        message,
    };
    let id = RepoId::parse(repo).map_err(|e| hub_err(e.to_string()))?;
    let commit = hub
        .resolve(&id, revision.unwrap_or("main"))
        .map_err(|e| hub_err(e.to_string()))?;
    let tree = hub.tree(&id, &commit).map_err(|e| hub_err(e.to_string()))?;
    let root = cache.join(format!(
        "{}@{}",
        repo.replace('/', "__"),
        &commit.as_str()[..12.min(commit.as_str().len())]
    ));
    for file in tree.iter().filter(|f| f.path.ends_with(".jsonl")) {
        let dest = root.join(&file.path);
        if std::fs::metadata(&dest).is_ok_and(|m| m.len() == file.size) {
            continue;
        }
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).map_err(|e| hub_err(e.to_string()))?;
        }
        let part = dest.with_extension("jsonl.part");
        let path = RepoPath::parse(&file.path).map_err(|e| hub_err(e.to_string()))?;
        let (sha, size) = hub
            .download(&id, &commit, &path, &part, FILE_LIMIT)
            .map_err(|e| hub_err(e.to_string()))?;
        if size != file.size
            || file
                .lfs_sha256
                .as_ref()
                .is_some_and(|expected| *expected != sha)
        {
            let _ = std::fs::remove_file(&part);
            return Err(hub_err(format!(
                "{} did not download as the hub lists it",
                file.path
            )));
        }
        std::fs::rename(&part, &dest).map_err(|e| hub_err(e.to_string()))?;
    }
    Ok((root, commit.as_str().to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load::ChatRecord;

    fn record(task: &str) -> String {
        format!(
            r#"{{"messages":[{{"role":"user","content":"x","train":true}}],"metadata":{{"task":"{task}"}}}}"#
        )
    }

    fn write(dir: &Path, name: &str, tasks: &[&str]) {
        std::fs::create_dir_all(dir).unwrap();
        let text: String = tasks.iter().map(|t| format!("{}\n", record(t))).collect();
        std::fs::write(dir.join(name), text).unwrap();
    }

    #[test]
    fn a_directory_loads_as_splits_with_validation_under_its_aliases() {
        let dir = tempfile::tempdir().unwrap();
        write(&dir.path().join("data"), "train.jsonl", &["a", "b"]);
        write(&dir.path().join("data"), "eval.jsonl", &["c"]);
        let dict =
            load_dataset::<ChatRecord>(dir.path().to_str().unwrap(), None, &LoadOptions::default())
                .unwrap();
        assert_eq!(
            dict.split_names().collect::<Vec<_>>(),
            ["train", "validation"]
        );
        assert_eq!(dict.get("train").unwrap().len(), 2);
        assert_eq!(dict.get("eval").unwrap().get(0).unwrap().task(), Some("c"));
        assert_eq!(dict.get("validation").unwrap().split(), "validation");
    }

    #[test]
    fn a_configuration_is_a_subdirectory_and_an_unknown_one_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "train.jsonl", &["root"]);
        write(&dir.path().join("hi"), "train.jsonl", &["hi1", "hi2"]);
        let path = dir.path().to_str().unwrap();
        let hi = load_dataset::<ChatRecord>(path, Some("hi"), &LoadOptions::default()).unwrap();
        assert_eq!(hi.get("train").unwrap().len(), 2);
        let none = load_dataset::<ChatRecord>(path, Some("xx"), &LoadOptions::default());
        assert!(matches!(none, Err(LoadError::NoConfig(_))), "{none:?}");
    }

    #[test]
    fn a_bad_line_is_named_by_file_and_line() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("train.jsonl"),
            format!("{}\nnot json\n", record("a")),
        )
        .unwrap();
        let err =
            load_dataset::<ChatRecord>(dir.path().to_str().unwrap(), None, &LoadOptions::default())
                .unwrap_err()
                .to_string();
        assert!(err.contains("train.jsonl line 2"), "{err}");
    }

    #[test]
    fn a_repository_name_needs_a_hub_and_a_missing_path_is_refused() {
        assert_eq!(
            Reference::parse("hf:a/b@v4").unwrap(),
            Reference::Hub {
                repo: "a/b".into(),
                revision: Some("v4".into())
            }
        );
        let needs = load_dataset::<ChatRecord>("owner/name", None, &LoadOptions::default());
        assert!(matches!(needs, Err(LoadError::NeedsHub(_))), "{needs:?}");
        assert!(matches!(
            Reference::parse("./no/such/dir/at/all"),
            Err(LoadError::NoSuchDataset(_))
        ));
    }
}
