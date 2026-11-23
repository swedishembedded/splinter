// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that turn an examiner's practice
// material into graded agent experience for its clients. If your team needs
// expertise in on-policy experience collection for agent training you can
// procure our services by sending an email to info@swedishembedded.com.

//! Practice material as the examiner published it (`bench.practice/2`): a
//! directory holding `practice.json` and the staged task folders, read and
//! held to what the learner may practise on. Nothing here runs Bench: it
//! reads material already recorded, for the capability report and for the
//! dataset builds that shape a reward by a task's naive floor.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::bench::artifact::sha256_hex;
use crate::episodes::EpisodeError;

/// The practice manifest schema this reader understands.
pub const PRACTICE_SCHEMA: &str = "bench.practice/2";

/// The practice manifest's file name.
pub const PRACTICE_FILE: &str = "practice.json";

/// One staged task, by digest.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StagedTask {
    /// The task folder's name.
    pub name: String,
    /// The sha256 of the staged folder.
    pub digest: String,
    /// The sha256 of the task as drawn or as the collection holds it.
    pub source_digest: String,
    /// The capability ids the examiner's taxonomy labels the task with;
    /// empty in material staged before the examiner named them. Data: the
    /// ids are the examiner's, never a list this reader knows.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub capabilities: Vec<String>,
    /// The reporting track the taxonomy puts the task in, when it does.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub track: Option<String>,
    /// The best share of the task's weighted checks a naive answer earns, in
    /// `[0, 1]`, as the examiner measured it when it admitted the task; absent
    /// in material staged before the examiner exported it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub naive_floor: Option<f64>,
}

/// The fields of `practice.json` this reader uses.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PracticeManifest {
    /// Must be [`PRACTICE_SCHEMA`].
    pub schema: String,
    /// Always `train`: the examiner never writes another side.
    pub set: String,
    /// The staged task folders' directory, relative to the practice
    /// directory.
    #[serde(default)]
    pub tasks_dir: Option<String>,
    /// The staged tasks.
    pub tasks: Vec<StagedTask>,
}

/// The key of a staged task's `task.toml` `[metadata]` that names the
/// lineage it was generated from (`breach/defective`): the family a seed of
/// it belongs to.
const LINEAGE_KEY: &str = "bench_defect_lineage";

/// What practice material says of a task beyond its folder: its naive floor
/// and its lineage.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct TaskFacts {
    /// The staged folder's name.
    pub name: String,
    /// [`StagedTask::naive_floor`].
    pub naive_floor: Option<f64>,
    /// The lineage its `task.toml` names, when it names one.
    pub lineage: Option<String>,
}

/// The facts of every task of some practice directories, by the task's
/// source digest (the same task, byte for byte, as a trial's verdict names it
/// by `task.digest`), with the `practice.json` sha256 of each directory read.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct PracticeFacts {
    /// Per task, by source digest.
    pub tasks: BTreeMap<String, TaskFacts>,
    /// The sha256 of each `practice.json` read, in the order given.
    pub sources: Vec<String>,
}

impl PracticeFacts {
    /// The facts of the practice directories `dirs`; none for none.
    ///
    /// # Errors
    /// A directory is not readable practice material ([`Practice::read`]).
    pub fn read(dirs: &[PathBuf]) -> Result<Self, EpisodeError> {
        let mut facts = Self::default();
        for dir in dirs {
            let practice = Practice::read(dir)?;
            facts.tasks.extend(practice.task_facts());
            facts.sources.push(practice.sha256);
        }
        Ok(facts)
    }

    /// Each task's naive floor, by source digest; a task without one is absent.
    #[must_use]
    pub fn naive_floors(&self) -> BTreeMap<String, f64> {
        self.tasks
            .iter()
            .filter_map(|(digest, t)| Some((digest.clone(), t.naive_floor?)))
            .collect()
    }

    /// The lineage of the task whose source digest is `digest`, when known.
    #[must_use]
    pub fn lineage(&self, digest: Option<&str>) -> Option<&str> {
        self.tasks.get(digest?)?.lineage.as_deref()
    }
}

/// A practice directory read and checked.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Practice {
    /// The directory (the one holding `practice.json`).
    pub dir: PathBuf,
    /// The sha256 of its `practice.json`: what identifies the material.
    pub sha256: String,
    /// The manifest.
    pub manifest: PracticeManifest,
}

fn malformed(dir: &Path, reason: impl Into<String>) -> EpisodeError {
    EpisodeError::Malformed {
        directory: dir.to_path_buf(),
        reason: reason.into(),
    }
}

impl Practice {
    /// Reads the practice directory `dir`.
    ///
    /// # Errors
    /// [`EpisodeError::Sealed`] for material that is not the train side,
    /// [`EpisodeError::Malformed`] for a directory without a readable
    /// manifest of [`PRACTICE_SCHEMA`], or one that stages no task.
    pub fn read(dir: &Path) -> Result<Self, EpisodeError> {
        let path = dir.join(PRACTICE_FILE);
        let bytes =
            std::fs::read(&path).map_err(|e| malformed(dir, format!("{PRACTICE_FILE}: {e}")))?;
        let manifest: PracticeManifest = serde_json::from_slice(&bytes)
            .map_err(|e| malformed(dir, format!("{PRACTICE_FILE}: {e}")))?;
        if manifest.schema != PRACTICE_SCHEMA {
            return Err(malformed(
                dir,
                format!(
                    "{PRACTICE_FILE} is {:?}, not {PRACTICE_SCHEMA:?}",
                    manifest.schema
                ),
            ));
        }
        if manifest.set != "train" {
            return Err(EpisodeError::Sealed {
                directory: dir.to_path_buf(),
                reason: format!(
                    "the material is the {:?} side: only the train side is practised on",
                    manifest.set
                ),
            });
        }
        if manifest.tasks.is_empty() {
            return Err(malformed(dir, "it stages no task"));
        }
        if let Some(task) = manifest.tasks.iter().find(|t| {
            t.naive_floor
                .is_some_and(|f| !(f.is_finite() && (0.0..=1.0).contains(&f)))
        }) {
            return Err(malformed(
                dir,
                format!("task {}'s naive floor is not a share in [0, 1]", task.name),
            ));
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            sha256: sha256_hex(&bytes),
            manifest,
        })
    }

    /// The facts of every staged task, by source digest: its naive floor and
    /// the lineage its `task.toml` names (none when the file or key is
    /// absent or does not read).
    #[must_use]
    pub fn task_facts(&self) -> BTreeMap<String, TaskFacts> {
        let tasks = self.dir.join(self.tasks_dir());
        self.manifest
            .tasks
            .iter()
            .map(|t| {
                let lineage = std::fs::read_to_string(tasks.join(&t.name).join("task.toml"))
                    .ok()
                    .and_then(|text| text.parse::<toml::Table>().ok())
                    .and_then(|table| {
                        table
                            .get("metadata")?
                            .get(LINEAGE_KEY)?
                            .as_str()
                            .map(str::to_string)
                    });
                (
                    t.source_digest.clone(),
                    TaskFacts {
                        name: t.name.clone(),
                        naive_floor: t.naive_floor,
                        lineage,
                    },
                )
            })
            .collect()
    }

    /// The staged task folders' directory, relative to the practice
    /// directory (`tasks` unless the manifest names another).
    #[must_use]
    pub fn tasks_dir(&self) -> &str {
        self.manifest.tasks_dir.as_deref().unwrap_or("tasks")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn directory(set: &str, floor: f64) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let manifest = serde_json::json!({
            "schema": PRACTICE_SCHEMA,
            "set": set,
            "tasks": [{
                "name": "alpha", "digest": "d1", "source_digest": "s1",
                "capabilities": ["edit"], "naive_floor": floor
            }],
        });
        std::fs::write(dir.path().join(PRACTICE_FILE), manifest.to_string()).unwrap();
        let task = dir.path().join("tasks/alpha");
        std::fs::create_dir_all(&task).unwrap();
        std::fs::write(
            task.join("task.toml"),
            "[metadata]\nbench_defect_lineage = \"breach/defective\"\n",
        )
        .unwrap();
        dir
    }

    #[test]
    fn the_facts_of_a_train_side_directory_are_its_floor_and_lineage_by_source_digest() {
        let dir = directory("train", 0.25);
        let facts = PracticeFacts::read(&[dir.path().to_path_buf()]).unwrap();
        assert_eq!(facts.naive_floors()["s1"], 0.25);
        assert_eq!(facts.lineage(Some("s1")), Some("breach/defective"));
        assert_eq!(facts.sources.len(), 1);
    }

    #[test]
    fn material_of_another_side_or_with_a_floor_outside_zero_to_one_is_refused() {
        let sealed = directory("eval", 0.25);
        assert!(matches!(
            Practice::read(sealed.path()),
            Err(EpisodeError::Sealed { .. })
        ));
        let bad = directory("train", 1.5);
        assert!(matches!(
            Practice::read(bad.path()),
            Err(EpisodeError::Malformed { .. })
        ));
    }
}
