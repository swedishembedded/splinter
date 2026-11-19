// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that turn an examiner's practice
// material into graded agent experience for its clients. If your team needs
// expertise in on-policy experience collection for agent training you can
// procure our services by sending an email to info@swedishembedded.com.

//! Practice material as the examiner publishes it (`bench.practice/2`): a
//! directory holding `practice.json`, the staged task folders and the Harbor
//! job that runs them, read and held to what the learner may practise on.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::bench::artifact::sha256_hex;
use crate::episodes::EpisodeError;

/// The practice manifest schema this reader understands.
pub const PRACTICE_SCHEMA: &str = "bench.practice/2";

/// The practice manifest's file name.
pub const PRACTICE_FILE: &str = "practice.json";

/// The run manifest a profile's staging writes beside the job.
const RUN_FILE: &str = "bench-run.json";

/// One staged task, by digest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
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
}

impl StagedTask {
    /// What identifies the staged task: its name and digests, never its
    /// labels (a relabelled taxonomy stages the same material).
    fn identity(&self) -> (&str, &str, &str) {
        (&self.name, &self.digest, &self.source_digest)
    }
}

/// The fields of `practice.json` this reader uses; what identifies the
/// material is compared whole ([`Practice::same_material`]).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PracticeManifest {
    /// Must be [`PRACTICE_SCHEMA`].
    pub schema: String,
    /// The collection, as the examiner names it (`tasks/support`).
    pub collection: String,
    /// The sha256 of the collection's manifest; absent for a folder
    /// collection.
    #[serde(default)]
    pub manifest_sha256: Option<String>,
    /// The sha256 of the collection's content.
    #[serde(default)]
    pub content_sha256: Option<String>,
    /// Always `train`: the examiner never writes another side.
    pub set: String,
    /// The examiner's version and commit.
    #[serde(default)]
    pub bench: Value,
    /// The sandbox the job runs its tasks in (`userns`, `docker`).
    pub environment: String,
    /// The staged task folders' directory, relative to the practice
    /// directory.
    #[serde(default)]
    pub tasks_dir: Option<String>,
    /// A fresh draw from the collection's generator; absent for the
    /// committed train side.
    #[serde(default)]
    pub draw: Option<Value>,
    /// The staged tasks.
    pub tasks: Vec<StagedTask>,
    /// The taxonomy the tasks' capabilities were labelled by (`schema`,
    /// `sha256`); absent in material staged before the examiner named it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub taxonomy: Option<Value>,
    /// What a selection by capability kept (`capabilities`, `offered`);
    /// absent or null for material staged whole.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selection: Option<Value>,
}

/// The draw of a practice manifest made with `--count N --seed S`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Draw {
    /// Tasks drawn.
    pub count: u64,
    /// The seed they were drawn under.
    pub seed: u64,
}

/// The profile a practice job was staged under, as the examiner's run
/// manifest records it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct StagedProfile {
    /// The profile file.
    pub path: PathBuf,
    /// Its sha256 when staged.
    pub sha256: Option<String>,
}

/// Host tools a job's sandboxes are given: `(name, path)`.
type Tools = Vec<(String, String)>;

/// A practice directory read and checked.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Practice {
    /// The directory (the one holding `practice.json`).
    pub dir: PathBuf,
    /// The sha256 of its `practice.json`: what identifies the material.
    pub sha256: String,
    /// The manifest.
    pub manifest: PracticeManifest,
    /// The profile it was staged under, when it was.
    pub profile: Option<StagedProfile>,
    /// The host tools its sandboxes were given, by name.
    pub tools: Tools,
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
        let (profile, tools) = staged_run(dir)?;
        Ok(Self {
            dir: dir.to_path_buf(),
            sha256: sha256_hex(&bytes),
            manifest,
            profile,
            tools,
        })
    }

    /// The draw the material was made from, when it is one.
    ///
    /// # Errors
    /// The manifest's draw names no count or seed.
    pub fn draw(&self) -> Result<Option<Draw>, EpisodeError> {
        let Some(draw) = &self.manifest.draw else {
            return Ok(None);
        };
        let field = |key: &str| {
            draw.get(key)
                .and_then(Value::as_u64)
                .ok_or_else(|| malformed(&self.dir, format!("the draw records no {key}")))
        };
        Ok(Some(Draw {
            count: field("count")?,
            seed: field("seed")?,
        }))
    }

    /// The capability ids the material was selected by (`bench task
    /// practice --capability`); empty for material staged whole.
    ///
    /// # Errors
    /// A selection that names its capabilities as anything but strings.
    pub fn selected_capabilities(&self) -> Result<Vec<String>, EpisodeError> {
        let Some(selection) = self.manifest.selection.as_ref().filter(|s| !s.is_null()) else {
            return Ok(Vec::new());
        };
        selection
            .get("capabilities")
            .and_then(Value::as_array)
            .and_then(|ids| {
                ids.iter()
                    .map(|id| id.as_str().map(str::to_string))
                    .collect::<Option<Vec<_>>>()
            })
            .ok_or_else(|| malformed(&self.dir, "the selection names no capabilities"))
    }

    /// The capability ids of each staged task, by folder name; a task the
    /// examiner labelled with none is absent.
    #[must_use]
    pub fn capabilities(&self) -> BTreeMap<String, Vec<String>> {
        self.manifest
            .tasks
            .iter()
            .filter(|t| !t.capabilities.is_empty())
            .map(|t| (t.name.clone(), t.capabilities.clone()))
            .collect()
    }

    /// The staged task folders' directory, relative to the practice
    /// directory (`tasks` unless the manifest names another).
    #[must_use]
    pub fn tasks_dir(&self) -> &str {
        self.manifest.tasks_dir.as_deref().unwrap_or("tasks")
    }

    /// The name the examiner stages the collection under: its last part.
    #[must_use]
    pub fn staged_name(&self) -> &str {
        self.manifest
            .collection
            .rsplit('/')
            .find(|part| !part.is_empty())
            .unwrap_or(&self.manifest.collection)
    }

    /// Why `other` (the same collection staged again) is not this material,
    /// when it is not: the tasks, their digests, the draw and the
    /// collection's digests must all be the same.
    #[must_use]
    pub fn same_material(&self, other: &PracticeManifest) -> Option<String> {
        let ours = &self.manifest;
        if ours.collection != other.collection
            || ours.manifest_sha256 != other.manifest_sha256
            || ours.content_sha256 != other.content_sha256
            || ours.draw != other.draw
            || ours.selection.as_ref().filter(|s| !s.is_null())
                != other.selection.as_ref().filter(|s| !s.is_null())
        {
            return Some(format!(
                "the collection was staged as {} (manifest {:?}, content {:?}), not as {} \
                 (manifest {:?}, content {:?}) with the same draw",
                other.collection,
                other.manifest_sha256,
                other.content_sha256,
                ours.collection,
                ours.manifest_sha256,
                ours.content_sha256
            ));
        }
        let theirs: Vec<_> = other.tasks.iter().map(StagedTask::identity).collect();
        if ours
            .tasks
            .iter()
            .map(StagedTask::identity)
            .ne(theirs.iter().copied())
        {
            let differs = ours
                .tasks
                .iter()
                .find(|t| !theirs.contains(&t.identity()))
                .map_or_else(
                    || format!("{} tasks, not {}", other.tasks.len(), ours.tasks.len()),
                    |t| format!("task {} is not staged with digest {}", t.name, t.digest),
                );
            return Some(format!("the staged tasks differ: {differs}"));
        }
        None
    }
}

/// The profile and host tools of the run manifest beside a practice job
/// staged under a profile; nothing for one staged without.
fn staged_run(dir: &Path) -> Result<(Option<StagedProfile>, Tools), EpisodeError> {
    let path = dir.join(RUN_FILE);
    if !path.is_file() {
        return Ok((None, Vec::new()));
    }
    let run: Value = std::fs::read(&path)
        .map_err(|e| e.to_string())
        .and_then(|bytes| serde_json::from_slice(&bytes).map_err(|e| e.to_string()))
        .map_err(|e| malformed(dir, format!("{RUN_FILE}: {e}")))?;
    let profile = run
        .get("profile")
        .and_then(Value::as_str)
        .map(|path| StagedProfile {
            path: PathBuf::from(path),
            sha256: run
                .get("profile_sha256")
                .and_then(Value::as_str)
                .map(str::to_string),
        });
    let tools = run
        .get("tools")
        .and_then(Value::as_object)
        .into_iter()
        .flatten()
        .filter_map(|(name, path)| Some((name.clone(), path.as_str()?.to_string())))
        .collect();
    Ok((profile, tools))
}
