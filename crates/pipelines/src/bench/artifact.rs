// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements tamper-evident hand-over of agent benchmark
// results into training pipelines, for its clients. If your team needs
// expertise in turning benchmark runs into verified training data, you can
// procure our services by sending an email to info@swedishembedded.com.

//! A trial artifact exported by the benchmark harness, read and verified.
//!
//! An export directory holds `index.jsonl` (one row per trial, naming the
//! digest of the trial's manifest) and one immutable directory per trial.
//! A trial is read only when every claim the artifact makes about itself
//! holds: the manifest is the one the index names, it agrees with the index
//! row, and every file in the directory is exactly one the manifest lists,
//! with the digest it lists. Anything else is refused with the reason, and
//! nothing of a refused trial reaches the store.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufReader};
use std::path::{Component, Path};

use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest as _, Sha256};

/// The manifest schema this reader understands.
pub const MANIFEST_SCHEMA: &str = "bench.trial-artifact/1";

/// The file name of the manifest in a trial directory.
const MANIFEST: &str = "manifest.json";

/// The file that is the trial's record.
const TRAJECTORY: &str = "trajectory.atif.json";

/// One row of an export's `index.jsonl`.
#[derive(Clone, Debug, Deserialize)]
pub struct IndexRow {
    /// The trial's name.
    pub trial: String,
    /// The benchmark task's id.
    pub task: String,
    /// The reward the grade gave.
    pub reward: f64,
    /// `pass` or `fail`.
    pub verdict: String,
    /// The split the task belongs to, when the export states it.
    #[serde(default)]
    pub set: Option<String>,
    /// The sha256 of the trial's `manifest.json`.
    pub manifest_sha256: String,
    /// The trial directory relative to the export.
    pub path: String,
}

/// The task a trial attempted, as the manifest names it.
#[derive(Clone, Debug, Deserialize)]
pub struct ManifestTask {
    /// The task's id.
    pub id: String,
    /// The task's version.
    pub version: String,
    /// The digest of the task's content.
    pub digest: String,
    /// The capability the task exercises.
    #[serde(default)]
    pub dimension: Option<String>,
    /// The split the task belongs to, when the export states it.
    #[serde(default)]
    pub set: Option<String>,
}

/// The agent that ran the trial.
#[derive(Clone, Debug, Deserialize)]
pub struct ManifestAgent {
    /// The agent's name.
    pub name: String,
    /// The agent's version.
    pub version: String,
    /// The digest of the agent build.
    #[serde(default)]
    pub sven_build_sha256: Option<String>,
}

/// The verifier that graded the trial.
#[derive(Clone, Debug, Deserialize)]
pub struct ManifestVerifier {
    /// The digest of the verifier.
    pub digest: String,
    /// How the checks combine into a reward.
    #[serde(default)]
    pub scoring: Option<String>,
}

/// The grade the verifier gave.
#[derive(Clone, Debug, Deserialize)]
pub struct ManifestGrade {
    /// The reward.
    pub reward: f64,
    /// `pass` or `fail`.
    pub verdict: String,
    /// The per-check metrics.
    #[serde(default)]
    pub metrics: Option<Value>,
}

/// One model the trial's agent used.
#[derive(Clone, Debug, Deserialize)]
pub struct ManifestModel {
    /// The model's name as the server knew it.
    pub model: String,
}

/// The fields of `manifest.json` this reader uses; the rest are ignored.
#[derive(Clone, Debug, Deserialize)]
pub struct Manifest {
    /// Must be [`MANIFEST_SCHEMA`].
    pub schema: String,
    /// The trial's name.
    pub trial: String,
    /// The task attempted.
    pub task: ManifestTask,
    /// The repository snapshot the trial started from.
    #[serde(default)]
    pub snapshot: Option<String>,
    /// The container image and environment.
    #[serde(default)]
    pub image: Option<Value>,
    /// The agent.
    pub agent: ManifestAgent,
    /// The models, by role.
    #[serde(default)]
    pub models: BTreeMap<String, ManifestModel>,
    /// The verifier.
    pub verifier: ManifestVerifier,
    /// The grade.
    pub grade: ManifestGrade,
    /// Tokens the trial used.
    #[serde(default)]
    pub tokens: Option<Value>,
    /// Wall-clock seconds the trial took.
    #[serde(default)]
    pub wall_seconds: Option<f64>,
    /// The sha256 of every file of the trial directory but the manifest.
    pub files: BTreeMap<String, String>,
}

/// A trial whose every file matched its digest.
#[derive(Debug)]
pub struct Artifact {
    /// The index row it was read for.
    pub row: IndexRow,
    /// The verified manifest.
    pub manifest: Manifest,
    /// The trajectory, parsed from the bytes that were verified.
    pub trajectory: atif::Trajectory,
}

/// Reads the rows of `directory`'s `index.jsonl`; a row that does not parse
/// is returned as the reason it was refused, with its line number.
pub fn read_index(directory: &Path) -> Result<Vec<Result<IndexRow, String>>, String> {
    let path = directory.join("index.jsonl");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(text
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(n, line)| {
            serde_json::from_str::<IndexRow>(line)
                .map_err(|e| format!("index.jsonl line {}: {e}", n + 1))
        })
        .collect())
}

fn hex(digest: impl AsRef<[u8]>) -> String {
    digest.as_ref().iter().map(|b| format!("{b:02x}")).collect()
}

/// The sha256 of `bytes`, in lower-case hex.
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex(Sha256::digest(bytes))
}

/// The sha256 of the file at `path`, read without holding it in memory.
fn sha256_file(path: &Path) -> io::Result<String> {
    let mut hasher = Sha256::new();
    io::copy(&mut BufReader::new(File::open(path)?), &mut hasher)?;
    Ok(hex(hasher.finalize()))
}

/// Whether `name` is one plain path component: no separator, no `..`.
fn is_plain_name(name: &str) -> bool {
    let mut parts = Path::new(name).components();
    matches!(
        (parts.next(), parts.next()),
        (Some(Component::Normal(_)), None)
    )
}

/// Reads and verifies the trial `row` names under `export`.
///
/// # Errors
///
/// The reason the artifact is refused: a path that leaves the export, a
/// missing or altered file, a file the manifest does not list, a manifest
/// that is not the indexed one or disagrees with the row, an unknown
/// schema, or a trajectory that is not ATIF.
pub fn read_trial(export: &Path, row: &IndexRow) -> Result<Artifact, String> {
    if !is_plain_name(&row.path) {
        return Err(format!(
            "the index path {:?} is not a directory directly under the export",
            row.path
        ));
    }
    let dir = export.join(&row.path);
    if !dir.is_dir() {
        return Err(format!(
            "{} is not a directory (partial export)",
            dir.display()
        ));
    }
    let manifest_bytes =
        std::fs::read(dir.join(MANIFEST)).map_err(|e| format!("{MANIFEST}: {e}"))?;
    let found = sha256_hex(&manifest_bytes);
    if !found.eq_ignore_ascii_case(&row.manifest_sha256) {
        return Err(format!(
            "{MANIFEST} is altered: the index names sha256 {}, the file is {found}",
            row.manifest_sha256
        ));
    }
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|e| format!("{MANIFEST} is not a trial manifest: {e}"))?;
    agree(row, &manifest)?;
    let trajectory = verify_files(&dir, &manifest)?;
    Ok(Artifact {
        row: row.clone(),
        manifest,
        trajectory,
    })
}

/// The manifest and the index row must say the same about the trial.
fn agree(row: &IndexRow, manifest: &Manifest) -> Result<(), String> {
    if manifest.schema != MANIFEST_SCHEMA {
        return Err(format!(
            "manifest schema {:?} is not {MANIFEST_SCHEMA:?}",
            manifest.schema
        ));
    }
    let differs = |what: &str, index: &dyn std::fmt::Debug, manifest: &dyn std::fmt::Debug| {
        Err(format!(
            "the index and the manifest disagree on the {what}: {index:?} against {manifest:?}"
        ))
    };
    if manifest.trial != row.trial {
        return differs("trial", &row.trial, &manifest.trial);
    }
    if manifest.task.id != row.task {
        return differs("task", &row.task, &manifest.task.id);
    }
    if manifest.grade.verdict != row.verdict {
        return differs("verdict", &row.verdict, &manifest.grade.verdict);
    }
    if (manifest.grade.reward - row.reward).abs() > f64::EPSILON {
        return differs("reward", &row.reward, &manifest.grade.reward);
    }
    if manifest.task.set != row.set {
        return differs("set", &row.set, &manifest.task.set);
    }
    Ok(())
}

/// Checks every file of `dir` against `manifest.files`, both ways, and
/// returns the trajectory parsed from the verified bytes.
fn verify_files(dir: &Path, manifest: &Manifest) -> Result<atif::Trajectory, String> {
    let mut present: Vec<String> = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))? {
        let entry = entry.map_err(|e| format!("{}: {e}", dir.display()))?;
        present.push(entry.file_name().to_string_lossy().into_owned());
    }
    for name in &present {
        if name != MANIFEST && !manifest.files.contains_key(name) {
            return Err(format!(
                "{name} is in the directory but not in the manifest"
            ));
        }
    }
    let mut trajectory: Option<Vec<u8>> = None;
    for (name, expected) in &manifest.files {
        if !is_plain_name(name) {
            return Err(format!(
                "the manifest lists {name:?}, which is not a plain file name"
            ));
        }
        let path = dir.join(name);
        let unreadable = |e: io::Error| format!("{name}: {e} (partial artifact)");
        // The trajectory is parsed from the very bytes that were hashed.
        let found = if name == TRAJECTORY {
            let bytes = std::fs::read(&path).map_err(unreadable)?;
            let found = sha256_hex(&bytes);
            trajectory = Some(bytes);
            found
        } else {
            sha256_file(&path).map_err(unreadable)?
        };
        if !found.eq_ignore_ascii_case(expected) {
            return Err(format!(
                "{name} is altered: the manifest names sha256 {expected}, the file is {found}"
            ));
        }
    }
    let bytes = trajectory.ok_or_else(|| format!("the manifest lists no {TRAJECTORY}"))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("{TRAJECTORY} is not ATIF: {e}"))
}
