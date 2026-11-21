// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the loop from a failed agent attempt to a
// verified corrected one, for its clients. If your team needs expertise in
// keeping evaluation material out of self-generated training data you can
// procure our services by sending an email to info@swedishembedded.com.

//! What a rationalisation run reads: the compiled task folders of a dataset,
//! each with the side its `bench-compile.json` record puts it on, and the
//! student's graded Harbor trials of them.
//!
//! A hint is a task's reference solution, so it comes only from a task the
//! compile record names on the `train` side and that no marker seals
//! ([`splinter_knowledge::tasks::workspace::exam_material`]); any other
//! folder, or one the record does not name, is refused by name and why.
//! Nothing of a task folder is executed here: its files are only read.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use splinter_knowledge::tasks::workspace::exam_material;

use super::filters::TaskFiles;
use crate::bench::repair::select::rank;

/// The file a compiled dataset records its tasks and their sides in.
pub const COMPILE_FILE: &str = "bench-compile.json";

/// The schema of [`COMPILE_FILE`] this reader knows.
pub const COMPILE_SCHEMA: &str = "bench.compile/1";

/// The only side hints come from.
const TRAIN: &str = "train";

/// The largest task file read, in bytes.
const MAX_FILE_BYTES: u64 = 1 << 20;

/// One task of the compile record.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompileEntry {
    /// The task's name.
    pub name: String,
    /// Its folder, relative to the dataset.
    pub path: String,
    /// Its digest, as the compiler recorded it.
    pub digest: String,
    /// The side it is on (`train` or `eval`).
    pub set: String,
}

#[derive(Deserialize)]
struct CompileRecord {
    schema: String,
    #[serde(default)]
    bench_commit: Option<String>,
    tasks: Vec<CompileEntry>,
}

/// A compiled dataset: its directory and the side of each task.
#[derive(Clone, Debug)]
pub struct CompiledDataset {
    /// The dataset directory, resolved.
    pub dir: PathBuf,
    /// The sha256 of its compile record.
    pub sha256: String,
    /// The examiner commit that compiled it, when recorded.
    pub bench_commit: Option<String>,
    entries: BTreeMap<String, CompileEntry>,
}

/// A train-side task, read: what a hinted attempt and the filters need.
#[derive(Clone, Debug)]
pub struct TrainTask {
    /// The task folder in the dataset.
    pub folder: PathBuf,
    /// Its path relative to the dataset (`train/040-paste-above`).
    pub rel: String,
    /// Its compile record.
    pub entry: CompileEntry,
    /// `[metadata].bench_difficulty`, the rung a selection stratifies by.
    pub difficulty: Option<i64>,
    /// `[metadata].bench_dimension`.
    pub dimension: Option<String>,
    /// `instruction.md`, as it is.
    pub instruction: String,
    /// `solution/solve.sh`, the hint.
    pub hint: String,
    /// The starting files (`environment/workspace/`) and target files
    /// (`tests/expected/`).
    pub files: TaskFiles,
}

fn read_text(path: &Path) -> Result<String, String> {
    let meta = std::fs::symlink_metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if !meta.is_file() {
        return Err(format!("{} is not a regular file", path.display()));
    }
    if meta.len() > MAX_FILE_BYTES {
        return Err(format!(
            "{} is {} bytes, more than the {MAX_FILE_BYTES} read",
            path.display(),
            meta.len()
        ));
    }
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// The regular files under `dir` by path relative to it; symbolic links are
/// refused, never followed.
fn files_under(dir: &Path) -> Result<BTreeMap<String, String>, String> {
    let mut found = BTreeMap::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(at) = pending.pop() {
        let entries = std::fs::read_dir(&at).map_err(|e| format!("{}: {e}", at.display()))?;
        for entry in entries {
            let path = entry.map_err(|e| format!("{}: {e}", at.display()))?.path();
            let kind = std::fs::symlink_metadata(&path)
                .map_err(|e| format!("{}: {e}", path.display()))?
                .file_type();
            if kind.is_symlink() {
                return Err(format!("{} is a symbolic link", path.display()));
            }
            if kind.is_dir() {
                pending.push(path);
                continue;
            }
            let rel = path
                .strip_prefix(dir)
                .map_err(|e| e.to_string())?
                .to_string_lossy()
                .replace('\\', "/");
            found.insert(rel, read_text(&path)?);
        }
    }
    Ok(found)
}

impl CompiledDataset {
    /// The dataset at `dir`, by its compile record.
    ///
    /// # Errors
    /// The record is missing, unreadable or of another schema.
    pub fn read(dir: &Path) -> Result<Self, String> {
        let dir = dir
            .canonicalize()
            .map_err(|e| format!("{}: {e}", dir.display()))?;
        let path = dir.join(COMPILE_FILE);
        let bytes = std::fs::read(&path).map_err(|e| {
            format!(
                "{}: {e}; without it no task's side is known, and no hint is taken",
                path.display()
            )
        })?;
        let record: CompileRecord =
            serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
        if record.schema != COMPILE_SCHEMA {
            return Err(format!(
                "{} is {}, not {COMPILE_SCHEMA}",
                path.display(),
                record.schema
            ));
        }
        let entries = record
            .tasks
            .into_iter()
            .map(|e| (e.path.trim_start_matches("./").to_string(), e))
            .collect();
        Ok(Self {
            sha256: crate::bench::artifact::sha256_hex(&bytes),
            bench_commit: record.bench_commit,
            dir,
            entries,
        })
    }

    /// The train-side task in `folder`, read.
    ///
    /// # Errors
    /// Naming the task and why: it is not a folder of this dataset, the
    /// compile record does not name it, it is not on the train side, it is
    /// sealed, or it has no reference solution to hint with.
    pub fn train_task(&self, folder: &Path) -> Result<TrainTask, String> {
        let resolved = folder
            .canonicalize()
            .map_err(|e| format!("task {}: {e}", folder.display()))?;
        let rel = resolved
            .strip_prefix(&self.dir)
            .map_err(|_| {
                format!(
                    "task {} is not a folder of the dataset {}: its side is unknown, so it is \
                     never a hint's source",
                    folder.display(),
                    self.dir.display()
                )
            })?
            .to_string_lossy()
            .replace('\\', "/");
        let entry = self.entries.get(&rel).ok_or_else(|| {
            format!(
                "task {rel} has no record in {COMPILE_FILE}: its side is unknown, so it is never \
                 a hint's source"
            )
        })?;
        let name = &entry.name;
        if entry.set != TRAIN {
            return Err(format!(
                "task {name} ({rel}) is on the {} side: hints come from train-side tasks only",
                entry.set
            ));
        }
        let config: toml::Table = read_text(&resolved.join("task.toml"))?
            .parse()
            .map_err(|e: toml::de::Error| format!("task {name}: task.toml: {e}"))?;
        if let Some(why) = exam_material(&resolved, &config)
            .map_err(|e| format!("task {name}: its seal cannot be read: {e}"))?
        {
            return Err(format!(
                "task {name} ({rel}) is sealed material ({why}): never a hint's source"
            ));
        }
        let hint = read_text(&resolved.join("solution").join("solve.sh")).map_err(|e| {
            format!("task {name} ({rel}) has no reference solution to hint with: {e}")
        })?;
        let metadata = config.get("metadata").and_then(toml::Value::as_table);
        let starter_dir = resolved.join("environment").join("workspace");
        let expected_dir = resolved.join("tests").join("expected");
        let files = TaskFiles {
            starter: if starter_dir.is_dir() {
                files_under(&starter_dir).map_err(|e| format!("task {name}: {e}"))?
            } else {
                BTreeMap::new()
            },
            expected: if expected_dir.is_dir() {
                Some(files_under(&expected_dir).map_err(|e| format!("task {name}: {e}"))?)
            } else {
                None
            },
        };
        Ok(TrainTask {
            instruction: read_text(&resolved.join("instruction.md"))?,
            difficulty: metadata
                .and_then(|m| m.get("bench_difficulty"))
                .and_then(toml::Value::as_integer),
            dimension: metadata
                .and_then(|m| m.get("bench_dimension"))
                .and_then(toml::Value::as_str)
                .map(str::to_string),
            folder: resolved,
            rel,
            entry: entry.clone(),
            hint,
            files,
        })
    }
}

/// One graded Harbor trial of the student.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct StudentTrial {
    /// The job it is of.
    pub job: PathBuf,
    /// Its name.
    pub trial: String,
    /// The task folder it attempted, as its result records it.
    pub task: PathBuf,
    /// The verifier's reward; `None` when it gave none or the trial ended
    /// in an exception.
    pub reward: Option<f64>,
}

impl StudentTrial {
    /// Whether the verifier passed it.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.reward.is_some_and(|r| (r - 1.0).abs() < f64::EPSILON)
    }

    /// Whether the verifier graded it and did not pass it.
    #[must_use]
    pub fn failed(&self) -> bool {
        self.reward.is_some() && !self.passed()
    }
}

/// The reward of a Harbor trial's `result.json`; `None` when it ended in an
/// exception or the verifier gave none.
#[must_use]
pub fn reward_of(result: &Value) -> Option<f64> {
    if result.get("exception_info").is_some_and(|e| !e.is_null()) {
        return None;
    }
    result
        .pointer("/verifier_result/rewards/reward")
        .and_then(Value::as_f64)
}

/// The trial a Harbor `result.json` records.
fn trial_of(job: &Path, result: &Value) -> Result<StudentTrial, String> {
    let trial = result
        .get("trial_name")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{}: a trial result names no trial", job.display()))?;
    let task = result
        .pointer("/config/task/path")
        .or_else(|| result.pointer("/task_id/path"))
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{}: trial {trial} names no task folder", job.display()))?;
    Ok(StudentTrial {
        job: job.to_path_buf(),
        trial: trial.to_string(),
        task: PathBuf::from(task),
        reward: reward_of(result),
    })
}

/// Whether `job` is packed (`bench pack`): its trials are in `trials.jsonl`.
#[must_use]
pub fn is_packed(job: &Path) -> bool {
    job.join("trials.jsonl").is_file()
}

/// The trials of the Harbor job `job`, packed or not, in name order.
///
/// # Errors
/// The job, or a trial of it, cannot be read.
pub fn student_trials(job: &Path) -> Result<Vec<StudentTrial>, String> {
    let mut trials = Vec::new();
    if is_packed(job) {
        let path = job.join("trials.jsonl");
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        for (n, line) in text
            .lines()
            .enumerate()
            .filter(|(_, l)| !l.trim().is_empty())
        {
            let row: Value = serde_json::from_str(line)
                .map_err(|e| format!("{} line {}: {e}", path.display(), n + 1))?;
            let result = row
                .get("result")
                .ok_or_else(|| format!("{} line {}: no result", path.display(), n + 1))?;
            trials.push(trial_of(job, result)?);
        }
    } else {
        let entries = std::fs::read_dir(job).map_err(|e| format!("{}: {e}", job.display()))?;
        for entry in entries.filter_map(Result::ok) {
            let result = entry.path().join("result.json");
            if !result.is_file() {
                continue;
            }
            let value: Value = serde_json::from_slice(
                &std::fs::read(&result).map_err(|e| format!("{}: {e}", result.display()))?,
            )
            .map_err(|e| format!("{}: {e}", result.display()))?;
            trials.push(trial_of(job, &value)?);
        }
    }
    trials.sort_by(|a, b| a.trial.cmp(&b.trial));
    Ok(trials)
}

/// `count` of `items` (all when there are fewer), stratified by `stratum`:
/// each stratum gets its share of `count` by its size (largest remainder,
/// ties to the stratum ranked first under `seed`), and within a stratum the
/// items are taken in the order of their seeded rank by `name`. Returned by
/// stratum, then rank.
pub fn stratified<T>(
    items: Vec<T>,
    stratum: impl Fn(&T) -> Option<i64>,
    name: impl Fn(&T) -> &str,
    count: usize,
    seed: u64,
) -> Vec<T> {
    let total = items.len();
    let mut strata: BTreeMap<Option<i64>, Vec<(String, T)>> = BTreeMap::new();
    for item in items {
        let key = rank(seed, name(&item));
        strata.entry(stratum(&item)).or_default().push((key, item));
    }
    for group in strata.values_mut() {
        group.sort_by(|a, b| a.0.cmp(&b.0));
    }
    let count = count.min(total);
    let mut quotas: BTreeMap<Option<i64>, usize> = BTreeMap::new();
    let mut remainders = Vec::new();
    for (key, group) in &strata {
        let exact = count * group.len();
        quotas.insert(*key, exact / total.max(1));
        let stratum_rank = rank(seed, &format!("stratum:{key:?}"));
        remainders.push((exact % total.max(1), stratum_rank, *key));
    }
    // Largest remainder first; equal remainders in seeded stratum order.
    remainders.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    let mut left = count - quotas.values().sum::<usize>();
    for (_, _, key) in remainders {
        if left == 0 {
            break;
        }
        quotas.entry(key).and_modify(|q| *q += 1);
        left -= 1;
    }
    strata
        .into_iter()
        .flat_map(|(key, group)| {
            let take = quotas.get(&key).copied().unwrap_or(0);
            group.into_iter().take(take).map(|(_, item)| item)
        })
        .collect()
}

#[cfg(test)]
#[path = "material_tests.rs"]
mod tests;
