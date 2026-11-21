// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the loop from a failed agent attempt to a
// verified corrected one, for its clients. If your team needs expertise in
// running agent attempts on staged benchmark tasks you can procure our
// services by sending an email to info@swedishembedded.com.

//! The Harbor side of a rationalisation: staged copies of task folders run
//! once each through the examiner's sandbox, gateway and Sven adapter, and
//! what each trial left read back.
//!
//! A staged copy is the task folder with another `instruction.md` and
//! without its `solution/`: the environment, the verifier and the task's
//! name are the task's own, so the examiner's image cache and verifier serve
//! it as they serve the task. Nothing of the dataset is changed or run here;
//! the copy runs only inside the sandbox.
//!
//! A job that finished is read again, never run again; one that did not is
//! removed and run whole, because the gateway that served it is gone.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::Value;

use super::material::{reward_of, TrainTask};
use crate::bench::artifact::{first_request, sha256_hex, AgentContext};
use crate::bench::practice::{finished, harbor_dataset, task_of, BenchCheckout, DatasetRun};

/// The folder of a task a staged copy leaves out: the reference solution.
const SOLUTION: &str = "solution";

/// What every job of a run shares: the examiner, the profile, the gateway
/// and the sandboxes' limits.
pub(super) struct JobEnv<'a> {
    pub(super) checkout: &'a BenchCheckout,
    pub(super) profile: &'a Path,
    pub(super) gateway_state: &'a Path,
    pub(super) port: u16,
    pub(super) images: &'a Path,
    pub(super) jobs: usize,
}

/// One graded trial of a staged job.
#[derive(Clone, Debug)]
pub(super) struct Attempt {
    /// Its directory.
    pub(super) dir: PathBuf,
    /// The verifier's reward; `None` when it gave none.
    pub(super) reward: Option<f64>,
}

/// Copies `from` into `to`, refusing symbolic links, leaving out the
/// top-level entries `skip` names.
fn copy_tree(from: &Path, to: &Path, skip: &[&str]) -> Result<(), String> {
    std::fs::create_dir_all(to).map_err(|e| format!("{}: {e}", to.display()))?;
    let entries = std::fs::read_dir(from).map_err(|e| format!("{}: {e}", from.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("{}: {e}", from.display()))?;
        let name = entry.file_name();
        if skip.iter().any(|s| name == *s) {
            continue;
        }
        let path = entry.path();
        let kind = entry
            .file_type()
            .map_err(|e| format!("{}: {e}", path.display()))?;
        if kind.is_symlink() {
            return Err(format!(
                "{} is a symbolic link: a staged copy never follows one",
                path.display()
            ));
        }
        let target = to.join(&name);
        if kind.is_dir() {
            copy_tree(&path, &target, &[])?;
        } else {
            std::fs::copy(&path, &target).map_err(|e| format!("{}: {e}", path.display()))?;
        }
    }
    Ok(())
}

/// Stages a copy of `task` as `into/<name>` whose instruction is
/// `instruction`, without its reference solution. A copy left by an earlier
/// run is replaced.
pub(super) fn stage_copy(
    task: &TrainTask,
    instruction: &str,
    into: &Path,
    name: &str,
) -> Result<PathBuf, String> {
    let dir = into.join(name);
    if dir.exists() {
        std::fs::remove_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    copy_tree(&task.folder, &dir, &[SOLUTION, "instruction.md"])?;
    let path = dir.join("instruction.md");
    std::fs::write(&path, instruction).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(dir)
}

/// The name of the staged copy of `folder` for the attempt keyed `key`: the
/// key's prefix first, so it survives Harbor shortening trial names.
pub(super) fn staged_name(key: &str, folder: &str) -> String {
    format!("{}-{folder}", &key[..key.len().min(16)])
}

/// Runs the staged folders in `tasks` as Harbor job `name` under
/// `jobs_dir`, unless that job finished before, and returns its graded
/// trials by staged folder name.
pub(super) fn run_job(
    env: &JobEnv<'_>,
    tasks: &Path,
    jobs_dir: &Path,
    name: &str,
) -> Result<BTreeMap<String, Attempt>, String> {
    let job = jobs_dir.join(name);
    if !finished(&job) {
        if job.exists() {
            std::fs::remove_dir_all(&job).map_err(|e| format!("{}: {e}", job.display()))?;
        }
        std::fs::create_dir_all(jobs_dir).map_err(|e| format!("{}: {e}", jobs_dir.display()))?;
        let ran = harbor_dataset(
            env.checkout,
            &DatasetRun {
                dataset: tasks,
                jobs_dir,
                job_name: name,
                profile: env.profile,
                gateway_state: env.gateway_state,
                port: env.port,
                images: env.images,
                jobs: env.jobs,
            },
            &jobs_dir.join(format!("{name}.log")),
        );
        // Harbor can exit non-zero over one trial's exception; a job it
        // finished is read, and a trial with no grade is that trial's record.
        if let Err(why) = ran {
            if !finished(&job) {
                return Err(why);
            }
        }
    }
    attempts_of(&job)
}

/// The graded trials of the Harbor job at `job`, by the staged folder each
/// attempted.
fn attempts_of(job: &Path) -> Result<BTreeMap<String, Attempt>, String> {
    let mut attempts = BTreeMap::new();
    let entries = std::fs::read_dir(job).map_err(|e| format!("{}: {e}", job.display()))?;
    for entry in entries.filter_map(Result::ok) {
        let dir = entry.path();
        let Some(folder) = task_of(&dir) else {
            continue;
        };
        let path = dir.join("result.json");
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        let result: Value =
            serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
        attempts.insert(
            folder,
            Attempt {
                reward: reward_of(&result),
                dir,
            },
        );
    }
    Ok(attempts)
}

/// The trajectory a trial's agent left (`agent/trajectory.json`, ATIF) and
/// what its model was given before the first turn, from the first request
/// it recorded (`agent/requests.jsonl`).
pub(super) fn read_attempt(dir: &Path) -> Result<(atif::Trajectory, AgentContext), String> {
    let path = dir.join("agent").join("trajectory.json");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let trajectory = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let path = dir.join("agent").join("requests.jsonl");
    let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let context = first_request(&bytes, &sha256_hex(&bytes))
        .map_err(|e| format!("{}: {e}", path.display()))?;
    Ok((trajectory, context))
}

/// The directory of trial `trial` of the student job `job`: in place when
/// the job is unpacked, else in a copy under `copies` that the examiner
/// unpacks once (`bench unpack`); the job itself is never changed.
pub(super) fn student_trial_dir(
    checkout: &BenchCheckout,
    job: &Path,
    trial: &str,
    copies: &Path,
) -> Result<PathBuf, String> {
    if !super::material::is_packed(job) {
        return Ok(job.join(trial));
    }
    let resolved = job
        .canonicalize()
        .map_err(|e| format!("{}: {e}", job.display()))?;
    let digest = sha256_hex(resolved.to_string_lossy().as_bytes());
    let copy = copies.join(&digest[..16]);
    let done = copy.join(".unpacked");
    if !done.exists() {
        if copy.exists() {
            std::fs::remove_dir_all(&copy).map_err(|e| format!("{}: {e}", copy.display()))?;
        }
        copy_tree(&resolved, &copy, &[])?;
        crate::bench::practice::unpack(
            checkout,
            &copy,
            &copies.join(format!("{}.log", &digest[..16])),
        )?;
        std::fs::write(&done, resolved.to_string_lossy().as_bytes())
            .map_err(|e| format!("{}: {e}", done.display()))?;
    }
    Ok(copy.join(trial))
}
