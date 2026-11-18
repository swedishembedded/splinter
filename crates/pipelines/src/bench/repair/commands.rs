// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the loop from a failed agent attempt to a
// verified corrected one, for its clients. If your team needs expertise in
// turning agent failures into training data you can procure our services by
// sending an email to info@swedishembedded.com.

//! The examiner's two commands a repair runs: the pristine workspace of a
//! practice task (`bench task workspace`), and the grade of a changed one by
//! the task's own verifier, outside any agent (`bench task grade`).

use std::path::Path;
use std::process::Stdio;

use serde::{Deserialize, Serialize};

use crate::bench::practice::BenchCheckout;

/// The grade record the examiner prints.
pub const GRADE_SCHEMA: &str = "bench.grade/1";

/// How the task's verifier graded a workspace.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Grade {
    /// Always [`GRADE_SCHEMA`].
    pub schema: String,
    /// The task, as its `task.toml` names it.
    pub task: String,
    /// The staged task's digest, as the practice record names it.
    pub digest: String,
    /// `pass`, `fail` or `environment_error`.
    pub verdict: String,
    /// The verifier's reward in `[0, 1]`.
    #[serde(default)]
    pub reward: Option<f64>,
    /// The verifier's metrics.
    #[serde(default)]
    pub metrics: serde_json::Value,
    /// The verifier's reason.
    #[serde(default)]
    pub reason: Option<String>,
}

impl Grade {
    /// Whether the verifier passed the work.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.verdict == "pass"
    }
}

fn run_json(
    checkout: &BenchCheckout,
    args: &[&std::ffi::OsStr],
    what: &str,
) -> Result<Vec<u8>, String> {
    let output = checkout
        .bench()
        .args(args)
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("starting the examiner to {what}: {e}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let tail: String = stderr
            .lines()
            .rev()
            .take(4)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join(" | ");
        return Err(format!(
            "the examiner could not {what} ({}): {tail}",
            output.status
        ));
    }
    Ok(output.stdout)
}

/// Lays out at `out` the pristine workspace the task `task` of the practice
/// material at `practice` gives its agent.
///
/// # Errors
/// The examiner's refusal or failure, with its words.
pub fn workspace(
    checkout: &BenchCheckout,
    practice: &Path,
    task: &str,
    out: &Path,
) -> Result<(), String> {
    run_json(
        checkout,
        &[
            "task".as_ref(),
            "workspace".as_ref(),
            practice.as_os_str(),
            "--task".as_ref(),
            task.as_ref(),
            "--out".as_ref(),
            out.as_os_str(),
            "--json".as_ref(),
        ],
        "lay out the workspace",
    )
    .map(|_| ())
}

/// Grades the workspace `tree` (made by [`workspace`], then changed) with the
/// task's own verifier, the trajectory at `trajectory` in the agent's logs.
///
/// # Errors
/// The examiner's refusal, or a verifier that wrote no reward.
pub fn grade(
    checkout: &BenchCheckout,
    practice: &Path,
    task: &str,
    tree: &Path,
    trajectory: &Path,
) -> Result<Grade, String> {
    let stdout = run_json(
        checkout,
        &[
            "task".as_ref(),
            "grade".as_ref(),
            practice.as_os_str(),
            "--task".as_ref(),
            task.as_ref(),
            "--worktree".as_ref(),
            tree.as_os_str(),
            "--trajectory".as_ref(),
            trajectory.as_os_str(),
            "--json".as_ref(),
        ],
        "grade the repaired workspace",
    )?;
    let grade: Grade = serde_json::from_slice(&stdout)
        .map_err(|e| format!("the examiner's grade is not one this reader knows: {e}"))?;
    if grade.schema != GRADE_SCHEMA {
        return Err(format!(
            "the examiner's grade is of schema {:?}, not {GRADE_SCHEMA:?}",
            grade.schema
        ));
    }
    Ok(grade)
}
