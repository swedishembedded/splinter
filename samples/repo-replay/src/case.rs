// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The hidden side of one attempt, and the judging that reads it.
//!
//! A case directory holds what the agent must not reach: the commit's tree,
//! the diff, the parent's vocabulary, the gate's bounds. The host writes it
//! outside the agent's tree before the attempt; the harness document's gates
//! (`repo-replay check`, `repo-replay check-message`) read it through
//! `cwd_input`, and the host reads it again to judge the finished attempt.
//! One implementation judges in all three places.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};
use splinter_sdk::agent::sven::atif::{StepOrigin, Trajectory};

use crate::gate::{evaluate, GateConfig, GateInput, Verdict};
use crate::generated::is_generated_path;
use crate::git::{Change, Git};
use crate::supervise::{check_message, MessageContext};
use crate::trace::Trace;
use crate::tree::{compare, read_tree, Residual};

/// Gate classes a correction can repair: a wrong or incomplete tree.
pub const CORRECTABLE: &[&str] = &["tree_mismatch", "junk", "missing_edit"];

/// Longest agent diff shown to the supervisor, in characters.
pub const MAX_AGENT_CHANGES: usize = 24_000;

/// Heading of the agent's changes in what `check` prints for the supervisor.
pub const AGENT_CHANGES_HEADING: &str = "What the engineer changed:\n";
/// Heading of the differences in what `check` prints for the supervisor.
pub const DIFFERENCES_HEADING: &str = "Where the engineer's result differs from it:\n";

const FILE: &str = "case.json";
const TARGET: &str = "tree";

/// Exit status of `check` when the attempt fails in a way a correction can repair.
pub const EXIT_CORRECTABLE: i32 = 1;
/// Exit status of `check` when the attempt fails and a correction cannot help.
pub const EXIT_FAILED: i32 = 2;

/// What the judging of one attempt needs besides its tree and trajectory.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Case {
    /// The task the agent was given.
    pub plan: String,
    /// The commit's diff, unmasked.
    pub diff: String,
    /// The commit's changes, generated paths removed.
    pub changes: Vec<Change>,
    /// Identifier-shaped words the parent tree contains.
    pub parent_words: Vec<String>,
    /// Paths of the parent tree.
    pub parent_paths: Vec<String>,
    /// Paths of the source, mirror and state the shell must not name.
    pub forbidden_locations: Vec<String>,
    /// The attempt tree's path, spelled every way the agent was told it.
    pub attempt_roots: Vec<String>,
    /// The gate's bounds.
    pub gate: GateConfig,
    /// Tool calls of the first round.
    pub calls: usize,
    /// Tool calls each correction round adds.
    pub per_correction: usize,
}

impl Case {
    /// Writes the case into `dir`; the commit's tree is laid out beside it by the caller at [`Self::target_dir`].
    pub fn write(&self, dir: &Path) -> Result<()> {
        std::fs::write(dir.join(FILE), serde_json::to_vec(self)?)
            .with_context(|| format!("writing the case in {}", dir.display()))
    }

    /// Reads the case in `dir`.
    pub fn read(dir: &Path) -> Result<Self> {
        let bytes = std::fs::read(dir.join(FILE))
            .with_context(|| format!("reading the case in {}", dir.display()))?;
        serde_json::from_slice(&bytes).context("the case file is malformed")
    }

    /// Where the commit's tree lies in the case directory `dir`.
    #[must_use]
    pub fn target_dir(dir: &Path) -> PathBuf {
        dir.join(TARGET)
    }

    /// Where the tree in `workspace` differs from the commit's.
    pub fn residual(&self, dir: &Path, workspace: &Path) -> Result<Vec<Residual>> {
        let target = read_tree(&Self::target_dir(dir))?;
        let parent: HashSet<&str> = self.parent_paths.iter().map(String::as_str).collect();
        let mut residual = compare(&target, &read_tree(workspace)?, &|p| is_generated_path(p));
        for r in &mut residual {
            if let Residual::Extra { path } = r {
                if parent.contains(path.as_str()) {
                    *r = Residual::Stale {
                        path: std::mem::take(path),
                    };
                }
            }
        }
        Ok(residual)
    }

    /// The gate's decision on an attempt. `guidance` is each correction sent
    /// with the calls made before it.
    #[must_use]
    pub fn judge(
        &self,
        trace: &Trace,
        residual: &[Residual],
        stopped: Option<&str>,
        guidance: &[(usize, String)],
    ) -> Verdict {
        let parent_words: HashSet<String> = self.parent_words.iter().cloned().collect();
        let added_text = crate::plan::changed_lines(&self.diff).join("\n");
        evaluate(
            &GateInput {
                changes: &self.changes,
                plan: &self.plan,
                parent_words: &parent_words,
                diff: &self.diff,
                added_text: &added_text,
                trace,
                residual,
                forbidden_locations: &self.forbidden_locations,
                attempt_roots: &self.attempt_roots,
                stopped,
                guidance,
            },
            &GateConfig {
                max_tool_calls: self.calls + guidance.len() * self.per_correction,
                ..self.gate
            },
        )
    }

    /// Whether a failed attempt can be repaired by a correction.
    #[must_use]
    pub fn correctable(verdict: &Verdict, residual: &[Residual]) -> bool {
        !residual.is_empty()
            && verdict
                .reasons
                .iter()
                .all(|r| CORRECTABLE.contains(&r.class.as_str()))
    }
}

/// Why a turn that ended with the stop status `stop` was not a finished
/// attempt; `None` for a turn that concluded by itself.
#[must_use]
pub fn stopped_reason(stop: &str) -> Option<String> {
    match stop {
        "concluded" | "" => None,
        "tool_cap" => Some("the tool-call cap stopped the run".to_string()),
        "repetition" => Some("a call kept returning the same answer".to_string()),
        "output_truncated" => Some("a reply was cut off at the output limit".to_string()),
        "stopped_without_conclusion" => {
            Some("the run ended as StoppedWithoutConclusion".to_string())
        }
        other => Some(format!("the run ended as {other}")),
    }
}

/// The corrections in `trajectory` (user turns the supervisor wrote), each with
/// the number of tool calls made before it.
#[must_use]
pub fn guidance_of(trajectory: &Trajectory) -> Vec<(usize, String)> {
    let mut calls = 0;
    let mut out = Vec::new();
    for step in &trajectory.steps {
        let origin = step
            .extra
            .as_ref()
            .and_then(|e| e.pointer("/sven/origin"))
            .and_then(|o| o.as_str());
        if step.source == StepOrigin::User && origin == Some("correction") {
            let text = match &step.message {
                splinter_sdk::agent::sven::atif::MessageBody::Text(t) => t.clone(),
                splinter_sdk::agent::sven::atif::MessageBody::Segments(_) => String::new(),
            };
            out.push((calls, text));
        }
        calls += step.tool_calls.iter().flatten().count();
    }
    out
}

/// For each user turn of `trajectory` in order, whether it is a correction.
#[must_use]
pub fn user_turns_are_corrections(trajectory: &Trajectory) -> Vec<bool> {
    trajectory
        .steps
        .iter()
        .filter(|s| s.source == StepOrigin::User)
        .map(|s| {
            s.extra
                .as_ref()
                .and_then(|e| e.pointer("/sven/origin"))
                .and_then(|o| o.as_str())
                == Some("correction")
        })
        .collect()
}

/// One line telling the supervisor how a path of the agent's tree differs.
#[must_use]
pub fn describe(residual: &Residual) -> String {
    match residual {
        Residual::Missing { path } => format!("{path}: the engineer's tree lacks this file"),
        Residual::Extra { path } => {
            format!("{path}: exists, and the wanted change has no such file")
        }
        Residual::Stale { path } => {
            format!("{path}: still present, and the wanted change removes it")
        }
        Residual::Differs {
            path,
            first_difference,
        } => format!("{path}: first difference: {first_difference}"),
    }
}

fn read_trajectory(file: &Path) -> Result<Trajectory> {
    serde_json::from_slice(
        &std::fs::read(file).with_context(|| format!("reading {}", file.display()))?,
    )
    .context("the trajectory file is malformed")
}

/// The harness document's check gate. Returns the exit status (0 passed,
/// [`EXIT_CORRECTABLE`], [`EXIT_FAILED`]) and what to print: for a failure,
/// the evidence the supervisor is shown (the agent's changes and where its
/// tree differs).
pub fn check(dir: &Path, workspace: &Path, trajectory: &Path, stop: &str) -> Result<(i32, String)> {
    let case = Case::read(dir)?;
    let trajectory = read_trajectory(trajectory)?;
    let trace = Trace::of(&trajectory);
    let residual = case.residual(dir, workspace)?;
    let stopped = stopped_reason(stop);
    let verdict = case.judge(
        &trace,
        &residual,
        stopped.as_deref(),
        &guidance_of(&trajectory),
    );
    if verdict.passed {
        return Ok((0, String::new()));
    }
    let changes = Git::at(workspace).working_diff().unwrap_or_default();
    let changes: String = changes.chars().take(MAX_AGENT_CHANGES).collect();
    let differences: Vec<String> = residual.iter().map(describe).collect();
    let status = if Case::correctable(&verdict, &residual) {
        EXIT_CORRECTABLE
    } else {
        EXIT_FAILED
    };
    Ok((
        status,
        format!(
            "{AGENT_CHANGES_HEADING}{changes}\n\n{DIFFERENCES_HEADING}{}",
            differences.join("\n")
        ),
    ))
}

/// The harness document's accept gate for a supervisor's message. Returns
/// whether it may be sent and, if not, why.
pub fn check_correction(dir: &Path, trajectory: &Path, message: &str) -> Result<(bool, String)> {
    let case = Case::read(dir)?;
    let message = message.trim();
    if message.is_empty() {
        return Ok((false, "it is empty".into()));
    }
    let trajectory = read_trajectory(trajectory)?;
    let trace = Trace::of(&trajectory);
    let support =
        crate::justify::support_of_run(&case.plan, &trace.calls, &guidance_of(&trajectory));
    let parent_words: HashSet<String> = case.parent_words.iter().cloned().collect();
    let ctx = MessageContext::new(&case.plan, &case.diff, &parent_words, &support);
    Ok(match check_message(message, &ctx) {
        Ok(()) => (true, String::new()),
        Err(why) => (false, why.to_string()),
    })
}

/// The class a refused supervisor message is counted under, by the rules as
/// they judge it against `trajectory`.
#[must_use]
pub fn refusal_class(dir: &Path, trajectory: &Trajectory, message: &str) -> Option<String> {
    let case = Case::read(dir).ok()?;
    if message.trim().is_empty() {
        return Some("empty".into());
    }
    let trace = Trace::of(trajectory);
    let support =
        crate::justify::support_of_run(&case.plan, &trace.calls, &guidance_of(trajectory));
    let parent_words: HashSet<String> = case.parent_words.iter().cloned().collect();
    let ctx = MessageContext::new(&case.plan, &case.diff, &parent_words, &support);
    check_message(message.trim(), &ctx)
        .err()
        .map(|why| why.class().to_string())
}
