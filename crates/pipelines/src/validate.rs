// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agent trajectories verified by execution
// for training coding models, for its clients. If your team needs expertise
// in deciding which verified trajectories are worth training on, then you
// can procure our services by sending an email to info@swedishembedded.com.

//! The validate stage: which passing attempts at invented tasks are kept.
//!
//! The tests decide whether an attempt solved its task; the validate harness
//! (see [`splinter_agent::flywheel::Validate`]) decides whether a passing
//! attempt is worth training on, once per focus (outcome, process,
//! teaching). An attempt is kept only when every decision is keep. A
//! validator can remove a passing attempt and never admit a failing one: the
//! attempts that did not pass are left as they are.
//!
//! A validator run that gave no decision is not a discard. The attempt is
//! left out of the kept set, because nothing vouches for it, and reported
//! apart in [`Validated::undecided`] with the focus and why.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use futures::StreamExt;
use serde::Serialize;
use splinter_agent::flywheel::{conversations_text, Attempt, Decision, Validate, Verdict, FOCI};
use splinter_agent::CancelToken;
use splinter_core::experience::ExperienceId;
use splinter_eval::verifiers::workspace::{lay_over, WorkspaceCheck};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_sandbox::{Patch, ResolvedEnvironment};
use splinter_store::experiences::{ExperienceSet, SetId};

use crate::tasks::{remaining, TaskHarness};

/// How many validator runs are in flight at once.
const RUNS_IN_FLIGHT: usize = 8;

/// The directory in a laid-out tree the hidden tests are laid over at.
const TESTS_AT: &str = ".tests";

/// What the validate stage is asked.
pub struct ValidateRequest<'a> {
    /// The attempts, verified.
    pub attempts: &'a SetId,
    /// The harnesses; its validate harness is run.
    pub harness: &'a TaskHarness,
    /// No validator run starts after this, and none runs past it.
    pub deadline: Option<Instant>,
    /// Stops the stage.
    pub cancel: CancelToken,
}

/// A passing attempt a validator discarded.
#[derive(Clone, Debug, Serialize)]
pub struct Discarded {
    /// The attempt.
    pub experience: ExperienceId,
    /// The focus whose decision was discard.
    pub focus: String,
    /// Why.
    pub reasons: Vec<String>,
}

/// A validator run that gave no decision.
#[derive(Clone, Debug, Serialize)]
pub struct Undecided {
    /// The attempt.
    pub experience: ExperienceId,
    /// The focus.
    pub focus: String,
    /// Why there is no decision.
    pub why: String,
}

/// What the validate stage did.
#[derive(Clone, Debug, Serialize)]
pub struct Validated {
    /// The attempts with the passing ones that were not kept left out
    /// (`select` reads it).
    pub experience_set: SetId,
    /// Attempts that passed their tests.
    pub passing: usize,
    /// Of those, the attempts every focus decided keep.
    pub kept: usize,
    /// Passing attempts a validator discarded.
    pub discarded: Vec<Discarded>,
    /// Validator runs that gave no decision, apart from the discards.
    pub undecided: Vec<Undecided>,
    /// Why the stage stopped before every attempt was decided, if it did.
    pub stopped: Option<String>,
}

/// Validates the passing attempts of `request.attempts`.
///
/// # Errors
///
/// [`OrchestratorError`] when the store cannot be read or written, an
/// attempt's tree cannot be laid out, or the stage has no validate harness.
pub fn validate_attempts(
    ctx: &Context,
    request: &ValidateRequest<'_>,
) -> Result<Validated, OrchestratorError> {
    let Some(harness) = &request.harness.validate else {
        return Err(OrchestratorError::Refused(
            "no validate harness was named".into(),
        ));
    };
    let validate = Validate::new(
        request
            .harness
            .launch(TaskHarness::runs_root(ctx).join("validate")),
        harness.clone(),
    )
    .with_cancel(request.cancel.clone());
    let store = ctx.experiences();
    let members = store.get_set(request.attempts)?.members;
    let decisions = store.decisions(&members)?;
    let passing: Vec<ExperienceId> = members
        .iter()
        .filter(|id| decisions.get(*id).is_some_and(|d| d.passed))
        .cloned()
        .collect();

    let mut shown = Vec::new();
    for id in &passing {
        shown.push((id.clone(), shown_attempt(ctx, id)?));
    }
    let mut runs = Vec::new();
    for (index, (id, attempt)) in shown.iter().enumerate() {
        for focus in FOCI {
            runs.push((index, focus, id, attempt));
        }
    }
    let timeout = request.harness.run_timeout();
    let results: Vec<Ran> = ctx.block_on(async {
        futures::stream::iter(runs)
            .map(|(index, focus, id, attempt)| {
                let validate = &validate;
                async move {
                    let left = remaining(request.deadline, timeout);
                    if request.cancel.is_cancelled() || left.is_zero() {
                        return Ran {
                            attempt: index,
                            focus,
                            outcome: None,
                        };
                    }
                    let label = format!("{}-{focus}", id.hex());
                    let tree = attempt.workspace.as_path();
                    let decision = validate
                        .decide(
                            &Attempt {
                                task: &attempt.task,
                                workspace: tree,
                                tests: &attempt.tests,
                                trajectory: &attempt.trajectory,
                            },
                            focus,
                            &label,
                            left,
                        )
                        .await
                        .map_err(|e| e.to_string());
                    Ran {
                        attempt: index,
                        focus,
                        outcome: Some(decision),
                    }
                }
            })
            .buffer_unordered(RUNS_IN_FLIGHT)
            .collect()
            .await
    });

    let mut decided: BTreeMap<usize, Vec<Decision>> = BTreeMap::new();
    let mut discarded = Vec::new();
    let mut undecided = Vec::new();
    let mut skipped = false;
    for Ran {
        attempt: index,
        focus,
        outcome,
    } in results
    {
        let experience = shown[index].0.clone();
        match outcome {
            None => skipped = true,
            Some(Err(why)) => undecided.push(Undecided {
                experience,
                focus: focus.to_string(),
                why,
            }),
            Some(Ok(decision)) => {
                if decision.decision == Verdict::Discard {
                    discarded.push(Discarded {
                        experience,
                        focus: focus.to_string(),
                        reasons: decision.rejection_reasons.clone(),
                    });
                }
                decided.entry(index).or_default().push(decision);
            }
        }
    }
    let kept_ids: Vec<ExperienceId> = shown
        .iter()
        .enumerate()
        .filter(|(index, _)| {
            decided.get(index).is_some_and(|all| {
                all.len() == FOCI.len() && all.iter().all(|d| d.decision == Verdict::Keep)
            })
        })
        .map(|(_, (id, _))| id.clone())
        .collect();
    let experience_set = store.put_set(&ExperienceSet {
        name: format!("validated attempts of {}", request.attempts),
        members: members
            .into_iter()
            .filter(|id| !passing.contains(id) || kept_ids.contains(id))
            .collect(),
    })?;
    Ok(Validated {
        experience_set,
        passing: passing.len(),
        kept: kept_ids.len(),
        discarded,
        undecided,
        stopped: skipped.then(|| {
            "the budget was spent or the stage cancelled before every attempt was decided".into()
        }),
    })
}

/// One validator run: the attempt (by position), the focus, and what came of
/// it - `None` when the budget or a cancel kept it from starting, an error
/// when it ended without a decision.
struct Ran<'a> {
    attempt: usize,
    focus: &'a str,
    outcome: Option<Result<Decision, String>>,
}

/// What a validator is shown of one attempt, with the tree it works on.
struct Shown {
    task: String,
    tests: String,
    trajectory: String,
    /// The finished tree; removed with the attempt.
    workspace: PathBuf,
    _dir: tempfile::TempDir,
}

/// What `id`'s attempt left: its finished tree (the task's tree with the
/// attempt's change applied), what the hidden tests are and print on it, and
/// its conversation.
fn shown_attempt(ctx: &Context, id: &ExperienceId) -> Result<Shown, OrchestratorError> {
    let experience = ctx.experiences().get(id)?;
    let task = experience.to_task();
    let refused = |what: &str, e: &dyn std::fmt::Display| {
        OrchestratorError::Refused(format!("attempt {}: {what}: {e}", id.hex()))
    };
    let ResolvedEnvironment::Workspace(env) = ctx.environments().for_record(&task.environment)?
    else {
        return Err(refused(
            "laying out the tree",
            &"it is not a workspace task",
        ));
    };
    let Some((_, check)) =
        WorkspaceCheck::of_task(&task).map_err(|e| refused("reading the check", &e))?
    else {
        return Err(refused("laying out the tree", &"the task has no check"));
    };
    let tree = tempfile::Builder::new()
        .prefix("validate-tree-")
        .tempdir()
        .map_err(|e| refused("making a directory", &e))?;
    let root = tree.path().join("tree");
    let archive = ctx
        .artifacts()
        .path(&env.snapshot().digest)
        .map_err(|e| refused("finding the snapshot", &e))?;
    env.materialize(&archive, &root)
        .map_err(|e| refused("laying out the snapshot", &e))?;
    if let Some(diff) = experience
        .final_output
        .as_deref()
        .filter(|d| !d.trim().is_empty())
    {
        env.apply(
            &root,
            &Patch {
                diff: diff.as_bytes().to_vec(),
                files: Vec::new(),
            },
        )
        .map_err(|e| refused("applying the attempt's change", &e))?;
    }
    if let Some(hidden) = &check.hidden {
        let archive = ctx
            .artifacts()
            .path(hidden)
            .map_err(|e| refused("finding the hidden tests", &e))?;
        lay_over(&archive, hidden, &root).map_err(|e| refused("laying out the tests", &e))?;
    }
    let mut tests = String::new();
    let mut files = Vec::new();
    collect_files(&root.join(TESTS_AT), &root, &mut files);
    files.sort();
    for file in files {
        let text = std::fs::read_to_string(root.join(&file)).unwrap_or_default();
        tests.push_str(&format!("## {file}\n```\n{text}\n```\n\n"));
    }
    for call in &check.fail_to_pass {
        let result = env
            .run(&root, call)
            .map_err(|e| refused("running the tests", &e))?;
        tests.push_str(&format!(
            "## What the tests printed on this work (exit {:?})\n```\n{}{}\n```\n",
            result.exit_code, result.stdout, result.stderr
        ));
    }
    // The validator is shown the work, not the tests or the request twice.
    let _ = std::fs::remove_dir_all(root.join(TESTS_AT));
    let _ = std::fs::remove_file(root.join("task.md"));
    let trajectory = serde_json::to_value(&experience.trajectory)
        .map_err(|e| refused("reading the conversation", &e))?;
    Ok(Shown {
        task: experience.instruction,
        tests,
        trajectory: conversations_text(&[trajectory]),
        workspace: root,
        _dir: tree,
    })
}

/// Every file under `dir`, as a path relative to `root`.
fn collect_files(dir: &Path, root: &Path, into: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        if path.is_dir() {
            collect_files(&path, root, into);
        } else if let Ok(relative) = path.strip_prefix(root) {
            into.push(relative.to_string_lossy().into_owned());
        }
    }
}
