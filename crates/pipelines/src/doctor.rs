// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning pipelines that tell you what is
// wrong before an expensive run starts, for its clients. If your team needs
// expertise in making long-running ML jobs fail early and clearly, you can
// procure our services by sending an email to info@swedishembedded.com.

//! `doctor`: whether a learning run can start here, answered without
//! loading a model, touching the network or changing what the state root
//! holds. Each check is cheap and independent; a failed one says what to do.

use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;
use splinter_core::role::{Role, RoleOverrides};
use splinter_model::ModelSelection;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::roles;
use splinter_store::space::available_bytes;

/// Free space below which a run cannot be expected to finish: a checkpoint
/// copy, a dataset and an adapter do not fit.
pub const FAIL_BELOW_FREE_BYTES: u64 = 1 << 30;

/// Free space below which a run may run out part-way through. An estimate of
/// what a run on a small base needs, not a bound.
pub const WARN_BELOW_FREE_BYTES: u64 = 10 << 30;

/// How a check came out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// Nothing to do.
    Ok,
    /// A run can start, and may meet trouble.
    Warn,
    /// A run cannot start.
    Fail,
}

/// One thing checked.
#[derive(Clone, Debug, Serialize)]
pub struct Check {
    /// What was checked: `state-writable`, `free-space`, `role:teacher`.
    pub name: String,
    /// How it came out.
    pub verdict: Verdict,
    /// What was found.
    pub detail: String,
    /// What to do about it, when it is not `ok`.
    pub fix: Option<String>,
}

impl Check {
    fn ok(name: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            verdict: Verdict::Ok,
            detail: detail.into(),
            fix: None,
        }
    }

    fn problem(
        verdict: Verdict,
        name: impl Into<String>,
        detail: impl Into<String>,
        fix: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            verdict,
            detail: detail.into(),
            fix: Some(fix.into()),
        }
    }
}

/// What `doctor` reports.
#[derive(Clone, Debug, Serialize)]
pub struct Doctor {
    /// Every check, in the order made.
    pub checks: Vec<Check>,
}

impl Doctor {
    /// Whether a run can start: no check failed. Warnings do not stop one.
    #[must_use]
    pub fn ready(&self) -> bool {
        self.checks.iter().all(|c| c.verdict != Verdict::Fail)
    }
}

/// Checks that a run can start in the state root and with the models `ctx`
/// is configured with.
pub fn doctor(ctx: &Context) -> Doctor {
    let root = ctx.root().path();
    // A state root not yet created is where the first run will create it:
    // what matters is the nearest directory that exists, and nothing is
    // created here.
    let anchor = nearest_existing(root);
    let mut checks = vec![state_writable(root, anchor)];
    // Free space needs a directory that exists; without a writable one the
    // first check has already said why.
    if checks[0].verdict == Verdict::Ok {
        checks.push(free_space(anchor));
    }
    checks.extend(roles_resolve(ctx));
    Doctor { checks }
}

/// `path`, else its nearest ancestor that exists.
fn nearest_existing(path: &Path) -> &Path {
    path.ancestors()
        .find(|ancestor| ancestor.exists())
        .unwrap_or(path)
}

fn state_writable(root: &Path, anchor: &Path) -> Check {
    let name = "state-writable";
    let probe = anchor.join(format!(".doctor-{}", std::process::id()));
    let outcome = fs::write(&probe, b"probe").and_then(|()| fs::remove_file(&probe));
    match outcome {
        Ok(()) => Check::ok(name, format!("{} accepts writes", anchor.display())),
        Err(e) => Check::problem(
            Verdict::Fail,
            name,
            format!(
                "cannot write under {} (for the state root {}): {e}",
                anchor.display(),
                root.display()
            ),
            "choose a writable directory with --state or SPLINTER_STATE",
        ),
    }
}

fn free_space(root: &Path) -> Check {
    let name = "free-space";
    let free = match available_bytes(root) {
        Ok(free) => free,
        Err(e) => {
            return Check::problem(
                Verdict::Warn,
                name,
                format!(
                    "cannot measure the free space under {}: {e}",
                    root.display()
                ),
                "make sure the state root's filesystem is mounted and readable",
            )
        }
    };
    let detail = format!("{} free under {}", gib(free), root.display());
    let fix = "free space or point --state at a larger filesystem";
    if free < FAIL_BELOW_FREE_BYTES {
        Check::problem(Verdict::Fail, name, detail, fix)
    } else if free < WARN_BELOW_FREE_BYTES {
        Check::problem(Verdict::Warn, name, detail, fix)
    } else {
        Check::ok(name, detail)
    }
}

fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / (1u64 << 30) as f64)
}

/// The first local weights file or directory `selection` needs that is not
/// there. A selection only names paths; whether they exist is for this check.
fn missing_weights(selection: &ModelSelection) -> Option<PathBuf> {
    let ModelSelection::Local(weights) = selection else {
        return None;
    };
    std::iter::once(&weights.base)
        .chain(weights.adapter.as_ref())
        .find(|path| !path.exists())
        .cloned()
}

/// Whether the model each role would get resolves to weights or an endpoint
/// that exists, without loading any of them.
fn roles_resolve(ctx: &Context) -> Vec<Check> {
    let assignments = match roles::assignments(ctx.config(), &RoleOverrides::new()) {
        Ok(assignments) => assignments,
        Err(e) => {
            return vec![Check::problem(
                Verdict::Fail,
                "roles",
                e.to_string(),
                "fix the model settings named in the message",
            )]
        }
    };
    Role::ALL
        .into_iter()
        .map(|role| {
            let name = format!("role:{role}");
            let reference = assignments.get(role);
            match ctx.selection(reference) {
                Ok(selection) => match missing_weights(&selection) {
                    None => Check::ok(name, format!("{reference} is {}", selection.identity())),
                    Some(path) => Check::problem(
                        Verdict::Fail,
                        name,
                        format!("{reference} needs {}, which does not exist", path.display()),
                        "fetch the base checkpoint, or set the model store (see `splinter status`)",
                    ),
                },
                Err(e) => Check::problem(
                    Verdict::Fail,
                    name,
                    format!("{reference} does not resolve: {e}"),
                    "point the role at a model that exists (see `splinter status`)",
                ),
            }
        })
        .collect()
}
