// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements anchor suites that catch a model losing
// general behaviour while it learns something specific, for its clients.
// If your team needs expertise in regression testing for model releases,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! The anchor suite: a frozen, versioned set of general tasks every release
//! is held to, so learning one thing does not cost general behaviour.
//!
//! A version is a document in the experience database, named by its digest,
//! and the pointer `anchor` names the version in force, so which version was in
//! force when is on record. A version is frozen from a file ([`freeze`]) and
//! never changes; freezing other tasks makes the next version, and freezing the
//! tasks already in force changes nothing. A release records the version and digest it was
//! held to.
//!
//! The file is JSON Lines, one task per line: `{"instruction": ...,
//! "reference": ..., "kind": ...}`, where `kind` (default `recall`) is a
//! task kind solved closed-book whose verifiers include the formal one, so
//! every anchor task can be graded without a judge.

use std::path::Path;

use serde::{Deserialize, Serialize};
use splinter_core::digest::Digest;
use splinter_core::experience::{Environment, Privileged, PrivilegedKind, Task};
use splinter_knowledge::tasks::{Catalogue, VerifierKind};

use crate::context::Context;
use crate::error::{io, CampaignError};
use crate::release::probe::Suite;

/// The most tasks an anchor file may hold.
pub const MAX_ANCHOR_TASKS: usize = 10_000;
/// The kind an anchor task is when its line names none.
pub const DEFAULT_ANCHOR_KIND: &str = "recall";
/// The `format` every frozen version carries.
pub const ANCHOR_FORMAT: &str = "splinter-anchor-v1";

/// One frozen version of the anchor suite.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnchorSuite {
    /// Always [`ANCHOR_FORMAT`].
    pub format: String,
    /// Its version: one more than the version it replaced.
    pub version: u32,
    /// Its tasks.
    pub tasks: Vec<Task>,
}

/// A frozen version and its digest.
#[derive(Clone, Debug, PartialEq)]
pub struct FrozenAnchor {
    /// The digest of its canonical JSON.
    pub digest: Digest,
    /// The version.
    pub suite: AnchorSuite,
}

impl FrozenAnchor {
    /// Its tasks as a probe suite.
    #[must_use]
    pub fn probe_suite(&self) -> Suite {
        Suite::of_tasks(
            format!("anchor v{}", self.suite.version),
            self.suite.tasks.clone(),
        )
    }
}

/// One line of an anchor file.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AnchorLine {
    instruction: String,
    reference: String,
    #[serde(default)]
    kind: Option<String>,
}

const VERSION: &str = "anchor_version";
const POINTER: &str = "anchor";

/// The anchor suite version in force, verified; `None` before one is
/// frozen.
pub fn current(ctx: &Context) -> Result<Option<FrozenAnchor>, CampaignError> {
    ctx.workspace().refresh()?;
    let Some((_, value)) = ctx.workspace().pointer(POINTER)? else {
        return Ok(None);
    };
    let digest = Digest::parse(&value)
        .map_err(|e| CampaignError::Refused(format!("the anchor pointer is corrupt: {e}")))?;
    let suite = ctx
        .workspace()
        .get_document(VERSION, &digest)?
        .ok_or_else(|| CampaignError::NotFound {
            what: "anchor suite version",
            id: digest.to_string(),
        })?;
    Ok(Some(FrozenAnchor { digest, suite }))
}

/// Reads anchor tasks from `file` (see the module documentation), refusing
/// the whole file on its first bad line.
pub fn read_tasks(file: &Path) -> Result<Vec<Task>, CampaignError> {
    let text = std::fs::read_to_string(file).map_err(io(file))?;
    let catalogue = Catalogue::builtin();
    let mut tasks = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let refuse = |why: String| {
            CampaignError::Refused(format!("{} line {}: {why}", file.display(), index + 1))
        };
        let parsed: AnchorLine = serde_json::from_str(line).map_err(|e| {
            refuse(format!(
                "{e}; a line is {{\"instruction\", \"reference\", \"kind\"?}}"
            ))
        })?;
        if parsed.instruction.trim().is_empty() || parsed.reference.trim().is_empty() {
            return Err(refuse(
                "the instruction and the reference must not be empty".into(),
            ));
        }
        let kind = parsed.kind.unwrap_or_else(|| DEFAULT_ANCHOR_KIND.into());
        let gradable = catalogue.get(&kind).is_some_and(|spec| {
            spec.environment == splinter_knowledge::tasks::SolverEnvironment::ClosedBook
                && (spec.verifiers.contains(&VerifierKind::Formal)
                    || spec.verifiers.contains(&VerifierKind::Stated))
        });
        if !gradable {
            return Err(refuse(format!(
                "kind {kind:?} is not a closed-book kind a formal verifier grades"
            )));
        }
        if tasks.len() == MAX_ANCHOR_TASKS {
            return Err(refuse(format!("more than {MAX_ANCHOR_TASKS} tasks")));
        }
        let reference = Privileged {
            kind: PrivilegedKind::Reference,
            content: parsed.reference,
            span: None,
        };
        tasks.push(Task::new(
            kind,
            Vec::new(),
            Environment::closed_book(),
            parsed.instruction,
            vec![reference],
        )?);
    }
    if tasks.is_empty() {
        return Err(CampaignError::Refused(format!(
            "{} holds no task",
            file.display()
        )));
    }
    Ok(tasks)
}

/// Freezes the tasks in `file` as the anchor suite's next version and puts
/// it in force; the version in force is kept when it holds the same tasks.
/// Refused if another process froze one in the meantime.
pub fn freeze(ctx: &Context, file: &Path) -> Result<FrozenAnchor, CampaignError> {
    let tasks = read_tasks(file)?;
    let previous = current(ctx)?;
    if let Some(previous) = previous.as_ref().filter(|p| p.suite.tasks == tasks) {
        return Ok(previous.clone());
    }
    let version = previous.as_ref().map_or(1, |p| p.suite.version + 1);
    let suite = AnchorSuite {
        format: ANCHOR_FORMAT.into(),
        version,
        tasks,
    };
    let digest = ctx.workspace().put_document(VERSION, &suite)?;
    ctx.workspace().move_pointer(
        POINTER,
        previous.as_ref().map(|p| p.digest.as_str()),
        digest.as_str(),
        &ctx.clock().utc_now(),
    )?;
    Ok(FrozenAnchor { digest, suite })
}
