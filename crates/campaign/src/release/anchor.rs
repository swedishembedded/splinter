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
//! ```text
//! <root>/suites/anchor/<hex>.json   one version, canonical JSON; <hex> is its digest
//! <root>/suites/anchor/current      `sha256:<hex>` of the version in force
//! ```
//!
//! A version is frozen from a file ([`freeze`]) and never changes; freezing
//! other tasks makes the next version, and freezing the tasks already in
//! force changes nothing. A release records the version and digest it was
//! held to.
//!
//! The file is JSON Lines, one task per line: `{"instruction": ...,
//! "reference": ..., "kind": ...}`, where `kind` (default `recall`) is a
//! task kind solved closed-book whose verifiers include the formal one, so
//! every anchor task can be graded without a judge.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use splinter_knowledge::tasks::{Catalogue, VerifierKind};
use splinter_store::digest::{canonical_json, Digest};
use splinter_store::experience::{Environment, Privileged, PrivilegedKind, Task};
use splinter_store::{write_atomic, write_once};

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

fn dir(ctx: &Context) -> PathBuf {
    ctx.root().suites().join("anchor")
}

/// The anchor suite version in force, verified; `None` before one is
/// frozen.
pub fn current(ctx: &Context) -> Result<Option<FrozenAnchor>, CampaignError> {
    let pointer = dir(ctx).join("current");
    let text = match std::fs::read_to_string(&pointer) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(io(&pointer)(e)),
    };
    let digest = Digest::parse(text.trim())
        .map_err(|e| CampaignError::Refused(format!("{} is corrupt: {e}", pointer.display())))?;
    let path = dir(ctx).join(format!("{}.json", digest.hex()));
    let bytes = std::fs::read(&path).map_err(io(&path))?;
    let found = Digest::of(&bytes);
    if found != digest {
        return Err(CampaignError::Store(
            splinter_store::experiences::StoreError::Corrupt {
                path,
                expected: digest,
                found,
            },
        ));
    }
    let suite = serde_json::from_slice(&bytes).map_err(|source| CampaignError::Json {
        what: path.display().to_string(),
        source,
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
pub fn freeze(ctx: &Context, file: &Path) -> Result<FrozenAnchor, CampaignError> {
    let tasks = read_tasks(file)?;
    let previous = current(ctx)?;
    if let Some(previous) = previous.filter(|p| p.suite.tasks == tasks) {
        return Ok(previous);
    }
    let version = current(ctx)?.map_or(1, |p| p.suite.version + 1);
    let suite = AnchorSuite {
        format: ANCHOR_FORMAT.into(),
        version,
        tasks,
    };
    let bytes = canonical_json(&suite).map_err(|source| CampaignError::Json {
        what: "anchor suite".into(),
        source,
    })?;
    let digest = Digest::of(&bytes);
    let path = dir(ctx).join(format!("{}.json", digest.hex()));
    write_once(&path, &bytes).map_err(io(&path))?;
    let pointer = dir(ctx).join("current");
    write_atomic(&pointer, &format!("{digest}\n")).map_err(io(&pointer))?;
    Ok(FrozenAnchor { digest, suite })
}
