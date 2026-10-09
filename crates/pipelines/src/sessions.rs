// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning from the sessions a person holds
// with an agent, for its clients. If your team needs expertise in
// continual learning from user feedback or secret hygiene for training data,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! The intake stage: recorded sessions become session sources.
//!
//! A path is an ATIF file, or a directory searched (recursively, in path
//! order) for files named `*.atif.json`. Each file is captured by
//! [`splinter_knowledge::capture::capture_session`]: validated, refused with
//! the reason when the training projection cannot render it, stripped of
//! secrets, and stored once by content - the same file again is the same
//! source. A refused file does not stop the others; the report names it and
//! why. Nothing but the redacted session is stored, and nothing else is
//! ever shown to a model.

use std::path::{Path, PathBuf};

use serde::Serialize;
use splinter_core::source::{Origin, Redaction, SourceId};
use splinter_knowledge::capture::capture_session_file;
pub use splinter_knowledge::capture::DEFAULT_MAX_SESSION_BYTES;
use splinter_knowledge::session::SessionView;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// What a directory's files are named to be taken as sessions.
pub const SESSION_SUFFIX: &str = ".atif.json";

/// One intake request.
pub struct IntakeRequest<'a> {
    /// ATIF files, or directories of them.
    pub paths: &'a [PathBuf],
    /// The largest session file taken, in bytes.
    pub max_bytes: u64,
}

/// A session taken in.
#[derive(Clone, Debug, Serialize)]
pub struct SessionTaken {
    /// The file it came from.
    pub path: String,
    /// The session source.
    pub source: SourceId,
    /// Whether it was new to the store.
    pub new: bool,
    /// The steps with text in it.
    pub steps: usize,
    /// What was removed from it before it was stored.
    pub redactions: Vec<Redaction>,
}

/// A file that was not taken, and why.
#[derive(Clone, Debug, Serialize)]
pub struct SessionRefused {
    /// The file.
    pub path: String,
    /// Why.
    pub reason: String,
}

/// What the intake stage reports.
#[derive(Clone, Debug, Serialize)]
pub struct SessionsIntake {
    /// The sessions taken in, in file order.
    pub sessions: Vec<SessionTaken>,
    /// The files refused, with the reason each.
    pub refused: Vec<SessionRefused>,
    /// Sessions new to the store.
    pub new: usize,
    /// Secrets removed across all of them.
    pub redactions: u32,
}

/// Takes the sessions `request.paths` name into the source store.
pub fn intake(
    ctx: &Context,
    request: &IntakeRequest<'_>,
) -> Result<SessionsIntake, OrchestratorError> {
    let files = session_files(request.paths)?;
    let store = ctx.sources();
    let mut report = SessionsIntake {
        sessions: Vec::new(),
        refused: Vec::new(),
        new: 0,
        redactions: 0,
    };
    for file in files {
        let captured = match capture_session_file(&file, request.max_bytes, ctx.clock()) {
            Ok(captured) => captured,
            Err(e) => {
                report.refused.push(SessionRefused {
                    path: file.display().to_string(),
                    reason: reason_of(&e),
                });
                continue;
            }
        };
        let id = captured.source().id.clone();
        let new = !store.contains(&id)?;
        store.put_source(&captured)?;
        let steps = SessionView::of(&captured)
            .map_err(|e| OrchestratorError::Refused(format!("{}: {e}", file.display())))?
            .steps()
            .count();
        let redactions = match &captured.source().origin {
            Origin::Session { redactions, .. } => redactions.clone(),
            _ => Vec::new(),
        };
        report.new += usize::from(new);
        report.redactions += redactions.iter().map(|r| r.count).sum::<u32>();
        report.sessions.push(SessionTaken {
            path: file.display().to_string(),
            source: id,
            new,
            steps,
            redactions,
        });
    }
    Ok(report)
}

/// The refusal as a sentence without the file's name, which the report
/// carries beside it.
fn reason_of(error: &splinter_knowledge::capture::CaptureError) -> String {
    use splinter_knowledge::capture::CaptureError;
    match error {
        CaptureError::Session { source, .. } => source.to_string(),
        other => other.to_string(),
    }
}

/// The files `paths` name: a file as given, a directory's `*.atif.json`
/// files at any depth, in path order.
fn session_files(paths: &[PathBuf]) -> Result<Vec<PathBuf>, OrchestratorError> {
    let mut files = Vec::new();
    for path in paths {
        let metadata = std::fs::metadata(path).map_err(splinter_orchestrator::error::io(path))?;
        if metadata.is_dir() {
            let before = files.len();
            collect(path, &mut files)?;
            if files.len() == before {
                return Err(OrchestratorError::Refused(format!(
                    "{} holds no {SESSION_SUFFIX} file",
                    path.display()
                )));
            }
        } else {
            files.push(path.clone());
        }
    }
    Ok(files)
}

fn collect(dir: &Path, files: &mut Vec<PathBuf>) -> Result<(), OrchestratorError> {
    let io = splinter_orchestrator::error::io;
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(io(dir))?
        .map(|e| e.map(|e| e.path()))
        .collect::<Result<_, _>>()
        .map_err(io(dir))?;
    entries.sort();
    for entry in entries {
        if entry.is_dir() {
            collect(&entry, files)?;
        } else if entry
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.to_ascii_lowercase().ends_with(SESSION_SUFFIX))
        {
            files.push(entry);
        }
    }
    Ok(())
}

/// The stored session sources `given` name, by id or unique prefix; a
/// source that is not a session is refused.
pub fn resolve(ctx: &Context, given: &[String]) -> Result<Vec<SourceId>, OrchestratorError> {
    let store = ctx.sources();
    given
        .iter()
        .map(|id| {
            let source = crate::sources::resolve(ctx, id)?;
            let stored = store.get_source(&source)?;
            if matches!(stored.origin, Origin::Session { .. }) {
                Ok(source)
            } else {
                Err(OrchestratorError::Refused(format!(
                    "source {source} is a {}, not a session (add sessions with `session add`)",
                    stored.kind()
                )))
            }
        })
        .collect()
}

/// Every stored session source.
pub fn list(ctx: &Context) -> Result<Vec<SourceId>, OrchestratorError> {
    let store = ctx.sources();
    let mut sessions = Vec::new();
    for id in store.list()? {
        if matches!(store.get_source(&id)?.origin, Origin::Session { .. }) {
            sessions.push(id);
        }
    }
    Ok(sessions)
}
