// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements session intake that grounds every learned
// fact in exact, traceable bytes, for its clients. If your team needs
// expertise in training-data provenance or secret hygiene, you can
// procure our services by sending an email to info@swedishembedded.com.

//! One recorded session of an agent with a person, captured as a source.
//!
//! The bytes are an ATIF trajectory. They are accepted when they parse, pass
//! ATIF's own validation, can be rendered by the training projection
//! ([`splinter_data::projection_refusal`] names the step that cannot) and
//! hold a user step. Every string in the trajectory then has its secrets
//! removed ([`crate::redact`]) before anything is addressed, so a stored
//! part, the origin and anything later shown to a model never hold one; the
//! origin records which step each removal was in. The parts are those of
//! [`crate::session`].
//!
//! The same file captured again is the same source: its address covers only
//! the redacted content.

use std::collections::BTreeMap;

use std::path::Path;

use atif::{validate_trajectory, ContentSegment, MessageBody, StepOrigin, TraceStep, Trajectory};
use serde_json::Value;
use splinter_core::clock::Clock;
use splinter_core::digest::canonical_json;
use splinter_core::source::{CapturedSource, Origin, PartContent, Redaction, SourceError};
use splinter_data::{projection_refusal, Unprojectable};

use super::{read_capped, CaptureError};
use crate::redact::{redact, secret_value_in, Redacted, KIND_CREDENTIAL};
use crate::session::{part_name, Role, TRAJECTORY_PART};

/// The default cap on one session file's size, in bytes: a session is a whole
/// working conversation with its tool output, far larger than a document.
pub const DEFAULT_MAX_SESSION_BYTES: u64 = 64 * 1024 * 1024;

/// Why a session is not captured.
#[derive(Debug, thiserror::Error)]
pub enum SessionRefusal {
    /// The bytes are not an ATIF trajectory.
    #[error("not an ATIF trajectory: {0}")]
    NotATrajectory(String),
    /// A trajectory that breaks ATIF's rules.
    #[error("invalid ATIF: {}", .0.join("; "))]
    Invalid(Vec<String>),
    /// The training projection cannot render it.
    #[error("the training projection refuses it: {0}")]
    Unprojectable(Unprojectable),
    /// Nothing the person said.
    #[error("it has no user step, so nothing in it was taught")]
    NoUserStep,
    /// The source could not be built.
    #[error(transparent)]
    Source(#[from] SourceError),
    /// The redacted trajectory could not be written.
    #[error("the redacted trajectory could not be written: {0}")]
    Write(serde_json::Error),
}

/// Captures the session file at `path`, refused when it is larger than
/// `max_bytes`; the error names the file.
pub fn capture_session_file(
    path: &Path,
    max_bytes: u64,
    clock: &dyn Clock,
) -> Result<CapturedSource, CaptureError> {
    let bytes = read_capped(path, max_bytes)?;
    if bytes.len() as u64 > max_bytes {
        return Err(CaptureError::TooLarge {
            path: path.to_path_buf(),
            limit: max_bytes,
        });
    }
    capture_session(&bytes, clock).map_err(|source| CaptureError::Session {
        path: path.to_path_buf(),
        source,
    })
}

/// Captures the ATIF trajectory `bytes` as a session source; see the module
/// documentation.
pub fn capture_session(bytes: &[u8], clock: &dyn Clock) -> Result<CapturedSource, SessionRefusal> {
    let raw: Trajectory =
        serde_json::from_slice(bytes).map_err(|e| SessionRefusal::NotATrajectory(e.to_string()))?;
    validate_trajectory(&raw).map_err(|errors| {
        SessionRefusal::Invalid(errors.iter().map(ToString::to_string).collect())
    })?;
    if let Some(refusal) = projection_refusal(&raw) {
        return Err(SessionRefusal::Unprojectable(refusal));
    }
    if !raw.steps.iter().any(|s| s.source == StepOrigin::User) {
        return Err(SessionRefusal::NoUserStep);
    }

    let mut value = serde_json::to_value(&raw).map_err(SessionRefusal::Write)?;
    let mut found = Found::default();
    redact_trajectory(&mut value, &mut found);
    let trajectory: Trajectory = serde_json::from_value(value)
        .map_err(|e| SessionRefusal::NotATrajectory(format!("after redaction: {e}")))?;

    let mut parts = vec![PartContent {
        name: TRAJECTORY_PART.into(),
        media_type: "application/json".into(),
        bytes: canonical_json(&trajectory).map_err(SessionRefusal::Write)?,
    }];
    for step in &trajectory.steps {
        parts.extend(step_parts(step));
    }
    let origin = Origin::Session {
        session_id: trajectory.session_id.clone(),
        agent: format!("{} {}", trajectory.agent.name, trajectory.agent.version),
        redactions: found.into_redactions(),
    };
    Ok(CapturedSource::new(origin, parts, clock)?)
}

/// The parts of one step: its text, its calls, its observation.
fn step_parts(step: &TraceStep) -> Vec<PartContent> {
    let mut parts = Vec::new();
    let mut push = |role: Role, text: String| {
        if !text.is_empty() {
            parts.push(PartContent {
                name: part_name(step.step_id, role),
                media_type: "text/plain".into(),
                bytes: text.into_bytes(),
            });
        }
    };
    match step.source {
        StepOrigin::System => {}
        StepOrigin::User => push(Role::User, text_of(&step.message)),
        StepOrigin::Agent => {
            push(Role::Agent, text_of(&step.message));
            let calls: Vec<String> = step
                .tool_calls
                .iter()
                .flatten()
                .map(|c| format!("{} {}", c.function_name, c.arguments))
                .collect();
            push(Role::Calls, calls.join("\n"));
            let observed: Vec<String> = step
                .observation
                .iter()
                .flat_map(|o| &o.results)
                .filter_map(|r| r.content.as_ref().map(text_of))
                .collect();
            push(Role::Observation, observed.join("\n"));
        }
    }
    parts
}

/// The text of `body`; images carry none.
fn text_of(body: &MessageBody) -> String {
    match body {
        MessageBody::Text(text) => text.clone(),
        MessageBody::Segments(segments) => segments
            .iter()
            .filter_map(|s| match s {
                ContentSegment::Text { text } => Some(text.as_str()),
                ContentSegment::Image { .. } => None,
            })
            .collect(),
    }
}

/// What redaction removed, by step and kind.
#[derive(Default)]
struct Found(BTreeMap<(Option<u64>, String), u32>);

impl Found {
    fn add(&mut self, step: Option<u64>, kind: &str, count: u32) {
        *self.0.entry((step, kind.to_string())).or_default() += count;
    }

    fn into_redactions(self) -> Vec<Redaction> {
        self.0
            .into_iter()
            .map(|((step, kind), count)| Redaction { step, kind, count })
            .collect()
    }
}

/// Redacts every string of the trajectory `value`, remembering which step it
/// was in.
fn redact_trajectory(value: &mut Value, found: &mut Found) {
    let Value::Object(root) = value else {
        return;
    };
    for (key, child) in root.iter_mut() {
        match (key.as_str(), child) {
            ("steps", Value::Array(steps)) => {
                for step in steps {
                    let id = step.get("step_id").and_then(Value::as_u64);
                    redact_value(step, id, found);
                }
            }
            (_, child) => redact_value(child, None, found),
        }
    }
}

fn redact_value(value: &mut Value, step: Option<u64>, found: &mut Found) {
    match value {
        Value::String(text) => {
            let Redacted {
                text: clean,
                found: removed,
            } = redact(text);
            if !removed.is_empty() {
                *text = clean;
                for (kind, count) in removed {
                    found.add(step, kind, count);
                }
            }
        }
        Value::Array(items) => items.iter_mut().for_each(|v| redact_value(v, step, found)),
        Value::Object(map) => {
            for (key, child) in map.iter_mut() {
                match child {
                    Value::String(text) if secret_value_in(key, text) => {
                        *child = Value::String(crate::redact::marker(KIND_CREDENTIAL));
                        found.add(step, KIND_CREDENTIAL, 1);
                    }
                    child => redact_value(child, step, found),
                }
            }
        }
        _ => {}
    }
}
