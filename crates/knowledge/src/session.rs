// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements session intake that grounds every learned
// fact in exact, traceable bytes, for its clients. If your team needs
// expertise in training-data provenance or secret hygiene, you can
// procure our services by sending an email to info@swedishembedded.com.

//! A captured session as text addressable by step.
//!
//! A session source ([`splinter_core::source::Origin::Session`]) has one part
//! for each piece of text a step carries, named by [`part_name`]:
//!
//! * `step-NNNNNN-user` - what the person said;
//! * `step-NNNNNN-agent` - what the agent said;
//! * `step-NNNNNN-calls` - the agent's tool calls, one `name arguments` line
//!   each (arguments as compact JSON);
//! * `step-NNNNNN-observation` - what the tools returned;
//!
//! and `session.atif.json`, the trajectory with its secrets removed, from
//! which the training projection is made. System steps carry no part. A
//! claim cites a step by its ATIF `step_id`; [`SessionView`] resolves it to
//! the part, its content address and its text, and so to a
//! [`splinter_core::experience::Span`] of exact bytes.

use std::collections::BTreeMap;

use splinter_core::digest::Digest;
use splinter_core::source::{CapturedSource, Origin, Part, Source, SourceId};
use splinter_store::error::StoreError;
use splinter_store::sources::SourceStore;

/// The part holding the trajectory with secrets removed.
pub const TRAJECTORY_PART: &str = "session.atif.json";

/// Which text of a step a part holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// What the person said.
    User,
    /// What the agent said.
    Agent,
    /// The agent's tool calls.
    Calls,
    /// What the tools returned.
    Observation,
}

impl Role {
    fn suffix(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Agent => "agent",
            Self::Calls => "calls",
            Self::Observation => "observation",
        }
    }
}

/// The name of the part holding `role`'s text of `step`: the step id padded
/// so that parts sort in step order.
#[must_use]
pub fn part_name(step: u64, role: Role) -> String {
    format!("step-{step:06}-{}", role.suffix())
}

/// The step and role a part name addresses; `None` for any other name.
#[must_use]
pub fn parse_part_name(name: &str) -> Option<(u64, Role)> {
    let rest = name.strip_prefix("step-")?;
    let (digits, suffix) = rest.split_once('-')?;
    let role = [Role::User, Role::Agent, Role::Calls, Role::Observation]
        .into_iter()
        .find(|r| r.suffix() == suffix)?;
    Some((digits.parse().ok()?, role))
}

/// Who a step is from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Speaker {
    /// The person.
    User,
    /// The agent.
    Agent,
}

/// One part's text, with what a span into it needs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextPart {
    /// The part's name in the session source.
    pub name: String,
    /// The digest of its content.
    pub content: Digest,
    /// Its text.
    pub text: String,
}

/// One step of a session.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Step {
    /// The ATIF `step_id`.
    pub id: u64,
    /// Who it is from.
    pub speaker: Speaker,
    /// What was said.
    pub message: Option<TextPart>,
    /// The tool calls made (agent steps).
    pub calls: Option<TextPart>,
    /// What the tools returned (agent steps).
    pub observation: Option<TextPart>,
}

/// Why a session source cannot be read as a session.
#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    /// The source is not a session.
    #[error("source {0} is not a session")]
    NotASession(SourceId),
    /// A part could not be read.
    #[error(transparent)]
    Store(#[from] StoreError),
    /// A part the source lists has no content.
    #[error("part {0:?} has no content")]
    Missing(String),
    /// A part is not UTF-8 text.
    #[error("part {0:?} is not UTF-8 text")]
    NotText(String),
}

/// A session's steps, by id.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionView {
    source: SourceId,
    steps: BTreeMap<u64, Step>,
    subject: Option<String>,
}

/// The key of the trajectory's root `extra` object under which a recording
/// names the fact a session is known to be about.
pub const SUBJECT_KEY: &str = "absorb_subject";

/// The subject a trajectory's root `extra` names, when it does.
fn subject_of(trajectory: &[u8]) -> Option<String> {
    let value: serde_json::Value = serde_json::from_slice(trajectory).ok()?;
    value
        .get("extra")?
        .get(SUBJECT_KEY)?
        .as_str()
        .map(str::to_string)
}

impl SessionView {
    /// The session `source` is, its parts' content read by `read`.
    pub fn from_source(
        source: &Source,
        mut read: impl FnMut(&Part) -> Result<Vec<u8>, SessionError>,
    ) -> Result<Self, SessionError> {
        if !matches!(source.origin, Origin::Session { .. }) {
            return Err(SessionError::NotASession(source.id.clone()));
        }
        let mut steps: BTreeMap<u64, Step> = BTreeMap::new();
        let mut subject = None;
        for part in &source.parts {
            if part.name == TRAJECTORY_PART {
                subject = subject_of(&read(part)?);
                continue;
            }
            let Some((id, role)) = parse_part_name(&part.name) else {
                continue;
            };
            let text = String::from_utf8(read(part)?)
                .map_err(|_| SessionError::NotText(part.name.clone()))?;
            let text_part = TextPart {
                name: part.name.clone(),
                content: part.content.clone(),
                text,
            };
            let step = steps.entry(id).or_insert_with(|| Step {
                id,
                speaker: if role == Role::User {
                    Speaker::User
                } else {
                    Speaker::Agent
                },
                message: None,
                calls: None,
                observation: None,
            });
            match role {
                Role::User | Role::Agent => step.message = Some(text_part),
                Role::Calls => step.calls = Some(text_part),
                Role::Observation => step.observation = Some(text_part),
            }
        }
        Ok(Self {
            source: source.id.clone(),
            steps,
            subject,
        })
    }

    /// The session just captured.
    pub fn of(captured: &CapturedSource) -> Result<Self, SessionError> {
        Self::from_source(captured.source(), |part| {
            captured
                .content(&part.name)
                .map(<[u8]>::to_vec)
                .ok_or_else(|| SessionError::Missing(part.name.clone()))
        })
    }

    /// The stored session `id`.
    pub fn load(store: &SourceStore, id: &SourceId) -> Result<Self, SessionError> {
        let source = store.get_source(id)?;
        Self::from_source(&source, |part| Ok(store.read_blob(&part.content)?))
    }

    /// The session source.
    #[must_use]
    pub fn source(&self) -> &SourceId {
        &self.source
    }

    /// The fact the recording says the session is about (the trajectory's
    /// root `extra.absorb_subject`); `None` for a session that carries none.
    #[must_use]
    pub fn subject(&self) -> Option<&str> {
        self.subject.as_deref()
    }

    /// The step `id`; `None` when the session has no such step with text.
    #[must_use]
    pub fn step(&self, id: u64) -> Option<&Step> {
        self.steps.get(&id)
    }

    /// Every step with text, in step order.
    pub fn steps(&self) -> impl Iterator<Item = &Step> {
        self.steps.values()
    }
}
