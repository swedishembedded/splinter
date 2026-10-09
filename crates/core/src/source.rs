// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements content-addressed sources that every
// learned fact traces back to, for its clients. If your team needs expertise
// in training-data provenance or knowledge acquisition, you can procure our
// services by sending an email to info@swedishembedded.com.

//! A source: anything Splinter learns from, recorded immutably and
//! addressable down to a byte range.
//!
//! A [`Source`] is its [`Origin`] (a document, a repository tree, a command
//! run) and its named [`Part`]s, each a media type and the [`Digest`] of its
//! content. The content itself is stored once per digest, shared by every
//! source that holds it (see [`crate::sources`]).
//!
//! The [`SourceId`] is the digest of the canonical JSON (see
//! [`crate::digest`]) of the origin and the parts, ordered by name. The
//! capture time is recorded but is not part of the address: capturing
//! unchanged content again is the same source, not a new one.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::clock::Clock;
use crate::digest::{canonical_json, Digest};
use crate::terms::Terms;

/// The content address of a [`Source`].
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SourceId(pub Digest);

impl SourceId {
    /// `blake3:<hex>`.
    #[must_use]
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }

    /// The hex part alone, which names the source's file.
    #[must_use]
    pub fn hex(&self) -> &str {
        self.0.hex()
    }
}

impl std::fmt::Display for SourceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// One named part of a [`Source`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Part {
    /// The part's name within the source: a `/`-separated path for a file
    /// (`README.md`, `src/main.rs`), or a stream (`stdout`, `stderr`).
    pub name: String,
    /// Its media type: `text/markdown`, `text/plain`, ...
    pub media_type: String,
    /// The digest of its content.
    pub content: Digest,
    /// Its length in bytes.
    pub bytes: u64,
}

/// A part of a source, named: what a [`crate::experience::Span`] points
/// into.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PartRef {
    /// The source.
    pub source: SourceId,
    /// The part's name within it.
    pub name: String,
}

/// The state of a git work tree when it was captured.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Revision {
    /// `HEAD`'s commit, absent in a repository with no commit yet.
    pub commit: Option<String>,
    /// Whether the captured tree differs from that commit (modified,
    /// staged or untracked files).
    pub dirty: bool,
}

/// Why a file of a repository tree is not one of its parts.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkipReason {
    /// Larger than the capture's per-file cap.
    TooLarge {
        /// Its size.
        bytes: u64,
    },
    /// Not UTF-8 text.
    NotText,
    /// Not a regular file or directory (a symbolic link, a socket, ...).
    NotRegular,
    /// Its path is not UTF-8, so it cannot be named as a part.
    NonUtf8Name,
    /// Held back from what is learned from: its text is reserved for the
    /// exam, with every other text that overlaps it.
    Reserved,
}

/// A file of a repository tree left out of the capture, and why.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Skipped {
    /// Its path relative to the tree's root (lossily decoded when it is
    /// not UTF-8).
    pub path: String,
    /// Why it was left out.
    pub reason: SkipReason,
}

/// What was removed from a session before it was stored: `count` secrets of
/// one `kind` in one step. Only the fact of the removal is recorded, never
/// the secret.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Redaction {
    /// The ATIF `step_id` the secret was in; `None` outside the steps
    /// (the session's own fields).
    pub step: Option<u64>,
    /// What it looked like: `private_key`, `token`, `credential`, ...
    pub kind: String,
    /// How many were removed.
    pub count: u32,
}

/// Where a source came from.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Origin {
    /// A single text file.
    Document {
        /// Its absolute path.
        path: String,
    },
    /// A directory tree of text files.
    Repository {
        /// The tree's absolute path.
        path: String,
        /// Its git state, when it is in a git work tree.
        revision: Option<Revision>,
        /// Files under it that are not parts, in path order. Directories
        /// the capture always ignores (version control, build output) are
        /// not listed.
        skipped: Vec<Skipped>,
    },
    /// One run of a program; its output streams are the parts.
    Command {
        /// The program and its arguments, run without a shell.
        argv: Vec<String>,
        /// The absolute working directory it ran in.
        cwd: String,
        /// Its exit code; absent when a signal ended it (a timeout
        /// included).
        exit_code: Option<i32>,
        /// Whether the run outlived its timeout and its process group was
        /// killed: the program itself, or a process it left holding its
        /// output open.
        timed_out: bool,
        /// Whether standard output exceeded the capture's cap and was cut.
        stdout_truncated: bool,
        /// Whether standard error exceeded the capture's cap and was cut.
        stderr_truncated: bool,
    },
    /// One recorded session of an agent with a person (an ATIF trajectory,
    /// secrets removed); each step's text is a part.
    Session {
        /// The trajectory's own session id, when it states one.
        session_id: Option<String>,
        /// The agent that held it: its name and version.
        agent: String,
        /// What was removed from it, in step order.
        redactions: Vec<Redaction>,
    },
}

impl Origin {
    /// The origin's kind as it is serialized: `document`, `repository`,
    /// `command` or `session`.
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Document { .. } => "document",
            Self::Session { .. } => "session",
            Self::Repository { .. } => "repository",
            Self::Command { .. } => "command",
        }
    }
}

/// A source: where it came from, when, and its parts.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Source {
    /// The content address of `origin` and `parts`.
    pub id: SourceId,
    /// Where it came from.
    pub origin: Origin,
    /// When it was first captured, from the injected [`Clock`].
    pub captured_at: String,
    /// Its parts, ordered by name, each name once.
    pub parts: Vec<Part>,
    /// The terms the content came under; `None` when none were stated, which
    /// every consumer treats as unknown ([`crate::terms::Terms::unknown`]).
    /// Part of the address: the same bytes under other terms are another
    /// source, so a stronger restriction can never be lost by capturing the
    /// content again, and a source stated before terms existed keeps its id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terms: Option<Terms>,
}

/// The fields a source's address is computed over.
#[derive(Serialize)]
struct SourceBody<'a> {
    origin: &'a Origin,
    parts: &'a [Part],
    #[serde(skip_serializing_if = "Option::is_none")]
    terms: Option<&'a Terms>,
}

impl Source {
    /// The origin's kind: `document`, `repository`, `command` or `session`.
    #[must_use]
    pub fn kind(&self) -> &'static str {
        self.origin.kind()
    }

    /// The part named `name`.
    #[must_use]
    pub fn part(&self, name: &str) -> Option<&Part> {
        self.parts.iter().find(|p| p.name == name)
    }

    /// The canonical form the source is stored as.
    pub fn canonical(&self) -> Result<Vec<u8>, SourceError> {
        Ok(canonical_json(self)?)
    }

    /// Checks what the type system cannot: parts ordered by name with each
    /// name once and non-empty, a capture time, and an id that addresses
    /// the origin and parts this record carries.
    pub fn validate(&self) -> Result<(), SourceError> {
        if self.captured_at.trim().is_empty() {
            return Err(SourceError::Missing("captured_at"));
        }
        for pair in self.parts.windows(2) {
            if pair[0].name >= pair[1].name {
                return Err(if pair[0].name == pair[1].name {
                    SourceError::DuplicatePart(pair[0].name.clone())
                } else {
                    SourceError::Unordered
                });
            }
        }
        if self.parts.iter().any(|p| p.name.is_empty()) {
            return Err(SourceError::Missing("part name"));
        }
        let expected = address(&self.origin, &self.parts, self.terms.as_ref())?;
        if expected != self.id {
            return Err(SourceError::Address {
                recorded: self.id.clone(),
                expected,
            });
        }
        Ok(())
    }
}

fn address(
    origin: &Origin,
    parts: &[Part],
    terms: Option<&Terms>,
) -> Result<SourceId, SourceError> {
    Ok(SourceId(Digest::of(&canonical_json(&SourceBody {
        origin,
        parts,
        terms,
    })?)))
}

/// One part's content as a capture read it, before it is addressed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartContent {
    /// The part's name (see [`Part::name`]).
    pub name: String,
    /// Its media type.
    pub media_type: String,
    /// Its content.
    pub bytes: Vec<u8>,
}

/// A freshly captured source together with its parts' content: what a
/// capture hands the source store. Built only by [`CapturedSource::new`],
/// so every part's digest is the digest of the content it carries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapturedSource {
    source: Source,
    contents: BTreeMap<String, Vec<u8>>,
}

impl CapturedSource {
    /// The source of `origin` with `parts` (in any order; they are ordered
    /// by name), stamped by `clock`. Refused when a part name is empty or
    /// appears twice.
    pub fn new(
        origin: Origin,
        parts: Vec<PartContent>,
        clock: &dyn Clock,
    ) -> Result<Self, SourceError> {
        let mut contents = BTreeMap::new();
        for part in parts {
            if part.name.is_empty() {
                return Err(SourceError::Missing("part name"));
            }
            if contents.contains_key(&part.name) {
                return Err(SourceError::DuplicatePart(part.name));
            }
            contents.insert(part.name, (part.media_type, part.bytes));
        }
        let parts: Vec<Part> = contents
            .iter()
            .map(|(name, (media_type, bytes))| Part {
                name: name.clone(),
                media_type: media_type.clone(),
                content: Digest::of(bytes),
                bytes: bytes.len() as u64,
            })
            .collect();
        let source = Source {
            id: address(&origin, &parts, None)?,
            origin,
            captured_at: clock.utc_now(),
            parts,
            terms: None,
        };
        source.validate()?;
        Ok(Self {
            source,
            contents: contents
                .into_iter()
                .map(|(name, (_, bytes))| (name, bytes))
                .collect(),
        })
    }

    /// The same capture stated as coming under `terms`; its id changes with
    /// them (see [`Source::terms`]).
    pub fn with_terms(mut self, terms: Terms) -> Result<Self, SourceError> {
        self.source.id = address(&self.source.origin, &self.source.parts, Some(&terms))?;
        self.source.terms = Some(terms);
        self.source.validate()?;
        Ok(self)
    }

    /// The source.
    #[must_use]
    pub fn source(&self) -> &Source {
        &self.source
    }

    /// The content of the part named `name`.
    #[must_use]
    pub fn content(&self, name: &str) -> Option<&[u8]> {
        self.contents.get(name).map(Vec::as_slice)
    }

    /// The source, without its content.
    #[must_use]
    pub fn into_source(self) -> Source {
        self.source
    }
}

/// Why a source is not valid.
#[derive(Debug, thiserror::Error)]
pub enum SourceError {
    /// A required field is empty.
    #[error("{0} is empty")]
    Missing(&'static str),
    /// Two parts share a name.
    #[error("part {0:?} appears more than once")]
    DuplicatePart(String),
    /// The parts are not ordered by name.
    #[error("the parts are not ordered by name")]
    Unordered,
    /// The id does not address the origin and parts the record carries.
    #[error(
        "source id {recorded} does not address this source (its content hashes to {expected})"
    )]
    Address {
        /// The id the record carries.
        recorded: SourceId,
        /// The id its content hashes to.
        expected: SourceId,
    },
    /// The record cannot be serialized.
    #[error("cannot serialize the source: {0}")]
    Serialize(#[from] serde_json::Error),
}
