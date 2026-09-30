// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements durable, content-addressed experience
// stores for learning agents, for its clients. If your team needs expertise
// in training-data lineage or crash-safe storage, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The experience store: immutable experiences under their content
//! address, append-only annotations beside them, and named sets of ids.
//!
//! ```text
//! <root>/experiences/
//!   objects/<hex>.json         one experience, its canonical form; written once
//!   annotations/<hex>.jsonl    its annotations, one JSON object per line, appended
//!   sets/<hex>.json            one experience set, its canonical form; written once
//! ```
//!
//! `<hex>` is the hex part of the content address, so a file's name is
//! also the digest its bytes must hash to; [`ExperienceStore::get`] checks
//! that on every read. An object is created with [`write_once`] and never
//! replaced. An annotation is appended with a single write and fsynced
//! before [`ExperienceStore::annotate`] returns. Appends are serialized
//! within one store handle; separate processes appending to one log rely on
//! the file being opened in append mode, and nothing coordinates them
//! beyond that.

use std::fs::{self, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::annotation::{Annotation, AnnotationBody};
use crate::digest::{canonical_json, Digest};
use crate::experience::{Experience, ExperienceError, ExperienceId};
use crate::{sync_dir, write_once, StateRoot};

/// The content address of an [`ExperienceSet`].
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SetId(pub Digest);

impl std::fmt::Display for SetId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// A named, ordered list of experiences: how a pipeline stage names its
/// input. Content-addressed like an experience, name included.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExperienceSet {
    /// What the set is, for a reader.
    pub name: String,
    /// The experiences, in order; each at most once.
    pub members: Vec<ExperienceId>,
}

/// An experience's annotations, as far as they could be read.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AnnotationLog {
    /// The readable annotations, in append order.
    pub annotations: Vec<Annotation>,
    /// Lines that could not be read - a line torn by a crash mid-append -
    /// and were skipped.
    pub unreadable: usize,
}

/// Why a store operation failed.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    /// The experience is not valid.
    #[error("invalid experience: {0}")]
    Invalid(#[from] ExperienceError),
    /// An annotation or set is not valid.
    #[error("invalid {what}: {reason}")]
    Rejected {
        /// What was refused.
        what: &'static str,
        /// Why.
        reason: String,
    },
    /// The store holds no experience with this id.
    #[error("no experience {0} in the store")]
    UnknownExperience(ExperienceId),
    /// The store holds no set with this id.
    #[error("no experience set {0} in the store")]
    UnknownSet(SetId),
    /// A stored object's bytes do not hash to its address.
    #[error("{path} is corrupt: it should hash to {expected}, it hashes to {found}")]
    Corrupt {
        /// The object file.
        path: PathBuf,
        /// Its address.
        expected: Digest,
        /// What its bytes hash to.
        found: Digest,
    },
    /// A stored object hashes correctly but does not read back as the
    /// record it addresses (written by an incompatible schema).
    #[error("{path} does not decode to the record it addresses: {reason}")]
    Undecodable {
        /// The object file.
        path: PathBuf,
        /// Why.
        reason: String,
    },
    /// A record cannot be serialized.
    #[error("cannot serialize {what}: {source}")]
    Serialize {
        /// What was being serialized.
        what: &'static str,
        /// The serializer's error.
        source: serde_json::Error,
    },
    /// A file operation failed.
    #[error("{path}: {source}")]
    Io {
        /// The file or directory.
        path: PathBuf,
        /// The underlying error.
        source: std::io::Error,
    },
}

fn io(path: &Path) -> impl FnOnce(std::io::Error) -> StoreError + '_ {
    move |source| StoreError::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// The experience store under one state root.
#[derive(Debug)]
pub struct ExperienceStore {
    dir: PathBuf,
    append: Mutex<()>,
}

impl ExperienceStore {
    /// The store under `root`. Nothing is created until something is
    /// written.
    #[must_use]
    pub fn open(root: &StateRoot) -> Self {
        Self {
            dir: root.experiences(),
            append: Mutex::new(()),
        }
    }

    fn object(&self, id: &ExperienceId) -> PathBuf {
        self.dir.join("objects").join(format!("{}.json", id.hex()))
    }

    fn log(&self, id: &ExperienceId) -> PathBuf {
        self.dir
            .join("annotations")
            .join(format!("{}.jsonl", id.hex()))
    }

    fn set(&self, id: &SetId) -> PathBuf {
        self.dir.join("sets").join(format!("{}.json", id.0.hex()))
    }

    /// Stores `experience` and returns its id. Write-once: storing the same
    /// content again is a no-op returning the same id; an existing object
    /// is never replaced, and one that no longer matches its address is
    /// reported as corrupt rather than overwritten.
    pub fn put(&self, experience: &Experience) -> Result<ExperienceId, StoreError> {
        experience.validate()?;
        let bytes = experience.canonical()?;
        let id = ExperienceId(Digest::of(&bytes));
        let path = self.object(&id);
        if !write_once(&path, &bytes).map_err(io(&path))? {
            read_verified(&path, &id.0)?;
        }
        Ok(id)
    }

    /// Whether the store holds `id` (without verifying its content).
    #[must_use]
    pub fn contains(&self, id: &ExperienceId) -> bool {
        self.object(id).is_file()
    }

    /// The experience stored under `id`, verified against its address.
    pub fn get(&self, id: &ExperienceId) -> Result<Experience, StoreError> {
        let path = self.object(id);
        if !path.is_file() {
            return Err(StoreError::UnknownExperience(id.clone()));
        }
        let bytes = read_verified(&path, &id.0)?;
        let experience: Experience = decode(&path, &bytes)?;
        // The address must be the address of the value handed back, not
        // only of the bytes: a field this build does not know would be
        // dropped on decode, and the caller would hold a different record.
        if experience.canonical()? != bytes {
            return Err(StoreError::Undecodable {
                path,
                reason: "it does not re-encode to its stored form".into(),
            });
        }
        Ok(experience)
    }

    /// Appends `note` to its experience's log. Refused for an experience
    /// the store does not hold (or a relation to one). Durable once this
    /// returns.
    pub fn annotate(&self, note: &Annotation) -> Result<(), StoreError> {
        let rejected = |reason: &str| StoreError::Rejected {
            what: "annotation",
            reason: reason.to_string(),
        };
        if note.producer.name.trim().is_empty() || note.producer.version.trim().is_empty() {
            return Err(rejected("the producer's name and version are required"));
        }
        if !self.contains(&note.experience) {
            return Err(StoreError::UnknownExperience(note.experience.clone()));
        }
        if let AnnotationBody::Relation { other, .. } = &note.body {
            if *other == note.experience {
                return Err(rejected("an experience cannot be related to itself"));
            }
            if !self.contains(other) {
                return Err(StoreError::UnknownExperience(other.clone()));
            }
        }
        let mut line = serde_json::to_vec(note).map_err(|source| StoreError::Serialize {
            what: "annotation",
            source,
        })?;
        line.push(b'\n');

        let path = self.log(&note.experience);
        // A poisoned lock only means another append panicked; the file is
        // still the source of truth, so carry on.
        let _guard = self
            .append
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let parent = self.dir.join("annotations");
        fs::create_dir_all(&parent).map_err(io(&parent))?;
        let created = !path.exists();
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&path)
            .map_err(io(&path))?;
        // A crash mid-append leaves a line without its newline; start on a
        // fresh line so the torn fragment stays its own unreadable line
        // instead of corrupting this one.
        if ends_torn(&mut file).map_err(io(&path))? {
            line.insert(0, b'\n');
        }
        file.write_all(&line).map_err(io(&path))?;
        file.sync_data().map_err(io(&path))?;
        if created {
            sync_dir(&parent).map_err(io(&parent))?;
        }
        Ok(())
    }

    /// `id`'s annotations in append order. Lines that cannot be read are
    /// skipped and counted in [`AnnotationLog::unreadable`]; an annotation
    /// that names a different experience is corruption, and an error.
    pub fn annotations(&self, id: &ExperienceId) -> Result<AnnotationLog, StoreError> {
        if !self.contains(id) {
            return Err(StoreError::UnknownExperience(id.clone()));
        }
        let path = self.log(id);
        let text = match fs::read_to_string(&path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
            Err(e) => return Err(io(&path)(e)),
        };
        let mut log = AnnotationLog::default();
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            match serde_json::from_str::<Annotation>(line) {
                Ok(note) if note.experience == *id => log.annotations.push(note),
                Ok(note) => {
                    return Err(StoreError::Undecodable {
                        path,
                        reason: format!("it holds an annotation of {}", note.experience),
                    })
                }
                Err(_) => log.unreadable += 1,
            }
        }
        Ok(log)
    }

    /// Stores `set` and returns its id; write-once like [`Self::put`].
    /// Refused when a member is unknown or listed twice.
    pub fn put_set(&self, set: &ExperienceSet) -> Result<SetId, StoreError> {
        let rejected = |reason: String| StoreError::Rejected {
            what: "experience set",
            reason,
        };
        if set.name.trim().is_empty() {
            return Err(rejected("the name is empty".into()));
        }
        let mut seen = std::collections::HashSet::new();
        for member in &set.members {
            if !seen.insert(member) {
                return Err(rejected(format!("{member} is listed twice")));
            }
            if !self.contains(member) {
                return Err(StoreError::UnknownExperience(member.clone()));
            }
        }
        let bytes = canonical_json(set).map_err(|source| StoreError::Serialize {
            what: "experience set",
            source,
        })?;
        let id = SetId(Digest::of(&bytes));
        let path = self.set(&id);
        if !write_once(&path, &bytes).map_err(io(&path))? {
            read_verified(&path, &id.0)?;
        }
        Ok(id)
    }

    /// The set stored under `id`, verified against its address.
    pub fn get_set(&self, id: &SetId) -> Result<ExperienceSet, StoreError> {
        let path = self.set(id);
        if !path.is_file() {
            return Err(StoreError::UnknownSet(id.clone()));
        }
        let bytes = read_verified(&path, &id.0)?;
        decode(&path, &bytes)
    }
}

/// The bytes at `path`, refused unless they hash to `address`.
fn read_verified(path: &Path, address: &Digest) -> Result<Vec<u8>, StoreError> {
    let bytes = fs::read(path).map_err(io(path))?;
    let found = Digest::of(&bytes);
    if found != *address {
        return Err(StoreError::Corrupt {
            path: path.to_path_buf(),
            expected: address.clone(),
            found,
        });
    }
    Ok(bytes)
}

fn decode<T: serde::de::DeserializeOwned>(path: &Path, bytes: &[u8]) -> Result<T, StoreError> {
    serde_json::from_slice(bytes).map_err(|e| StoreError::Undecodable {
        path: path.to_path_buf(),
        reason: e.to_string(),
    })
}

/// Whether a non-empty `file` lacks its final newline.
fn ends_torn(file: &mut fs::File) -> std::io::Result<bool> {
    if file.metadata()?.len() == 0 {
        return Ok(false);
    }
    file.seek(SeekFrom::End(-1))?;
    let mut last = [0u8; 1];
    file.read_exact(&mut last)?;
    Ok(last[0] != b'\n')
}
