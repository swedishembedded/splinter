// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Describing a damaged database instead of refusing to open it: which files
//! are missing or corrupt and what each held, filling a hole from a copy, and
//! quarantining a file nothing can restore.

use std::collections::BTreeSet;

use super::model::{ObjectKind, ObjectRef};
use super::resolve::load_manifest;
use crate::backend::{Key, Kind};
use crate::database::Database;
use crate::error::{Error, Result};
use crate::id::ContentId;

/// What kind of file a problem is about.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProblemKind {
    /// A ref that cannot be read as a manifest id.
    Ref,
    /// A manifest.
    Manifest,
    /// A segment of records.
    Segment,
    /// A pack of blob chunks.
    BlobPack,
    /// An index run.
    Index,
    /// A shard of embedding vectors.
    Vector,
    /// A shard of an inverted text index.
    Text,
}

/// What is wrong with a file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fault {
    /// There is no such file.
    Missing,
    /// The file has another size than the manifest recorded.
    SizeMismatch {
        /// The size recorded.
        expected: u64,
        /// The size found.
        found: u64,
    },
    /// The file's bytes are not what its name, or its format, says.
    Corrupt(String),
}

/// One file the database names and cannot use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Problem {
    /// What kind of file.
    pub kind: ProblemKind,
    /// Its name: a content id, or a ref's name.
    pub id: String,
    /// Where it lies under the database root, for filling it from a copy.
    pub path: String,
    /// What is wrong.
    pub fault: Fault,
    /// Records it held, for a segment; zero otherwise.
    pub records: u64,
    /// The manifest entry for it, when it is an object rather than a ref or a
    /// manifest.
    pub object: Option<ObjectRef>,
}

/// What a verification found.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VerifyReport {
    /// Manifests walked.
    pub manifests: usize,
    /// Segments the database names.
    pub segments: usize,
    /// Blob packs the database names.
    pub blob_packs: usize,
    /// Index runs and shards the database names.
    pub index_runs: usize,
    /// Everything wrong, in the order found.
    pub problems: Vec<Problem>,
}

fn problem_kind(kind: ObjectKind) -> ProblemKind {
    match kind {
        ObjectKind::Segment => ProblemKind::Segment,
        ObjectKind::BlobPack => ProblemKind::BlobPack,
        ObjectKind::Index => ProblemKind::Index,
        ObjectKind::Vector => ProblemKind::Vector,
        ObjectKind::Text => ProblemKind::Text,
    }
}

impl Database {
    /// Walks every ref, every manifest they reach and every file those name,
    /// and reports what is missing or damaged instead of stopping at the first
    /// of it. Without `deep` a file is checked for existing at the recorded
    /// size; with it every file is read and hashed, because each one is named
    /// by the hash of its bytes.
    pub fn verify(&self, deep: bool) -> Result<VerifyReport> {
        let mut report = VerifyReport::default();
        let mut heads = Vec::new();
        for key in self.backend().list(Kind::Ref)? {
            let name = key.name();
            if !(name.starts_with("jobs/")
                || name.starts_with("catalog/")
                || name.starts_with("retired/"))
            {
                continue;
            }
            let parsed = self
                .backend()
                .read(&key)
                .map_err(|e| e.to_string())
                .and_then(|bytes| {
                    ContentId::parse(String::from_utf8_lossy(&bytes).trim())
                        .map_err(|e| e.to_string())
                });
            match parsed {
                Ok(head) => heads.push(head),
                Err(reason) => report.problems.push(Problem {
                    kind: ProblemKind::Ref,
                    id: name.to_owned(),
                    path: key.relative_path(),
                    fault: Fault::Corrupt(reason),
                    records: 0,
                    object: None,
                }),
            }
        }

        let (mut added, mut removed) = (BTreeSet::new(), BTreeSet::new());
        let mut seen = BTreeSet::new();
        let mut pending = heads;
        while let Some(id) = pending.pop() {
            if !seen.insert(id) {
                continue;
            }
            let key = Key::new(Kind::Manifest, &id.to_string())?;
            match load_manifest(self, id) {
                Ok(manifest) => {
                    report.manifests += 1;
                    added.extend(manifest.add.iter().cloned());
                    removed.extend(manifest.remove.iter().cloned());
                    if !manifest.squash {
                        pending.extend(manifest.parents.iter().copied());
                    }
                }
                Err(error) => report.problems.push(Problem {
                    kind: ProblemKind::Manifest,
                    id: id.to_string(),
                    path: key.relative_path(),
                    fault: match error {
                        Error::NotFound { .. } => Fault::Missing,
                        other => Fault::Corrupt(other.to_string()),
                    },
                    records: 0,
                    object: None,
                }),
            }
        }

        for object in added.difference(&removed) {
            match object.kind {
                ObjectKind::Segment => report.segments += 1,
                ObjectKind::BlobPack => report.blob_packs += 1,
                _ => report.index_runs += 1,
            }
            if let Some(fault) = self.object_fault(object, deep)? {
                let key = object.key()?;
                report.problems.push(Problem {
                    kind: problem_kind(object.kind),
                    id: object.id.to_string(),
                    path: key.relative_path(),
                    fault,
                    records: object.records,
                    object: Some(object.clone()),
                });
            }
        }
        Ok(report)
    }

    fn object_fault(&self, object: &ObjectRef, deep: bool) -> Result<Option<Fault>> {
        let key = object.key()?;
        let found = match self.backend().len(&key) {
            Ok(found) => found,
            Err(Error::NotFound { .. }) => return Ok(Some(Fault::Missing)),
            Err(other) => return Err(other),
        };
        if found != object.bytes {
            return Ok(Some(Fault::SizeMismatch {
                expected: object.bytes,
                found,
            }));
        }
        if deep {
            let bytes = self.backend().read(&key)?;
            if ContentId::of(&bytes) != object.id {
                return Ok(Some(Fault::Corrupt(
                    "its bytes do not hash to its name".into(),
                )));
            }
        }
        Ok(None)
    }

    /// Writes the file at `path` under the database root (as
    /// [`Problem::path`] names it) from `bytes`, refused unless the bytes are
    /// what its name says: every object is named by the hash of its bytes, so a
    /// hole can be filled from any copy and from nowhere else. Returns whether
    /// a file was written; `false` when a sound one is already there. A file
    /// whose bytes are not what its name says is replaced.
    pub fn fill(&self, path: &str, bytes: &[u8]) -> Result<bool> {
        let key = Key::from_relative(path).ok_or_else(|| {
            Error::invalid(
                "file to fill",
                format!("`{path}` is not a file of a database"),
            )
        })?;
        if key.kind().is_addressed() {
            let name = key.name().split('.').next().unwrap_or_default();
            if ContentId::of(bytes).to_string() != name {
                return Err(Error::invalid(
                    "file to fill",
                    format!("these bytes are not {path}: they hash to another name"),
                ));
            }
        }
        if self.backend().write_once(&key, bytes)? {
            return Ok(true);
        }
        if !key.kind().is_addressed() || self.backend().read(&key)? == bytes {
            return Ok(false);
        }
        self.backend().replace(&key, bytes)?;
        Ok(true)
    }

    /// Withdraws the file a problem is about from the database, for good, so
    /// what remains opens again. Whatever it held is lost, and the report says
    /// how much. Only for a file no copy can restore.
    pub fn quarantine(&self, problem: &Problem) -> Result<Quarantined> {
        let Some(object) = problem.object.clone() else {
            return Err(Error::invalid(
                "quarantine",
                "only a segment, pack or index can be withdrawn",
            ));
        };
        self.publish_once("quarantine", Vec::new(), vec![object.clone()])?;
        Ok(Quarantined {
            kind: problem.kind,
            id: problem.id.clone(),
            records: object.records,
        })
    }
}

/// What quarantining a file cost.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Quarantined {
    /// What kind of file it was.
    pub kind: ProblemKind,
    /// Its name.
    pub id: String,
    /// The records it held, now lost.
    pub records: u64,
}
