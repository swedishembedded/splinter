// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements durable, recoverable state for learning
// agents, for its clients. If your team needs expertise in backup and
// disaster recovery of training data, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Finding what is missing or damaged, and recovering it: filling holes from
//! copies, rebuilding what can be rebuilt, and writing off, only when told to,
//! what nothing can restore.

use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

use splinter_expdb::manifest::{Problem, ProblemKind, VerifyReport};
use splinter_expdb::Database;

use super::archive::{place_artifact, read_member, scan};
use super::{loss_name, Loss};
use crate::artifacts::{Artifact, ArtifactState, ArtifactStore};
use crate::digest::Digest;
use crate::error::StoreError;
use crate::workspace::Workspace;

/// What is wrong with an artifact's file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArtifactFault {
    /// There is no file.
    Missing,
    /// The file has another size than was recorded.
    SizeChanged,
    /// The file has the recorded size but other bytes.
    Corrupt,
}

/// An artifact that cannot be used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactProblem {
    /// Its digest.
    pub digest: Digest,
    /// What it is for.
    pub role: String,
    /// What is wrong.
    pub fault: ArtifactFault,
}

/// What a verification found.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StateVerify {
    /// The database's files.
    pub database: VerifyReport,
    /// The artifacts' files, other than those written off.
    pub artifacts: Vec<ArtifactProblem>,
    /// The artifacts could not be listed because the database is damaged;
    /// they are checked once it is repaired.
    pub artifacts_unchecked: bool,
}

impl StateVerify {
    /// Whether nothing is wrong.
    #[must_use]
    pub fn is_sound(&self) -> bool {
        self.database.problems.is_empty() && self.artifacts.is_empty() && !self.artifacts_unchecked
    }
}

/// What a repair may use and do.
#[derive(Clone, Debug, Default)]
pub struct RepairOptions {
    /// Copies to fill holes from: archive files, or other state roots.
    pub from: Vec<PathBuf>,
    /// Write off what no copy has: withdraw damaged database files and record
    /// lost artifacts in the ledger. Without it nothing is given up.
    pub accept_loss: bool,
}

/// What a repair did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Repaired {
    /// Files restored from a copy.
    pub filled: usize,
    /// Index files dropped and built again from the records.
    pub rebuilt: usize,
    /// Database files withdrawn because nothing could restore them.
    pub quarantined: usize,
    /// Everything written off, database files and artifacts.
    pub lost: Vec<Loss>,
    /// Problems still open.
    pub unresolved: usize,
}

/// A file to find in a copy.
struct Want {
    /// Its member path in an archive, which is also its path under a state root.
    member: String,
    /// The digest its bytes must have, for an artifact; a database file is
    /// named by its own.
    hex: String,
    /// Where it goes, for an artifact.
    dest: Option<PathBuf>,
    /// Where it goes under the database root, for a database file.
    database: Option<String>,
}

fn kind_name(kind: ProblemKind) -> &'static str {
    match kind {
        ProblemKind::Ref => "ref",
        ProblemKind::Manifest => "manifest",
        ProblemKind::Segment => "segment",
        ProblemKind::BlobPack => "blobpack",
        ProblemKind::Index => "index",
        ProblemKind::Vector => "vector",
        ProblemKind::Text => "text",
    }
}

fn fault_of(state: ArtifactState) -> Option<ArtifactFault> {
    match state {
        ArtifactState::Sound => None,
        ArtifactState::Missing => Some(ArtifactFault::Missing),
        ArtifactState::SizeChanged { .. } => Some(ArtifactFault::SizeChanged),
        ArtifactState::Corrupt { .. } => Some(ArtifactFault::Corrupt),
    }
}

impl Workspace {
    /// Reports every missing or damaged file of the database and of the
    /// artifacts without stopping at the first. Without `deep` a file is
    /// checked for existing at the recorded size; with it every byte is hashed.
    pub fn verify(&self, deep: bool) -> Result<StateVerify, StoreError> {
        self.commit()?;
        if !self.initialised() {
            return Ok(StateVerify::default());
        }
        let database = self.database()?.verify(deep)?;
        self.refresh().ok();
        let mut report = StateVerify {
            database,
            ..StateVerify::default()
        };
        match self.artifact_problems(deep) {
            Ok(problems) => report.artifacts = problems,
            Err(_) if !report.database.problems.is_empty() => report.artifacts_unchecked = true,
            Err(e) => return Err(e),
        }
        Ok(report)
    }

    fn artifact_problems(&self, deep: bool) -> Result<Vec<ArtifactProblem>, StoreError> {
        let store = ArtifactStore::new(self, self.root());
        let written_off = self.written_off_artifacts()?;
        let mut problems = Vec::new();
        for artifact in store.list()? {
            if written_off.contains(artifact.digest.as_str()) {
                continue;
            }
            if let Some(fault) = fault_of(store.check(&artifact.digest, deep)?) {
                problems.push(ArtifactProblem {
                    digest: artifact.digest,
                    role: artifact.role,
                    fault,
                });
            }
        }
        Ok(problems)
    }

    fn written_off_artifacts(&self) -> Result<HashSet<String>, StoreError> {
        Ok(self
            .losses()?
            .into_iter()
            .filter(|l| l.kind == "artifact")
            .map(|l| l.id)
            .collect())
    }

    /// Recovers what is wrong, in order: holes are filled from `options.from`,
    /// damaged index files are rebuilt from the records, and what is left is
    /// given up on only with `options.accept_loss`, each item recorded in the
    /// loss ledger. Content addressing means a copy is used only if its bytes
    /// are exactly right.
    pub fn repair(&self, options: &RepairOptions) -> Result<Repaired, StoreError> {
        self.commit()?;
        let mut done = Repaired::default();
        if !self.initialised() {
            return Ok(done);
        }
        let db = self.database()?;

        let report = db.verify(true)?;
        let wants: Vec<Want> = report
            .problems
            .iter()
            .filter(|p| p.kind != ProblemKind::Ref)
            .map(|p| Want {
                member: format!("expdb/{}", p.path),
                hex: p.id.clone(),
                dest: None,
                database: Some(p.path.clone()),
            })
            .collect();
        let filled = fetch(&db, &wants, &options.from)?;
        done.filled += filled.len();
        let remaining = if filled.is_empty() {
            report
        } else {
            db.verify(true)?
        };

        let mut stuck: Vec<Problem> = Vec::new();
        let mut rebuilt = false;
        for problem in remaining.problems {
            if matches!(
                problem.kind,
                ProblemKind::Index | ProblemKind::Vector | ProblemKind::Text
            ) {
                db.quarantine(&problem)?;
                done.rebuilt += 1;
                rebuilt = true;
            } else {
                stuck.push(problem);
            }
        }
        if rebuilt {
            db.build_indexes()?;
        }
        for problem in stuck {
            if !options.accept_loss {
                done.unresolved += 1;
                continue;
            }
            let withdrawn = db.quarantine(&problem)?;
            let loss = Loss {
                kind: kind_name(problem.kind).into(),
                id: problem.id.clone(),
                detail: format!(
                    "{} of the database, {:?}",
                    kind_name(problem.kind),
                    problem.fault
                ),
                records: withdrawn.records,
            };
            self.write_off(&loss_name(&loss.kind, &loss.id), &loss)?;
            done.quarantined += 1;
            done.lost.push(loss);
        }
        self.refresh().ok();

        if done.unresolved > 0 {
            // The artifacts are named by records that cannot be read yet.
            return Ok(done);
        }
        self.repair_artifacts(options, &mut done)?;
        Ok(done)
    }

    fn repair_artifacts(
        &self,
        options: &RepairOptions,
        done: &mut Repaired,
    ) -> Result<(), StoreError> {
        let store = ArtifactStore::new(self, self.root());
        let broken = self.artifact_problems(true)?;
        if broken.is_empty() {
            return Ok(());
        }
        let records: Vec<Artifact> = broken
            .iter()
            .map(|p| store.get(&p.digest))
            .collect::<Result<_, _>>()?;
        let wants: Vec<Want> = records
            .iter()
            .map(|a| Want {
                member: a.relative_path(),
                hex: a.digest.hex().to_owned(),
                dest: Some(self.root().path().join(a.relative_path())),
                database: None,
            })
            .collect();
        let filled = fetch(&self.database()?, &wants, &options.from)?;
        done.filled += filled.len();
        for (index, artifact) in records.iter().enumerate() {
            if filled.contains(&index) {
                continue;
            }
            if !options.accept_loss {
                done.unresolved += 1;
                continue;
            }
            let loss = Loss {
                kind: "artifact".into(),
                id: artifact.digest.to_string(),
                detail: format!("{} made by {}", artifact.role, artifact.producer),
                records: 0,
            };
            self.write_off(&loss_name("artifact", artifact.digest.hex()), &loss)?;
            done.lost.push(loss);
        }
        Ok(())
    }
}

/// Puts what `reader` holds in place as the wanted file, if its bytes are
/// exactly that file's; false when they are not.
fn place(
    db: &Database,
    want: &Want,
    reader: &mut dyn std::io::Read,
    size: u64,
) -> Result<bool, StoreError> {
    if let Some(dest) = &want.dest {
        return place_artifact(dest, reader, &want.hex);
    }
    let path = want.database.as_deref().unwrap_or_default();
    let bytes = read_member(reader, size).map_err(|source| StoreError::Io {
        path: PathBuf::from(&want.member),
        source,
    })?;
    // A database file is named by its own digest, which `fill` enforces.
    Ok(db.fill(path, &bytes).unwrap_or(false))
}

/// Looks for each wanted file in each copy and puts the ones found in place.
/// Returns the indexes of the wants that were filled. A copy that has the file
/// with the wrong bytes is as good as one that lacks it.
fn fetch(db: &Database, wants: &[Want], copies: &[PathBuf]) -> Result<HashSet<usize>, StoreError> {
    let mut filled = HashSet::new();
    for copy in copies {
        if filled.len() == wants.len() {
            break;
        }
        if copy.is_dir() {
            for (index, want) in wants.iter().enumerate() {
                let candidate = copy.join(&want.member);
                if filled.contains(&index) || !candidate.is_file() {
                    continue;
                }
                let io = |source| StoreError::Io {
                    path: candidate.clone(),
                    source,
                };
                let size = fs::metadata(&candidate).map_err(io)?.len();
                let mut file = fs::File::open(&candidate).map_err(io)?;
                if place(db, want, &mut file, size)? {
                    filled.insert(index);
                }
            }
        } else {
            scan(copy, |member, size, data| {
                if let Some(index) = wants.iter().position(|w| w.member == member) {
                    if !filled.contains(&index) && place(db, &wants[index], data, size)? {
                        filled.insert(index);
                    }
                }
                Ok(())
            })?;
        }
    }
    Ok(filled)
}
