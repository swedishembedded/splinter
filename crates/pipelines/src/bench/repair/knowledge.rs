// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements teachers that correct agent failures with
// the documentation a practitioner would read, for its clients. If your team
// needs expertise in retrieval-assisted agent training then you can procure
// our services by sending an email to info@swedishembedded.com.

//! The knowledge-assisted correction: the teacher continues from the last
//! good step with task-relevant public reference material in front of it.
//!
//! The material is a [`KnowledgeBase`] of documents the operator names (a
//! domain's documentation, a task's public reference material), held in
//! Splinter's knowledge store ([`Knowledge`]) and searched by word with the
//! task's instruction and the diagnosis. It never holds sealed material: a
//! path under a verifier's `tests`, a `solution`, an `oracle`, a
//! `truth.json`, a run's data or the examiner's private store is refused
//! when named, and left out (and counted) when a named directory holds it.
//!
//! What the teacher was shown is recorded ([`KnowledgeUse`]): the snapshot
//! of every document, the query, and each passage with its source, unit and
//! digest. The intervention is its own (`knowledge_correction`, keyed by
//! the snapshot), so a repair it makes is told apart from the plain fork
//! correction by the same teacher on the same failure: a gain from the
//! knowledge is the difference between the two, the teacher held fixed.
//! The passages are shown beside the student's system prompt and are in no
//! step of the trajectory kept: a student trained on the repair learns the
//! teacher's steps without the material.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use splinter_agent::knowledge::{Knowledge, Passage};
use splinter_agent::sven::memory::{MemoryStore, SqliteMemory};
use splinter_core::digest::Digest;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

use super::super::artifact::sha256_hex;
use super::fork::ForkCorrection;
use super::{Attempted, Case, Intervention, InterventionKind, Notes};

/// Passages one correction is shown.
pub const KNOWLEDGE_PASSAGES: usize = 4;

/// The space the documents are held in.
const SPACE: &str = "repair-knowledge";

/// Path components that are sealed material, never knowledge: a task's
/// verifier and its hidden truth, a reference solution, a generator's
/// oracle, a run's data, the examiner's private store.
const SEALED: [&str; 7] = [
    "tests",
    "solution",
    "oracle",
    "truth.json",
    "bench-private",
    ".bench-data",
    "jobs",
];

/// The file kinds read as documents.
const DOCUMENT_EXTENSIONS: [&str; 3] = ["md", "txt", "rst"];

/// One document of a knowledge base.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Source {
    /// Its id in the store.
    pub id: String,
    /// The file it was read from.
    pub path: PathBuf,
    /// The SHA-256 of its bytes.
    pub sha256: String,
}

/// The knowledge a correction was shown, and where it came from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KnowledgeUse {
    /// The digest of every document the base held: what was searched.
    pub snapshot: String,
    /// Every document the base held.
    pub sources: Vec<Source>,
    /// Files a named directory held that are sealed material, left out.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub withheld: Vec<PathBuf>,
    /// What was searched for.
    pub query: String,
    /// The passages shown, in rank order, each with the SHA-256 of its text.
    pub passages: Vec<ShownPassage>,
}

/// One passage a correction was shown.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShownPassage {
    /// The source document's id.
    pub source: String,
    /// The unit's id in the store.
    pub unit: String,
    /// The revision of the source it was cut from.
    pub revision: String,
    /// Its rank, from 1.
    pub rank: usize,
    /// Its score.
    pub score: f32,
    /// The SHA-256 of its text.
    pub sha256: String,
}

/// Public reference documents, held in Splinter's knowledge store.
pub struct KnowledgeBase {
    knowledge: Knowledge,
    sources: Vec<Source>,
    withheld: Vec<PathBuf>,
    snapshot: String,
}

/// Whether `path` is sealed material: a component of it is one.
#[must_use]
pub fn is_sealed(path: &Path) -> bool {
    path.components().any(|c| {
        let name = c.as_os_str().to_string_lossy();
        SEALED.iter().any(|sealed| name == *sealed)
    })
}

impl KnowledgeBase {
    /// The documents at `paths`: a file is read as it is, a directory for
    /// its documents (`.md`, `.txt`, `.rst`), sealed material left out.
    ///
    /// # Errors
    /// When a named path is sealed material, cannot be read, or the base
    /// would hold no document.
    pub fn load(ctx: &Context, paths: &[PathBuf]) -> Result<Self, OrchestratorError> {
        let refused = |why: String| OrchestratorError::Refused(format!("knowledge: {why}"));
        let mut files = Vec::new();
        let mut withheld = Vec::new();
        for path in paths {
            let path = path
                .canonicalize()
                .map_err(|e| refused(format!("{}: {e}", path.display())))?;
            if is_sealed(&path) {
                return Err(refused(format!(
                    "{} is sealed material (a verifier, a solution, an oracle or hidden truth) \
                     and is never knowledge",
                    path.display()
                )));
            }
            if path.is_dir() {
                documents(&path, &path, &mut files, &mut withheld)
                    .map_err(|e| refused(e.to_string()))?;
            } else {
                files.push(path);
            }
        }
        files.sort();
        files.dedup();
        withheld.sort();
        if files.is_empty() {
            return Err(refused("the paths named hold no document".into()));
        }
        let store: Arc<dyn MemoryStore> = Arc::new(
            SqliteMemory::open_in_memory()
                .map_err(|e| refused(format!("the knowledge store: {e}")))?,
        );
        let knowledge = Knowledge::new(store, SPACE);
        let mut sources = Vec::new();
        for (n, path) in files.iter().enumerate() {
            let bytes =
                std::fs::read(path).map_err(|e| refused(format!("{}: {e}", path.display())))?;
            let text = String::from_utf8(bytes.clone())
                .map_err(|_| refused(format!("{} is not text", path.display())))?;
            let source = Source {
                id: format!("doc-{n}"),
                path: path.clone(),
                sha256: sha256_hex(&bytes),
            };
            let label = path.display().to_string();
            ctx.block_on(knowledge.add_document(&source.id, &label, &text))
                .map_err(|e| refused(format!("{}: {e}", path.display())))?;
            sources.push(source);
        }
        let listing: Vec<String> = sources
            .iter()
            .map(|s| format!("{} {}", s.sha256, s.path.display()))
            .collect();
        let snapshot = Digest::of(listing.join("\n").as_bytes()).to_string();
        Ok(Self {
            knowledge,
            sources,
            withheld,
            snapshot,
        })
    }

    /// The digest of every document the base holds.
    #[must_use]
    pub fn snapshot(&self) -> &str {
        &self.snapshot
    }

    /// The passages for `query`, and the record of them.
    fn retrieve(&self, ctx: &Context, query: &str) -> Result<(Vec<Passage>, KnowledgeUse), String> {
        let passages = ctx
            .block_on(self.knowledge.retrieve(query, KNOWLEDGE_PASSAGES))
            .map_err(|e| format!("the knowledge store: {e}"))?;
        let used = KnowledgeUse {
            snapshot: self.snapshot.clone(),
            sources: self.sources.clone(),
            withheld: self.withheld.clone(),
            query: query.to_string(),
            passages: passages
                .iter()
                .map(|p| ShownPassage {
                    source: p.source.clone(),
                    unit: p.unit.clone(),
                    revision: p.revision.clone(),
                    rank: p.rank,
                    score: p.score,
                    sha256: sha256_hex(p.text.as_bytes()),
                })
                .collect(),
        };
        Ok((passages, used))
    }

    fn path_of(&self, source: &str) -> String {
        self.sources
            .iter()
            .find(|s| s.id == source)
            .map_or_else(|| source.to_string(), |s| s.path.display().to_string())
    }
}

/// The documents under `dir`, sealed ones counted in `withheld`.
fn documents(
    root: &Path,
    dir: &Path,
    files: &mut Vec<PathBuf>,
    withheld: &mut Vec<PathBuf>,
) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        let relative = path.strip_prefix(root).unwrap_or(&path);
        if is_sealed(relative) {
            withheld.push(path);
            continue;
        }
        if path.is_dir() {
            documents(root, &path, files, withheld)?;
        } else if path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| DOCUMENT_EXTENSIONS.contains(&e))
        {
            files.push(path);
        }
    }
    Ok(())
}

/// The fork correction with the knowledge's passages in front of the
/// teacher.
pub struct KnowledgeCorrection<'a> {
    /// The teacher's correction it extends.
    pub fork: &'a ForkCorrection<'a>,
    /// What it retrieves from.
    pub knowledge: &'a KnowledgeBase,
}

impl Intervention for KnowledgeCorrection<'_> {
    fn name(&self) -> &'static str {
        InterventionKind::KnowledgeCorrection.as_str()
    }

    fn actor(&self) -> &str {
        &self.fork.teacher.identity
    }

    fn variant(&self) -> Option<String> {
        Some(self.knowledge.snapshot.clone())
    }

    fn attempt(&self, ctx: &Context, case: &Case<'_>) -> Result<Attempted, String> {
        let reason: Vec<&str> = case
            .diagnosis
            .evidence
            .iter()
            .map(|e| e.excerpt.as_str())
            .collect();
        let query = format!("{}\n{}", case.student.instruction, reason.join("\n"));
        let (passages, used) = self.knowledge.retrieve(ctx, &query)?;
        if passages.is_empty() {
            return Err("the knowledge holds nothing that matches the task".into());
        }
        let mut briefing = String::from(
            "Reference material for this task, from its public documentation (not part of the \
             conversation; use it where it applies):",
        );
        for passage in &passages {
            briefing.push_str(&format!(
                "\n\n[{}] {}\n{}",
                passage.rank,
                self.knowledge.path_of(&passage.source),
                passage.text.trim()
            ));
        }
        let notes = Notes {
            knowledge: Some(used),
            ..Notes::default()
        };
        self.fork
            .attempt_briefed(ctx, case, Some(&briefing), self.name(), notes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sealed_material_is_recognized_by_any_component() {
        for sealed in [
            "/b/tasks/legal/t1/tests/truth.json",
            "/b/tasks/legal/t1/solution/solve.sh",
            "/b/environments/legal/oracle/rules.json",
            "/srv/bench-private/seeds.toml",
            "/b/.bench-data/jobs/j1/trials.jsonl",
        ] {
            assert!(is_sealed(Path::new(sealed)), "{sealed}");
        }
        for public in [
            "/b/environments/legal/README.md",
            "/b/tasks/legal/t1/instruction.md",
        ] {
            assert!(!is_sealed(Path::new(public)), "{public}");
        }
    }
}
