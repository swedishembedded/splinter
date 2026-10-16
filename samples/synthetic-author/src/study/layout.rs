// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the measurement of fine-tuned language
// models for its clients. If your team needs expertise in pre-registering a
// replication study so that no choice can follow its results, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Where a study keeps what its stages write, and the manifests that tie each
//! piece of work to the protocol it was done under.
//!
//! ```text
//! STUDY/protocol.json                       the frozen protocol
//! STUDY/authors/<seed>/                     one author's corpus (a synthetic-author cache)
//! STUDY/authors/<seed>/study.json           the author's manifest
//! STUDY/authors/<seed>/exam-items.jsonl     the frozen examination, as the models are asked it
//! STUDY/authors/<seed>/exam/<slot>/         the examination's outcomes, one slot per adapter
//! STUDY/authors/<seed>/runs/<arm>-s<seed>/  one training run and its manifest
//! ```

use std::path::{Path, PathBuf};

use anyhow::Context as _;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use splinter_sdk::vocabulary::digest::Digest;

use super::protocol::{self, AuthorPlan, ModelRoles, Protocol};
use crate::binding::write_whole;
use crate::cache::{Cache, Usage};
use crate::dataset::Briefings;
use crate::exam::Version;
use crate::survivors::Survivors;
use crate::training::Settings;

/// The author's manifest, in the author's directory.
pub const AUTHOR_MANIFEST: &str = "study.json";
/// The frozen examination, as the models are asked it, in the author's
/// directory.
pub const EXAM_ITEMS: &str = "exam-items.jsonl";

/// What the corpus of one author came to.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusRecord {
    /// Articles planned.
    pub articles_planned: usize,
    /// Of those, articles kept.
    pub articles_kept: usize,
    /// Cases read out of the kept articles.
    pub cases_read: usize,
    /// Of those, decisions that passed.
    pub decisions_kept: usize,
    /// What the writer and the gate reader cost, as estimated.
    pub usage: Usage,
}

/// The records one setting of the briefings made.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatasetRecord {
    /// The setting.
    pub briefings: Briefings,
    /// Records written.
    pub records: usize,
    /// Decisions that made no record.
    pub without_record: usize,
    /// The digest of the file of records.
    pub digest: String,
}

/// The examination of one author, drawn and hashed before any adapter is
/// trained on the corpus.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrozenExam {
    /// Which examination.
    pub version: Version,
    /// Articles the corpus was planned with.
    pub articles: usize,
    /// Items of each kind of difficulty.
    pub per_cell: usize,
    /// Items in all.
    pub items: usize,
    /// The digest of the items: what every arm's outcomes are bound to.
    pub digest: String,
    /// The digest of the file that shows the items.
    pub file_digest: String,
}

/// What the corpus stage of one author wrote.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AuthorManifest {
    /// The digest of the protocol this was done under.
    pub protocol_digest: String,
    /// The author's seed.
    pub author_seed: u64,
    /// The author's name.
    pub author_name: String,
    /// The models that wrote, read and mined.
    pub models: ModelRoles,
    /// The revision of the code that did it.
    pub code_revision: String,
    /// What the corpus came to.
    pub corpus: CorpusRecord,
    /// The records made from the decisions, one per setting.
    pub datasets: Vec<DatasetRecord>,
    /// What the gates took out of the corpus.
    pub survivors: Survivors,
    /// The examination, frozen.
    pub exam: FrozenExam,
}

/// One adapter, trained: what made it and what came of the training. The
/// run's directory holds this as `run.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunManifest {
    /// The digest of the protocol this was done under.
    pub protocol_digest: String,
    /// The author's seed.
    pub author_seed: u64,
    /// The seed this adapter was trained with.
    pub training_seed: u64,
    /// The digest of the base checkpoint.
    pub base_digest: String,
    /// The digest of the examination frozen for the author before training.
    pub exam_digest: String,
    /// The adapter file, relative to the study directory.
    pub adapter: PathBuf,
    /// The digest of the adapter file: what its outcomes are bound to.
    pub adapter_digest: String,
    /// Every setting of the training, including its seed and the digest of the
    /// records.
    pub settings: Settings,
    /// Records trained on.
    pub records: usize,
    /// Held-out loss and token accuracy before training.
    pub held_out_before: serde_json::Value,
    /// Held-out loss and token accuracy after training.
    pub held_out_after: serde_json::Value,
    /// Seconds the training took.
    pub seconds: u64,
    /// The revision of the code that trained it.
    pub code_revision: String,
}

/// The file in a run's directory that holds its [`RunManifest`].
pub const RUN_MANIFEST: &str = "run.json";

/// A study directory with its protocol read and checked.
pub struct Study {
    root: PathBuf,
    protocol: Protocol,
    digest: Digest,
}

impl Study {
    /// The study frozen in `root`.
    ///
    /// # Errors
    /// There is no valid protocol there.
    pub fn open(root: &Path) -> anyhow::Result<Self> {
        let frozen = protocol::read(root)?;
        Ok(Self {
            root: root.to_path_buf(),
            protocol: frozen.protocol,
            digest: frozen.digest,
        })
    }

    /// The study directory.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The frozen protocol.
    #[must_use]
    pub fn protocol(&self) -> &Protocol {
        &self.protocol
    }

    /// The digest every manifest of the study records.
    #[must_use]
    pub fn digest(&self) -> &Digest {
        &self.digest
    }

    /// The plan of the author with seed `seed`.
    ///
    /// # Errors
    /// The protocol has no such author.
    pub fn author_plan(&self, seed: u64) -> anyhow::Result<&AuthorPlan> {
        self.protocol
            .authors
            .iter()
            .find(|a| a.seed == seed)
            .with_context(|| format!("author seed {seed} is not in the protocol"))
    }

    /// The directory of the author with seed `seed`.
    #[must_use]
    pub fn author_dir(&self, seed: u64) -> PathBuf {
        self.root.join("authors").join(seed.to_string())
    }

    /// The corpus of the author with seed `seed`, created if absent.
    ///
    /// # Errors
    /// The author is not in the protocol or the directory cannot be made.
    pub fn cache(&self, seed: u64) -> anyhow::Result<Cache> {
        self.author_plan(seed)?;
        Cache::open(self.author_dir(seed))
    }

    /// The directory of one training run: the adapter trained on `briefings`
    /// with `training_seed`.
    #[must_use]
    pub fn run_dir(&self, author: u64, briefings: Briefings, training_seed: u64) -> PathBuf {
        self.author_dir(author)
            .join("runs")
            .join(format!("tuned{}-s{training_seed}", briefings.suffix()))
    }

    /// The manifest of the training run of the adapter trained on `briefings`
    /// with `training_seed`, if it has finished.
    ///
    /// # Errors
    /// The manifest cannot be read, or was written under another protocol.
    pub fn run_manifest(
        &self,
        author: u64,
        briefings: Briefings,
        training_seed: u64,
    ) -> anyhow::Result<Option<RunManifest>> {
        let path = self
            .run_dir(author, briefings, training_seed)
            .join(RUN_MANIFEST);
        let manifest: Option<RunManifest> = self.read_manifest(&path)?;
        if let Some(m) = &manifest {
            self.ensure_same_protocol(&path, &m.protocol_digest)?;
        }
        Ok(manifest)
    }

    /// Writes the manifest of a finished run, whole or not at all.
    ///
    /// # Errors
    /// The file cannot be written.
    pub fn write_run_manifest(
        &self,
        briefings: Briefings,
        manifest: &RunManifest,
    ) -> anyhow::Result<()> {
        write_whole(
            &self
                .run_dir(manifest.author_seed, briefings, manifest.training_seed)
                .join(RUN_MANIFEST),
            &serde_json::to_vec_pretty(manifest)?,
        )
    }

    /// The manifest the corpus stage wrote for `author`, if it has.
    ///
    /// # Errors
    /// The manifest cannot be read, or was written under another protocol.
    pub fn author_manifest(&self, author: u64) -> anyhow::Result<Option<AuthorManifest>> {
        let path = self.author_dir(author).join(AUTHOR_MANIFEST);
        let manifest: Option<AuthorManifest> = self.read_manifest(&path)?;
        if let Some(m) = &manifest {
            // The corpus stage comes before any training, so a corpus written
            // under the protocol an amendment replaced is still this study's.
            if !self
                .protocol
                .earlier_digests()
                .any(|d| d == m.protocol_digest)
            {
                self.ensure_same_protocol(&path, &m.protocol_digest)?;
            }
        }
        Ok(manifest)
    }

    /// Whether any adapter of the study has finished training.
    ///
    /// # Errors
    /// A run manifest cannot be read.
    pub fn any_trained(&self) -> anyhow::Result<bool> {
        for author in &self.protocol.authors {
            let runs = self.author_dir(author.seed).join("runs");
            let Ok(entries) = std::fs::read_dir(&runs) else {
                continue;
            };
            for entry in entries {
                let entry = entry.with_context(|| format!("reading {}", runs.display()))?;
                if entry.path().join(RUN_MANIFEST).exists() {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    /// The manifest of `author`, which the corpus stage must have written.
    ///
    /// # Errors
    /// As [`Self::author_manifest`], or there is none: the message says to run
    /// `study corpus` first.
    pub fn require_author_manifest(&self, author: u64) -> anyhow::Result<AuthorManifest> {
        self.author_manifest(author)?.with_context(|| {
            format!(
                "author {author} has no {AUTHOR_MANIFEST}: run `study corpus` for it first, \
                 which freezes the examination before anything is trained"
            )
        })
    }

    /// Writes the manifest of its author, whole or not at all.
    ///
    /// # Errors
    /// The file cannot be written.
    pub fn write_author_manifest(&self, manifest: &AuthorManifest) -> anyhow::Result<()> {
        write_whole(
            &self.author_dir(manifest.author_seed).join(AUTHOR_MANIFEST),
            &serde_json::to_vec_pretty(manifest)?,
        )
    }

    /// Reads the JSON document at `path`, if there is one.
    ///
    /// # Errors
    /// It cannot be read or parsed.
    pub fn read_manifest<T: DeserializeOwned>(&self, path: &Path) -> anyhow::Result<Option<T>> {
        match std::fs::read_to_string(path) {
            Ok(text) => Ok(Some(
                serde_json::from_str(&text)
                    .with_context(|| format!("parsing {}", path.display()))?,
            )),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
        }
    }

    /// Checks that work recorded under `recorded`, found in `path`, was done
    /// under the protocol that is frozen now.
    ///
    /// # Errors
    /// The digests differ; both are named.
    pub fn ensure_same_protocol(&self, path: &Path, recorded: &str) -> anyhow::Result<()> {
        anyhow::ensure!(
            recorded == self.digest.as_str(),
            "{} was written under protocol {recorded} and the protocol frozen in {} is {}; \
             a study is never continued under another protocol",
            path.display(),
            self.root.display(),
            self.digest,
        );
        Ok(())
    }
}
