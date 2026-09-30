// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems whose every answer traces
// back to the model release that gave it, for its clients. If your team
// needs expertise in model provenance or auditable AI, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The answers `ask` gave, recorded so an answer traces back to the model
//! and the release that gave it.
//!
//! ```text
//! <root>/answers/<hex>.json    one answer in canonical JSON; <hex> is its digest
//! ```
//!
//! An [`AnswerId`] is the digest of the record's canonical form (see
//! `splinter_store::digest`), the time it was asked included, so asking the
//! same question twice records two answers. A record is written once and
//! checked against its address on every read. Every record carries
//! [`ANSWER_FORMAT`]; a later format adds fields beside it rather than
//! changing what these mean.

use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use splinter_store::digest::{canonical_json, Digest};
use splinter_store::experiences::StoreError;
use splinter_store::source::SourceId;
use splinter_store::{write_once, StateRoot};

use crate::error::{io, CampaignError};
use crate::release::ReleaseId;

/// The `format` every answer record carries.
pub const ANSWER_FORMAT: &str = "splinter-answer-v1";

/// An answer's id: the digest of its record's canonical form.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AnswerId(pub Digest);

impl AnswerId {
    /// `sha256:<hex>`.
    #[must_use]
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl std::fmt::Display for AnswerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// One answer, and what gave it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnswerRecord {
    /// Always [`ANSWER_FORMAT`].
    pub format: String,
    /// The question, as asked.
    pub question: String,
    /// The model's answer.
    pub answer: String,
    /// The identity of the model that answered.
    pub model: String,
    /// The model reference it was asked through (`policy:default`, ...).
    pub policy: String,
    /// The release a `policy:` reference resolved to; `None` for any other
    /// reference, and for the base before any release.
    pub release: Option<ReleaseId>,
    /// The source shown with the question, if any.
    pub open_book: Option<SourceId>,
    /// When it was asked, from the injected clock.
    pub asked_at: String,
}

/// The recorded answers under one state root.
#[derive(Clone, Debug)]
pub struct AnswerStore {
    dir: PathBuf,
}

impl AnswerStore {
    /// The store under `root`; nothing is created until something is
    /// written.
    #[must_use]
    pub fn open(root: &StateRoot) -> Self {
        Self {
            dir: root.answers(),
        }
    }

    fn path(&self, id: &AnswerId) -> PathBuf {
        self.dir.join(format!("{}.json", id.0.hex()))
    }

    /// Records `record` and returns its id; recording the same record
    /// again is a no-op, and an existing file that no longer matches its
    /// address is reported as corrupt rather than replaced.
    pub fn put(&self, record: &AnswerRecord) -> Result<AnswerId, CampaignError> {
        let bytes = canonical_json(record).map_err(|source| CampaignError::Json {
            what: "answer record".into(),
            source,
        })?;
        let id = AnswerId(Digest::of(&bytes));
        let path = self.path(&id);
        if !write_once(&path, &bytes).map_err(io(&path))? {
            self.get(&id)?;
        }
        Ok(id)
    }

    /// The answer `id`, verified against its address.
    pub fn get(&self, id: &AnswerId) -> Result<AnswerRecord, CampaignError> {
        let path = self.path(id);
        if !path.is_file() {
            return Err(CampaignError::NotFound {
                what: "answer",
                id: id.to_string(),
            });
        }
        let bytes = fs::read(&path).map_err(io(&path))?;
        let found = Digest::of(&bytes);
        if found != id.0 {
            return Err(CampaignError::Store(StoreError::Corrupt {
                path,
                expected: id.0.clone(),
                found,
            }));
        }
        serde_json::from_slice(&bytes).map_err(|source| CampaignError::Json {
            what: path.display().to_string(),
            source,
        })
    }

    /// Every recorded answer's id, in id order (without verifying them).
    pub fn list(&self) -> Result<Vec<AnswerId>, CampaignError> {
        let entries = match fs::read_dir(&self.dir) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(io(&self.dir)(e)),
        };
        let mut ids = Vec::new();
        for entry in entries {
            let name = entry.map_err(io(&self.dir))?.file_name();
            // Only `<64 hex>.json` is a record; a write-once temporary is not.
            let digest = name
                .to_str()
                .and_then(|n| n.strip_suffix(".json"))
                .and_then(|hex| Digest::parse(&format!("sha256:{hex}")).ok());
            if let Some(digest) = digest {
                ids.push(AnswerId(digest));
            }
        }
        ids.sort();
        Ok(ids)
    }
}
