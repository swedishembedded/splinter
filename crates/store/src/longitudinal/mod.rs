// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for importing sensitive
// longitudinal datasets as immutable, traceable episodes, for its clients. If
// your team needs expertise in privacy-preserving data pipelines you can
// procure our services by sending an email to info@swedishembedded.com.

//! Longitudinal datasets kept as episodes of the experience database: one
//! immutable, content-addressed episode per participant ([`episode`] says how
//! it is laid out), imported from a `timeline-v1`-shaped JSON-lines file.
//!
//! An import reads the file twice, one line at a time, so memory does not
//! grow with the file (only with the number of participants, one key each,
//! to refuse a participant who appears twice). The first pass validates
//! every row, counts rows and participants and digests the file; nothing is
//! written unless every row is acceptable. The second writes the episodes,
//! each in one commit with the marker that makes it findable by its address,
//! skips the ones already present (so importing again adds nothing and a
//! crashed import resumes where it stopped), and checks the file did not
//! change underneath it and that the store grew by exactly what was added.

mod episode;

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use serde_json::json;
use splinter_core::digest::Digest;
use splinter_core::longitudinal::{History, ParticipantKey, ParticipantKeying, Record};
use splinter_core::terms::Terms;
use splinter_expdb::model::Entity;
use splinter_expdb::RecordId;

pub use episode::{calendar_clock, subject_clock};

use crate::address::digest_of;
use crate::error::StoreError;
use crate::workspace::Workspace;

/// The class of the entity that makes an episode findable by its address.
const EPISODE: &str = "longitudinal_episode";

/// The longest line an import reads: a record larger than this is refused
/// rather than buffered.
const MAX_LINE_BYTES: u64 = 16 << 20;

/// What an import is told about the file it reads.
#[derive(Clone, Debug)]
pub struct ImportSpec {
    /// The source dataset's id: part of every participant key and of every
    /// item's provenance.
    pub dataset: String,
    /// The terms the dataset came under; stated by the caller, never assumed.
    pub terms: Terms,
    /// The secret the opaque participant keys are derived with.
    pub keying: ParticipantKeying,
}

/// What an import did and verified.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportReport {
    /// The dataset imported.
    pub dataset: String,
    /// The digest of the file's bytes.
    pub file: Digest,
    /// Non-blank rows in the file.
    pub rows: u64,
    /// Distinct participants in the file; equal to `rows`, since a repeat is refused.
    pub participants: u64,
    /// Episodes this import wrote.
    pub added: u64,
    /// Episodes that were already in the store.
    pub already_present: u64,
}

/// What names an episode in its marker entity.
#[derive(Serialize, Deserialize)]
struct Marker {
    episode: String,
    participant: ParticipantKey,
    dataset: String,
}

/// The longitudinal episodes of a workspace.
#[derive(Clone, Debug)]
pub struct LongitudinalStore {
    workspace: Workspace,
}

fn refused(reason: String) -> StoreError {
    StoreError::Rejected {
        what: "longitudinal import",
        reason,
    }
}

/// Calls `each` with the number and text of every non-blank line of `path`,
/// reading one line at a time, and returns the digest of the whole file.
fn for_each_row(
    path: &Path,
    mut each: impl FnMut(u64, &str) -> Result<(), StoreError>,
) -> Result<Digest, StoreError> {
    let file = std::fs::File::open(path)
        .map_err(|e| refused(format!("cannot open {}: {e}", path.display())))?;
    let mut reader = BufReader::new(file);
    let mut hasher = blake3::Hasher::new();
    let mut buffer = Vec::new();
    let mut number = 0u64;
    loop {
        buffer.clear();
        let read = (&mut reader)
            .take(MAX_LINE_BYTES + 1)
            .read_until(b'\n', &mut buffer)
            .map_err(|e| refused(format!("cannot read {}: {e}", path.display())))?;
        if read == 0 {
            break;
        }
        number += 1;
        if buffer.len() as u64 > MAX_LINE_BYTES {
            return Err(refused(format!(
                "{}: line {number} is longer than {MAX_LINE_BYTES} bytes",
                path.display()
            )));
        }
        hasher.update(&buffer);
        let text = std::str::from_utf8(&buffer)
            .map_err(|_| refused(format!("{}: line {number} is not UTF-8", path.display())))?;
        if !text.trim().is_empty() {
            each(number, text.trim())?;
        }
    }
    Digest::from_content_hex(hasher.finalize().to_hex().as_ref())
        .map_err(|e| refused(format!("{}: {e}", path.display())))
}

impl LongitudinalStore {
    /// The episodes kept in `workspace`.
    #[must_use]
    pub fn new(workspace: &Workspace) -> Self {
        Self {
            workspace: workspace.clone(),
        }
    }

    fn count(&self) -> Result<usize, StoreError> {
        Ok(self.workspace.ids_of(EPISODE)?.len())
    }

    /// Imports the JSON-lines file at `path` as one episode per participant.
    /// See the module documentation for what is checked; a file that cannot
    /// be imported whole imports nothing.
    pub fn import_jsonl(&self, path: &Path, spec: &ImportSpec) -> Result<ImportReport, StoreError> {
        let shown = path.display();
        let mut seen: HashMap<ParticipantKey, u64> = HashMap::new();
        let mut rows = 0u64;
        let file = for_each_row(path, |line, text| {
            let record = Record::from_json_line(text)
                .map_err(|e| refused(format!("{shown}: line {line}: {e}")))?;
            let key = spec.keying.participant(&spec.dataset, &record.subject_id);
            if let Some(first) = seen.insert(key, line) {
                return Err(refused(format!(
                    "{shown}: line {line} repeats the participant of line {first}"
                )));
            }
            rows += 1;
            Ok(())
        })?;
        if rows == 0 {
            return Err(refused(format!("{shown} holds no rows")));
        }
        let participants = seen.len() as u64;
        drop(seen);

        let before = self.count()?;
        let (mut added, mut already_present) = (0u64, 0u64);
        let batch = self.workspace.batch();
        let again = for_each_row(path, |line, text| {
            let record = Record::from_json_line(text)
                .map_err(|e| refused(format!("{shown}: line {line}: {e}")))?;
            let history = History::from_record(
                record,
                &spec.keying,
                &spec.dataset,
                &file,
                line,
                &spec.terms,
            )
            .map_err(|e| refused(format!("{shown}: line {line}: {e}")))?;
            if self.write_episode(&history)? {
                added += 1;
            } else {
                already_present += 1;
            }
            Ok(())
        })?;
        batch.commit()?;
        if again != file {
            return Err(refused(format!(
                "{shown} changed while it was being imported"
            )));
        }
        if added + already_present != rows {
            return Err(refused(format!(
                "{shown} has {rows} rows but {} episodes were accounted for",
                added + already_present
            )));
        }
        let after = self.count()?;
        if after != before + added as usize {
            return Err(refused(format!(
                "the store held {before} episodes before {shown} and {after} after, but {added} were added"
            )));
        }
        Ok(ImportReport {
            dataset: spec.dataset.clone(),
            file,
            rows,
            participants,
            added,
            already_present,
        })
    }

    /// Writes the episode of `history` unless one with its address is there;
    /// whether it was written. The episode and its marker are one commit.
    fn write_episode(&self, history: &History) -> Result<bool, StoreError> {
        let address = history
            .address()
            .map_err(|e| refused(format!("cannot address a history: {e}")))?;
        if self.workspace.has(EPISODE, &address)? {
            return Ok(false);
        }
        let key = crate::workspace::content_id(&address)?;
        self.workspace.write(|s| {
            let episode = s.with_collector(|c| episode::write_history(c, history, &address))?;
            let marker = Marker {
                episode: episode.to_string(),
                participant: history.participant.clone(),
                dataset: history.provenance.dataset.clone(),
            };
            let value =
                serde_json::to_value(&marker).map_err(|source| splinter_expdb::Error::Encode {
                    what: "episode marker",
                    source,
                })?;
            s.put_entity(&Entity::keyed(EPISODE, key, value))?;
            Ok(())
        })?;
        Ok(true)
    }

    /// The address of every episode, in address order.
    pub fn addresses(&self) -> Result<Vec<Digest>, StoreError> {
        self.workspace.ids_of(EPISODE)
    }

    /// The episodes as `(address, episode record)`, in address order.
    fn episodes(&self) -> Result<Vec<(Digest, RecordId)>, StoreError> {
        let stored = self.workspace.read_or_default(|s| {
            s.snapshot()?;
            s.entities(EPISODE)
        })?;
        let mut found = Vec::with_capacity(stored.len());
        for item in stored {
            let marker: Marker = serde_json::from_value(item.entity.value).map_err(|e| {
                StoreError::UndecodableObject {
                    what: "an episode marker".into(),
                    reason: e.to_string(),
                }
            })?;
            let record = RecordId::from_str(&marker.episode)?;
            found.push((digest_of(item.id), record));
        }
        found.sort();
        Ok(found)
    }

    /// Calls `each` with every episode's address and history, in address
    /// order, reading them a bounded batch at a time.
    pub fn for_each_history(
        &self,
        mut each: impl FnMut(&Digest, History) -> Result<(), StoreError>,
    ) -> Result<(), StoreError> {
        for batch in self.episodes()?.chunks(1024) {
            let histories = self.workspace.read(|s| {
                let snapshot = s.snapshot()?;
                batch
                    .iter()
                    .map(|(_, record)| episode::read_history(snapshot, *record))
                    .collect::<splinter_expdb::Result<Vec<_>>>()
            })?;
            for ((address, _), history) in batch.iter().zip(histories) {
                let computed = history
                    .address()
                    .map_err(|e| refused(format!("cannot address episode {address}: {e}")))?;
                if &computed != address {
                    return Err(StoreError::UndecodableObject {
                        what: format!("episode {address}"),
                        reason: format!("its content addresses as {computed}"),
                    });
                }
                each(address, history)?;
            }
        }
        Ok(())
    }

    /// Records that the dataset `dataset` was projected from the episodes at
    /// `episodes`, so lineage from a release reaches them, and pins the
    /// database as it is. Recording the same dataset again changes nothing.
    pub fn record_dataset(
        &self,
        dataset: &Digest,
        recipe: serde_json::Value,
        records: u64,
        episodes: &[Digest],
    ) -> Result<(), StoreError> {
        let known: HashMap<Digest, RecordId> = self.episodes()?.into_iter().collect();
        let mut attempts = Vec::with_capacity(episodes.len());
        for address in episodes {
            attempts.push(*known.get(address).ok_or_else(|| StoreError::Rejected {
                what: "dataset lineage",
                reason: format!("episode {address} is not in the store"),
            })?);
        }
        let lineage = crate::lineage::DatasetLineage {
            recipe: json!({ "view": "timeline-v1", "recipe": recipe }),
            records,
        };
        self.workspace
            .record_dataset_from_attempts(dataset, &lineage, attempts)
    }
}
