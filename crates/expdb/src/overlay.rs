// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The replay overlay: what a learner currently thinks is important.
//!
//! Sampling priority, error estimates and how often a record has been used
//! change with every training step. They are kept in a small per-trainer
//! file, replaced atomically, and never written into the canonical segments,
//! which stay exactly as the experience was recorded.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::backend::{Key, Kind};
use crate::database::Database;
use crate::error::{Error, Result};
use crate::id::RecordId;
use crate::train::Rng;

/// The weight of a record nobody has prioritised.
pub const DEFAULT_PRIORITY: f64 = 1.0;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct Entry {
    priority: Option<f64>,
    td_error: Option<f64>,
    times_sampled: u64,
}

/// One trainer's view of how much each record matters.
#[derive(Debug, Clone)]
pub struct ReplayOverlay {
    trainer: String,
    entries: BTreeMap<RecordId, Entry>,
}

impl ReplayOverlay {
    /// Loads a trainer's overlay, empty if it has none yet.
    pub fn load(db: &Database, trainer: &str) -> Result<Self> {
        let key = Key::new(Kind::Overlay, trainer)?;
        let entries = if db.backend().exists(&key)? {
            let bytes = db.backend().read(&key)?;
            serde_json::from_slice(&bytes).map_err(|source| Error::Decode {
                what: format!("overlay of `{trainer}`"),
                source,
            })?
        } else {
            BTreeMap::new()
        };
        Ok(Self {
            trainer: trainer.to_owned(),
            entries,
        })
    }

    /// Writes the overlay, atomically replacing the previous one.
    pub fn save(&self, db: &Database) -> Result<()> {
        let bytes = serde_json::to_vec(&self.entries).map_err(|source| Error::Encode {
            what: "overlay",
            source,
        })?;
        db.backend()
            .replace(&Key::new(Kind::Overlay, &self.trainer)?, &bytes)
    }

    /// Sets how much a record should be sampled.
    pub fn set_priority(&mut self, id: RecordId, priority: f64) {
        self.entries.entry(id).or_default().priority = Some(priority);
    }

    /// Notes that a record was used and what its error estimate was.
    pub fn record_sample(&mut self, id: RecordId, td_error: f64) {
        let entry = self.entries.entry(id).or_default();
        entry.times_sampled += 1;
        entry.td_error = Some(td_error);
    }

    /// A record's priority, if one was set.
    pub fn priority(&self, id: RecordId) -> Option<f64> {
        self.entries.get(&id).and_then(|e| e.priority)
    }

    /// A record's last error estimate, if it was sampled.
    pub fn td_error(&self, id: RecordId) -> Option<f64> {
        self.entries.get(&id).and_then(|e| e.td_error)
    }

    /// How often a record has been sampled.
    pub fn times_sampled(&self, id: RecordId) -> u64 {
        self.entries.get(&id).map_or(0, |e| e.times_sampled)
    }

    /// `k` of the candidates drawn without replacement with probability in
    /// proportion to priority, fixed by `seed`.
    pub fn sample(&self, candidates: &[RecordId], k: usize, seed: u64) -> Vec<RecordId> {
        let mut rng = Rng::new(seed);
        let mut keyed: Vec<(f64, RecordId)> = candidates
            .iter()
            .map(|id| {
                let weight = self
                    .priority(*id)
                    .filter(|p| *p > 0.0)
                    .unwrap_or(DEFAULT_PRIORITY);
                (rng.unit().powf(1.0 / weight), *id)
            })
            .collect();
        keyed.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        keyed.into_iter().take(k).map(|(_, id)| id).collect()
    }
}
