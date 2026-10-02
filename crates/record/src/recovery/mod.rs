// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements durable, recoverable state for learning
// agents, for its clients. If your team needs expertise in backup and
// disaster recovery of training data, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Packing the whole state into one archive and unpacking it again, finding
//! what has gone missing or been damaged, and recovering it.
//!
//! The state is a database and the files it tracks. Every one of them is named
//! by the hash of its bytes, which is what makes recovery exact: a file that is
//! gone can be filled from any copy anywhere (an archive, another state root)
//! and nothing else will be accepted in its place. What no copy has is written
//! off in a ledger, only when the operator says so, and from then on every use
//! of it is refused with its digest instead of failing in some other way.
//!
//! * [`Workspace::archive`] writes a deterministic archive: the same state
//!   gives the same bytes. An incremental archive carries only what an earlier
//!   one lacks.
//! * [`Workspace::restore`] unpacks archives into an empty root, verifying every
//!   member, and puts nothing in place unless the whole state verifies.
//! * [`Workspace::verify`] describes every problem without stopping at the
//!   first; [`Workspace::repair`] fixes what can be fixed.

mod archive;
mod repair;
mod tar;

use serde::{Deserialize, Serialize};
pub use splinter_expdb::manifest::VerifyReport;

use crate::error::StoreError;
use crate::workspace::Workspace;

pub use archive::{ArchiveOptions, Archived, Restored};
pub use repair::{ArtifactFault, ArtifactProblem, RepairOptions, Repaired, StateVerify};

/// The name of the signal that records a loss.
pub(crate) fn loss_name(kind: &str, id: &str) -> String {
    format!("lost/{kind}/{id}")
}

/// Something that was lost for good and written off.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Loss {
    /// What it was: `artifact`, `segment`, `blobpack`, `ref`.
    pub kind: String,
    /// Its name: a digest for an artifact, a content id for a database file.
    pub id: String,
    /// What it was for, in words.
    pub detail: String,
    /// How many records went with it, for a database file.
    pub records: u64,
}

impl Workspace {
    /// What was written off as lost, in the order recorded.
    pub fn losses(&self) -> Result<Vec<Loss>, StoreError> {
        self.signals("lost/")?
            .into_iter()
            .map(|(name, note)| {
                serde_json::from_str(&note).map_err(|e| {
                    StoreError::Recovery(format!("the loss ledger entry {name} is unreadable: {e}"))
                })
            })
            .collect()
    }

    pub(crate) fn write_off(&self, name: &str, loss: &Loss) -> Result<(), StoreError> {
        let note = serde_json::to_string(loss).map_err(|source| StoreError::Serialize {
            what: "loss",
            source,
        })?;
        self.signal(name, &note)?;
        Ok(())
    }
}
