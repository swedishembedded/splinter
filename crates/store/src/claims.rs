// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements durable, content-addressed experience
// stores for learning agents, for its clients. If your team needs expertise
// in training-data lineage or crash-safe storage, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Claims: what an extraction proposed from sessions, and the ledger of
//! rulings on it.
//!
//! A [`ClaimSet`] is stored once under the digest of its canonical form. A
//! ledger entry is stored once too, and the ledger is its entries in the
//! order they were first stored: a ruling is never rewritten or removed, and
//! what supersedes what is read from the entries that say so.

use splinter_core::claim::{ClaimSet, LedgerEntry};
use splinter_core::digest::Digest;

use crate::documents::encode;
use crate::error::StoreError;
use crate::workspace::Workspace;

const CLAIM_SET: &str = "claim_set";
const LEDGER_ENTRY: &str = "claim_ledger_entry";

/// The claim sets and the claim ledger.
#[derive(Clone, Debug)]
pub struct ClaimStore {
    workspace: Workspace,
}

impl ClaimStore {
    /// The claim store over `workspace`.
    #[must_use]
    pub fn new(workspace: &Workspace) -> Self {
        Self {
            workspace: workspace.clone(),
        }
    }

    /// Stores `set` and returns its address; the same set again is the same
    /// address and writes nothing.
    pub fn put_set(&self, set: &ClaimSet) -> Result<Digest, StoreError> {
        self.workspace.put_document(CLAIM_SET, set)
    }

    /// The set stored under `id`, checked against its address.
    pub fn get_set(&self, id: &Digest) -> Result<ClaimSet, StoreError> {
        self.workspace
            .get_document(CLAIM_SET, id)?
            .ok_or_else(|| StoreError::UnknownClaimSet(id.clone()))
    }

    /// Every stored claim set's address, in address order.
    pub fn list_sets(&self) -> Result<Vec<Digest>, StoreError> {
        self.workspace.document_ids(CLAIM_SET)
    }

    /// Appends `entries` to the ledger; one already in it is not appended
    /// again. Returns how many were new.
    pub fn append(&self, entries: &[LedgerEntry]) -> Result<usize, StoreError> {
        let mut appended = 0;
        for entry in entries {
            let (id, _) = encode(LEDGER_ENTRY, entry)?;
            if !self.workspace.has_document(LEDGER_ENTRY, &id)? {
                self.workspace.put_document(LEDGER_ENTRY, entry)?;
                appended += 1;
            }
        }
        Ok(appended)
    }

    /// Every ledger entry, in the order they were appended.
    pub fn entries(&self) -> Result<Vec<LedgerEntry>, StoreError> {
        self.workspace
            .documents_in_order(LEDGER_ENTRY)?
            .iter()
            .map(|id| {
                self.workspace
                    .get_document(LEDGER_ENTRY, id)?
                    .ok_or_else(|| StoreError::Rejected {
                        what: "claim ledger entry",
                        reason: format!("{id} is listed and cannot be read"),
                    })
            })
            .collect()
    }
}
