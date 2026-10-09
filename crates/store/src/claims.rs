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

use splinter_core::claim::{Absorption, ClaimId, ClaimSet, ClaimTaskLink, LedgerEntry};
use splinter_core::digest::Digest;

use crate::documents::encode;
use crate::error::StoreError;
use crate::workspace::Workspace;

const CLAIM_SET: &str = "claim_set";
const LEDGER_ENTRY: &str = "claim_ledger_entry";
const TASK_LINK: &str = "claim_task_link";
const ABSORPTION: &str = "claim_absorption";

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
        self.append_all(LEDGER_ENTRY, entries)
    }

    /// Every ledger entry, in the order they were appended.
    pub fn entries(&self) -> Result<Vec<LedgerEntry>, StoreError> {
        self.in_order(LEDGER_ENTRY, "claim ledger entry")
    }

    /// Records which task each claim was made into; a link already recorded
    /// is not recorded again. Returns how many were new.
    pub fn link_tasks(&self, links: &[ClaimTaskLink]) -> Result<usize, StoreError> {
        self.append_all(TASK_LINK, links)
    }

    /// Every task link, in the order recorded.
    pub fn task_links(&self) -> Result<Vec<ClaimTaskLink>, StoreError> {
        self.in_order(TASK_LINK, "claim task link")
    }

    /// Records the release that took each claim in. A claim already taken
    /// in keeps the release that first did. Returns how many were new.
    pub fn absorb(&self, absorptions: &[Absorption]) -> Result<usize, StoreError> {
        let known: Vec<ClaimId> = self.absorptions()?.into_iter().map(|a| a.claim).collect();
        let fresh: Vec<Absorption> = absorptions
            .iter()
            .filter(|a| !known.contains(&a.claim))
            .cloned()
            .collect();
        self.append_all(ABSORPTION, &fresh)
    }

    /// Every claim taken in and the release that first did, in the order
    /// recorded.
    pub fn absorptions(&self) -> Result<Vec<Absorption>, StoreError> {
        self.in_order(ABSORPTION, "claim absorption")
    }

    fn append_all<T: serde::Serialize>(
        &self,
        kind: &'static str,
        items: &[T],
    ) -> Result<usize, StoreError> {
        let mut appended = 0;
        for item in items {
            let (id, _) = encode(kind, item)?;
            if !self.workspace.has_document(kind, &id)? {
                self.workspace.put_document(kind, item)?;
                appended += 1;
            }
        }
        Ok(appended)
    }

    fn in_order<T: serde::de::DeserializeOwned + serde::Serialize>(
        &self,
        kind: &'static str,
        what: &'static str,
    ) -> Result<Vec<T>, StoreError> {
        self.workspace
            .documents_in_order(kind)?
            .iter()
            .map(|id| {
                self.workspace
                    .get_document(kind, id)?
                    .ok_or_else(|| StoreError::Rejected {
                        what,
                        reason: format!("{id} is listed and cannot be read"),
                    })
            })
            .collect()
    }
}
