// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements durable, content-addressed experience
// stores for learning agents, for its clients. If your team needs expertise
// in training-data lineage or crash-safe storage, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Documents: records Splinter keeps by their content address, such as an
//! answer, a release manifest, a dataset manifest or a frozen suite.
//!
//! A document is a value of any serializable type, stored once under the digest
//! of its canonical JSON. Storing the same document again changes nothing, and
//! reading one checks that it still hashes to the address it is stored under.

use serde::de::DeserializeOwned;
use serde::Serialize;
use splinter_expdb::model::Entity;

use crate::error::StoreError;
use crate::workspace::{content_id, put_spilling, Workspace};
use splinter_core::digest::{canonical_json, Digest};

impl Workspace {
    /// Stores `document` of `class` and returns its address. Storing the same
    /// document again writes nothing.
    pub fn put_document<T: Serialize>(
        &self,
        class: &str,
        document: &T,
    ) -> Result<Digest, StoreError> {
        let (id, entity) = encode(class, document)?;
        if !self.has(class, &id)? {
            self.write(|s| put_spilling(s, entity.clone()).map(|_| ()))?;
        }
        Ok(id)
    }

    /// The document of `class` stored under `id`, checked against its address;
    /// `None` when there is none.
    pub fn get_document<T: DeserializeOwned + Serialize>(
        &self,
        class: &str,
        id: &Digest,
    ) -> Result<Option<T>, StoreError> {
        let Some(entity) = self.find(class, id)? else {
            return Ok(None);
        };
        let document: T =
            serde_json::from_value(entity.value).map_err(|e| StoreError::UndecodableObject {
                what: format!("{class} {id}"),
                reason: e.to_string(),
            })?;
        let found = digest_of(&document)?;
        if found != *id {
            return Err(StoreError::Altered {
                what: format!("{class} {id}"),
                expected: id.clone(),
                found,
            });
        }
        Ok(Some(document))
    }

    /// Whether a document of `class` is stored under `id`.
    pub fn has_document(&self, class: &str, id: &Digest) -> Result<bool, StoreError> {
        self.has(class, id)
    }

    /// The addresses of every document of `class`, in address order.
    pub fn document_ids(&self, class: &str) -> Result<Vec<Digest>, StoreError> {
        self.ids_of(class)
    }

    /// The addresses of every document of `class`, in the order they were
    /// stored.
    pub fn documents_in_order(&self, class: &str) -> Result<Vec<Digest>, StoreError> {
        Ok(self
            .read_or_default(|s| s.entity_ids(class))?
            .into_iter()
            .map(crate::address::digest_of)
            .collect())
    }
}

fn digest_of<T: Serialize>(document: &T) -> Result<Digest, StoreError> {
    let bytes = canonical_json(document).map_err(|source| StoreError::Serialize {
        what: "document",
        source,
    })?;
    Ok(Digest::of(&bytes))
}

pub(crate) fn encode<T: Serialize>(
    class: &str,
    document: &T,
) -> Result<(Digest, Entity), StoreError> {
    let id = digest_of(document)?;
    let value = serde_json::to_value(document).map_err(|source| StoreError::Serialize {
        what: "document",
        source,
    })?;
    let entity = Entity::keyed(class, content_id(&id)?, value);
    Ok((id, entity))
}
