// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Finding content-addressed entities by their id, and application entities
//! by their class, through the index: no record body is read to answer
//! whether an entity exists or which ones a class holds.

use crate::error::Result;
use crate::format::class_hash;
use crate::id::{ContentId, RecordId};
use crate::manifest::Snapshot;
use crate::model::{Body, Entity};

/// An application entity as stored: where, and what.
#[derive(Debug, Clone, PartialEq)]
pub struct StoredEntity {
    /// The content id.
    pub id: ContentId,
    /// The record that holds it.
    pub record: RecordId,
    /// The entity itself.
    pub entity: Entity,
}

impl Snapshot {
    /// The first record that defines the entity with content id `id`: a task
    /// definition, task instance, state, episode family or application
    /// entity. The same content from two writers is one entity.
    pub fn entity(&self, id: &ContentId) -> Result<Option<RecordId>> {
        Ok(self.index()?.entity(id))
    }

    /// Whether the entity `id` is an application entity of `class`.
    pub fn entity_in_class(&self, id: &ContentId, class: &str) -> Result<bool> {
        Ok(self.index()?.entity_has_class(id, class_hash(class)))
    }

    /// How many distinct entities the snapshot holds.
    pub fn entity_count(&self) -> Result<usize> {
        Ok(self.index()?.entity_count())
    }

    /// The bytes of the blob with content id `id`, verified against it.
    pub fn read_blob(&self, id: &ContentId) -> Result<Vec<u8>> {
        self.blob_store()?.get_by_id(id)
    }

    /// The application entity with content id `id`.
    pub fn entity_body(&self, id: &ContentId) -> Result<Option<Entity>> {
        let Some(record) = self.entity(id)? else {
            return Ok(None);
        };
        Ok(match self.get(record)? {
            Some(record) => match record.body {
                Body::Entity(entity) => Some(entity),
                _ => None,
            },
            None => None,
        })
    }

    /// The ids of every application entity of `class`, in the order first
    /// written, without reading any body.
    pub fn entity_ids(&self, class: &str) -> Result<Vec<(ContentId, RecordId)>> {
        Ok(self.index()?.entities_of_class(class_hash(class)))
    }

    /// Every application entity of `class`, each once, in the order first
    /// written.
    pub fn entities(&self, class: &str) -> Result<Vec<StoredEntity>> {
        let mut out = Vec::new();
        for (id, record_id) in self.entity_ids(class)? {
            if let Some(record) = self.get(record_id)? {
                if let Body::Entity(entity) = record.body {
                    if entity.class == class {
                        out.push(StoredEntity {
                            id,
                            record: record_id,
                            entity,
                        });
                    }
                }
            }
        }
        Ok(out)
    }
}
