// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Finding content-addressed entities by their id, and application entities
//! by their class.

use std::collections::HashMap;
use std::sync::Arc;

use super::read::locked;
use crate::error::Result;
use crate::id::{ContentId, RecordId};
use crate::manifest::Snapshot;
use crate::model::{Body, Entity, RecordKind};

/// Every entity of a snapshot, by id and by class.
#[derive(Default)]
pub(crate) struct EntityMap {
    by_id: HashMap<ContentId, RecordId>,
    by_class: HashMap<String, Vec<(ContentId, RecordId)>>,
}

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

const ENTITY_KINDS: [RecordKind; 5] = [
    RecordKind::TaskDefinition,
    RecordKind::TaskInstance,
    RecordKind::State,
    RecordKind::Family,
    RecordKind::Entity,
];

impl Snapshot {
    fn entity_map(&self) -> Result<Arc<EntityMap>> {
        if let Some(found) = locked(&self.cache().entities).as_ref() {
            return Ok(Arc::clone(found));
        }
        let index = self.index()?;
        let mut map = EntityMap::default();
        for kind in ENTITY_KINDS {
            for record_id in index.by_kind(kind) {
                let Some(record) = self.get(record_id)? else {
                    continue;
                };
                let Some(id) = record.body.entity_id()? else {
                    continue;
                };
                if map.by_id.contains_key(&id) {
                    continue;
                }
                map.by_id.insert(id, record_id);
                if let Body::Entity(entity) = &record.body {
                    map.by_class
                        .entry(entity.class.clone())
                        .or_default()
                        .push((id, record_id));
                }
            }
        }
        for found in map.by_class.values_mut() {
            found.sort_by_key(|(_, record)| *record);
        }
        let map = Arc::new(map);
        *locked(&self.cache().entities) = Some(Arc::clone(&map));
        Ok(map)
    }

    /// The first record that defines the entity with content id `id`: a task
    /// definition, task instance, state, episode family or application
    /// entity. The same content from two writers is one entity.
    pub fn entity(&self, id: &ContentId) -> Result<Option<RecordId>> {
        Ok(self.entity_map()?.by_id.get(id).copied())
    }

    /// How many entities the snapshot holds.
    pub fn entity_count(&self) -> Result<usize> {
        Ok(self.entity_map()?.by_id.len())
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

    /// Every application entity of `class`, each once, in the order first
    /// written.
    pub fn entities(&self, class: &str) -> Result<Vec<StoredEntity>> {
        let map = self.entity_map()?;
        let mut out = Vec::new();
        for (id, record_id) in map.by_class.get(class).into_iter().flatten() {
            if let Some(record) = self.get(*record_id)? {
                if let Body::Entity(entity) = record.body {
                    out.push(StoredEntity {
                        id: *id,
                        record: *record_id,
                        entity,
                    });
                }
            }
        }
        Ok(out)
    }
}
