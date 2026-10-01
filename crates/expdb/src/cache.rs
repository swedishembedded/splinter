// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! A cache of data derived from experience.
//!
//! Raw experience stays canonical. Anything computed from it, such as token
//! ids, is cached under a key made from everything that went into it: change
//! the tokenizer, the transform or the source and the key changes, so a stale
//! entry can never be read as current and an old one is never damaged.

use crate::backend::{Key, Kind};
use crate::database::Database;
use crate::error::Result;
use crate::id::ContentId;

/// One namespace of derived data.
pub struct DerivedCache {
    db: Database,
    namespace: String,
}

impl DerivedCache {
    /// The cache for one kind of derived data, such as `tokens`.
    pub fn new(db: &Database, namespace: &str) -> Self {
        Self {
            db: db.clone(),
            namespace: namespace.to_owned(),
        }
    }

    /// The key for what `tool`, at `transform` version, makes of `source`.
    pub fn key(tool: &str, transform: &str, source: &ContentId) -> ContentId {
        let mut bytes = Vec::new();
        for part in [tool.as_bytes(), transform.as_bytes()] {
            bytes.extend_from_slice(&(part.len() as u64).to_le_bytes());
            bytes.extend_from_slice(part);
        }
        bytes.extend_from_slice(source.as_bytes());
        ContentId::of(&bytes)
    }

    fn object(&self, key: &ContentId) -> Result<Key> {
        Key::new(Kind::Cache, &format!("{key}.{}", self.namespace))
    }

    /// Stores derived bytes; storing under a key again changes nothing.
    pub fn put(&self, key: &ContentId, bytes: &[u8]) -> Result<()> {
        self.db
            .backend()
            .write_once(&self.object(key)?, bytes)
            .map(|_| ())
    }

    /// The bytes under a key, if they were ever derived.
    pub fn get(&self, key: &ContentId) -> Result<Option<Vec<u8>>> {
        let object = self.object(key)?;
        if self.db.backend().exists(&object)? {
            self.db.backend().read(&object).map(Some)
        } else {
            Ok(None)
        }
    }
}
