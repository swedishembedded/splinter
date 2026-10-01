// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The manifest itself and the objects it names.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::backend::{Key, Kind};
use crate::error::{Error, Result};
use crate::id::ContentId;

/// What kind of immutable file an object is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectKind {
    /// A sealed segment of records.
    Segment,
    /// A pack of blob chunks.
    BlobPack,
    /// An index run.
    Index,
    /// A shard of embedding vectors.
    Vector,
    /// A shard of an inverted text index.
    Text,
}

/// One immutable file a manifest adds or removes.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ObjectRef {
    /// What the file is.
    pub kind: ObjectKind,
    /// Its content id.
    pub id: ContentId,
    /// Its size in bytes.
    pub bytes: u64,
    /// Records inside, for segments; zero for other objects.
    pub records: u64,
}

impl ObjectRef {
    /// The backend key of the file.
    pub fn key(&self) -> Result<Key> {
        match self.kind {
            ObjectKind::Segment => Key::new(Kind::Segment, &format!("{}.seg", self.id)),
            ObjectKind::BlobPack => Key::new(Kind::BlobPack, &format!("{}.pack", self.id)),
            ObjectKind::Index => Key::new(Kind::Index, &format!("{}.idx", self.id)),
            ObjectKind::Vector => Key::new(Kind::Index, &format!("{}.vec", self.id)),
            ObjectKind::Text => Key::new(Kind::Index, &format!("{}.txt", self.id)),
        }
    }

    /// A reference to a blob pack.
    pub fn blob_pack(id: ContentId, bytes: u64) -> Self {
        Self {
            kind: ObjectKind::BlobPack,
            id,
            bytes,
            records: 0,
        }
    }
}

/// A set of additions and removals on top of parent manifests.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    /// The manifests this builds on, sorted and unique.
    pub parents: Vec<ContentId>,
    /// Files made part of the database.
    pub add: BTreeSet<ObjectRef>,
    /// Files withdrawn from it, for good.
    pub remove: BTreeSet<ObjectRef>,
    /// A checkpoint carries the whole state, so resolving it needs no
    /// ancestors.
    pub squash: bool,
    /// When it was published, in nanoseconds since the Unix epoch.
    pub created_ns: u64,
    /// What it is for.
    pub label: String,
}

impl Manifest {
    /// A manifest adding and removing files on top of `parents`.
    pub fn new(
        parents: Vec<ContentId>,
        add: BTreeSet<ObjectRef>,
        remove: BTreeSet<ObjectRef>,
        created_ns: u64,
        label: &str,
    ) -> Self {
        Self {
            parents: normalised(parents),
            add,
            remove,
            squash: false,
            created_ns,
            label: label.into(),
        }
    }

    /// The merge of several histories. It carries nothing of its own and no
    /// timestamp, so everyone merging the same heads makes the same manifest.
    pub fn merge(heads: Vec<ContentId>) -> Self {
        Self::new(heads, BTreeSet::new(), BTreeSet::new(), 0, "merge")
    }

    /// The canonical bytes the manifest is stored as.
    pub fn encode(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(|source| Error::Encode {
            what: "manifest",
            source,
        })
    }

    /// The manifest's id: the content id of its canonical bytes.
    pub fn id(&self) -> Result<ContentId> {
        Ok(ContentId::of(&self.encode()?))
    }
}

fn normalised(mut parents: Vec<ContentId>) -> Vec<ContentId> {
    parents.sort();
    parents.dedup();
    parents
}
