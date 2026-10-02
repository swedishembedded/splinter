// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Resolving heads to the set of files that exist.

use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;

use super::model::{Manifest, ObjectRef};
use crate::backend::{Key, Kind};
use crate::database::Database;
use crate::error::{Error, Result};
use crate::id::ContentId;

/// What a set of heads makes visible.
pub(crate) struct Resolved {
    /// Files that are added and never removed.
    pub(crate) live: BTreeSet<ObjectRef>,
    /// Files removed somewhere in the history.
    pub(crate) removed: BTreeSet<ObjectRef>,
    /// Every manifest the walk touched.
    pub(crate) manifests: HashSet<ContentId>,
}

/// Reads a manifest, checking its bytes against its id.
pub(crate) fn load_manifest(db: &Database, id: ContentId) -> Result<Arc<Manifest>> {
    if let Some(found) = db.manifest_cache().get(&id) {
        return Ok(Arc::clone(found));
    }
    let bytes = db
        .backend()
        .read(&Key::new(Kind::Manifest, &id.to_string())?)?;
    if ContentId::of(&bytes) != id {
        return Err(Error::corrupt(
            format!("manifest {id}"),
            "bytes do not match the content id",
        ));
    }
    let manifest: Manifest = serde_json::from_slice(&bytes).map_err(|source| Error::Decode {
        what: format!("manifest {id}"),
        source,
    })?;
    let manifest = Arc::new(manifest);
    db.manifest_cache().insert(id, Arc::clone(&manifest));
    Ok(manifest)
}

impl Database {
    /// Everything the union of `heads` makes visible. A checkpoint ends the
    /// walk down its branch, because it carries the whole state.
    pub(crate) fn resolve(&self, heads: &[ContentId]) -> Result<Resolved> {
        let mut pending: Vec<ContentId> = heads.to_vec();
        let mut seen = HashSet::new();
        let (mut added, mut removed) = (BTreeSet::new(), BTreeSet::new());
        while let Some(id) = pending.pop() {
            if !seen.insert(id) {
                continue;
            }
            let manifest = load_manifest(self, id)?;
            added.extend(manifest.add.iter().cloned());
            removed.extend(manifest.remove.iter().cloned());
            if !manifest.squash {
                pending.extend(manifest.parents.iter().copied());
            }
        }
        let live = added.difference(&removed).cloned().collect();
        Ok(Resolved {
            live,
            removed,
            manifests: seen,
        })
    }
}

impl Database {
    /// Every manifest reachable from `heads`, walking through checkpoints as
    /// well. [`resolve`](Database::resolve) stops at a checkpoint because it
    /// carries the whole state; this is for asking whether a manifest has
    /// been absorbed at all.
    pub(crate) fn ancestry_all(&self, heads: &[ContentId]) -> Result<HashSet<ContentId>> {
        let mut pending: Vec<ContentId> = heads.to_vec();
        let mut seen = HashSet::new();
        while let Some(id) = pending.pop() {
            if seen.insert(id) {
                pending.extend(load_manifest(self, id)?.parents.iter().copied());
            }
        }
        Ok(seen)
    }
}
