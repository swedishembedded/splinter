// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Garbage collection: remove only what no ref and no pin can reach, and
//! never a file young enough that its writer may be about to publish it.
//!
//! A collection reads what is needed, sets the candidates aside, reads what is
//! needed again, puts back anything needed now, and only then discards the
//! rest. A pin written while the collection was running is therefore honoured:
//! either it was written before the second read, and its files are put back,
//! or after, and validating the pin finds them already gone and withdraws it.

use std::collections::{BTreeSet, HashSet};
use std::time::{Duration, SystemTime};

use super::model::ObjectRef;
use crate::backend::{Key, Kind};
use crate::database::Database;
use crate::error::Result;
use crate::id::ContentId;

/// What a collection did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GcReport {
    /// Files removed.
    pub removed: usize,
    /// Files left because something reaches them.
    pub kept: usize,
    /// Unreachable files left because they are younger than the grace period.
    pub too_young: usize,
}

impl Database {
    /// What the database needs right now: the objects and manifests live
    /// under every ref taken together (so a removal published by one process
    /// applies to what another still names), and under every pin on its own.
    fn needed(&self) -> Result<(HashSet<String>, HashSet<String>)> {
        let mut heads: Vec<ContentId> = Vec::new();
        for key in self.backend().list(Kind::Ref)? {
            if key.name() != "format" {
                heads.extend(self.get_ref(key.name())?);
            }
        }
        let mut objects: BTreeSet<ObjectRef> = BTreeSet::new();
        let mut manifests: HashSet<ContentId> = HashSet::new();
        let current = self.resolve(&heads)?;
        objects.extend(current.live);
        manifests.extend(current.manifests);
        // A pin keeps its own snapshot whole, whatever came after.
        for (_, pinned) in self.pins()? {
            let resolved = self.resolve(&[pinned])?;
            objects.extend(resolved.live);
            manifests.extend(resolved.manifests);
        }
        let object_names = objects
            .iter()
            .map(|o| Ok(o.key()?.name().to_owned()))
            .collect::<Result<_>>()?;
        Ok((
            object_names,
            manifests.iter().map(ToString::to_string).collect(),
        ))
    }

    fn trash_key(key: &Key) -> Result<Key> {
        Key::new(Kind::Trash, &format!("{}.{}", key.kind().dir(), key.name()))
    }

    /// The key a set-aside file came from, if the name says.
    fn restore_key(trash: &Key) -> Option<Key> {
        let (dir, name) = trash.name().split_once('.')?;
        let kind = [Kind::Segment, Kind::BlobPack, Kind::Index, Kind::Manifest]
            .into_iter()
            .find(|k| k.dir() == dir)?;
        Key::new(kind, name).ok()
    }

    /// Removes unreachable segments, blob packs, index files and manifests
    /// older than [`Config::orphan_grace`](crate::Config::orphan_grace).
    /// Reachable means live under some ref or pin.
    pub fn gc(&self) -> Result<GcReport> {
        let (objects, manifests) = self.needed()?;
        let mut report = GcReport::default();
        // Files a collection that died set aside are judged like the rest.
        let mut aside: Vec<Key> = self
            .backend()
            .list(Kind::Trash)?
            .iter()
            .filter_map(Self::restore_key)
            .collect();
        for (kind, wanted) in [
            (Kind::Segment, &objects),
            (Kind::BlobPack, &objects),
            (Kind::Index, &objects),
            (Kind::Manifest, &manifests),
        ] {
            for key in self.backend().list(kind)? {
                if wanted.contains(key.name()) {
                    report.kept += 1;
                } else if self.past_grace(&key)? {
                    if self.backend().rename(&key, &Self::trash_key(&key)?)? {
                        aside.push(key);
                    }
                } else {
                    report.too_young += 1;
                }
            }
        }
        // Look again, now that the candidates are out of the way: anything a
        // pin or a ref needs by now goes back, and so does anything that was
        // rewritten (and so made young) in the meantime.
        let (objects, manifests) = self.needed()?;
        for key in aside {
            let wanted = if key.kind() == Kind::Manifest {
                &manifests
            } else {
                &objects
            };
            let trash = Self::trash_key(&key)?;
            if wanted.contains(key.name()) {
                if !self.backend().exists(&key)? {
                    self.backend().rename(&trash, &key)?;
                } else {
                    self.backend().remove(&trash)?;
                }
                report.kept += 1;
            } else {
                self.backend().remove(&trash)?;
                report.removed += 1;
            }
        }
        Ok(report)
    }

    fn past_grace(&self, key: &Key) -> Result<bool> {
        let modified = match self.backend().modified(key) {
            Ok(time) => time,
            // Gone already: nothing to collect, and not worth failing over.
            Err(crate::error::Error::NotFound { .. }) => return Ok(false),
            Err(other) => return Err(other),
        };
        let age = SystemTime::now()
            .duration_since(modified)
            .unwrap_or(Duration::ZERO);
        Ok(age >= self.config().orphan_grace)
    }
}

/// What absorbing idle job refs did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AbsorbReport {
    /// Job refs retired because the catalog now holds their history.
    pub pruned: usize,
}

impl Database {
    /// Folds every job's history into the catalog and retires the refs of
    /// jobs that have been idle for the orphan grace period, so ten thousand
    /// finished writers do not leave ten thousand refs behind.
    ///
    /// A ref is retired by moving it atomically aside, so exactly the value it
    /// held at that instant is known; if the catalog does not hold it (the
    /// writer published since the merge) it is folded in before the retired
    /// ref is discarded. Retired refs count as heads until then, so a crash
    /// at any point loses nothing. A writer that publishes after the retirement
    /// starts a new chain.
    pub fn absorb_jobs(&self) -> Result<AbsorbReport> {
        self.merge_catalog()?;
        let mut report = AbsorbReport::default();
        for key in self.backend().list(Kind::Ref)? {
            let Some(job) = key.name().strip_prefix("jobs/") else {
                continue;
            };
            if !self.past_grace(&key)? {
                continue;
            }
            static RETIREMENT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let mut unique = self.clock().now_ns().to_le_bytes().to_vec();
            unique.extend_from_slice(
                &RETIREMENT
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                    .to_le_bytes(),
            );
            let tag = &ContentId::of(&unique).to_string()[..12];
            let retired = Key::new(Kind::Ref, &format!("retired/{job}-{tag}"))?;
            if !self.backend().rename(&key, &retired)? {
                continue;
            }
            if let Some(head) = self.get_ref(retired.name())? {
                self.fold_into_catalog(head)?;
            }
            self.backend().remove(&retired)?;
            report.pruned += 1;
        }
        Ok(report)
    }
}
