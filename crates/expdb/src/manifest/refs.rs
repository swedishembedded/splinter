// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Publishing: writing manifests and moving the small refs that name heads.

use std::collections::BTreeSet;

use super::model::{Manifest, ObjectRef};
use super::resolve::load_manifest;
use crate::backend::{Key, Kind};
use crate::database::Database;
use crate::error::Result;
use crate::id::ContentId;

const CATALOG_PREFIX: &str = "catalog/";
const JOBS_PREFIX: &str = "jobs/";

impl Database {
    /// Writes a manifest; publishing the same one again changes nothing.
    pub(crate) fn put_manifest(&self, manifest: &Manifest) -> Result<ContentId> {
        let id = manifest.id()?;
        self.backend().write_once(
            &Key::new(Kind::Manifest, &id.to_string())?,
            &manifest.encode()?,
        )?;
        Ok(id)
    }

    /// The value of a ref, if it has been set.
    pub fn get_ref(&self, name: &str) -> Result<Option<ContentId>> {
        let key = Key::new(Kind::Ref, name)?;
        // Read directly: a ref can be retired between an existence check and
        // the read, and that is not an error, only a ref that is gone.
        let bytes = match self.backend().read(&key) {
            Ok(bytes) => bytes,
            Err(crate::error::Error::NotFound { .. }) => return Ok(None),
            Err(other) => return Err(other),
        };
        let text = String::from_utf8_lossy(&bytes);
        ContentId::parse(text.trim()).map(Some)
    }

    /// Points a ref at a manifest. Each job owns its ref, so jobs never
    /// contend; the catalog ref is advanced by merging.
    pub fn set_ref(&self, name: &str, manifest: ContentId) -> Result<()> {
        self.backend()
            .replace(&Key::new(Kind::Ref, name)?, manifest.to_string().as_bytes())
    }

    /// The manifest each job last published.
    pub fn job_heads(&self) -> Result<Vec<ContentId>> {
        let mut heads = Vec::new();
        for key in self.backend().list(Kind::Ref)? {
            // Refs being retired still count: until the catalog holds their
            // history they are the only record of it.
            if key.name().starts_with(JOBS_PREFIX) || key.name().starts_with("retired/") {
                if let Some(head) = self.get_ref(key.name())? {
                    heads.push(head);
                }
            }
        }
        Ok(heads)
    }

    /// Publishes files on top of the job's previous manifest and moves the
    /// job's ref. Nothing is visible to readers before this returns, and
    /// nothing is partly visible after.
    pub fn publish(
        &self,
        job: &str,
        add: Vec<ObjectRef>,
        remove: Vec<ObjectRef>,
    ) -> Result<ContentId> {
        if add.iter().any(|a| remove.contains(a)) {
            // A removal is permanent, so adding what is removed in the same
            // breath would delete it for good.
            return Err(crate::error::Error::invalid(
                "manifest",
                "a file cannot be added and removed together",
            ));
        }
        let ref_name = format!("{JOBS_PREFIX}{job}");
        let parents = self.get_ref(&ref_name)?.into_iter().collect();
        let manifest = Manifest::new(
            parents,
            add.into_iter().collect::<BTreeSet<_>>(),
            remove.into_iter().collect::<BTreeSet<_>>(),
            self.clock().now_ns(),
            job,
        );
        let id = self.put_manifest(&manifest)?;
        self.set_ref(&ref_name, id)?;
        Ok(id)
    }

    /// Publishes on a ref of its own, so any number of processes can do it at
    /// once. [`publish`](Database::publish) chains each manifest after the
    /// job's previous one and so needs a single publisher per job name; two
    /// publishers on one name could overwrite each other's head. Maintenance
    /// tasks that any process may run (indexing, compaction, search shards)
    /// use this instead.
    pub fn publish_once(
        &self,
        prefix: &str,
        add: Vec<ObjectRef>,
        remove: Vec<ObjectRef>,
    ) -> Result<ContentId> {
        static CALL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let mut unique = Vec::new();
        unique.extend_from_slice(&self.clock().now_ns().to_le_bytes());
        unique.extend_from_slice(&u64::from(std::process::id()).to_le_bytes());
        unique.extend_from_slice(
            &CALL
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                .to_le_bytes(),
        );
        let tag = &ContentId::of(&unique).to_string()[..16];
        self.publish(&format!("{prefix}-{tag}"), add, remove)
    }

    /// The heads of the catalog. The catalog is a set of immutable head files
    /// that only ever grows or is trimmed of heads another head contains. It is
    /// never overwritten, so two processes merging at once cannot lose what
    /// either absorbed: both heads exist until a later merge joins them.
    pub fn catalog_heads(&self) -> Result<Vec<ContentId>> {
        let mut heads = Vec::new();
        for key in self.backend().list(Kind::Ref)? {
            if key.name().starts_with(CATALOG_PREFIX) {
                heads.extend(self.get_ref(key.name())?);
            }
        }
        heads.sort();
        heads.dedup();
        Ok(heads)
    }

    fn add_catalog_head(&self, head: ContentId) -> Result<()> {
        let key = Key::new(Kind::Ref, &format!("{CATALOG_PREFIX}{head}"))?;
        self.backend()
            .write_once(&key, head.to_string().as_bytes())
            .map(|_| ())
    }

    /// Drops catalog heads that `newest` contains. A head is dropped only
    /// because another holds all of it, so nothing the catalog knew is lost.
    fn trim_catalog(&self, newest: ContentId) -> Result<()> {
        let inside = self.ancestry_all(&[newest])?;
        for head in self.catalog_heads()? {
            if head != newest && inside.contains(&head) {
                self.backend()
                    .remove(&Key::new(Kind::Ref, &format!("{CATALOG_PREFIX}{head}"))?)?;
            }
        }
        Ok(())
    }

    /// Puts `head` into the catalog and returns once the catalog holds it.
    pub(crate) fn fold_into_catalog(&self, head: ContentId) -> Result<()> {
        let heads = self.catalog_heads()?;
        if self.ancestry_all(&heads)?.contains(&head) {
            return Ok(());
        }
        let merged = self.merged_head(heads.into_iter().chain([head]).collect())?;
        self.add_catalog_head(merged)?;
        self.trim_catalog(merged)
    }

    /// Merges every job head the catalog does not yet hold into it, leaving
    /// one catalog head. Merging when the catalog already holds every head
    /// changes nothing, so merging twice in a row gives the same catalog.
    pub fn merge_catalog(&self) -> Result<ContentId> {
        let catalog = self.catalog_heads()?;
        // Everything the catalog has ever absorbed, below any checkpoint too,
        // so a checkpoint does not make settled history look new.
        let known = self.ancestry_all(&catalog)?;
        let fresh: Vec<ContentId> = self
            .job_heads()?
            .into_iter()
            .filter(|h| !known.contains(h))
            .collect();
        if fresh.is_empty() {
            if let [only] = catalog.as_slice() {
                return Ok(*only);
            }
        }
        let id = self.merged_head(fresh.into_iter().chain(catalog).collect())?;
        self.add_catalog_head(id)?;
        self.trim_catalog(id)?;
        Ok(id)
    }

    /// Replaces the head of `job` with a checkpoint that carries everything
    /// the job's chain made visible, so reading the job never walks further
    /// back than this. Only the job's own writer may call it.
    pub fn checkpoint_job(&self, job: &str) -> Result<ContentId> {
        let ref_name = format!("{JOBS_PREFIX}{job}");
        let head = self
            .get_ref(&ref_name)?
            .ok_or_else(|| crate::error::Error::NotFound {
                what: format!("ref {ref_name}"),
            })?;
        let resolved = self.resolve(&[head])?;
        let mut manifest = Manifest::new(
            vec![head],
            resolved.live,
            resolved.removed,
            self.clock().now_ns(),
            "checkpoint",
        );
        manifest.squash = true;
        let id = self.put_manifest(&manifest)?;
        self.set_ref(&ref_name, id)?;
        Ok(id)
    }

    /// How many manifests resolving the database walks: what every open
    /// reads. It grows with every commit until a checkpoint cuts it.
    pub fn history_depth(&self) -> Result<usize> {
        let mut heads = self.job_heads()?;
        heads.extend(self.catalog_heads()?);
        Ok(self.resolve(&heads)?.manifests.len())
    }

    /// Adds a checkpoint that carries the whole state, so resolving the
    /// catalog never walks history, and trims the heads it contains.
    pub fn checkpoint(&self) -> Result<ContentId> {
        let mut heads = self.job_heads()?;
        heads.extend(self.catalog_heads()?);
        let head = self.merged_head(heads)?;
        let resolved = self.resolve(&[head])?;
        let mut manifest = Manifest::new(
            vec![head],
            resolved.live.clone(),
            resolved.removed,
            self.clock().now_ns(),
            "checkpoint",
        );
        manifest.squash = true;
        let id = self.put_manifest(&manifest)?;
        self.add_catalog_head(id)?;
        self.trim_catalog(id)?;
        Ok(id)
    }

    /// One head naming all of `heads`: the head itself if there is one,
    /// a stored merge manifest if there are several.
    pub(crate) fn merged_head(&self, mut heads: Vec<ContentId>) -> Result<ContentId> {
        heads.sort();
        heads.dedup();
        match heads.as_slice() {
            [only] => Ok(*only),
            _ => self.put_manifest(&Manifest::merge(heads)),
        }
    }

    /// The manifest with `id`.
    pub fn manifest(&self, id: ContentId) -> Result<std::sync::Arc<Manifest>> {
        load_manifest(self, id)
    }
}
