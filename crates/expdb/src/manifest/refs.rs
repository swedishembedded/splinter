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

const CATALOG_REF: &str = "catalog";
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
        if !self.backend().exists(&key)? {
            return Ok(None);
        }
        let bytes = self.backend().read(&key)?;
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
            if let Some(job) = key.name().strip_prefix(JOBS_PREFIX) {
                if let Some(head) = self.get_ref(&format!("{JOBS_PREFIX}{job}"))? {
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

    /// Merges the catalog and every job head into the catalog ref. If two
    /// processes merge at once their manifests are identical, and a merge
    /// that loses the race to move the ref is picked up by the next one.
    pub fn merge_catalog(&self) -> Result<ContentId> {
        let mut heads = self.job_heads()?;
        heads.extend(self.get_ref(CATALOG_REF)?);
        let id = self.merged_head(heads)?;
        self.set_ref(CATALOG_REF, id)?;
        Ok(id)
    }

    /// Replaces the catalog with a checkpoint that carries the whole state,
    /// so resolving it never walks history.
    pub fn checkpoint(&self) -> Result<ContentId> {
        let mut heads = self.job_heads()?;
        heads.extend(self.get_ref(CATALOG_REF)?);
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
        self.set_ref(CATALOG_REF, id)?;
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
