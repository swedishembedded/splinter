// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements experiment stores whose products move
// between runs without being recomputed or altered, for its clients. If your
// team needs expertise in reproducible machine-learning pipelines, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Listing a store's committed products and bringing one over from another
//! run's store, in the order publishing keeps.

use std::fs;
use std::path::Path;

use splinter_core::digest::Digest;

use super::super::fsutil::write_atomic;
use super::{
    io_error, malformed, pretty, read_dir_if_exists, PendingPublish, ProductError, ProductManifest,
    ProductStore, PRODUCT_DIR,
};

impl ProductStore {
    /// Every committed product's manifest, by stage.
    ///
    /// # Errors
    /// The products directory or a manifest in it cannot be read.
    pub fn list(&self) -> Result<Vec<ProductManifest>, ProductError> {
        let mut out = Vec::new();
        for path in read_dir_if_exists(&self.root.join(PRODUCT_DIR))? {
            if path.extension().is_none_or(|e| e != "json") {
                continue;
            }
            let bytes = fs::read(&path).map_err(io_error(format!("read {}", path.display())))?;
            let manifest: ProductManifest = serde_json::from_slice(&bytes)
                .map_err(|e| malformed(&path.display().to_string(), e))?;
            // The commit point is named by its stage; read it the way manifest() does.
            out.extend(self.committed(&manifest.stage)?);
        }
        out.sort_by(|a, b| a.stage.cmp(&b.stage));
        Ok(out)
    }

    /// Brings `source`'s committed product of `stage_id` into this store:
    /// its blobs (hard-linked where the filesystem allows, verified either
    /// way), its view at `view`, then its commit point. A product this
    /// store already holds is left as it is.
    ///
    /// # Errors
    /// `source` holds no such product, a blob does not verify, or a file
    /// cannot be written.
    pub fn import(
        &self,
        source: &Self,
        stage_id: &Digest,
        view: &Path,
    ) -> Result<ProductManifest, ProductError> {
        if let Some(existing) = self.committed(stage_id)? {
            return Ok(existing);
        }
        let manifest = source.manifest(stage_id)?;
        let relative_view = self.relative_view(view)?;
        for file in &manifest.files {
            let target = self.blob_path(&file.digest)?;
            if target.exists() {
                continue;
            }
            let from = source.blob_path(&file.digest)?;
            if let Some(dir) = target.parent() {
                fs::create_dir_all(dir).map_err(io_error(format!("create {}", dir.display())))?;
            }
            if fs::hard_link(&from, &target).is_err() {
                self.put_file(&from)?;
            }
            self.verify(&file.digest)?;
        }
        let intent = serde_json::to_vec(&PendingPublish {
            view: relative_view,
        })
        .map_err(|e| malformed("publish intent", e))?;
        let pending = self.pending_path(stage_id);
        write_atomic(&pending, &intent)
            .map_err(io_error(format!("write {}", pending.display())))?;
        self.materialise_view(view, &manifest)?;
        let path = self.product_path(stage_id);
        write_atomic(&path, &pretty(&manifest)?)
            .map_err(io_error(format!("commit {}", path.display())))?;
        let _ = fs::remove_file(pending);
        Ok(manifest)
    }

    /// Makes the view at `view` show the committed product of `stage_id`:
    /// left alone when it already does, materialised from the blobs
    /// otherwise (a view another product of the same stage key last took,
    /// as a fork of a run that holds both leaves it).
    ///
    /// # Errors
    /// No such product, or the view cannot be written.
    pub fn ensure_view(
        &self,
        stage_id: &Digest,
        view: &Path,
    ) -> Result<ProductManifest, ProductError> {
        let manifest = self.manifest(stage_id)?;
        // The view names the product it shows in its own manifest; reading
        // it is enough here (`verify_product` re-hashes every file).
        let shown = fs::read(view.join(super::VIEW_MANIFEST))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<ProductManifest>(&bytes).ok());
        if shown.as_ref() == Some(&manifest) {
            return Ok(manifest);
        }
        self.relative_view(view)?;
        self.materialise_view(view, &manifest)?;
        Ok(manifest)
    }
}
