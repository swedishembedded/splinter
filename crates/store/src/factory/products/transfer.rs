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
use std::io;
use std::path::Path;

use splinter_core::digest::Digest;

use super::super::fsutil::write_atomic;
use super::{
    io_error, malformed, pretty, read_dir_if_exists, PendingPublish, ProductError, ProductManifest,
    ProductStore, PRODUCT_DIR,
};

/// A product brought over from another run's store.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Imported {
    /// The product's committed manifest.
    pub manifest: ProductManifest,
    /// Blobs now shared with the source store as hard links.
    pub blobs_linked: usize,
    /// Blobs copied (and verified) because linking was not possible, as
    /// across file systems; zero when every blob was linked or already held.
    pub blobs_copied: usize,
}

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
    /// its blobs (hard-linked where the filesystem allows, copied and
    /// verified where it does not), its view at `view`, then its commit
    /// point. A product this store already holds is left as it is.
    ///
    /// Blobs are immutable and read-only, so a linked blob is the source's
    /// own file and costs no I/O; it is verified when the product is reused
    /// ([`Self::ensure_view`]) or checked ([`Self::verify_product`]), like
    /// every committed blob. The result counts linked and copied blobs so a
    /// caller can show when linking was not possible.
    ///
    /// # Errors
    /// `source` holds no such product, a copied blob does not verify, or a
    /// file cannot be written.
    pub fn import(
        &self,
        source: &Self,
        stage_id: &Digest,
        view: &Path,
    ) -> Result<Imported, ProductError> {
        self.import_linking_with(source, stage_id, view, &|from, to| fs::hard_link(from, to))
    }

    /// [`Self::import`] with the way a blob is linked supplied, so a file
    /// system that cannot link can be exercised.
    ///
    /// # Errors
    /// As [`Self::import`].
    pub fn import_linking_with(
        &self,
        source: &Self,
        stage_id: &Digest,
        view: &Path,
        link: &dyn Fn(&Path, &Path) -> io::Result<()>,
    ) -> Result<Imported, ProductError> {
        if let Some(manifest) = self.committed(stage_id)? {
            return Ok(Imported {
                manifest,
                blobs_linked: 0,
                blobs_copied: 0,
            });
        }
        let manifest = source.manifest(stage_id)?;
        let relative_view = self.relative_view(view)?;
        let (mut blobs_linked, mut blobs_copied) = (0, 0);
        for file in &manifest.files {
            let target = self.blob_path(&file.digest)?;
            if target.exists() {
                continue;
            }
            let from = source.blob_path(&file.digest)?;
            if let Some(dir) = target.parent() {
                fs::create_dir_all(dir).map_err(io_error(format!("create {}", dir.display())))?;
            }
            if link(&from, &target).is_ok() {
                blobs_linked += 1;
            } else {
                // The copy is hashed as it is written, so it is verified
                // without a second read.
                let actual = self.put_file(&from)?;
                if actual != file.digest {
                    return Err(ProductError::BlobCorrupt {
                        digest: file.digest.clone(),
                        actual,
                    });
                }
                blobs_copied += 1;
            }
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
        Ok(Imported {
            manifest,
            blobs_linked,
            blobs_copied,
        })
    }

    /// Makes the view at `view` show the committed product of `stage_id`,
    /// byte for byte, as a product is reused: every blob is re-hashed (a
    /// rotted one is refused), and a view that does not hold exactly the
    /// committed files (another product of the same stage key took it, as a
    /// fork of a run that holds both leaves it, or a file of it was edited)
    /// is materialised from the blobs again.
    ///
    /// # Errors
    /// No such product, a blob that no longer hashes to its address, or the
    /// view cannot be written.
    pub fn ensure_view(
        &self,
        stage_id: &Digest,
        view: &Path,
    ) -> Result<ProductManifest, ProductError> {
        let manifest = self.manifest(stage_id)?;
        match self.verify_product(stage_id, view) {
            Ok(()) => return Ok(manifest),
            Err(rotted @ (ProductError::BlobCorrupt { .. } | ProductError::UnknownBlob(_))) => {
                return Err(rotted)
            }
            Err(_) => {}
        }
        self.relative_view(view)?;
        self.materialise_view(view, &manifest)?;
        // What was written is checked like what was found: every blob and
        // every file of the view.
        self.verify_product(stage_id, view)?;
        Ok(manifest)
    }
}
