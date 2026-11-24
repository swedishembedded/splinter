// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements crash-safe, immutable artifact stores for
// its clients. If your team needs expertise in storage that never loses a
// committed record, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The commit point of a publish: writing `products/<stage>.json`, once.
//!
//! [`ProductStore::prepare`] finds a stage uncommitted before it builds the
//! view, but two publishers of one stage (a controller whose lease lapsed
//! mid-stage, and the one that took over) can both find it so. The commit
//! therefore never replaces a manifest: it links the new one into place only
//! where none is, and the publisher that comes second reads the first's
//! product and is refused when it differs, as a publish of a committed stage
//! always is.

use super::*;
use crate::factory::fsutil::write_new;

impl ProductStore {
    /// Commits `manifest` as the product of `stage_id`, unless a product was
    /// committed in the meantime: then the committed one when it holds the
    /// same files, [`ProductError::Immutable`] when it does not.
    pub(super) fn commit(
        &self,
        stage_id: &Digest,
        manifest: ProductManifest,
    ) -> Result<ProductManifest, ProductError> {
        let bytes = pretty(&manifest)?;
        let path = self.product_path(stage_id);
        if write_new(&path, &bytes).map_err(io_error(format!("commit {}", path.display())))? {
            return Ok(manifest);
        }
        let committed = self
            .committed(stage_id)?
            .ok_or_else(|| malformed(&path.display().to_string(), "vanished after a commit"))?;
        if committed.name == manifest.name && committed.files == manifest.files {
            return Ok(committed);
        }
        Err(ProductError::Immutable {
            stage: stage_id.clone(),
            reason: "another publish committed different files first".to_string(),
        })
    }
}
