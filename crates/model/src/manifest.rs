// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements release steps that publish a fine-tuned
// adapter to a running inference server without a restart, for its clients.
// If your team needs expertise in rolling fine-tuned models out to serving,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! The release manifest `brain serve --adapter-manifest` follows.
//!
//! The manifest is brain's own type, [`brain::AdapterManifest`]: the file the
//! server reads is written with the type the server reads it with, so the two
//! cannot disagree about its shape. The server verifies, when the manifest
//! changes, that the file it names hashes to the digest the manifest states
//! and binds to the base it serves; a manifest that fails either check is
//! reported and the server keeps the adapter it had.

use std::io::{Error, ErrorKind, Result};
use std::path::Path;

use brain::AdapterManifest;
use splinter_core::digest::Digest;

/// The algorithm prefix brain's manifest names an adapter's digest by.
const SHA256_PREFIX: &str = "sha256:";

/// Publishes the adapter at `adapter`, whose SHA-256 is `digest`, as the one
/// a server following `manifest` serves: the manifest is written beside its
/// path and renamed over it, so a server polling it sees the old manifest or
/// the new one and never a torn one.
///
/// `adapter` must already be in place and must not change afterwards, and
/// must not be reused for another adapter: the server verifies it when the
/// manifest changes and reads it again when the model next loads. A release
/// kept by content address satisfies both.
///
/// # Errors
///
/// `digest` is not a SHA-256 (brain's manifest names nothing else),
/// `adapter` is not an absolute path (a relative one would be read relative
/// to the manifest, which is not where the caller means), or the manifest
/// could not be written; nothing is left beside it on failure.
pub fn write_adapter_manifest(manifest: &Path, adapter: &Path, digest: &Digest) -> Result<()> {
    if !digest.as_str().starts_with(SHA256_PREFIX) {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            format!("{digest} is not a SHA-256: a manifest names the adapter by one"),
        ));
    }
    if !adapter.is_absolute() {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            format!("adapter path {} is not absolute", adapter.display()),
        ));
    }
    if let Some(dir) = manifest.parent() {
        std::fs::create_dir_all(dir)?;
    }
    AdapterManifest {
        adapter: adapter.to_path_buf(),
        digest: digest.to_string(),
    }
    .write_atomically(manifest)
}

/// Withdraws the manifest at `manifest`: a server following it then keeps
/// serving the adapter it has, and one starting finds none and serves the
/// base. Nothing to withdraw is not an error.
pub fn withdraw_adapter_manifest(manifest: &Path) -> Result<()> {
    match std::fs::remove_file(manifest) {
        Err(e) if e.kind() != ErrorKind::NotFound => Err(e),
        _ => Ok(()),
    }
}
