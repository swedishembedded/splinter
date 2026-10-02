// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Where bytes live. The database names objects by kind and name, never by
//! path, so the same format can sit on a local disk, a parallel filesystem or
//! an object store.

mod key;
mod posix;

pub use key::{Key, Kind};
pub use posix::PosixBackend;

use std::time::SystemTime;

use crate::error::Result;

/// What the database needs from storage.
///
/// Published objects are immutable: [`write_once`](StorageBackend::write_once)
/// never replaces, and writing content that is already there restarts its
/// grace period so a collection in progress cannot take it. Only refs and pins, which are tiny pointers, are replaced,
/// and replacement is atomic.
pub trait StorageBackend: Send + Sync {
    /// Publishes `bytes` under `key` unless it exists. Returns whether this
    /// call published it. A reader never sees a partly written object.
    fn write_once(&self, key: &Key, bytes: &[u8]) -> Result<bool>;
    /// Atomically replaces the value under `key`, creating it if needed.
    fn replace(&self, key: &Key, bytes: &[u8]) -> Result<()>;
    /// Atomically moves an object (or a ref) to another key, replacing
    /// anything there. Returns whether it was there to move. It is how a
    /// collection sets a file aside, and how an idle ref is retired: the
    /// mover knows exactly which value it took, and whatever is written under
    /// the old key afterwards is new.
    fn rename(&self, from: &Key, to: &Key) -> Result<bool>;
    /// All of the object.
    fn read(&self, key: &Key) -> Result<Vec<u8>>;
    /// Exactly `len` bytes from `offset`; reading past the end is an error.
    fn read_range(&self, key: &Key, offset: u64, len: usize) -> Result<Vec<u8>>;
    /// The object's size in bytes.
    fn len(&self, key: &Key) -> Result<u64>;
    /// Whether the object exists.
    fn exists(&self, key: &Key) -> Result<bool>;
    /// Every published object of a kind, never a temporary file.
    fn list(&self, kind: Kind) -> Result<Vec<Key>>;
    /// Deletes the object. Deleting a missing object is not an error.
    fn remove(&self, key: &Key) -> Result<()>;
    /// When the object was last written.
    fn modified(&self, key: &Key) -> Result<SystemTime>;
    /// The write size this storage prefers; writers batch up to it.
    fn preferred_write_bytes(&self) -> usize;
}
