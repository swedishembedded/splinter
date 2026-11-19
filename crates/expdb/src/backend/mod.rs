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
mod layout;
mod packed;
mod posix;
mod repack;

pub use key::{Key, Kind};
pub(crate) use layout::FORMAT_NAME;
pub use layout::{open_existing, Layout};
pub use packed::PackedBackend;
pub use posix::PosixBackend;
pub use repack::{repack, Repacked};

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
    /// Marks the object as written now, so it is as young as that when a
    /// collection judges it. The default does nothing, for storage that keeps
    /// no modification time.
    fn touch(&self, _key: &Key) -> Result<()> {
        Ok(())
    }
    /// The write size this storage prefers; writers batch up to it.
    fn preferred_write_bytes(&self) -> usize;
    /// How the objects lie on disk, which the database's format marker
    /// records so a binary that knows only another layout refuses the root.
    fn layout(&self) -> Layout {
        Layout::Loose
    }
}

/// Reads one object in pieces of the backend's preferred size, so an object
/// of gigabytes can be streamed (into an archive, say) without holding it.
pub struct ObjectReader<'a> {
    backend: &'a dyn StorageBackend,
    key: Key,
    at: u64,
    len: u64,
}

impl<'a> ObjectReader<'a> {
    /// A reader over the object under `key`, which must exist.
    pub fn new(backend: &'a dyn StorageBackend, key: Key) -> Result<Self> {
        let len = backend.len(&key)?;
        Ok(Self {
            backend,
            key,
            at: 0,
            len,
        })
    }

    /// The object's size in bytes.
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Whether the object is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl std::io::Read for ObjectReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let left = self.len - self.at;
        let want = usize::try_from(left)
            .unwrap_or(usize::MAX)
            .min(buf.len())
            .min(self.backend.preferred_write_bytes());
        if want == 0 {
            return Ok(0);
        }
        let bytes = self
            .backend
            .read_range(&self.key, self.at, want)
            .map_err(std::io::Error::other)?;
        buf[..want].copy_from_slice(&bytes);
        self.at += want as u64;
        Ok(want)
    }
}
