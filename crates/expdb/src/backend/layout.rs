// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in keeping
// file counts bounded on quota-limited parallel filesystems, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Which layout a database root holds, recorded in its format marker.
//!
//! The marker is the file `refs/format` in either layout, so a binary that
//! knows only the loose layout reads `splinter-expdb 3` and refuses the root
//! instead of writing loose files beside the packs.

use std::path::Path;
use std::sync::Arc;

use super::{Key, Kind, PackedBackend, PosixBackend, StorageBackend};
use crate::error::{Error, Result};

/// The name of the format marker among the refs.
pub(crate) const FORMAT_NAME: &str = "format";

/// How a database's objects lie on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Layout {
    /// Layout 2: one file per object, fanned out over two-character
    /// directories. Read and written for roots that hold it.
    Loose,
    /// Layout 3: immutable objects appended to a few pack files with an
    /// append-only journal as their index; refs, pins and overlays stay
    /// small files. The default for a new root.
    Packed,
}

impl Layout {
    /// The bytes of the format marker for this layout.
    pub fn marker(self) -> &'static [u8] {
        match self {
            Layout::Loose => b"splinter-expdb 2\n",
            Layout::Packed => b"splinter-expdb 3\n",
        }
    }

    /// The layout a marker names.
    fn of_marker(bytes: &[u8]) -> Result<Self> {
        [Layout::Loose, Layout::Packed]
            .into_iter()
            .find(|layout| layout.marker() == bytes)
            .ok_or_else(|| {
                Error::corrupt(
                    "database format marker",
                    format!(
                        "`{}` is not a splinter-expdb 2 or 3 store",
                        String::from_utf8_lossy(bytes).trim()
                    ),
                )
            })
    }

    /// The layout of the database under `root`; `None` when there is none.
    pub fn of_root(root: &Path) -> Result<Option<Self>> {
        let marker = Key::new(Kind::Ref, FORMAT_NAME)?;
        match PosixBackend::new(root).read(&marker) {
            Ok(bytes) => Self::of_marker(&bytes).map(Some),
            Err(Error::NotFound { .. }) => Ok(None),
            Err(other) => Err(other),
        }
    }

    /// The layout of the database under `root`, recording `wanted` first
    /// when the root holds none. Racing openers of a new root agree: the
    /// marker is written once and every one of them reads the winner's. A
    /// root that holds one is only read, so a read-only one opens.
    pub fn establish(root: &Path, wanted: Layout) -> Result<Self> {
        if let Some(held) = Self::of_root(root)? {
            return Ok(held);
        }
        let marker = Key::new(Kind::Ref, FORMAT_NAME)?;
        let files = PosixBackend::new(root);
        if files.write_once(&marker, wanted.marker())? {
            return Ok(wanted);
        }
        Self::of_marker(&files.read(&marker)?)
    }

    /// A backend over `root` in this layout.
    pub fn backend(self, root: &Path) -> Arc<dyn StorageBackend> {
        match self {
            Layout::Loose => Arc::new(PosixBackend::new(root)),
            Layout::Packed => Arc::new(PackedBackend::new(root)),
        }
    }
}

/// A backend over the database that already exists under `root`, in its own
/// layout, without writing anything; `None` when there is no database there.
/// For reading a copy, as a repair does.
pub fn open_existing(root: &Path) -> Result<Option<Arc<dyn StorageBackend>>> {
    Ok(Layout::of_root(root)?.map(|layout| layout.backend(root)))
}
