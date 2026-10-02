// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Object keys: a kind and a validated name.

use crate::error::{Error, Result};

/// What an object is, which decides where a backend puts it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    /// A sealed segment of records.
    Segment,
    /// A pack of content-addressed blob chunks.
    BlobPack,
    /// An immutable manifest.
    Manifest,
    /// An immutable index run.
    Index,
    /// A named pointer to a manifest, replaced atomically.
    Ref,
    /// A marker that keeps a manifest's files alive.
    Pin,
    /// A microsegment waiting for an aggregator.
    Spool,
    /// A trainer's mutable replay state, kept apart from the experience.
    Overlay,
    /// A cache of data derived from experience, keyed by what it came from.
    Cache,
    /// Files a collection has set aside and may still restore.
    Trash,
    /// A write-once marker one process raises for others to notice.
    Signal,
}

impl Kind {
    /// The directory the kind lives in.
    pub fn dir(self) -> &'static str {
        match self {
            Kind::Segment => "segments",
            Kind::BlobPack => "blobs",
            Kind::Manifest => "manifests",
            Kind::Index => "indexes",
            Kind::Ref => "refs",
            Kind::Pin => "pins",
            Kind::Spool => "spool",
            Kind::Overlay => "overlay",
            Kind::Cache => "cache",
            Kind::Trash => "trash",
            Kind::Signal => "signals",
        }
    }

    /// Whether names are spread over two-character subdirectories, so no
    /// directory grows without bound. Pointer kinds may nest by `/` instead.
    pub fn fans_out(self) -> bool {
        matches!(
            self,
            Kind::Segment | Kind::BlobPack | Kind::Manifest | Kind::Spool
        )
    }

    /// Whether the name is the hash of the bytes, so the bytes can be checked
    /// against it.
    pub fn is_addressed(self) -> bool {
        matches!(
            self,
            Kind::Segment | Kind::BlobPack | Kind::Manifest | Kind::Index
        )
    }

    /// Every kind.
    pub const ALL: [Kind; 11] = [
        Kind::Segment,
        Kind::BlobPack,
        Kind::Manifest,
        Kind::Index,
        Kind::Ref,
        Kind::Pin,
        Kind::Spool,
        Kind::Overlay,
        Kind::Cache,
        Kind::Trash,
        Kind::Signal,
    ];
}

/// The address of one object.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Key {
    kind: Kind,
    name: String,
}

impl Key {
    /// A key, refusing names that could leave the root or collide with a
    /// temporary file.
    pub fn new(kind: Kind, name: &str) -> Result<Self> {
        let valid_part = |part: &str| {
            !part.is_empty()
                && part != "."
                && part != ".."
                && !part.starts_with('.')
                && part
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'_'))
        };
        let ok = if kind.fans_out() {
            valid_part(name)
        } else {
            name.split('/').all(valid_part)
        };
        if !ok {
            return Err(Error::invalid(
                "object name",
                format!("`{name}` is not a valid {} name", kind.dir()),
            ));
        }
        Ok(Self {
            kind,
            name: name.to_owned(),
        })
    }

    /// The kind.
    pub fn kind(&self) -> Kind {
        self.kind
    }

    /// The name within the kind.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Where the object lies under a database root, with `/` separators:
    /// the kind's directory, a two-character fan-out directory where the
    /// kind has one, then the name.
    pub fn relative_path(&self) -> String {
        if self.kind.fans_out() {
            let fan: String = self.name.chars().take(2).collect();
            format!("{}/{fan}/{}", self.kind.dir(), self.name)
        } else {
            format!("{}/{}", self.kind.dir(), self.name)
        }
    }

    /// The key of the object at `path` under a database root, the inverse of
    /// [`Key::relative_path`]; `None` for a path that is not an object's.
    pub fn from_relative(path: &str) -> Option<Self> {
        let (dir, rest) = path.split_once('/')?;
        let kind = Kind::ALL.into_iter().find(|k| k.dir() == dir)?;
        let name = if kind.fans_out() {
            let (fan, name) = rest.split_once('/')?;
            if name.chars().take(2).collect::<String>() != fan {
                return None;
            }
            name
        } else {
            rest
        };
        Self::new(kind, name).ok()
    }
}
