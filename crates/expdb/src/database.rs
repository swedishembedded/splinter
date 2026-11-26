// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The database handle: a backend, the settings and a clock.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::backend::{Key, Kind, Layout, StorageBackend, FORMAT_NAME};
use crate::clock::{Clock, SystemClock};
use crate::config::Config;
use crate::error::{Error, Result};
use crate::id::ContentId;
use crate::manifest::Manifest;

/// An open database. Cheap to clone; clones share nothing but the files.
#[derive(Clone)]
pub struct Database {
    backend: Arc<dyn StorageBackend>,
    config: Arc<Config>,
    clock: Arc<dyn Clock>,
    manifests: Arc<Mutex<HashMap<ContentId, Arc<Manifest>>>>,
}

impl Database {
    /// Opens the database under `root` in the layout it holds, creating a
    /// packed one if the directory holds none.
    pub fn open(root: impl AsRef<Path>, config: Config) -> Result<Self> {
        Self::open_as(root, Layout::Packed, config)
    }

    /// Opens the database under `root` in the layout it holds, creating one
    /// in `layout` if the directory holds none. A root that already holds
    /// the other layout opens in that one: the marker decides, not the
    /// caller.
    pub fn open_as(root: impl AsRef<Path>, layout: Layout, config: Config) -> Result<Self> {
        let root = root.as_ref();
        let layout = Layout::establish(root, layout)?;
        Self::with_backend(layout.backend(root), config, Arc::new(SystemClock))
    }

    /// Opens a database over any backend and clock. A store whose format
    /// marker names another layout than the backend's is refused rather
    /// than written into.
    pub fn with_backend(
        backend: Arc<dyn StorageBackend>,
        config: Config,
        clock: Arc<dyn Clock>,
    ) -> Result<Self> {
        config.validate()?;
        let marker = Key::new(Kind::Ref, FORMAT_NAME)?;
        let wanted = backend.layout().marker();
        // An existing marker is only read: writing it again would touch a
        // committed, read-only database.
        let held = match backend.read(&marker) {
            Ok(bytes) => Some(bytes),
            Err(Error::NotFound { .. }) => None,
            Err(other) => return Err(other),
        };
        let matches = match held {
            Some(bytes) => bytes == wanted,
            None => backend.write_once(&marker, wanted)? || backend.read(&marker)? == wanted,
        };
        if !matches {
            return Err(Error::corrupt(
                "database format marker",
                format!(
                    "this is not a {} store",
                    String::from_utf8_lossy(wanted).trim()
                ),
            ));
        }
        Ok(Self {
            backend,
            config: Arc::new(config),
            clock,
            manifests: Arc::default(),
        })
    }

    /// The storage backend.
    pub fn backend(&self) -> &dyn StorageBackend {
        self.backend.as_ref()
    }

    /// A shared handle to the backend.
    pub fn backend_arc(&self) -> Arc<dyn StorageBackend> {
        Arc::clone(&self.backend)
    }

    /// The settings.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Manifests are immutable, so a handle remembers the ones it has read.
    pub(crate) fn manifest_cache(
        &self,
    ) -> std::sync::MutexGuard<'_, HashMap<ContentId, Arc<Manifest>>> {
        self.manifests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The clock records are stamped with.
    pub fn clock(&self) -> &dyn Clock {
        self.clock.as_ref()
    }
}
