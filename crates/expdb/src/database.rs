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

use crate::backend::{Key, Kind, PosixBackend, StorageBackend};
use crate::clock::{Clock, SystemClock};
use crate::config::Config;
use crate::error::{Error, Result};
use crate::id::ContentId;
use crate::manifest::Manifest;

/// The marker that says a directory is a database, and of which format.
const FORMAT_NAME: &str = "format";
const FORMAT_TEXT: &[u8] = b"splinter-expdb 2\n";

/// An open database. Cheap to clone; clones share nothing but the files.
#[derive(Clone)]
pub struct Database {
    backend: Arc<dyn StorageBackend>,
    config: Arc<Config>,
    clock: Arc<dyn Clock>,
    manifests: Arc<Mutex<HashMap<ContentId, Arc<Manifest>>>>,
}

impl Database {
    /// Opens the database under `root`, creating it if the directory is new.
    pub fn open(root: impl AsRef<Path>, config: Config) -> Result<Self> {
        Self::with_backend(
            Arc::new(PosixBackend::new(root.as_ref())),
            config,
            Arc::new(SystemClock),
        )
    }

    /// Opens a database over any backend and clock. A store holding another
    /// format is refused rather than written into.
    pub fn with_backend(
        backend: Arc<dyn StorageBackend>,
        config: Config,
        clock: Arc<dyn Clock>,
    ) -> Result<Self> {
        config.validate()?;
        let marker = Key::new(Kind::Ref, FORMAT_NAME)?;
        if !backend.write_once(&marker, FORMAT_TEXT)? && backend.read(&marker)? != FORMAT_TEXT {
            return Err(Error::corrupt(
                "database format marker",
                "this is not a splinter-expdb 2 store",
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
