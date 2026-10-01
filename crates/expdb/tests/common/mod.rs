// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Shared helpers for the specs: a scratch database root per test.
#![allow(dead_code, clippy::unwrap_used)]

use splinter_expdb::{Config, Database};
use tempfile::TempDir;

/// A database in its own temporary directory, removed on drop.
pub struct Scratch {
    pub dir: TempDir,
}

impl Scratch {
    pub fn new() -> Self {
        Self {
            dir: TempDir::new().unwrap(),
        }
    }

    pub fn open(&self) -> Database {
        Database::open(self.dir.path(), Config::default()).unwrap()
    }

    pub fn open_with(&self, config: Config) -> Database {
        Database::open(self.dir.path(), config).unwrap()
    }
}
