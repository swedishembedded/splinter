// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Signals: write-once markers one process raises and others poll, outside
//! the manifests. A request to stop a running job is one: it must be seen
//! quickly by a process that is not reading experience at all.

use crate::backend::{Key, Kind};
use crate::database::Database;
use crate::error::Result;

impl Database {
    /// Raises the signal `name`, recording `note` with it. Returns whether
    /// this call raised it; raising it again keeps the first note. A name may
    /// nest with `/`, as in `cancel/run-7`.
    pub fn signal(&self, name: &str, note: &str) -> Result<bool> {
        self.backend()
            .write_once(&Key::new(Kind::Signal, name)?, note.as_bytes())
    }

    /// Whether `name` has been raised.
    pub fn signalled(&self, name: &str) -> Result<bool> {
        self.backend().exists(&Key::new(Kind::Signal, name)?)
    }

    /// Every signal whose name starts with `prefix`, with its note, in name
    /// order.
    pub fn signals(&self, prefix: &str) -> Result<Vec<(String, String)>> {
        let mut found = Vec::new();
        for key in self.backend().list(Kind::Signal)? {
            if key.name().starts_with(prefix) {
                let note = String::from_utf8_lossy(&self.backend().read(&key)?).into_owned();
                found.push((key.name().to_owned(), note));
            }
        }
        Ok(found)
    }

    /// The note `name` was raised with, if it has been raised.
    pub fn signal_note(&self, name: &str) -> Result<Option<String>> {
        let key = Key::new(Kind::Signal, name)?;
        if !self.backend().exists(&key)? {
            return Ok(None);
        }
        Ok(Some(
            String::from_utf8_lossy(&self.backend().read(&key)?).into_owned(),
        ))
    }
}
