// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements durable, content-addressed experience
// stores for learning agents, for its clients. If your team needs expertise
// in training-data lineage or crash-safe storage, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Pointers: the few mutable names Splinter keeps, such as the release the
//! alias `default` means or the version of the anchor suite in force.
//!
//! A pointer is its whole history, never a value overwritten. Version `n+1` is
//! claimed by creating one file that cannot be created twice, so when two
//! processes move a pointer from the same value exactly one wins and the other
//! learns it lost; there is no lock file to go stale. Moving a pointer back is
//! moving it to an older value, and what it pointed at, and when, stays on
//! record.

use serde::{Deserialize, Serialize};

use crate::error::StoreError;
use crate::workspace::Workspace;

/// One move of a pointer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PointerMove {
    /// Its version, counting from one.
    pub version: u64,
    /// What it pointed at from then on.
    pub value: String,
    /// What it pointed at before; `None` for the first move.
    pub previous: Option<String>,
    /// When, as the caller stamped it.
    pub at: String,
}

fn prefix(name: &str) -> String {
    format!("pointer/{name}/")
}

fn signal_name(name: &str, version: u64) -> String {
    format!("{}{version:012}", prefix(name))
}

impl Workspace {
    /// Every move of the pointer `name`, oldest first.
    pub fn pointer_history(&self, name: &str) -> Result<Vec<PointerMove>, StoreError> {
        let mut moves = Vec::new();
        for (signal, note) in self.signals(&prefix(name))? {
            let version = signal[prefix(name).len()..].parse::<u64>().map_err(|_| {
                StoreError::UndecodableObject {
                    what: format!("pointer {name}"),
                    reason: format!("{signal} is not a version"),
                }
            })?;
            let mut found: PointerMove =
                serde_json::from_str(&note).map_err(|e| StoreError::UndecodableObject {
                    what: format!("pointer {name} version {version}"),
                    reason: e.to_string(),
                })?;
            found.version = version;
            moves.push(found);
        }
        moves.sort_by_key(|m| m.version);
        Ok(moves)
    }

    /// Every pointer whose name starts with `prefix`, with what it points at
    /// now, in name order.
    pub fn pointers(&self, prefix: &str) -> Result<Vec<(String, String)>, StoreError> {
        let mut latest: std::collections::BTreeMap<String, (u64, String)> = Default::default();
        for (signal, note) in self.signals(&format!("pointer/{prefix}"))? {
            let rest = &signal["pointer/".len()..];
            let Some((name, version)) = rest.rsplit_once('/') else {
                continue;
            };
            let version = version.parse::<u64>().unwrap_or(0);
            let value = serde_json::from_str::<PointerMove>(&note)
                .map(|m| m.value)
                .map_err(|e| StoreError::UndecodableObject {
                    what: format!("pointer {name}"),
                    reason: e.to_string(),
                })?;
            let entry = latest.entry(name.to_owned()).or_insert((0, String::new()));
            if version >= entry.0 {
                *entry = (version, value);
            }
        }
        Ok(latest
            .into_iter()
            .map(|(name, (_, value))| (name, value))
            .collect())
    }

    /// What the pointer `name` points at now, with its version.
    pub fn pointer(&self, name: &str) -> Result<Option<(u64, String)>, StoreError> {
        Ok(self
            .pointer_history(name)?
            .pop()
            .map(|latest| (latest.version, latest.value)))
    }

    /// Moves the pointer `name` to `to`, if it still points at `expected`
    /// (`None` for a pointer that has never been set). Returns the new
    /// version. Moving a pointer to the value it already has changes nothing.
    pub fn move_pointer(
        &self,
        name: &str,
        expected: Option<&str>,
        to: &str,
        at: &str,
    ) -> Result<u64, StoreError> {
        let current = self.pointer(name)?;
        let found = current.as_ref().map(|(_, value)| value.as_str());
        let conflict = |found: Option<&str>| StoreError::PointerConflict {
            pointer: name.to_owned(),
            expected: expected.map(str::to_owned),
            found: found.map(str::to_owned),
        };
        if found != expected {
            return Err(conflict(found));
        }
        let version = current.as_ref().map_or(0, |(version, _)| *version) + 1;
        let note = serde_json::to_string(&PointerMove {
            version,
            value: to.to_owned(),
            previous: expected.map(str::to_owned),
            at: at.to_owned(),
        })
        .map_err(|source| StoreError::Serialize {
            what: "pointer move",
            source,
        })?;
        if self.signal(&signal_name(name, version), &note)? {
            Ok(version)
        } else {
            // Another process claimed this version first.
            let now = self.pointer(name)?.map(|(_, value)| value);
            Err(conflict(now.as_deref()))
        }
    }
}
