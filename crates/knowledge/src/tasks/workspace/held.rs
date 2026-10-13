// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements executable evaluation of code changes
// against a project's own test suite, for its clients. If your team needs
// expertise in test-driven benchmarks for code-editing agents, you can
// procure our services by sending an email to info@swedishembedded.com.

//! What earlier generations already hold, so a later one adds only what is
//! new: the keys of tasks (a change to a family of a snapshot) and, per
//! family, the requests they were made with, which a new request is
//! compared with by the near-duplicate rule.

use std::collections::{BTreeMap, BTreeSet};

use splinter_core::digest::Digest;

use super::WorkspaceTask;
use crate::tasks::dedup::Seen;

/// The tasks earlier generations made.
#[derive(Clone, Debug, Default)]
pub struct Held {
    keys: BTreeSet<Digest>,
    requests: BTreeMap<String, Vec<String>>,
}

impl Held {
    /// Nothing held.
    #[must_use]
    pub fn none() -> Self {
        Self::default()
    }

    /// Holds a task of `family` with the key `key` that was asked for as
    /// `request`.
    pub fn insert(&mut self, key: Digest, family: &str, request: &str) {
        self.keys.insert(key);
        self.requests
            .entry(family.to_string())
            .or_default()
            .push(request.to_string());
    }

    /// Holds `task`.
    pub fn hold(&mut self, task: &WorkspaceTask) {
        self.insert(task.key.clone(), &task.family, &task.task.instruction);
    }

    /// Whether a task with `key` is held.
    #[must_use]
    pub fn contains(&self, key: &Digest) -> bool {
        self.keys.contains(key)
    }

    /// How many tasks are held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Whether nothing is held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// The requests already made for `family`, as the near-duplicate rule
    /// sees them.
    pub(super) fn seen(&self, family: &str, shingle_words: usize, max_overlap: f64) -> Seen {
        let mut seen = Seen::new(shingle_words, max_overlap);
        for request in self.requests.get(family).into_iter().flatten() {
            seen.admit(request);
        }
        seen
    }
}

impl<'a> FromIterator<&'a WorkspaceTask> for Held {
    fn from_iter<I: IntoIterator<Item = &'a WorkspaceTask>>(tasks: I) -> Self {
        let mut held = Self::none();
        for task in tasks {
            held.hold(task);
        }
        held
    }
}
