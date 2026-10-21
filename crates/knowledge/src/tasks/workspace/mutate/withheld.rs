// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements executable evaluation of code changes
// against a project's own test suite, for its clients. If your team needs
// expertise in test-driven benchmarks for code-editing agents, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The tests a solver's tree lacks, whole files or only failing cases.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use splinter_core::experience::{Privileged, PrivilegedKind};

use super::{Killed, Parent, HIDDEN_TEST};

/// The tests the solver's tree lacks.
pub(super) struct Withheld {
    /// Test files taken out whole: every test in them is hidden.
    pub(super) files: BTreeSet<String>,
    /// Test files that stay in the tree without the failing cases (the ids)
    /// named here, because tests that stay import them.
    pub(super) cases: BTreeMap<String, BTreeSet<String>>,
}

impl Withheld {
    /// Every test file that kills the mutant, whole.
    pub(super) fn files(killed: &Killed) -> Self {
        Self {
            files: killed.files.clone(),
            cases: BTreeMap::new(),
        }
    }

    /// Only the failing cases of the files that kill the mutant.
    pub(super) fn cases(base: &Parent<'_>, killed: &Killed) -> Self {
        let mut cases: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for id in &killed.failing {
            if let Some(file) = base.locate(id) {
                cases.entry(file).or_default().insert(id.clone());
            }
        }
        Self {
            files: BTreeSet::new(),
            cases,
        }
    }

    /// Whether the solver's tree lacks the test `id`.
    pub(super) fn hides(&self, base: &Parent<'_>, id: &str) -> bool {
        base.locate(id).is_some_and(|file| {
            self.files.contains(&file) || self.cases.get(&file).is_some_and(|ids| ids.contains(id))
        })
    }

    /// The text of the tests the solver's tree lacks: the files of `hidden`
    /// whole, and of the others the lines that `solver` no longer has.
    pub(super) fn text(
        &self,
        hidden: &Path,
        solver: &Path,
        files: &BTreeSet<String>,
    ) -> Privileged {
        let read = |tree: &Path, file: &String| {
            std::fs::read(tree.join(file))
                .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
                .ok()
        };
        let content: String = files
            .iter()
            .filter_map(|file| {
                let original = read(hidden, file)?;
                match read(solver, file) {
                    Some(kept) => Some(lines_missing(&original, &kept)),
                    None => Some(original),
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        Privileged {
            kind: PrivilegedKind::Other(HIDDEN_TEST.into()),
            content,
            span: None,
        }
    }
}

/// The lines of `original` that `kept` (a copy of it with some definitions
/// removed) has no occurrence left for, in order: what a reader of `kept`
/// cannot see.
fn lines_missing(original: &str, kept: &str) -> String {
    let mut left: BTreeMap<&str, usize> = BTreeMap::new();
    for line in kept.lines() {
        *left.entry(line).or_default() += 1;
    }
    original
        .lines()
        .filter(|line| match left.get_mut(line) {
            Some(count) if *count > 0 => {
                *count -= 1;
                false
            }
            _ => true,
        })
        .collect::<Vec<_>>()
        .join("\n")
}
