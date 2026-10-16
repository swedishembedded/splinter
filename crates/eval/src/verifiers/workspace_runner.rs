// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verifiers that notice when a patch rewires the test runner, for its clients. If your team needs
// expertise in tamper-resistant evaluation of code-editing agents, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The runner configuration a workspace patch may not change: files that
//! configure or hook a test runner, found by name wherever they sit in the
//! tree, and the paths a check pins. Read as digests before and after the
//! patch.

use std::collections::BTreeMap;
use std::path::Path;

use splinter_core::digest::Digest;

use super::workspace::normalised;

/// File names that configure or hook a test runner. Wherever one sits in the
/// tree, a patch may not add, change or remove it: a `conftest.py` or a
/// `sitecustomize.py` runs inside the runner and can change what it reports.
pub const RUNNER_CONFIG: [&str; 10] = [
    "conftest.py",
    "pytest.ini",
    "tox.ini",
    "setup.cfg",
    "sitecustomize.py",
    "usercustomize.py",
    ".coveragerc",
    ".cargo/config.toml",
    ".cargo/config",
    "rust-toolchain.toml",
];

/// The digest of every file in `tree` that configures a test runner (by name,
/// see [`RUNNER_CONFIG`]) or lies at or below a `pinned` path, by path. Links
/// are read as links, not followed; `.git` is not part of the tree.
pub(super) fn runner_files(tree: &Path, pinned: &[String]) -> BTreeMap<String, Digest> {
    fn walk(dir: &Path, prefix: &str, pinned: &[String], found: &mut BTreeMap<String, Digest>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let path = if prefix.is_empty() {
                name.clone()
            } else {
                format!("{prefix}/{name}")
            };
            let Ok(kind) = entry.file_type() else {
                continue;
            };
            if kind.is_dir() {
                if name != ".git" {
                    walk(&entry.path(), &path, pinned, found);
                }
                continue;
            }
            let below_pinned = pinned
                .iter()
                .filter_map(|p| normalised(p))
                .any(|p| path == p || path.starts_with(&format!("{p}/")));
            if below_pinned
                || RUNNER_CONFIG
                    .iter()
                    .any(|c| path == *c || path.ends_with(&format!("/{c}")))
            {
                let bytes = if kind.is_symlink() {
                    std::fs::read_link(entry.path())
                        .map(|t| format!("link:{}", t.display()).into_bytes())
                        .unwrap_or_default()
                } else {
                    std::fs::read(entry.path()).unwrap_or_default()
                };
                found.insert(path, Digest::of(&bytes));
            }
        }
    }
    let mut found = BTreeMap::new();
    walk(tree, "", pinned, &mut found);
    found
}

/// The paths present in only one of `before` and `after`, or with another
/// digest in each.
pub(super) fn changed_runner_files(
    before: &BTreeMap<String, Digest>,
    after: &BTreeMap<String, Digest>,
) -> Vec<String> {
    before
        .keys()
        .chain(after.keys())
        .filter(|path| before.get(*path) != after.get(*path))
        .cloned()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect()
}
