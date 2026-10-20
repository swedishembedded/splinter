// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Working trees as values, and the exact comparison the gate rests on.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// What one path holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Entry {
    /// A regular file: its bytes and whether it is executable.
    File {
        /// The content.
        bytes: Vec<u8>,
        /// The owner-execute bit.
        executable: bool,
    },
    /// A symbolic link and where it points.
    Link(String),
}

/// A tree: repository-relative path (`/`-separated) to what it holds. The
/// `.git` directory is not part of a tree.
pub type Tree = BTreeMap<String, Entry>;

/// Reads the tree under `root`.
pub fn read_tree(root: &Path) -> Result<Tree> {
    let mut tree = Tree::new();
    walk(root, root, &mut tree)?;
    Ok(tree)
}

fn walk(root: &Path, dir: &Path, tree: &mut Tree) -> Result<()> {
    for entry in fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name();
        if dir == root && name == ".git" {
            continue;
        }
        let meta = fs::symlink_metadata(&path)?;
        let rel = path
            .strip_prefix(root)
            .with_context(|| format!("{} is outside {}", path.display(), root.display()))?
            .components()
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join("/");
        if meta.file_type().is_symlink() {
            let target = fs::read_link(&path)?.to_string_lossy().into_owned();
            tree.insert(rel, Entry::Link(target));
        } else if meta.is_dir() {
            walk(root, &path, tree)?;
        } else {
            use std::os::unix::fs::PermissionsExt;
            tree.insert(
                rel,
                Entry::File {
                    bytes: fs::read(&path)?,
                    executable: meta.permissions().mode() & 0o100 != 0,
                },
            );
        }
    }
    Ok(())
}

/// How a path of the attempt's tree differs from the commit's.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Residual {
    /// The commit has the path and the attempt does not.
    Missing {
        /// The path.
        path: String,
    },
    /// The attempt has a path the commit does not, and the parent did not
    /// have it either (junk).
    Extra {
        /// The path.
        path: String,
    },
    /// The parent had the path, the commit removes it, and the attempt still
    /// has it: work not done, not junk.
    Stale {
        /// The path.
        path: String,
    },
    /// Both have it with other bytes, mode or link target.
    Differs {
        /// The path.
        path: String,
        /// The first differing line (1-based) and what each side has there.
        first_difference: String,
    },
}

impl Residual {
    /// The path concerned.
    #[must_use]
    pub fn path(&self) -> &str {
        match self {
            Self::Missing { path }
            | Self::Extra { path }
            | Self::Stale { path }
            | Self::Differs { path, .. } => path,
        }
    }
}

/// Every path where `actual` is not `expected`, in path order, skipping the
/// paths `ignored` names (generated and vendored ones).
pub fn compare(expected: &Tree, actual: &Tree, ignored: &dyn Fn(&str) -> bool) -> Vec<Residual> {
    let mut residual = Vec::new();
    for path in expected
        .keys()
        .chain(actual.keys())
        .collect::<std::collections::BTreeSet<_>>()
    {
        if ignored(path) {
            continue;
        }
        match (expected.get(path), actual.get(path)) {
            (Some(_), None) => residual.push(Residual::Missing { path: path.clone() }),
            (None, Some(_)) => residual.push(Residual::Extra { path: path.clone() }),
            (Some(want), Some(got)) if want != got => residual.push(Residual::Differs {
                path: path.clone(),
                first_difference: first_difference(want, got),
            }),
            _ => {}
        }
    }
    residual
}

fn first_difference(want: &Entry, got: &Entry) -> String {
    let (
        Entry::File {
            bytes: want_bytes,
            executable: want_exec,
        },
        Entry::File {
            bytes: got_bytes,
            executable: got_exec,
        },
    ) = (want, got)
    else {
        return "one side is a symbolic link".into();
    };
    if want_bytes == got_bytes {
        return format!("mode differs (executable: expected {want_exec}, found {got_exec})");
    }
    let (want_text, got_text) = (
        String::from_utf8_lossy(want_bytes),
        String::from_utf8_lossy(got_bytes),
    );
    let (mut want_lines, mut got_lines) = (want_text.lines(), got_text.lines());
    for number in 1.. {
        let (w, g) = (want_lines.next(), got_lines.next());
        if w != g {
            let show = |l: Option<&str>| {
                l.map_or("<end of file>".to_string(), |l| {
                    l.chars().take(120).collect()
                })
            };
            return format!("line {number}: expected {:?}, found {:?}", show(w), show(g));
        }
        if w.is_none() {
            break;
        }
    }
    "bytes differ (line endings or a final newline)".into()
}

/// Every identifier-shaped word (three or more characters) of the text files
/// of `tree`: what the parent's code already says, against which a
/// commit's new identifiers are told apart.
#[must_use]
pub fn vocabulary(tree: &Tree) -> std::collections::HashSet<String> {
    let mut words = std::collections::HashSet::new();
    for entry in tree.values() {
        if let Entry::File { bytes, .. } = entry {
            if let Ok(text) = std::str::from_utf8(bytes) {
                words.extend(identifiers(text).map(String::from));
                words.extend(dashed_names(text));
            }
        }
    }
    words
}

/// The identifier-shaped words of `text` (letters, digits and underscores, at
/// least four characters, not starting with a digit).
pub fn identifiers(text: &str) -> impl Iterator<Item = &str> {
    text.split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|w| w.chars().count() >= 4 && !w.starts_with(|c: char| c.is_ascii_digit()))
}

/// The names of `text` written with hyphens (`acme-core-m`, a crate or
/// package name), spelled as code spells them (`acme_core_m`): a manifest
/// says the one and the code that uses it the other, and they are one name.
#[must_use]
pub fn dashed_names(text: &str) -> Vec<String> {
    let dashed = regex::Regex::new(r"[A-Za-z_][A-Za-z0-9_]*(?:-[A-Za-z0-9_]+)+")
        .unwrap_or_else(|_| unreachable!("a fixed pattern"));
    dashed
        .find_iter(text)
        .map(|m| m.as_str().replace('-', "_"))
        .filter(|n| n.chars().count() >= 4)
        .collect()
}
