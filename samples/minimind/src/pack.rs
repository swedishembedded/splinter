// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training pipelines that keep their run
// directories small and verifiable on shared storage, for its clients. If
// your team needs expertise in reproducible run records on quota-bound file
// systems, you can procure our services by sending an email to
// info@swedishembedded.com.

//! A tree of small text files packed into one JSON-lines file, so a stage
//! with thousands of episodes or task folders leaves one file behind instead
//! of thousands (shared file systems bound the number of files).
//!
//! A pack holds one `{"path", "mode", "text"}` line per file, in path order,
//! so the same tree always makes the same bytes. [`unpack`] writes the tree
//! back; [`Pack::read`] reads one member without writing anything.
//!
//! A stage's episodes are [`append`]ed to its pack as each finishes, so a
//! run killed mid-stage keeps every finished episode in one file; a torn
//! last line (a kill during the write) is ignored, a path appended again is
//! read as its last copy, and [`canonical`] rewrites the pack in path order
//! once the stage is done.

use std::collections::BTreeMap;
use std::os::unix::fs::PermissionsExt as _;
use std::path::{Component, Path};

use anyhow::{bail, Context as _};
use serde::{Deserialize, Serialize};

use crate::stages::write;

/// One packed file.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Member {
    path: String,
    mode: u32,
    text: String,
}

/// Every file below `dir`, by relative path.
fn members(dir: &Path) -> anyhow::Result<Vec<Member>> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(at) = stack.pop() {
        for entry in std::fs::read_dir(&at).with_context(|| format!("reading {}", at.display()))? {
            let entry = entry?;
            let path = entry.path();
            let kind = entry.file_type()?;
            if kind.is_dir() {
                stack.push(path);
            } else if kind.is_file() {
                let rel = path
                    .strip_prefix(dir)
                    .context("a member is below its tree")?
                    .to_string_lossy()
                    .into_owned();
                let text = std::fs::read_to_string(&path).with_context(|| {
                    format!("{} is not text: it cannot be packed", path.display())
                })?;
                let mode = entry.metadata()?.permissions().mode() & 0o777;
                out.push(Member {
                    path: rel,
                    mode,
                    text,
                });
            } else {
                bail!(
                    "{} is neither a file nor a directory: it cannot be packed",
                    path.display()
                );
            }
        }
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(out)
}

/// Packs the tree `dir` into `out` (written atomically), then removes
/// `dir`. Returns how many files it holds.
///
/// # Errors
/// A file is not text, or the pack cannot be written.
pub fn pack_tree(dir: &Path, out: &Path) -> anyhow::Result<usize> {
    let members = members(dir)?;
    let mut text = String::new();
    for m in &members {
        text.push_str(&serde_json::to_string(m)?);
        text.push('\n');
    }
    write(out, &text)?;
    std::fs::remove_dir_all(dir).with_context(|| format!("removing {}", dir.display()))?;
    Ok(members.len())
}

/// Appends the files of `tree/rel` to the pack `archive` as members
/// `rel/<path>`, in one write. Returns how many it appended.
///
/// # Errors
/// A file is not text, or the pack cannot be written.
pub fn append(archive: &Path, tree: &Path, rel: &str) -> anyhow::Result<usize> {
    use std::io::Write as _;
    let members = members(&tree.join(rel))?;
    let mut text = String::new();
    for mut m in members.iter().cloned() {
        m.path = format!("{rel}/{}", m.path);
        text.push_str(&serde_json::to_string(&m)?);
        text.push('\n');
    }
    if let Some(parent) = archive.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(archive)
        .with_context(|| format!("opening {}", archive.display()))?;
    file.write_all(text.as_bytes())
        .with_context(|| format!("appending to {}", archive.display()))?;
    Ok(members.len())
}

/// Rewrites the pack `archive` as [`pack_tree`] would write its tree: each
/// path once (its last copy), in path order.
///
/// # Errors
/// The pack cannot be read or written.
pub fn canonical(archive: &Path) -> anyhow::Result<()> {
    let pack = Pack::open(archive)?;
    let mut text = String::new();
    for m in pack.0.values() {
        text.push_str(&serde_json::to_string(m)?);
        text.push('\n');
    }
    write(archive, &text)
}

/// A member path that stays inside its tree.
fn safe(rel: &str) -> bool {
    !rel.is_empty()
        && Path::new(rel)
            .components()
            .all(|c| matches!(c, Component::Normal(_)))
}

/// A pack, read into memory.
#[derive(Clone, Debug, Default)]
pub struct Pack(BTreeMap<String, Member>);

impl Pack {
    /// The pack at `path`.
    ///
    /// # Errors
    /// It cannot be read, a line is not a member, or a member names a path
    /// outside its tree.
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let mut members = BTreeMap::new();
        // Every complete line ends with a newline: a last line without one
        // was torn by a kill while it was appended, and is no member.
        let whole = text.rfind('\n').map_or("", |end| &text[..=end]);
        for line in whole.lines() {
            let m: Member =
                serde_json::from_str(line).with_context(|| format!("in {}", path.display()))?;
            if !safe(&m.path) {
                bail!("{} names {:?}, outside its tree", path.display(), m.path);
            }
            members.insert(m.path.clone(), m);
        }
        Ok(Self(members))
    }

    /// The text of member `rel`, if the pack holds it.
    #[must_use]
    pub fn read(&self, rel: &str) -> Option<&str> {
        self.0.get(rel).map(|m| m.text.as_str())
    }
}

/// Writes the pack at `pack` back as the tree `dir`, with its file modes.
/// The pack stays: it is the durable copy, the tree a working one.
///
/// # Errors
/// The pack cannot be read or a file cannot be written.
pub fn unpack(pack: &Path, dir: &Path) -> anyhow::Result<()> {
    for m in Pack::open(pack)?.0.values() {
        let path = dir.join(&m.path);
        write(&path, &m.text)?;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(m.mode))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn episodes_appended_as_they_finish_survive_a_kill_and_pack_as_a_tree_would() {
        let root = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let tree = root.path().join("tree");
        let archive = root.path().join("stage.jsonl");
        let put = |rel: &str, text: &str| {
            write(&tree.join(rel), text).unwrap_or_else(|e| panic!("{e}"));
        };
        put("b/outcome.json", "first");
        append(&archive, &tree, "b").unwrap_or_else(|e| panic!("{e}"));
        put("a/outcome.json", "a");
        append(&archive, &tree, "a").unwrap_or_else(|e| panic!("{e}"));
        // b ran again (a new attempt after an interruption), then a kill
        // tore the next append.
        put("b/outcome.json", "again");
        append(&archive, &tree, "b").unwrap_or_else(|e| panic!("{e}"));
        let mut torn = std::fs::read_to_string(&archive).unwrap_or_else(|e| panic!("{e}"));
        torn.push_str("{\"path\":\"c/outc");
        std::fs::write(&archive, torn).unwrap_or_else(|e| panic!("{e}"));
        let pack = Pack::open(&archive).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(pack.read("b/outcome.json"), Some("again"));
        assert_eq!(pack.read("a/outcome.json"), Some("a"));
        // Unpacking keeps the durable copy; once the stage is done the pack
        // is the bytes its tree packs into.
        let copy = root.path().join("copy");
        unpack(&archive, &copy).unwrap_or_else(|e| panic!("{e}"));
        assert!(archive.exists());
        canonical(&archive).unwrap_or_else(|e| panic!("{e}"));
        let packed = root.path().join("tree.jsonl");
        pack_tree(&copy, &packed).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            std::fs::read(&archive).unwrap_or_else(|e| panic!("{e}")),
            std::fs::read(&packed).unwrap_or_else(|e| panic!("{e}"))
        );
    }

    #[test]
    fn a_tree_packs_into_one_file_and_comes_back_byte_for_byte() {
        let root = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let tree = root.path().join("tree");
        write(&tree.join("a/outcome.json"), "{}").unwrap_or_else(|e| panic!("{e}"));
        write(&tree.join("b/run.sh"), "#!/bin/sh\n").unwrap_or_else(|e| panic!("{e}"));
        std::fs::set_permissions(
            tree.join("b/run.sh"),
            std::fs::Permissions::from_mode(0o755),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let pack = root.path().join("tree.jsonl");
        assert_eq!(pack_tree(&tree, &pack).unwrap_or_else(|e| panic!("{e}")), 2);
        assert!(!tree.exists(), "the tree is gone, one file is left");
        let first = std::fs::read(&pack).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            Pack::open(&pack)
                .unwrap_or_else(|e| panic!("{e}"))
                .read("a/outcome.json"),
            Some("{}")
        );
        unpack(&pack, &tree).unwrap_or_else(|e| panic!("{e}"));
        let mode = std::fs::metadata(tree.join("b/run.sh"))
            .unwrap_or_else(|e| panic!("{e}"))
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o755);
        pack_tree(&tree, &pack).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            std::fs::read(&pack).unwrap_or_else(|e| panic!("{e}")),
            first,
            "the same tree, the same bytes"
        );
        std::fs::write(&pack, "{\"path\":\"../x\",\"mode\":420,\"text\":\"\"}\n")
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(
            Pack::open(&pack).is_err(),
            "a member outside its tree is refused"
        );
    }
}
