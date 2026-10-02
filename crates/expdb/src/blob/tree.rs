// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Merkle trees of files: a directory is an object naming its children by
//! content id, so changing one file stores that file and its ancestors and
//! shares everything else with every earlier snapshot.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::store::{BlobRef, BlobStore};
use crate::error::{Error, Result};
use crate::id::ContentId;

/// One name in a directory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Entry {
    /// A file's bytes.
    File {
        /// Where the bytes are.
        blob: BlobRef,
    },
    /// A subdirectory, by the content id of its directory object.
    Dir {
        /// The subdirectory object.
        tree: ContentId,
    },
}

/// Operations on trees stored in a [`BlobStore`]. A tree is named by the
/// content id of its root directory object.
pub struct Tree;

type Dir = BTreeMap<String, Entry>;

impl Tree {
    /// Stores a whole tree from `path -> bytes` and returns its root.
    pub fn write(store: &mut BlobStore, files: &BTreeMap<String, Vec<u8>>) -> Result<ContentId> {
        let mut root = Dir::new();
        let mut subtrees: BTreeMap<String, BTreeMap<String, Vec<u8>>> = BTreeMap::new();
        for (path, bytes) in files {
            match path.split_once('/') {
                None => {
                    check_name(path)?;
                    root.insert(
                        path.clone(),
                        Entry::File {
                            blob: store.put(bytes)?,
                        },
                    );
                }
                Some((head, rest)) => {
                    check_name(head)?;
                    subtrees
                        .entry(head.to_owned())
                        .or_default()
                        .insert(rest.to_owned(), bytes.clone());
                }
            }
        }
        for (name, sub) in subtrees {
            if root.contains_key(&name) {
                return Err(Error::invalid(
                    "tree path",
                    format!("`{name}` is both a file and a directory"),
                ));
            }
            root.insert(
                name,
                Entry::Dir {
                    tree: Self::write(store, &sub)?,
                },
            );
        }
        put_dir(store, &root)
    }

    /// A new root with `path` set to `bytes`; only the directories on the
    /// path and the new file are stored.
    pub fn update(
        store: &mut BlobStore,
        root: ContentId,
        path: &str,
        bytes: &[u8],
    ) -> Result<ContentId> {
        let mut dir = load_dir(store, root)?;
        match path.split_once('/') {
            None => {
                check_name(path)?;
                dir.insert(
                    path.to_owned(),
                    Entry::File {
                        blob: store.put(bytes)?,
                    },
                );
            }
            Some((head, rest)) => {
                check_name(head)?;
                let child = match dir.get(head) {
                    Some(Entry::Dir { tree }) => *tree,
                    Some(Entry::File { .. }) => {
                        return Err(Error::invalid(
                            "tree path",
                            format!("`{head}` is a file, not a directory"),
                        ))
                    }
                    None => put_dir(store, &Dir::new())?,
                };
                dir.insert(
                    head.to_owned(),
                    Entry::Dir {
                        tree: Self::update(store, child, rest, bytes)?,
                    },
                );
            }
        }
        put_dir(store, &dir)
    }

    /// The bytes of the file at `path`.
    pub fn read_file(store: &BlobStore, root: ContentId, path: &str) -> Result<Vec<u8>> {
        let mut dir = load_dir(store, root)?;
        let mut parts = path.split('/').peekable();
        while let Some(part) = parts.next() {
            let entry = dir.get(part).ok_or_else(|| Error::NotFound {
                what: format!("tree path `{path}`"),
            })?;
            match (entry, parts.peek()) {
                (Entry::File { blob }, None) => return store.get(blob),
                (Entry::Dir { tree }, Some(_)) => dir = load_dir(store, *tree)?,
                _ => {
                    return Err(Error::NotFound {
                        what: format!("tree path `{path}`"),
                    })
                }
            }
        }
        Err(Error::NotFound {
            what: format!("tree path `{path}`"),
        })
    }

    /// Every file in the tree, as `path -> blob`.
    pub fn files(store: &BlobStore, root: ContentId) -> Result<BTreeMap<String, BlobRef>> {
        let mut found = BTreeMap::new();
        let mut pending = vec![(String::new(), root)];
        while let Some((prefix, tree)) = pending.pop() {
            for (name, entry) in load_dir(store, tree)? {
                let path = format!("{prefix}{name}");
                match entry {
                    Entry::File { blob } => {
                        found.insert(path, blob);
                    }
                    Entry::Dir { tree } => pending.push((format!("{path}/"), tree)),
                }
            }
        }
        Ok(found)
    }
}

fn check_name(name: &str) -> Result<()> {
    if name.is_empty() || name == "." || name == ".." {
        return Err(Error::invalid(
            "tree path",
            format!("`{name}` is not a valid name"),
        ));
    }
    Ok(())
}

fn put_dir(store: &mut BlobStore, dir: &Dir) -> Result<ContentId> {
    let bytes = serde_json::to_vec(dir).map_err(|source| Error::Encode {
        what: "directory",
        source,
    })?;
    Ok(store.put(&bytes)?.id)
}

fn load_dir(store: &BlobStore, id: ContentId) -> Result<Dir> {
    // A directory object is small, so its length is not needed to read it
    // back; the content id verifies the bytes.
    let bytes = store.get_by_id(&id)?;
    serde_json::from_slice(&bytes).map_err(|source| Error::Decode {
        what: format!("directory {id}"),
        source,
    })
}
