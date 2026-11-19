// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in migrating
// content-addressed stores without losing a byte, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Rewriting a database, in either layout, into a fresh packed one.
//!
//! Every object is copied under its own key, keeping its age, and checked on
//! both sides: an object named by its hash must hash to its name before it is
//! copied, and every copy must read back as the bytes that were read. A
//! packed source is compacted on the way, because only live objects are
//! copied. The source is never written.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use std::time::SystemTime;

use super::{Key, Kind, Layout, PackedBackend, StorageBackend, FORMAT_NAME};
use crate::error::{Error, Result};
use crate::id::ContentId;

/// Objects are copied in batches of about this many bytes: one sync of the
/// packs and one of the journal per batch, and bounded memory.
const BATCH_BYTES: usize = 64 * 1024 * 1024;

/// What a repack did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct Repacked {
    /// Objects copied: packed objects, refs, pins and overlays.
    pub objects: u64,
    /// Their bytes.
    pub bytes: u64,
    /// Files under the source root.
    pub files_before: u64,
    /// Files under the target root.
    pub files_after: u64,
}

/// Copies the database under `source` into a new packed database under
/// `target`, which must not hold one, and verifies the copy. Refused, with
/// `target` left for the caller to remove, when an object of the source does
/// not hash to its name or the source changed while it was copied.
pub fn repack(source: &Path, target: &Path) -> Result<Repacked> {
    let Some(layout) = Layout::of_root(source)? else {
        return Err(Error::NotFound {
            what: format!("a database under {}", source.display()),
        });
    };
    if target.exists()
        && fs::read_dir(target)
            .map_err(Error::io(target))?
            .next()
            .is_some()
    {
        return Err(Error::invalid(
            "repack target",
            format!("{} is not empty", target.display()),
        ));
    }
    let from = layout.backend(source);
    fs::create_dir_all(target).map_err(Error::io(target))?;
    Layout::establish(target, Layout::Packed)?;
    let to = PackedBackend::new(target);

    let mut report = Repacked {
        files_before: count_files(source)?,
        ..Repacked::default()
    };
    let mut copied: BTreeMap<Key, blake3::Hash> = BTreeMap::new();
    let mut listed: BTreeSet<Key> = BTreeSet::new();
    for kind in Kind::ALL {
        let mut batch: Vec<(Key, Vec<u8>, SystemTime)> = Vec::new();
        let mut batch_bytes = 0;
        for key in from.list(kind)? {
            if kind == Kind::Ref && key.name() == FORMAT_NAME {
                continue;
            }
            listed.insert(key.clone());
            let bytes = from.read(&key)?;
            if kind.is_addressed() {
                let name = key.name().split('.').next().unwrap_or_default();
                if ContentId::of(&bytes).to_string() != name {
                    return Err(Error::corrupt(
                        format!("{} `{}`", kind.dir(), key.name()),
                        "its bytes do not hash to its name; repair the state before packing it",
                    ));
                }
            }
            copied.insert(key.clone(), blake3::hash(&bytes));
            report.objects += 1;
            report.bytes += bytes.len() as u64;
            if matches!(kind, Kind::Ref | Kind::Pin | Kind::Overlay) {
                to.replace(&key, &bytes)?;
                continue;
            }
            let modified = from.modified(&key)?;
            batch_bytes += bytes.len();
            batch.push((key, bytes, modified));
            if batch_bytes >= BATCH_BYTES {
                to.import(&batch)?;
                batch.clear();
                batch_bytes = 0;
            }
        }
        to.import(&batch)?;
    }

    // The copy reads back as what was read, and holds nothing else.
    for (key, digest) in &copied {
        if blake3::hash(&to.read(key)?) != *digest {
            return Err(Error::corrupt(
                format!("the copy of {} `{}`", key.kind().dir(), key.name()),
                "it does not read back as the bytes copied",
            ));
        }
    }
    // Nothing was written to the source meanwhile: a writer's new object
    // would be missing from the copy.
    let mut again = BTreeSet::new();
    for kind in Kind::ALL {
        for key in from.list(kind)? {
            if !(kind == Kind::Ref && key.name() == FORMAT_NAME) {
                again.insert(key);
            }
        }
    }
    if again != listed {
        return Err(Error::invalid(
            "repack source",
            "the database changed while it was copied; stop every process using it and pack again",
        ));
    }
    for kind in Kind::ALL {
        let held: BTreeSet<Key> = to
            .list(kind)?
            .into_iter()
            .filter(|key| !(kind == Kind::Ref && key.name() == FORMAT_NAME))
            .collect();
        let wanted: BTreeSet<Key> = listed
            .iter()
            .filter(|k| k.kind() == kind)
            .cloned()
            .collect();
        if held != wanted {
            return Err(Error::corrupt(
                format!("the copy's {}", kind.dir()),
                "it does not hold exactly the source's objects",
            ));
        }
    }
    report.files_after = count_files(target)?;
    Ok(report)
}

/// Every file under `root`, at any depth.
pub(crate) fn count_files(root: &Path) -> Result<u64> {
    let mut count = 0;
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in fs::read_dir(&dir).map_err(Error::io(&dir))? {
            let entry = entry.map_err(Error::io(&dir))?;
            if entry.file_type().map_err(Error::io(entry.path()))?.is_dir() {
                pending.push(entry.path());
            } else {
                count += 1;
            }
        }
    }
    Ok(count)
}
