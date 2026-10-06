// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible packaging of trained models
// for regulated deployments, for its clients. If your team needs expertise in
// shipping model checkpoints that can be verified byte for byte, you can
// procure our services by sending an email to info@swedishembedded.com.

//! A flat directory of files (a saved model: its weights and its vocabulary)
//! packed into ONE file, deterministically, so a release store that keeps one
//! content-addressed file per release can keep a model that is a directory.
//!
//! The framing is the archive's own tar writer: fixed owner, mode and time,
//! members in name order, so the same files always make the same bytes.
//! Nothing else is in it: no compression (weights do not compress) and no
//! subdirectories, because a model directory has none. [`unpack_directory`]
//! accepts nothing but what [`pack_directory`] writes: a member that names a
//! path (rather than a file in the directory), a repeated name or a file that
//! ends early is refused, and nothing is left behind at the destination when
//! it is.

use std::collections::BTreeSet;
use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use splinter_core::digest::Digest;

use crate::error::StoreError;
use crate::recovery::tar::{Reader, Writer, MAX_PATH};

/// What a packed bundle is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Packed {
    /// The SHA-256 of the bundle's bytes, which a release manifest names.
    pub sha256: Digest,
    /// The bundle's size in bytes.
    pub bytes: u64,
    /// The files in it, in name order.
    pub files: Vec<String>,
}

fn refused(what: &'static str, reason: String) -> StoreError {
    StoreError::Rejected { what, reason }
}

fn io_failure(what: &'static str, path: &Path, e: &io::Error) -> StoreError {
    refused(what, format!("{}: {e}", path.display()))
}

/// Packs the regular files of `dir` into the single file `out`, written
/// beside its destination and renamed so a file that exists is whole.
/// Refuses a directory that has a subdirectory, no file at all or a file
/// whose name ustar cannot hold.
pub fn pack_directory(dir: &Path, out: &Path) -> Result<Packed, StoreError> {
    const WHAT: &str = "bundle";
    let mut files: Vec<(String, PathBuf)> = Vec::new();
    for entry in fs::read_dir(dir).map_err(|e| io_failure(WHAT, dir, &e))? {
        let entry = entry.map_err(|e| io_failure(WHAT, dir, &e))?;
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if !path.is_file() {
            return Err(refused(
                WHAT,
                format!(
                    "{} is not a regular file: a bundle is a flat directory",
                    path.display()
                ),
            ));
        }
        files.push((name, path));
    }
    if files.is_empty() {
        return Err(refused(
            WHAT,
            format!("{} holds no file to pack", dir.display()),
        ));
    }
    files.sort();
    let pending = out.with_extension("pending");
    let written = (|| -> io::Result<()> {
        let mut tar = Writer::new(BufWriter::new(fs::File::create(&pending)?));
        for (name, path) in &files {
            let size = fs::metadata(path)?.len();
            tar.member(name, size, fs::File::open(path)?)?;
        }
        tar.finish()?.flush()
    })();
    if let Err(e) = written {
        let _ = fs::remove_file(&pending);
        return Err(io_failure(WHAT, out, &e));
    }
    fs::rename(&pending, out).map_err(|e| io_failure(WHAT, out, &e))?;
    let bytes = fs::read(out).map_err(|e| io_failure(WHAT, out, &e))?;
    Ok(Packed {
        sha256: Digest::sha256_of(&bytes),
        bytes: bytes.len() as u64,
        files: files.into_iter().map(|(name, _)| name).collect(),
    })
}

/// Unpacks `bundle` into `dest`, which must not exist yet; it is created, and
/// removed again when the bundle is not one [`pack_directory`] wrote. Returns
/// the file names in the order they were stored.
pub fn unpack_directory(bundle: &Path, dest: &Path) -> Result<Vec<String>, StoreError> {
    const WHAT: &str = "bundle";
    if dest.exists() {
        return Err(refused(
            WHAT,
            format!(
                "{} exists: a bundle is unpacked into a new directory",
                dest.display()
            ),
        ));
    }
    fs::create_dir_all(dest).map_err(|e| io_failure(WHAT, dest, &e))?;
    let unpacked = unpack_into(bundle, dest);
    if unpacked.is_err() {
        let _ = fs::remove_dir_all(dest);
    }
    unpacked
}

fn unpack_into(bundle: &Path, dest: &Path) -> Result<Vec<String>, StoreError> {
    const WHAT: &str = "bundle";
    let file = fs::File::open(bundle).map_err(|e| io_failure(WHAT, bundle, &e))?;
    let mut tar = Reader::new(io::BufReader::new(file));
    let mut seen = BTreeSet::new();
    let mut names = Vec::new();
    while let Some(entry) = tar
        .next()
        .map_err(|e| refused(WHAT, format!("{}: {e}", bundle.display())))?
    {
        let name = entry.path;
        if name.is_empty()
            || name.len() > MAX_PATH
            || name == "."
            || name == ".."
            || name.contains(['/', '\\', '\0'])
        {
            return Err(refused(
                WHAT,
                format!(
                    "{}: member {name:?} is not a plain file name",
                    bundle.display()
                ),
            ));
        }
        if !seen.insert(name.clone()) {
            return Err(refused(
                WHAT,
                format!("{}: member {name:?} appears twice", bundle.display()),
            ));
        }
        let path = dest.join(&name);
        let mut out = fs::File::create(&path).map_err(|e| io_failure(WHAT, &path, &e))?;
        let copied = io::copy(&mut tar.data(), &mut out)
            .map_err(|e| refused(WHAT, format!("{}: member {name:?}: {e}", bundle.display())))?;
        if copied != entry.size {
            return Err(refused(
                WHAT,
                format!("{}: member {name:?} ended early", bundle.display()),
            ));
        }
        names.push(name);
    }
    Ok(names)
}
