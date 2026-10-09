// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Naming the parts of a speaking persona by content address.

use std::path::Path;

use anyhow::{Context, Result};
use splinter_sdk::vocabulary::digest::Digest;
use splinter_sdk::vocabulary::speech_bundle::Part;

/// The content address of a model directory: every file's name and size, and
/// the bytes of the small ones (configuration, tokenizer, weight index), in
/// name order. Hashing sixteen gigabytes of weights would say nothing the
/// weight index does not: it names every shard and its size.
pub fn digest_dir(dir: &Path) -> Result<Digest> {
    const SMALL: u64 = 8 * 1024 * 1024;
    let mut names: Vec<_> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    let mut listing = Vec::new();
    for name in &names {
        let path = dir.join(name);
        let size = std::fs::metadata(&path)?.len();
        listing.extend_from_slice(format!("{name}\t{size}\n").as_bytes());
        if size <= SMALL {
            listing.extend_from_slice(&std::fs::read(&path)?);
            listing.push(b'\n');
        }
    }
    Ok(Digest::of(&listing))
}

/// A part that is one file.
pub fn file_part(id: &str, path: &Path) -> Result<Part> {
    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    Ok(Part {
        id: id.to_string(),
        digest: Digest::of_reader(file)?,
    })
}

/// A part that is a model directory.
pub fn dir_part(id: &str, dir: &Path) -> Result<Part> {
    Ok(Part {
        id: id.to_string(),
        digest: digest_dir(dir)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_directory_digest_follows_its_files_and_not_their_order_of_creation() {
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        for (dir, order) in [
            (a.path(), ["x.json", "y.json"]),
            (b.path(), ["y.json", "x.json"]),
        ] {
            for name in order {
                std::fs::write(dir.join(name), name.as_bytes()).unwrap();
            }
        }
        assert_eq!(digest_dir(a.path()).unwrap(), digest_dir(b.path()).unwrap());
        std::fs::write(b.path().join("x.json"), b"changed").unwrap();
        assert_ne!(digest_dir(a.path()).unwrap(), digest_dir(b.path()).unwrap());
    }

    #[test]
    fn a_file_part_is_the_address_of_its_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("p.safetensors");
        std::fs::write(&path, b"weights").unwrap();
        assert_eq!(
            file_part("p", &path).unwrap().digest,
            Digest::of(b"weights")
        );
        assert!(file_part("p", &dir.path().join("absent")).is_err());
    }
}
