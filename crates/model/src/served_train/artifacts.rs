// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that hand datasets and
// trained adapters to a training service by content, never by path, for its
// clients. If your team needs expertise in content-addressed artifact
// exchange between a learning pipeline and a GPU cluster, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The content-addressed artifact store a brain service trains through
//! (`brain serve --artifact-store DIR`): a directory both sides can reach, in
//! which a file is named by the BLAKE3 of its bytes (64 lowercase hex digits).
//!
//! The client publishes a dataset by writing `<store>/<blake3>` and sends the
//! digest; brain resolves it in the store it was started with and verifies
//! it, and answers with the digest of the adapter it published there. Neither
//! side names a path to the other. A file appears under its name complete or
//! not at all (it is written to a temporary name in the store and renamed), so
//! a reader never sees half of it, and what is fetched is checked against the
//! digest it was asked by.

use std::fs::OpenOptions;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use splinter_core::digest::Digest;

use crate::error::PolicyError;

/// Makes the names of temporary files unique within this process.
static PUBLISHED: AtomicU64 = AtomicU64::new(0);

/// A directory of files named by their BLAKE3 digest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactStore {
    root: PathBuf,
}

impl ArtifactStore {
    /// The store in `dir`, which must exist: a mistyped flag is an error here,
    /// not a store created where no server looks.
    pub fn open(dir: &Path) -> Result<Self, PolicyError> {
        if !dir.is_dir() {
            return Err(PolicyError::Endpoint {
                url: dir.display().to_string(),
                reason: "the artifact store is not a directory that exists".into(),
            });
        }
        Ok(Self {
            root: dir.to_path_buf(),
        })
    }

    /// The directory.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn io(path: &Path) -> impl Fn(std::io::Error) -> PolicyError + '_ {
        move |source| PolicyError::Io {
            path: path.to_path_buf(),
            source,
        }
    }

    /// Publishes the contents of `file` and returns the digest that names
    /// them. Publishing what the store already holds changes nothing.
    pub fn publish(&self, file: &Path) -> Result<Digest, PolicyError> {
        let temp = self.root.join(format!(
            ".publishing-{}-{}",
            std::process::id(),
            PUBLISHED.fetch_add(1, Ordering::Relaxed)
        ));
        let published = (|| {
            let mut source = std::fs::File::open(file).map_err(Self::io(file))?;
            let mut out = OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o640)
                .open(&temp)
                .map_err(Self::io(&temp))?;
            std::io::copy(&mut source, &mut out).map_err(Self::io(&temp))?;
            out.sync_all().map_err(Self::io(&temp))?;
            drop(out);
            let digest = Digest::of_reader(std::fs::File::open(&temp).map_err(Self::io(&temp))?)
                .map_err(Self::io(&temp))?;
            let dest = self.root.join(digest.hex());
            std::fs::rename(&temp, &dest).map_err(Self::io(&dest))?;
            Ok(digest)
        })();
        if published.is_err() {
            let _ = std::fs::remove_file(&temp);
        }
        published
    }

    /// Copies the artifact `digest` to the new file `dest`, after checking
    /// that the copy hashes to `digest`. A digest that is not a BLAKE3 one is
    /// refused: the store names files by nothing else.
    pub fn fetch(&self, digest: &Digest, dest: &Path) -> Result<(), PolicyError> {
        if !digest.as_str().starts_with("blake3:") {
            return Err(PolicyError::Endpoint {
                url: self.root.display().to_string(),
                reason: format!("{digest} is not a blake3 digest, which names a file of the store"),
            });
        }
        let source = self.root.join(digest.hex());
        let temp = dest.with_extension("fetching");
        let fetched = (|| {
            std::fs::copy(&source, &temp).map_err(Self::io(&source))?;
            let actual = Digest::of_reader(std::fs::File::open(&temp).map_err(Self::io(&temp))?)
                .map_err(Self::io(&temp))?;
            if &actual != digest {
                return Err(PolicyError::Endpoint {
                    url: self.root.display().to_string(),
                    reason: format!("the artifact named {digest} hashes to {actual}"),
                });
            }
            std::fs::rename(&temp, dest).map_err(Self::io(dest))
        })();
        if fetched.is_err() {
            let _ = std::fs::remove_file(&temp);
        }
        fetched
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_file_is_published_under_the_blake3_of_its_bytes_and_comes_back_by_it() {
        let dir = tempfile::tempdir().unwrap();
        let store = ArtifactStore::open(&dir.path().join("store").tap_create()).unwrap();
        let file = dir.path().join("train.jsonl");
        std::fs::write(&file, b"{\"messages\":[]}\n").unwrap();
        let digest = store.publish(&file).unwrap();
        assert_eq!(digest, Digest::of(b"{\"messages\":[]}\n"));
        assert!(store.root().join(digest.hex()).is_file());
        // Again: the same artifact, and nothing else is left in the store.
        assert_eq!(store.publish(&file).unwrap(), digest);
        assert_eq!(std::fs::read_dir(store.root()).unwrap().count(), 1);

        let copy = dir.path().join("copy");
        store.fetch(&digest, &copy).unwrap();
        assert_eq!(std::fs::read(&copy).unwrap(), b"{\"messages\":[]}\n");
    }

    #[test]
    fn an_artifact_that_does_not_hash_to_its_name_is_refused_and_not_kept() {
        let dir = tempfile::tempdir().unwrap();
        let store = ArtifactStore::open(dir.path()).unwrap();
        let digest = Digest::of(b"the adapter");
        std::fs::write(store.root().join(digest.hex()), b"another file").unwrap();
        let dest = dir.path().join("adapter.safetensors");
        let error = store.fetch(&digest, &dest).unwrap_err().to_string();
        assert!(error.contains("hashes to"), "{error}");
        assert!(!dest.exists());
        assert!(!dest.with_extension("fetching").exists());
        // A digest the store does not hold, and one it cannot name a file by.
        let absent = Digest::of(b"absent");
        assert!(store.fetch(&absent, &dest).is_err());
        let sha = Digest::sha256_of(b"x");
        assert!(store
            .fetch(&sha, &dest)
            .unwrap_err()
            .to_string()
            .contains("not a blake3 digest"));
    }

    #[test]
    fn a_store_that_is_not_there_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        assert!(ArtifactStore::open(&dir.path().join("typo")).is_err());
    }

    /// Makes the directory a test names.
    trait Create {
        fn tap_create(self) -> Self;
    }
    impl Create for PathBuf {
        fn tap_create(self) -> Self {
            std::fs::create_dir_all(&self).unwrap();
            self
        }
    }
}
