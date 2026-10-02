// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements durable, content-addressed experience
// stores for learning agents, for its clients. If your team needs expertise
// in training-data lineage or crash-safe storage, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Artifacts: the bulk files a tool needs as real files, such as a LoRA adapter
//! brain loads from a path or a dataset it trains on.
//!
//! A file stays a plain file at a stable, content-addressed path, so a tool or
//! a person opens it directly. The database records each one precisely: what
//! it is, who made it, how big it is and what it was made from. A file is
//! written first (a temporary file, synced, renamed to its address) and made
//! official by one database commit, so a crash between the two leaves only an
//! orphan file, which [`ArtifactStore::sweep_orphans`] removes. A recorded
//! artifact whose file is gone or damaged is reported, never served.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use splinter_expdb::model::Entity;

use crate::digest::Digest;
use crate::error::StoreError;
use crate::workspace::{content_id, put_spilling, Workspace};
use crate::StateRoot;

const ARTIFACT: &str = "artifact";

/// What a file is for and who made it, as the caller says.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactSpec {
    role: String,
    producer: String,
    extension: String,
    sha256: bool,
}

impl ArtifactSpec {
    /// A file of `role` (for example `adapter`) made by `producer`.
    #[must_use]
    pub fn new(role: &str, producer: &str) -> Self {
        Self {
            role: role.into(),
            producer: producer.into(),
            extension: String::new(),
            sha256: false,
        }
    }

    /// The same file kept with `extension` (including the dot), for a tool
    /// that expects its files to be named a certain way.
    #[must_use]
    pub fn with_extension(mut self, extension: &str) -> Self {
        self.extension = extension.into();
        self
    }

    /// The same file, also hashed with SHA-256, the digest an external tool
    /// reports for it.
    #[must_use]
    pub fn with_sha256(mut self) -> Self {
        self.sha256 = true;
        self
    }
}

/// A file the database knows.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Artifact {
    /// The content address of its bytes.
    pub digest: Digest,
    /// Its size in bytes.
    pub size: u64,
    /// What it is for.
    pub role: String,
    /// Who made it.
    pub producer: String,
    /// The extension its file carries, with the dot, or empty.
    pub extension: String,
    /// The SHA-256 of its bytes, when it was asked for.
    pub sha256: Option<Digest>,
}

/// What is true of an artifact's file now.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArtifactState {
    /// The file is there and is what was recorded.
    Sound,
    /// There is no file.
    Missing,
    /// The file has a different size than was recorded.
    SizeChanged {
        /// The size found.
        found: u64,
    },
    /// The file has the recorded size but other bytes.
    Corrupt {
        /// What its bytes hash to.
        found: Digest,
    },
}

/// The artifacts under one state root.
#[derive(Clone, Debug)]
pub struct ArtifactStore {
    workspace: Workspace,
    dir: PathBuf,
}

fn io_error(path: &Path) -> impl FnOnce(std::io::Error) -> StoreError + '_ {
    move |source| StoreError::Io {
        path: path.to_path_buf(),
        source,
    }
}

impl ArtifactStore {
    /// The store over `workspace`, keeping files under `root`.
    #[must_use]
    pub fn new(workspace: &Workspace, root: &StateRoot) -> Self {
        Self {
            workspace: workspace.clone(),
            dir: root.artifacts(),
        }
    }

    fn file_of(&self, artifact: &Artifact) -> PathBuf {
        let hex = artifact.digest.hex();
        self.dir
            .join(&hex[..2])
            .join(format!("{hex}{}", artifact.extension))
    }

    fn temporary(&self) -> PathBuf {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        self.dir.join(format!(
            ".tmp-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ))
    }

    /// Keeps the file at `source` as an artifact. The bytes are copied, so the
    /// source can be removed. Keeping the same bytes again changes nothing.
    pub fn put_file(&self, source: &Path, spec: &ArtifactSpec) -> Result<Artifact, StoreError> {
        let mut input = fs::File::open(source).map_err(io_error(source))?;
        self.put_reader(&mut input, spec)
    }

    /// Keeps `bytes` as an artifact.
    pub fn put_bytes(&self, bytes: &[u8], spec: &ArtifactSpec) -> Result<Artifact, StoreError> {
        let mut input = bytes;
        self.put_reader(&mut input, spec)
    }

    fn put_reader(
        &self,
        input: &mut dyn Read,
        spec: &ArtifactSpec,
    ) -> Result<Artifact, StoreError> {
        fs::create_dir_all(&self.dir).map_err(io_error(&self.dir))?;
        let temporary = self.temporary();
        let mut blake = blake3::Hasher::new();
        let mut sha = spec.sha256.then(Sha256::new);
        let mut size = 0u64;
        {
            let mut out = fs::File::create(&temporary).map_err(io_error(&temporary))?;
            let mut buffer = vec![0u8; 1 << 20];
            loop {
                let n = match input.read(&mut buffer) {
                    Ok(0) => break,
                    Ok(n) => n,
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(e) => {
                        let _ = fs::remove_file(&temporary);
                        return Err(io_error(&temporary)(e));
                    }
                };
                blake.update(&buffer[..n]);
                if let Some(sha) = sha.as_mut() {
                    sha.update(&buffer[..n]);
                }
                out.write_all(&buffer[..n]).map_err(io_error(&temporary))?;
                size += n as u64;
            }
            out.sync_all().map_err(io_error(&temporary))?;
        }
        let artifact = Artifact {
            digest: Digest::from(splinter_expdb::ContentId::from_bytes(
                *blake.finalize().as_bytes(),
            )),
            size,
            role: spec.role.clone(),
            producer: spec.producer.clone(),
            extension: spec.extension.clone(),
            sha256: sha.map(|sha| Digest::sha256_from_hash(&sha.finalize())),
        };
        let target = self.file_of(&artifact);
        let parent = target.parent().unwrap_or(&self.dir);
        fs::create_dir_all(parent).map_err(io_error(parent))?;
        if target.exists() {
            // The same bytes are already here: this copy is redundant.
            fs::remove_file(&temporary).map_err(io_error(&temporary))?;
        } else {
            let mut permissions = fs::metadata(&temporary)
                .map_err(io_error(&temporary))?
                .permissions();
            permissions.set_readonly(true);
            fs::set_permissions(&temporary, permissions).map_err(io_error(&temporary))?;
            fs::rename(&temporary, &target).map_err(io_error(&target))?;
            fs::File::open(parent)
                .and_then(|dir| dir.sync_all())
                .map_err(io_error(parent))?;
        }
        // The file is in place; now the commit makes it official.
        if !self.workspace.has(ARTIFACT, &artifact.digest)? {
            let value =
                serde_json::to_value(&artifact).map_err(|source| StoreError::Serialize {
                    what: "artifact",
                    source,
                })?;
            let entity = Entity::keyed(ARTIFACT, content_id(&artifact.digest)?, value);
            self.workspace.write(|s| put_spilling(s, entity.clone()))?;
        }
        Ok(artifact)
    }

    /// The record of the artifact with this digest.
    pub fn get(&self, digest: &Digest) -> Result<Artifact, StoreError> {
        let entity = self
            .workspace
            .find(ARTIFACT, digest)?
            .ok_or_else(|| StoreError::UnknownArtifact(digest.clone()))?;
        serde_json::from_value(entity.value).map_err(|e| StoreError::UndecodableObject {
            what: format!("artifact {digest}"),
            reason: e.to_string(),
        })
    }

    /// The path of the artifact's file, for a tool to open. Refused when the
    /// artifact is not recorded, its file is gone or its size is not what was
    /// recorded; a check of its bytes is [`ArtifactStore::check`].
    pub fn path(&self, digest: &Digest) -> Result<PathBuf, StoreError> {
        let artifact = self.get(digest)?;
        let path = self.file_of(&artifact);
        match fs::metadata(&path) {
            Ok(meta) if meta.len() == artifact.size => Ok(path),
            Ok(_) | Err(_) => Err(StoreError::MissingArtifact {
                digest: artifact.digest,
                role: artifact.role,
                path,
            }),
        }
    }

    /// What is true of the artifact's file now. With `deep` the bytes are
    /// hashed; without, only the size is compared, which catches a missing or
    /// truncated file cheaply.
    pub fn check(&self, digest: &Digest, deep: bool) -> Result<ArtifactState, StoreError> {
        let artifact = self.get(digest)?;
        let path = self.file_of(&artifact);
        let meta = match fs::metadata(&path) {
            Ok(meta) => meta,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(ArtifactState::Missing)
            }
            Err(e) => return Err(io_error(&path)(e)),
        };
        if meta.len() != artifact.size {
            return Ok(ArtifactState::SizeChanged { found: meta.len() });
        }
        if deep {
            let found = Digest::of_reader(fs::File::open(&path).map_err(io_error(&path))?)
                .map_err(io_error(&path))?;
            if found != artifact.digest {
                return Ok(ArtifactState::Corrupt { found });
            }
        }
        Ok(ArtifactState::Sound)
    }

    /// Every recorded artifact, in the order recorded.
    pub fn list(&self) -> Result<Vec<Artifact>, StoreError> {
        let stored = self.workspace.read(|s| s.entities(ARTIFACT))?;
        stored
            .into_iter()
            .map(|stored| {
                serde_json::from_value(stored.entity.value).map_err(|e| {
                    StoreError::UndecodableObject {
                        what: "an artifact".into(),
                        reason: e.to_string(),
                    }
                })
            })
            .collect()
    }

    /// Removes files that are in the artifact directory and that no recorded
    /// artifact names, and temporary files left by a write that died, once they
    /// are older than `older_than` so one being written is left alone. Returns
    /// how many it removed.
    pub fn sweep_orphans(&self, older_than: Duration) -> Result<usize, StoreError> {
        let recorded: std::collections::HashSet<String> = self
            .list()?
            .iter()
            .map(|a| a.digest.hex().to_owned())
            .collect();
        let mut removed = 0;
        let Ok(top) = fs::read_dir(&self.dir) else {
            return Ok(0);
        };
        let old_enough = |meta: &fs::Metadata| {
            meta.modified()
                .ok()
                .and_then(|m| SystemTime::now().duration_since(m).ok())
                .is_some_and(|age| age >= older_than)
        };
        for entry in top.flatten() {
            let path = entry.path();
            let Ok(meta) = entry.metadata() else { continue };
            if meta.is_file() {
                if path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with(".tmp-"))
                    && old_enough(&meta)
                {
                    fs::remove_file(&path).map_err(io_error(&path))?;
                    removed += 1;
                }
                continue;
            }
            for file in fs::read_dir(&path).map_err(io_error(&path))?.flatten() {
                let name = file.file_name().to_string_lossy().into_owned();
                let hex = name.split('.').next().unwrap_or_default();
                let Ok(meta) = file.metadata() else { continue };
                if !recorded.contains(hex) && old_enough(&meta) {
                    let victim = file.path();
                    fs::remove_file(&victim).map_err(io_error(&victim))?;
                    removed += 1;
                }
            }
        }
        Ok(removed)
    }
}
