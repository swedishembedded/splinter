// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements immutable, content-addressed model
// releases with full lineage, for its clients. If your team needs
// expertise in model release management, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Releases under the state root, and the aliases that point at them.
//!
//! ```text
//! <root>/releases/<hex>/
//!   adapter.safetensors      the released adapter, copied once, read-only
//!   manifest.json            its manifest in canonical JSON; <hex> is its digest
//! <root>/releases/aliases/<name>
//!                            `sha256:<hex>` of the release the alias points at
//! ```
//!
//! A [`ReleaseId`] is the digest of the manifest's bytes, and the manifest
//! names the adapter by its digest, so one id pins both. A release is
//! written into a pending directory and moved into place whole; one that
//! already exists is never replaced, and its files are made read-only.
//! [`ReleaseStore::get`] checks both digests on every read.
//!
//! An alias is replaced atomically. Moving one is a compare-and-set under
//! a lock file beside the aliases ([`ReleaseStore::move_alias`]): it moves
//! only from the release its caller measured against, so two releases
//! decided against the same champion cannot both land. The lock
//! coordinates processes sharing one state root on one host.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use splinter_store::digest::{canonical_json, Digest};
use splinter_store::{sync_dir, write_atomic, write_once, StateRoot};
use splinter_views::DatasetId;

use crate::error::{io, CampaignError};
use crate::model_ref::is_alias_name;
use crate::release::gate::GateReport;
use crate::train::{ReplaySample, TrainingSummary};

/// The adapter file's name inside a release's directory.
pub const ADAPTER_FILE: &str = "adapter.safetensors";
/// The manifest's name inside a release's directory.
pub const MANIFEST_FILE: &str = "manifest.json";
/// The `format` every manifest carries.
pub const RELEASE_FORMAT: &str = "splinter-release-v1";

/// The longest lineage walked: a guard against a corrupt store, far past
/// any real history.
const MAX_LINEAGE: usize = 10_000;

/// A release's id: the digest of its manifest's bytes.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ReleaseId(pub Digest);

impl std::fmt::Display for ReleaseId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// What a release is and why it was released.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReleaseManifest {
    /// Always [`RELEASE_FORMAT`].
    pub format: String,
    /// The base model, as brain's model store names it.
    pub base_model: String,
    /// The digest of the base checkpoint file the adapter sits on.
    pub base_digest: Digest,
    /// The digest of the adapter file.
    pub adapter_digest: Digest,
    /// The release it was trained from; `None` for the first.
    pub parent: Option<ReleaseId>,
    /// The candidate it was.
    pub candidate: String,
    /// The new datasets it was trained on; their held-out records are its
    /// suite.
    pub datasets: Vec<DatasetId>,
    /// The earlier records replayed beside them.
    pub replay: Option<ReplaySample>,
    /// How it was trained.
    pub training: TrainingSummary,
    /// The gate it passed, with every number.
    pub gate: GateReport,
    /// When it was released, from the injected clock.
    pub created_at: String,
}

/// A release as stored and verified.
#[derive(Clone, Debug, PartialEq)]
pub struct StoredRelease {
    /// Its id.
    pub id: ReleaseId,
    /// Its directory.
    pub dir: PathBuf,
    /// Its adapter file.
    pub adapter: PathBuf,
    /// Its manifest.
    pub manifest: ReleaseManifest,
}

/// The releases under one state root.
#[derive(Clone, Debug)]
pub struct ReleaseStore {
    dir: PathBuf,
}

impl ReleaseStore {
    /// The store under `root`; nothing is created until something is
    /// written.
    #[must_use]
    pub fn open(root: &StateRoot) -> Self {
        Self {
            dir: root.releases(),
        }
    }

    fn release_dir(&self, id: &ReleaseId) -> PathBuf {
        self.dir.join(id.0.hex())
    }

    fn aliases_dir(&self) -> PathBuf {
        self.dir.join("aliases")
    }

    /// Stores `manifest` with a copy of `adapter`, refused unless the
    /// adapter's bytes hash to the manifest's adapter digest, and refused
    /// when the release already exists: a release is written once.
    pub fn put(
        &self,
        manifest: &ReleaseManifest,
        adapter: &Path,
    ) -> Result<StoredRelease, CampaignError> {
        let bytes = canonical_json(manifest).map_err(|source| CampaignError::Json {
            what: "release manifest".into(),
            source,
        })?;
        let id = ReleaseId(Digest::of(&bytes));
        let target = self.release_dir(&id);
        let exists = || {
            CampaignError::Refused(format!(
                "release {id} already exists; a release is written once and never replaced"
            ))
        };
        if target.exists() {
            return Err(exists());
        }
        let adapter_bytes = fs::read(adapter).map_err(io(adapter))?;
        let found = Digest::of(&adapter_bytes);
        if found != manifest.adapter_digest {
            return Err(CampaignError::Refused(format!(
                "{} hashes to {found}, not the {} the manifest names",
                adapter.display(),
                manifest.adapter_digest
            )));
        }
        let pending = self.pending_dir();
        fs::create_dir_all(&pending).map_err(io(&pending))?;
        let written = (|| {
            for (name, content) in [(ADAPTER_FILE, &adapter_bytes), (MANIFEST_FILE, &bytes)] {
                let path = pending.join(name);
                write_once(&path, content).map_err(io(&path))?;
                read_only(&path)?;
            }
            sync_dir(&pending).map_err(io(&pending))?;
            fs::rename(&pending, &target).map_err(|e| {
                if target.exists() {
                    exists()
                } else {
                    io(&target)(e)
                }
            })
        })();
        if let Err(e) = written {
            // The refusal is the error worth reporting; a leftover pending
            // directory is named for this process and never read.
            let _ = fs::remove_dir_all(&pending);
            return Err(e);
        }
        sync_dir(&self.dir).map_err(io(&self.dir))?;
        self.get(&id)
    }

    fn pending_dir(&self) -> PathBuf {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        self.dir.join(format!(
            ".pending-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ))
    }

    /// The release `id`, verified: the manifest's bytes hash to `id`, and
    /// the adapter's to the digest the manifest names.
    pub fn get(&self, id: &ReleaseId) -> Result<StoredRelease, CampaignError> {
        let dir = self.release_dir(id);
        let manifest_file = dir.join(MANIFEST_FILE);
        if !manifest_file.is_file() {
            return Err(CampaignError::NotFound {
                what: "release",
                id: id.to_string(),
            });
        }
        let bytes = fs::read(&manifest_file).map_err(io(&manifest_file))?;
        verify(&manifest_file, &id.0, &Digest::of(&bytes))?;
        let manifest: ReleaseManifest =
            serde_json::from_slice(&bytes).map_err(|source| CampaignError::Json {
                what: manifest_file.display().to_string(),
                source,
            })?;
        let adapter = dir.join(ADAPTER_FILE);
        let file = fs::File::open(&adapter).map_err(io(&adapter))?;
        let found = Digest::of_reader(file).map_err(io(&adapter))?;
        verify(&adapter, &manifest.adapter_digest, &found)?;
        Ok(StoredRelease {
            id: id.clone(),
            dir,
            adapter,
            manifest,
        })
    }

    /// Every stored release's id, in id order (without verifying them).
    pub fn list(&self) -> Result<Vec<ReleaseId>, CampaignError> {
        let entries = match fs::read_dir(&self.dir) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(io(&self.dir)(e)),
        };
        let mut ids = Vec::new();
        for entry in entries {
            let name = entry.map_err(io(&self.dir))?.file_name();
            // Only a `<64 hex>` directory is a release; `aliases` and a
            // pending write are not.
            if let Some(Ok(digest)) = name.to_str().map(|n| Digest::parse(&format!("sha256:{n}"))) {
                ids.push(ReleaseId(digest));
            }
        }
        ids.sort();
        Ok(ids)
    }

    /// The release `alias` points at; `None` when it points nowhere yet.
    pub fn alias(&self, alias: &str) -> Result<Option<ReleaseId>, CampaignError> {
        let path = self.alias_path(alias)?;
        let text = match fs::read_to_string(&path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(io(&path)(e)),
        };
        let digest = Digest::parse(text.trim()).map_err(|e| {
            CampaignError::Refused(format!("alias file {} is corrupt: {e}", path.display()))
        })?;
        Ok(Some(ReleaseId(digest)))
    }

    /// Every alias and the release it points at.
    pub fn aliases(&self) -> Result<BTreeMap<String, ReleaseId>, CampaignError> {
        let dir = self.aliases_dir();
        let entries = match fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(BTreeMap::new()),
            Err(e) => return Err(io(&dir)(e)),
        };
        let mut aliases = BTreeMap::new();
        for entry in entries {
            let name = entry.map_err(io(&dir))?.file_name();
            let Some(name) = name.to_str().filter(|n| is_alias_name(n)) else {
                continue;
            };
            if let Some(id) = self.alias(name)? {
                aliases.insert(name.to_string(), id);
            }
        }
        Ok(aliases)
    }

    /// Points `alias` at `to`, provided it still points at `from` (`None`:
    /// nowhere yet); refused otherwise, naming where it points now. `to`
    /// must be a stored release.
    pub fn move_alias(
        &self,
        alias: &str,
        from: Option<&ReleaseId>,
        to: &ReleaseId,
    ) -> Result<(), CampaignError> {
        self.get(to)?;
        let path = self.alias_path(alias)?;
        let _lock = AliasLock::acquire(&self.aliases_dir())?;
        let now = self.alias(alias)?;
        if now.as_ref() != from {
            return Err(CampaignError::Refused(format!(
                "alias {alias} moved to {} while this was decided against {}; decide again",
                now.map_or("nothing".into(), |id| id.to_string()),
                from.map_or("nothing".into(), ToString::to_string),
            )));
        }
        write_atomic(&path, &format!("{to}\n")).map_err(io(&path))?;
        let dir = self.aliases_dir();
        sync_dir(&dir).map_err(io(&dir))
    }

    /// `id` and the releases it descends from, newest first.
    pub fn lineage(&self, id: &ReleaseId) -> Result<Vec<StoredRelease>, CampaignError> {
        let mut lineage: Vec<StoredRelease> = Vec::new();
        let mut next = Some(id.clone());
        while let Some(id) = next {
            if lineage.len() >= MAX_LINEAGE || lineage.iter().any(|r| r.id == id) {
                return Err(CampaignError::Refused(format!(
                    "the lineage of release {id} does not end; the release store is corrupt"
                )));
            }
            let release = self.get(&id)?;
            next = release.manifest.parent.clone();
            lineage.push(release);
        }
        Ok(lineage)
    }

    fn alias_path(&self, alias: &str) -> Result<PathBuf, CampaignError> {
        if !is_alias_name(alias) {
            return Err(CampaignError::Refused(format!(
                "{alias:?} is not an alias name: a lowercase letter, then up to 63 lowercase \
                 letters, digits, - or _"
            )));
        }
        Ok(self.aliases_dir().join(alias))
    }
}

/// Makes `path` read-only.
fn read_only(path: &Path) -> Result<(), CampaignError> {
    let mut permissions = fs::metadata(path).map_err(io(path))?.permissions();
    permissions.set_readonly(true);
    fs::set_permissions(path, permissions).map_err(io(path))
}

/// Refuses a file of `path` whose digest `found` is not `expected`.
fn verify(path: &Path, expected: &Digest, found: &Digest) -> Result<(), CampaignError> {
    if found == expected {
        return Ok(());
    }
    Err(CampaignError::Store(
        splinter_store::experiences::StoreError::Corrupt {
            path: path.to_path_buf(),
            expected: expected.clone(),
            found: found.clone(),
        },
    ))
}

/// The lock an alias is moved under: a file created exclusively, removed
/// when dropped.
struct AliasLock(PathBuf);

impl AliasLock {
    const NAME: &'static str = ".lock";

    fn acquire(dir: &Path) -> Result<Self, CampaignError> {
        fs::create_dir_all(dir).map_err(io(dir))?;
        let path = dir.join(Self::NAME);
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(mut file) => {
                // The holder's pid, for whoever finds a stale lock.
                let _ = writeln!(file, "{}", std::process::id());
                Ok(Self(path))
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                Err(CampaignError::Refused(format!(
                    "another release or rollback holds {}; if no splinter process is running, \
                     remove it",
                    path.display()
                )))
            }
            Err(e) => Err(io(&path)(e)),
        }
    }
}

impl Drop for AliasLock {
    fn drop(&mut self) {
        // A lock left behind is reported by the next acquire, naming it.
        let _ = fs::remove_file(&self.0);
    }
}
