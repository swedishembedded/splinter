// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements immutable, content-addressed result stores
// whose products survive a killed process intact, for its clients. If your
// team needs expertise in crash-safe artifact publication, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The immutable product store of a run directory.
//!
//! A stage writes its output into a [`Staging`] directory. [`ProductStore::publish`]
//! turns that tree into a *product*:
//!
//! 1. every regular file is hashed into `blobs/sha256/<hex>` (temporary file,
//!    fsync, rename), so identical content is stored once;
//! 2. a [`ProductManifest`] lists the files by path, digest and size;
//! 3. an intent record is written to `pending/`, then the readable *view*
//!    directory is built next to its final place from hard links (copies
//!    across devices) and renamed into position;
//! 4. last, `products/<stage>.json` is written atomically. **That file is the
//!    commit point**: a product exists exactly when it does, and a crash at
//!    any earlier step leaves nothing a reader can mistake for a product.
//!
//! A product is immutable. Publishing the same stage again with identical
//! content returns the committed manifest; different content is refused.
//! [`ProductStore::recover`] removes what a crash left behind.
//!
//! Blobs are read-only files and views share their inodes, so editing a view
//! file in place would edit the blob; [`ProductStore::verify_product`] exists
//! to catch that.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use splinter_core::clock::Clock;
use splinter_core::digest::{digest_of, Digest};

use super::fsutil::{sync_dir, temp_sibling, write_atomic};

mod transfer;

/// The schema tag of a product manifest.
pub const MANIFEST_SCHEMA: &str = "splinter.product/1";

/// The manifest's file name inside a view directory.
pub const VIEW_MANIFEST: &str = "manifest.json";

const BLOB_DIR: &str = "blobs/sha256";
const PRODUCT_DIR: &str = "products";
const STAGING_DIR: &str = "staging";
const PENDING_DIR: &str = "pending";
const SHA256_PREFIX: &str = "sha256:";

/// Why a product-store operation failed.
#[derive(Debug, thiserror::Error)]
pub enum ProductError {
    /// A file-system operation failed.
    #[error("{context}: {source}")]
    Io {
        /// What was being done, with the path.
        context: String,
        /// The underlying error.
        #[source]
        source: io::Error,
    },
    /// No committed product exists for this stage.
    #[error("no product is committed for stage {0}")]
    UnknownProduct(Digest),
    /// No blob exists for this digest.
    #[error("no blob {0} in the product store")]
    UnknownBlob(Digest),
    /// A blob's bytes no longer hash to its address.
    #[error("blob {digest} is corrupt: its content hashes to {actual}")]
    BlobCorrupt {
        /// The address the blob is stored under.
        digest: Digest,
        /// What its bytes hash to now.
        actual: Digest,
    },
    /// Blobs are addressed by SHA-256 only.
    #[error("{0} is not a sha256 digest; product blobs are addressed by sha256")]
    UnsupportedDigest(Digest),
    /// The stage already has a product with different content.
    #[error("stage {stage} already has a committed product with different content ({reason})")]
    Immutable {
        /// The stage.
        stage: Digest,
        /// What differs.
        reason: String,
    },
    /// The staging tree holds something that is not a regular file or directory.
    #[error("staging entry {0} is not a regular file or directory; products hold files only")]
    UnsupportedEntry(PathBuf),
    /// A view path outside the run directory, or one that is the directory itself.
    #[error("view {view} is not a path inside the run directory {root}")]
    InvalidView {
        /// The requested view.
        view: PathBuf,
        /// The run directory.
        root: PathBuf,
    },
    /// A committed manifest or intent record cannot be read.
    #[error("{what} is malformed: {reason}")]
    Malformed {
        /// Which record.
        what: String,
        /// Why.
        reason: String,
    },
    /// The view directory does not match the committed manifest.
    #[error("view {view} does not match the product: {reason}")]
    ViewMismatch {
        /// The view.
        view: PathBuf,
        /// What differs.
        reason: String,
    },
}

fn io_error(context: impl Into<String>) -> impl FnOnce(io::Error) -> ProductError {
    let context = context.into();
    move |source| ProductError::Io { context, source }
}

/// One file of a product.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestFile {
    /// Path below the product root, `/`-separated.
    pub path: String,
    /// Content address of the file's bytes.
    pub digest: Digest,
    /// Size in bytes.
    pub size: u64,
    /// Whether the file is a program (any execute bit set when published).
    /// Its blob is shared with any file of the same bytes; a view gives an
    /// executable file a copy of its own, mode 555.
    #[serde(default, skip_serializing_if = "is_false")]
    pub executable: bool,
}

#[allow(clippy::trivially_copy_pass_by_ref)]
fn is_false(value: &bool) -> bool {
    !*value
}

/// Whether the file at `path` has any execute bit set.
fn is_executable(path: &Path) -> Result<bool, ProductError> {
    use std::os::unix::fs::PermissionsExt;
    let mode = fs::metadata(path)
        .map_err(io_error(format!("stat {}", path.display())))?
        .permissions()
        .mode();
    Ok(mode & 0o111 != 0)
}

/// What a committed product contains.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProductManifest {
    /// Always [`MANIFEST_SCHEMA`].
    pub schema: String,
    /// The stage that made the product.
    pub stage: Digest,
    /// The product's name, for people.
    pub name: String,
    /// The files, sorted by path.
    pub files: Vec<ManifestFile>,
    /// When the product was committed, from the store's clock.
    pub created_at: String,
}

impl ProductManifest {
    /// The content address of the manifest's canonical form.
    #[must_use]
    pub fn digest(&self) -> Digest {
        digest_of(self)
    }
}

/// A directory a stage writes its output into before it publishes.
///
/// Dropping it does nothing: a stage that dies leaves the directory behind,
/// and [`ProductStore::recover`] removes it.
#[derive(Debug)]
pub struct Staging {
    dir: PathBuf,
}

impl Staging {
    /// The directory to write into.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.dir
    }
}

/// The intent record of a publish in flight.
#[derive(Serialize, Deserialize)]
struct PendingPublish {
    /// The view, relative to the run directory.
    view: String,
}

/// A reader that copies what it yields into a writer, so a file is hashed and
/// stored in one pass over its bytes.
struct Tee<R, W> {
    inner: R,
    out: W,
}

impl<R: Read, W: Write> Read for Tee<R, W> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.out.write_all(&buf[..n])?;
        Ok(n)
    }
}

/// The immutable product store of one run directory.
pub struct ProductStore {
    root: PathBuf,
    clock: Box<dyn Clock + Send + Sync>,
}

impl ProductStore {
    /// A store rooted at the run directory `root`; directories are created
    /// as they are first needed. `clock` stamps manifests.
    pub fn new(root: impl Into<PathBuf>, clock: Box<dyn Clock + Send + Sync>) -> Self {
        Self {
            root: root.into(),
            clock,
        }
    }

    /// The run directory.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn blob_path(&self, digest: &Digest) -> Result<PathBuf, ProductError> {
        if !digest.as_str().starts_with(SHA256_PREFIX) {
            return Err(ProductError::UnsupportedDigest(digest.clone()));
        }
        Ok(self.root.join(BLOB_DIR).join(digest.hex()))
    }

    fn product_path(&self, stage: &Digest) -> PathBuf {
        self.root
            .join(PRODUCT_DIR)
            .join(format!("{}.json", stage.hex()))
    }

    fn pending_path(&self, stage: &Digest) -> PathBuf {
        self.root
            .join(PENDING_DIR)
            .join(format!("{}.json", stage.hex()))
    }

    /// Stores `bytes` as a blob and returns its address; storing the same
    /// bytes again is a no-op.
    pub fn put_bytes(&self, bytes: &[u8]) -> Result<Digest, ProductError> {
        let digest = Digest::sha256_of(bytes);
        let target = self.blob_path(&digest)?;
        if target.exists() {
            return Ok(digest);
        }
        let incoming = self.new_incoming()?;
        let mut file = create_new(&incoming)?;
        file.write_all(bytes)
            .map_err(io_error(format!("write {}", incoming.display())))?;
        self.seal_blob(file, &incoming, &target)?;
        Ok(digest)
    }

    /// Stores the contents of the file at `path` as a blob, hashing while it
    /// copies, and returns its address.
    pub fn put_file(&self, path: &Path) -> Result<Digest, ProductError> {
        let source = File::open(path).map_err(io_error(format!("open {}", path.display())))?;
        let incoming = self.new_incoming()?;
        let out = create_new(&incoming)?;
        let tee = Tee {
            inner: source,
            out: &out,
        };
        let hashed = Digest::sha256_of_reader(tee);
        let digest = match hashed {
            Ok(digest) => digest,
            Err(error) => {
                let _ = fs::remove_file(&incoming);
                return Err(ProductError::Io {
                    context: format!("copy {} into the blob store", path.display()),
                    source: error,
                });
            }
        };
        let target = self.blob_path(&digest)?;
        self.seal_blob(out, &incoming, &target)?;
        Ok(digest)
    }

    fn new_incoming(&self) -> Result<PathBuf, ProductError> {
        let dir = self.root.join(BLOB_DIR);
        fs::create_dir_all(&dir).map_err(io_error(format!("create {}", dir.display())))?;
        Ok(temp_sibling(&dir.join("incoming")))
    }

    /// Syncs `file` and moves `incoming` to its address, read-only; when the
    /// blob already exists the temporary copy is dropped.
    fn seal_blob(&self, file: File, incoming: &Path, target: &Path) -> Result<(), ProductError> {
        let sealed = (|| {
            file.sync_all()?;
            if target.exists() {
                return fs::remove_file(incoming);
            }
            let mut perms = file.metadata()?.permissions();
            perms.set_readonly(true);
            fs::set_permissions(incoming, perms)?;
            fs::rename(incoming, target)?;
            sync_dir(target.parent().unwrap_or(&self.root))
        })();
        if let Err(source) = sealed {
            let _ = fs::remove_file(incoming);
            return Err(ProductError::Io {
                context: format!("store blob {}", target.display()),
                source,
            });
        }
        Ok(())
    }

    /// Opens the blob stored under `digest`.
    pub fn open(&self, digest: &Digest) -> Result<File, ProductError> {
        let path = self.blob_path(digest)?;
        File::open(&path).map_err(|source| {
            if source.kind() == io::ErrorKind::NotFound {
                ProductError::UnknownBlob(digest.clone())
            } else {
                ProductError::Io {
                    context: format!("open {}", path.display()),
                    source,
                }
            }
        })
    }

    /// Re-hashes the blob and checks it still matches its address.
    pub fn verify(&self, digest: &Digest) -> Result<(), ProductError> {
        let file = self.open(digest)?;
        let actual =
            Digest::sha256_of_reader(file).map_err(io_error(format!("read blob {digest}")))?;
        if &actual == digest {
            Ok(())
        } else {
            Err(ProductError::BlobCorrupt {
                digest: digest.clone(),
                actual,
            })
        }
    }

    /// A fresh, empty staging directory under `<root>/staging/`.
    pub fn begin_staging(&self) -> Result<Staging, ProductError> {
        let parent = self.root.join(STAGING_DIR);
        fs::create_dir_all(&parent).map_err(io_error(format!("create {}", parent.display())))?;
        loop {
            let dir = temp_sibling(&parent.join("stage"));
            match fs::create_dir(&dir) {
                Ok(()) => return Ok(Staging { dir }),
                // A leftover of a dead process with the same id: try the next name.
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(io_error(format!("create {}", dir.display()))(error)),
            }
        }
    }

    /// Publishes the tree in `staging` as the product of `stage_id`, with its
    /// readable view at `view` (a path inside the run directory).
    ///
    /// Committed already with identical content: returns the committed
    /// manifest and changes nothing. Committed with different content:
    /// [`ProductError::Immutable`]. The staging directory is removed once the
    /// product is committed.
    pub fn publish(
        &self,
        stage_id: &Digest,
        name: &str,
        staging: Staging,
        view: &Path,
    ) -> Result<ProductManifest, ProductError> {
        let manifest = match self.prepare(stage_id, name, &staging, view)? {
            Prepared::Existing(existing) => existing,
            Prepared::Viewed(manifest) => {
                let bytes = pretty(&manifest)?;
                let path = self.product_path(stage_id);
                write_atomic(&path, &bytes)
                    .map_err(io_error(format!("commit {}", path.display())))?;
                manifest
            }
        };
        // The product is committed; leftovers are recover()'s to clean if this fails.
        let _ = fs::remove_file(self.pending_path(stage_id));
        let _ = fs::remove_dir_all(staging.path());
        Ok(manifest)
    }

    /// Fault-injection hook for crash tests: runs [`Self::publish`] up to but
    /// not including the commit, as a process killed there would leave it.
    #[doc(hidden)]
    pub fn publish_without_commit(
        &self,
        stage_id: &Digest,
        name: &str,
        staging: Staging,
        view: &Path,
    ) -> Result<(), ProductError> {
        self.prepare(stage_id, name, &staging, view).map(|_| ())
    }

    fn prepare(
        &self,
        stage_id: &Digest,
        name: &str,
        staging: &Staging,
        view: &Path,
    ) -> Result<Prepared, ProductError> {
        let relative_view = self.relative_view(view)?;
        let mut files = Vec::new();
        self.collect_files(&staging.dir, Path::new(""), &mut files)?;
        files.sort_by(|a, b| a.path.cmp(&b.path));

        if let Some(existing) = self.committed(stage_id)? {
            if existing.name == name && existing.files == files {
                return Ok(Prepared::Existing(existing));
            }
            return Err(ProductError::Immutable {
                stage: stage_id.clone(),
                reason: if existing.name == name {
                    "the files differ".to_string()
                } else {
                    format!("it is named {:?}, not {name:?}", existing.name)
                },
            });
        }

        let manifest = ProductManifest {
            schema: MANIFEST_SCHEMA.to_string(),
            stage: stage_id.clone(),
            name: name.to_string(),
            files,
            created_at: self.clock.utc_now(),
        };
        let intent = serde_json::to_vec(&PendingPublish {
            view: relative_view,
        })
        .map_err(|e| malformed("publish intent", e))?;
        let pending = self.pending_path(stage_id);
        write_atomic(&pending, &intent)
            .map_err(io_error(format!("write {}", pending.display())))?;
        self.materialise_view(view, &manifest)?;
        Ok(Prepared::Viewed(manifest))
    }

    /// `view` relative to the run directory, refusing anything that is not
    /// strictly inside it.
    fn relative_view(&self, view: &Path) -> Result<String, ProductError> {
        let invalid = || ProductError::InvalidView {
            view: view.to_path_buf(),
            root: self.root.clone(),
        };
        let relative = view.strip_prefix(&self.root).map_err(|_| invalid())?;
        let clean = relative
            .components()
            .all(|c| matches!(c, Component::Normal(_)));
        match relative.to_str() {
            Some(text) if clean && !text.is_empty() => Ok(text.to_string()),
            _ => Err(invalid()),
        }
    }

    fn collect_files(
        &self,
        base: &Path,
        relative: &Path,
        out: &mut Vec<ManifestFile>,
    ) -> Result<(), ProductError> {
        let dir = base.join(relative);
        let mut entries = fs::read_dir(&dir)
            .map_err(io_error(format!("read {}", dir.display())))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(io_error(format!("read {}", dir.display())))?;
        entries.sort_by_key(fs::DirEntry::file_name);
        for entry in entries {
            let child = relative.join(entry.file_name());
            let full = base.join(&child);
            let kind = entry
                .file_type()
                .map_err(io_error(format!("stat {}", full.display())))?;
            if kind.is_dir() {
                self.collect_files(base, &child, out)?;
            } else if kind.is_file() {
                let path = child
                    .to_str()
                    .ok_or_else(|| ProductError::UnsupportedEntry(full.clone()))?
                    .replace(std::path::MAIN_SEPARATOR, "/");
                let executable = is_executable(&full)?;
                let digest = self.put_file(&full)?;
                let size = fs::metadata(self.blob_path(&digest)?)
                    .map_err(io_error(format!("stat blob {digest}")))?
                    .len();
                out.push(ManifestFile {
                    path,
                    digest,
                    size,
                    executable,
                });
            } else {
                return Err(ProductError::UnsupportedEntry(full));
            }
        }
        Ok(())
    }

    /// Builds the view beside its final place and renames it into position.
    fn materialise_view(
        &self,
        view: &Path,
        manifest: &ProductManifest,
    ) -> Result<(), ProductError> {
        let parent = view.parent().unwrap_or(&self.root);
        fs::create_dir_all(parent).map_err(io_error(format!("create {}", parent.display())))?;
        let building = temp_sibling(view);
        fs::create_dir(&building).map_err(io_error(format!("create {}", building.display())))?;
        let built = (|| {
            for file in &manifest.files {
                let dest = building.join(&file.path);
                if let Some(dir) = dest.parent() {
                    fs::create_dir_all(dir)
                        .map_err(io_error(format!("create {}", dir.display())))?;
                }
                let blob = self.blob_path(&file.digest)?;
                if file.executable {
                    // A program gets its own copy: the blob is shared with
                    // every file of its bytes, and a mode is the inode's.
                    use std::os::unix::fs::PermissionsExt;
                    fs::copy(&blob, &dest).map_err(io_error(format!(
                        "copy {} to {}",
                        blob.display(),
                        dest.display()
                    )))?;
                    fs::set_permissions(&dest, fs::Permissions::from_mode(0o555))
                        .map_err(io_error(format!("make {} executable", dest.display())))?;
                } else if fs::hard_link(&blob, &dest).is_err() {
                    // Another device, or a file system without links: copy.
                    fs::copy(&blob, &dest).map_err(io_error(format!(
                        "copy {} to {}",
                        blob.display(),
                        dest.display()
                    )))?;
                }
            }
            let manifest_path = building.join(VIEW_MANIFEST);
            write_atomic(&manifest_path, &pretty(manifest)?)
                .map_err(io_error(format!("write {}", manifest_path.display())))?;
            sync_tree(&building)
        })();
        if let Err(error) = built {
            let _ = fs::remove_dir_all(&building);
            return Err(error);
        }
        // An earlier uncommitted or superseded view is moved aside, then dropped.
        let displaced = if view.exists() {
            let aside = temp_sibling(view);
            fs::rename(view, &aside).map_err(io_error(format!("replace {}", view.display())))?;
            Some(aside)
        } else {
            None
        };
        fs::rename(&building, view).map_err(io_error(format!("place {}", view.display())))?;
        sync_dir(parent).map_err(io_error(format!("sync {}", parent.display())))?;
        if let Some(aside) = displaced {
            let _ = fs::remove_dir_all(aside);
        }
        Ok(())
    }

    fn committed(&self, stage: &Digest) -> Result<Option<ProductManifest>, ProductError> {
        let path = self.product_path(stage);
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(io_error(format!("read {}", path.display()))(e)),
        };
        let manifest: ProductManifest = serde_json::from_slice(&bytes)
            .map_err(|e| malformed(&path.display().to_string(), e))?;
        if manifest.schema != MANIFEST_SCHEMA || &manifest.stage != stage {
            return Err(malformed(
                &path.display().to_string(),
                format!("schema {:?} for stage {}", manifest.schema, manifest.stage),
            ));
        }
        Ok(Some(manifest))
    }

    /// The committed manifest of `stage_id`.
    pub fn manifest(&self, stage_id: &Digest) -> Result<ProductManifest, ProductError> {
        self.committed(stage_id)?
            .ok_or_else(|| ProductError::UnknownProduct(stage_id.clone()))
    }

    /// Checks every file of the committed product against its blob and the
    /// view at `view` against the manifest.
    pub fn verify_product(&self, stage_id: &Digest, view: &Path) -> Result<(), ProductError> {
        let manifest = self.manifest(stage_id)?;
        let mismatch = |reason: String| ProductError::ViewMismatch {
            view: view.to_path_buf(),
            reason,
        };
        let recorded = fs::read(view.join(VIEW_MANIFEST)).map_err(io_error(format!(
            "read the manifest of view {}",
            view.display()
        )))?;
        let recorded: ProductManifest = serde_json::from_slice(&recorded)
            .map_err(|e| malformed(&format!("manifest of view {}", view.display()), e))?;
        if recorded != manifest {
            return Err(mismatch(
                "its manifest.json is not the committed manifest".into(),
            ));
        }
        for file in &manifest.files {
            self.verify(&file.digest)?;
            let path = view.join(&file.path);
            let handle = File::open(&path).map_err(|e| mismatch(format!("{}: {e}", file.path)))?;
            let actual = Digest::sha256_of_reader(handle)
                .map_err(io_error(format!("read {}", path.display())))?;
            if actual != file.digest {
                return Err(mismatch(format!(
                    "{} no longer hashes to {}",
                    file.path, file.digest
                )));
            }
            let executable = is_executable(&path)?;
            if executable != file.executable {
                return Err(mismatch(format!(
                    "{} is {}executable, and was published {}",
                    file.path,
                    if executable { "" } else { "not " },
                    if file.executable {
                        "executable"
                    } else {
                        "plain"
                    }
                )));
            }
        }
        Ok(())
    }

    /// Removes what a crash left behind: staging directories, interrupted
    /// temporary files, and the views of publishes that never committed.
    /// Run it at startup, while no stage is writing into a staging directory.
    pub fn recover(&self) -> Result<(), ProductError> {
        for entry in read_dir_if_exists(&self.root.join(STAGING_DIR))? {
            remove_any(&entry)?;
        }
        for dir in [BLOB_DIR, PRODUCT_DIR] {
            for entry in read_dir_if_exists(&self.root.join(dir))? {
                if entry
                    .file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with('.'))
                {
                    remove_any(&entry)?;
                }
            }
        }
        for intent in read_dir_if_exists(&self.root.join(PENDING_DIR))? {
            self.recover_publish(&intent)?;
        }
        Ok(())
    }

    fn recover_publish(&self, intent: &Path) -> Result<(), ProductError> {
        let stem = intent.file_stem().map(|s| s.to_string_lossy().into_owned());
        let committed = stem.as_deref().is_some_and(|hex| {
            self.root
                .join(PRODUCT_DIR)
                .join(format!("{hex}.json"))
                .exists()
        });
        if !committed {
            let text = fs::read(intent).map_err(io_error(format!("read {}", intent.display())))?;
            let pending: PendingPublish = serde_json::from_slice(&text)
                .map_err(|e| malformed(&intent.display().to_string(), e))?;
            // The intent was validated on write; re-check so an edited record
            // cannot point recovery outside the run directory.
            let view = self
                .root
                .join(self.relative_view(&self.root.join(&pending.view))?);
            if view_belongs_to_other_product(&view, stem.as_deref()) {
                // A committed product's view, displaced mid-swap: keep it.
            } else {
                remove_any(&view)?;
            }
            remove_siblings_of(&view)?;
        }
        remove_any(intent)
    }
}

enum Prepared {
    Existing(ProductManifest),
    Viewed(ProductManifest),
}

fn create_new(path: &Path) -> Result<File, ProductError> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(io_error(format!("create {}", path.display())))
}

fn malformed(what: &str, reason: impl std::fmt::Display) -> ProductError {
    ProductError::Malformed {
        what: what.to_string(),
        reason: reason.to_string(),
    }
}

fn pretty(manifest: &ProductManifest) -> Result<Vec<u8>, ProductError> {
    let mut bytes =
        serde_json::to_vec_pretty(manifest).map_err(|e| malformed("product manifest", e))?;
    bytes.push(b'\n');
    Ok(bytes)
}

/// Syncs `dir` and every directory below it.
fn sync_tree(dir: &Path) -> Result<(), ProductError> {
    for entry in fs::read_dir(dir).map_err(io_error(format!("read {}", dir.display())))? {
        let entry = entry.map_err(io_error(format!("read {}", dir.display())))?;
        if entry.file_type().is_ok_and(|t| t.is_dir()) {
            sync_tree(&entry.path())?;
        }
    }
    sync_dir(dir).map_err(io_error(format!("sync {}", dir.display())))
}

fn read_dir_if_exists(dir: &Path) -> Result<Vec<PathBuf>, ProductError> {
    match fs::read_dir(dir) {
        Ok(entries) => entries
            .map(|e| e.map(|e| e.path()))
            .collect::<Result<_, _>>()
            .map_err(io_error(format!("read {}", dir.display()))),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(io_error(format!("read {}", dir.display()))(e)),
    }
}

fn remove_any(path: &Path) -> Result<(), ProductError> {
    let removed = match fs::symlink_metadata(path) {
        Ok(meta) if meta.is_dir() => fs::remove_dir_all(path),
        Ok(_) => fs::remove_file(path),
        Err(e) => Err(e),
    };
    match removed {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(io_error(format!("remove {}", path.display()))(e)),
    }
}

/// True when `view` carries the manifest of a stage other than `stage_hex`.
fn view_belongs_to_other_product(view: &Path, stage_hex: Option<&str>) -> bool {
    let Ok(bytes) = fs::read(view.join(VIEW_MANIFEST)) else {
        return false;
    };
    serde_json::from_slice::<ProductManifest>(&bytes)
        .is_ok_and(|m| Some(m.stage.hex()) != stage_hex)
}

/// Removes the hidden temporary siblings a view build or replacement leaves.
fn remove_siblings_of(view: &Path) -> Result<(), ProductError> {
    let (Some(parent), Some(name)) = (view.parent(), view.file_name()) else {
        return Ok(());
    };
    let prefix = format!(".{}.tmp-", name.to_string_lossy());
    for entry in read_dir_if_exists(parent)? {
        if entry
            .file_name()
            .is_some_and(|n| n.to_string_lossy().starts_with(&prefix))
        {
            remove_any(&entry)?;
        }
    }
    Ok(())
}
