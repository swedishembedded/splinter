// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The POSIX backend: plain files, write-once by hard link, atomic replace
//! by rename, reads by `pread`.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

use super::{Key, Kind, StorageBackend};
use crate::error::{Error, Result};

/// Large sequential writes suit shared filesystems; writers batch to this.
const PREFERRED_WRITE_BYTES: usize = 8 * 1024 * 1024;

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// A backend over a directory.
#[derive(Debug, Clone)]
pub struct PosixBackend {
    root: PathBuf,
}

impl PosixBackend {
    /// A backend rooted at `root`; nothing is created until a write.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn path(&self, key: &Key) -> PathBuf {
        let base = self.root.join(key.kind().dir());
        if key.kind().fans_out() {
            let name = key.name();
            let fan: String = name.chars().take(2).collect();
            base.join(fan).join(name)
        } else {
            key.name()
                .split('/')
                .fold(base, |path, part| path.join(part))
        }
    }

    fn temp_beside(path: &Path) -> PathBuf {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("object");
        let n = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        path.with_file_name(format!(".{name}.{}.{n}.tmp", std::process::id()))
    }

    /// Writes `bytes` to a fresh temporary file beside `path` and syncs it.
    /// Returns the temporary file and whether a directory had to be created
    /// for it, in which case the directories above need syncing too.
    fn stage(path: &Path, bytes: &[u8]) -> Result<(PathBuf, bool)> {
        let dir = path.parent().unwrap_or(Path::new("."));
        let created = !dir.exists();
        fs::create_dir_all(dir).map_err(Error::io(dir))?;
        let temp = Self::temp_beside(path);
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)
            .map_err(Error::io(&temp))?;
        file.write_all(bytes).map_err(Error::io(&temp))?;
        file.sync_all().map_err(Error::io(&temp))?;
        Ok((temp, created))
    }

    fn sync_dir(dir: &Path) -> Result<()> {
        File::open(dir)
            .and_then(|d| d.sync_all())
            .map_err(Error::io(dir))
    }

    /// Syncs `dir` and, when it is new, every directory above it up to the
    /// root, so a power loss cannot keep a file's entry but lose the
    /// directory that holds it.
    fn sync_up(&self, dir: &Path, created: bool) -> Result<()> {
        Self::sync_dir(dir)?;
        if created {
            let mut at = dir.parent();
            while let Some(parent) = at {
                Self::sync_dir(parent)?;
                if parent == self.root {
                    break;
                }
                at = parent.parent();
            }
        }
        Ok(())
    }

    fn open(&self, key: &Key) -> Result<(File, PathBuf)> {
        let path = self.path(key);
        match File::open(&path) {
            Ok(file) => Ok((file, path)),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Err(Error::NotFound {
                what: format!("{} `{}`", key.kind().dir(), key.name()),
            }),
            Err(e) => Err(Error::Io { path, source: e }),
        }
    }
}

impl StorageBackend for PosixBackend {
    fn write_once(&self, key: &Key, bytes: &[u8]) -> Result<bool> {
        let path = self.path(key);
        // A copy that is already there may be an orphan about to be collected.
        // Touching it restarts its grace period, so the writer that is about
        // to publish it cannot lose it to a collection already in progress.
        // If it is collected between the two steps, write it again.
        for _ in 0..4 {
            let (temp, created) = Self::stage(&path, bytes)?;
            let linked = fs::hard_link(&temp, &path);
            let _ = fs::remove_file(&temp);
            match linked {
                Ok(()) => {
                    if let Some(dir) = path.parent() {
                        self.sync_up(dir, created)?;
                    }
                    return Ok(true);
                }
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
                    match OpenOptions::new()
                        .write(true)
                        .open(&path)
                        .and_then(|f| f.set_modified(SystemTime::now()))
                    {
                        Ok(()) => return Ok(false),
                        Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
                        Err(source) => return Err(Error::Io { path, source }),
                    }
                }
                Err(e) => return Err(Error::Io { path, source: e }),
            }
        }
        Err(Error::corrupt(
            format!("{} `{}`", key.kind().dir(), key.name()),
            "it was removed again and again while being written",
        ))
    }

    fn replace(&self, key: &Key, bytes: &[u8]) -> Result<()> {
        let path = self.path(key);
        let (temp, created) = Self::stage(&path, bytes)?;
        if let Err(e) = fs::rename(&temp, &path) {
            let _ = fs::remove_file(&temp);
            return Err(Error::Io { path, source: e });
        }
        match path.parent() {
            Some(dir) => self.sync_up(dir, created),
            None => Ok(()),
        }
    }

    fn rename(&self, from: &Key, to: &Key) -> Result<bool> {
        let (source, target) = (self.path(from), self.path(to));
        if let Some(dir) = target.parent() {
            fs::create_dir_all(dir).map_err(Error::io(dir))?;
        }
        match fs::rename(&source, &target) {
            Ok(()) => {
                if let Some(dir) = target.parent() {
                    self.sync_up(dir, true)?;
                }
                if let Some(dir) = source.parent() {
                    Self::sync_dir(dir)?;
                }
                Ok(true)
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(source_error) => Err(Error::Io {
                path: source,
                source: source_error,
            }),
        }
    }

    fn read(&self, key: &Key) -> Result<Vec<u8>> {
        let (_, path) = self.open(key)?;
        fs::read(&path).map_err(Error::io(path))
    }

    fn read_range(&self, key: &Key, offset: u64, len: usize) -> Result<Vec<u8>> {
        let (file, path) = self.open(key)?;
        let mut buf = vec![0u8; len];
        file.read_exact_at(&mut buf, offset).map_err(|e| {
            if e.kind() == io::ErrorKind::UnexpectedEof {
                Error::corrupt(
                    path.display().to_string(),
                    format!("{len} bytes at {offset} are past the end"),
                )
            } else {
                Error::Io {
                    path: path.clone(),
                    source: e,
                }
            }
        })?;
        Ok(buf)
    }

    fn len(&self, key: &Key) -> Result<u64> {
        let (file, path) = self.open(key)?;
        file.metadata().map(|m| m.len()).map_err(Error::io(path))
    }

    fn exists(&self, key: &Key) -> Result<bool> {
        let path = self.path(key);
        path.try_exists().map_err(Error::io(path))
    }

    fn list(&self, kind: Kind) -> Result<Vec<Key>> {
        let base = self.root.join(kind.dir());
        let mut found = Vec::new();
        let mut pending = vec![(base.clone(), String::new())];
        while let Some((dir, prefix)) = pending.pop() {
            let entries = match fs::read_dir(&dir) {
                Ok(entries) => entries,
                Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
                Err(e) => {
                    return Err(Error::Io {
                        path: dir,
                        source: e,
                    })
                }
            };
            for entry in entries {
                let entry = entry.map_err(Error::io(&dir))?;
                let file_name = entry.file_name().to_string_lossy().into_owned();
                if file_name.starts_with('.') {
                    continue;
                }
                let kind_of = entry.file_type().map_err(Error::io(entry.path()))?;
                if kind_of.is_dir() {
                    let next = if kind.fans_out() {
                        String::new()
                    } else {
                        format!("{prefix}{file_name}/")
                    };
                    pending.push((entry.path(), next));
                } else if let Ok(key) = Key::new(kind, &format!("{prefix}{file_name}")) {
                    found.push(key);
                }
            }
        }
        found.sort_by(|a, b| a.name().cmp(b.name()));
        Ok(found)
    }

    fn remove(&self, key: &Key) -> Result<()> {
        let path = self.path(key);
        match fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(Error::Io { path, source: e }),
        }
    }

    fn modified(&self, key: &Key) -> Result<SystemTime> {
        let (file, path) = self.open(key)?;
        file.metadata()
            .and_then(|m| m.modified())
            .map_err(Error::io(path))
    }

    fn touch(&self, key: &Key) -> Result<()> {
        match self.open(key) {
            Ok((file, path)) => {
                drop(file);
                fs::File::options()
                    .write(true)
                    .open(&path)
                    .and_then(|f| f.set_modified(SystemTime::now()))
                    .map_err(Error::io(path))
            }
            Err(Error::NotFound { .. }) => Ok(()),
            Err(other) => Err(other),
        }
    }

    fn preferred_write_bytes(&self) -> usize {
        PREFERRED_WRITE_BYTES
    }
}
