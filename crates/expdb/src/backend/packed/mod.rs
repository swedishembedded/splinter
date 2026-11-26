// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in keeping
// file counts bounded on quota-limited parallel filesystems, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The packed backend: immutable objects appended to a few pack files, with
//! an append-only journal as their index.
//!
//! A filesystem that counts files (a project quota, a parallel filesystem's
//! metadata servers) pays per object in the loose layout. Here the number of
//! files depends on the bytes stored, not on the number of objects: the
//! journal, one pack file per [`DEFAULT_PACK_FILE_BYTES`], and the refs, pins
//! and overlays, which stay small files beside them because they are few and
//! are replaced in place.
//!
//! Every change happens under an exclusive `flock` on the journal, in one
//! order: the object's bytes are written to the active pack and synced, then
//! one journal entry naming them is appended and synced. An object exists
//! once its entry is whole. A crash between the two leaves bytes no entry
//! names; a crash inside the append leaves a torn last entry. The next writer
//! cuts both off before it appends, and a reader stops at the first entry
//! that is not whole, so a crash never leaves a store that cannot be read.
//! Writers in other processes (or other handles in this one) wait for the
//! lock; a filesystem without locks refuses the write instead.
//!
//! Removal, renaming and touching are journal entries too, so the bytes of a
//! collected object stay in its pack until the state is packed again
//! (`splinter state pack`), which copies only what is live.

mod journal;

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use self::journal::{parse, Entry, Index, Location, Stop};
use super::layout::Layout;
use super::{Key, Kind, PosixBackend, StorageBackend};
use crate::error::{Error, Result};

/// The directory under the root that holds the journal and the packs.
const DIR: &str = "packs";
/// The journal's file name.
const JOURNAL: &str = "journal";
/// What a journal starts with.
const HEADER: &[u8; 16] = b"splinter-pack 1\n";
/// A pack file is closed for appends once it holds this much; an object
/// larger than this gets a pack of its own.
pub const DEFAULT_PACK_FILE_BYTES: u64 = 1 << 30;
/// Large sequential writes suit shared filesystems; writers batch to this.
const PREFERRED_WRITE_BYTES: usize = 8 * 1024 * 1024;
/// No journal entry is longer; a longer length field is damage.
const MAX_ENTRY: usize = 4096;

/// Whether objects of `kind` live in the packs. Refs, pins and overlays are
/// few, small and replaced in place, so they stay files, where a binary that
/// knows only the loose layout also finds the format marker.
fn packs(kind: Kind) -> bool {
    !matches!(kind, Kind::Ref | Kind::Pin | Kind::Overlay)
}

fn now_ns() -> u64 {
    ns_of(SystemTime::now())
}

fn ns_of(time: SystemTime) -> u64 {
    time.duration_since(UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

#[derive(Debug, Default)]
struct State {
    index: Index,
    /// The journal, once it exists.
    journal: Option<File>,
    /// Whether `journal` was opened for writing: a reader of a read-only
    /// store opens it for reading only, and a writer opens it again.
    journal_writable: bool,
    /// Pack files opened for reading, by number.
    readers: HashMap<u32, Arc<File>>,
}

/// Releases the journal's `flock` when dropped.
struct Flock(File);

impl Drop for Flock {
    fn drop(&mut self) {
        let _ = self.0.unlock();
    }
}

/// A backend that packs immutable objects; see the module documentation.
#[derive(Debug)]
pub struct PackedBackend {
    root: PathBuf,
    files: PosixBackend,
    pack_file_bytes: u64,
    state: Mutex<State>,
}

impl PackedBackend {
    /// A backend rooted at `root`; nothing is created until a write.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self::with_pack_file_bytes(root, DEFAULT_PACK_FILE_BYTES)
    }

    /// The same, closing pack files for appends at `bytes`, so a test can
    /// see packs roll over without writing gigabytes.
    pub fn with_pack_file_bytes(root: impl Into<PathBuf>, bytes: u64) -> Self {
        let root = root.into();
        Self {
            files: PosixBackend::new(&root),
            root,
            pack_file_bytes: bytes.max(1),
            state: Mutex::new(State::default()),
        }
    }

    /// The root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn dir(&self) -> PathBuf {
        self.root.join(DIR)
    }

    fn journal_path(&self) -> PathBuf {
        self.dir().join(JOURNAL)
    }

    fn pack_path(&self, pack: u32) -> PathBuf {
        self.dir().join(format!("{pack:08}.pack"))
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Applies what other writers appended to the journal since the last
    /// look. A reader stops at the first entry that is not whole: it is being
    /// written, or a crash tore it and the next writer cuts it off.
    fn refresh(&self, state: &mut State) -> Result<()> {
        if state.journal.is_none() {
            let path = self.journal_path();
            let opened = match OpenOptions::new().read(true).write(true).open(&path) {
                Ok(file) => Ok((file, true)),
                // A committed store is read-only; reading it needs no more.
                Err(e) if e.kind() == io::ErrorKind::PermissionDenied => {
                    File::open(&path).map(|file| (file, false))
                }
                Err(e) => Err(e),
            };
            match opened {
                Ok((file, writable)) => {
                    state.journal = Some(file);
                    state.journal_writable = writable;
                }
                Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
                Err(e) => return Err(Error::Io { path, source: e }),
            }
        }
        let Some(journal) = state.journal.as_ref() else {
            return Ok(());
        };
        let path = self.journal_path();
        let len = journal.metadata().map_err(Error::io(&path))?.len();
        if state.index.applied == 0 {
            if len < HEADER.len() as u64 {
                // Being created by a writer that holds the lock.
                return Ok(());
            }
            let mut header = [0u8; HEADER.len()];
            journal
                .read_exact_at(&mut header, 0)
                .map_err(Error::io(&path))?;
            if &header != HEADER {
                return Err(Error::corrupt(
                    path.display().to_string(),
                    "it does not start as a splinter pack journal",
                ));
            }
            state.index.applied = HEADER.len() as u64;
        }
        if len <= state.index.applied {
            state.index.stop = Some(Stop::End);
            return Ok(());
        }
        let unread = usize::try_from(len - state.index.applied)
            .map_err(|_| Error::corrupt(path.display().to_string(), "it is larger than memory"))?;
        let mut bytes = vec![0u8; unread];
        journal
            .read_exact_at(&mut bytes, state.index.applied)
            .map_err(Error::io(&path))?;
        let (entries, used, stop) = parse(&bytes);
        for entry in entries {
            state.index.apply(entry);
        }
        state.index.applied += used as u64;
        state.index.stop = Some(stop);
        Ok(())
    }

    /// Runs `change` holding this handle's mutex and the journal's `flock`,
    /// on an index that includes every other writer's appends and a store
    /// cut back to its last whole entry.
    fn change<T>(&self, change: impl FnOnce(&Self, &mut State) -> Result<T>) -> Result<T> {
        let mut state = self.state();
        let lock = self.lock_journal(&mut state)?;
        self.refresh(&mut state)?;
        self.cut_torn_journal(&mut state)?;
        let result = change(self, &mut state);
        drop(lock);
        result
    }

    /// Creates the journal if needed and takes its exclusive lock.
    fn lock_journal(&self, state: &mut State) -> Result<Flock> {
        let path = self.journal_path();
        if state.journal.is_none() || !state.journal_writable {
            let dir = self.dir();
            let created = !dir.exists();
            fs::create_dir_all(&dir).map_err(Error::io(&dir))?;
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(&path)
                .map_err(Error::io(&path))?;
            sync_dir(&dir)?;
            if created {
                sync_dir(&self.root)?;
            }
            state.journal = Some(file);
            state.journal_writable = true;
        }
        let journal = state
            .journal
            .as_ref()
            .ok_or_else(|| Error::corrupt(path.display().to_string(), "it vanished"))?;
        let guard = journal.try_clone().map_err(Error::io(&path))?;
        guard.lock().map_err(|e| match e.kind() {
            io::ErrorKind::Unsupported => Error::invalid(
                "storage",
                format!(
                    "{} cannot be locked, so concurrent writers could not be kept apart; \
                     put the state on a filesystem with flock support",
                    path.display()
                ),
            ),
            _ => Error::Io {
                path: path.clone(),
                source: e,
            },
        })?;
        let guard = Flock(guard);
        let len = journal.metadata().map_err(Error::io(&path))?.len();
        if len < HEADER.len() as u64 {
            // New, or a creator died before its header was whole.
            journal.set_len(0).map_err(Error::io(&path))?;
            journal.write_all_at(HEADER, 0).map_err(Error::io(&path))?;
            journal.sync_data().map_err(Error::io(&path))?;
        }
        Ok(guard)
    }

    /// Cuts a torn last entry off the journal. Only a writer holding the lock
    /// may: nobody else is appending, so an entry that is not whole was left
    /// by a crash. Damage followed by whole entries is not a torn write and
    /// is reported instead of cut.
    fn cut_torn_journal(&self, state: &mut State) -> Result<()> {
        let path = self.journal_path();
        let Some(journal) = state.journal.as_ref() else {
            return Ok(());
        };
        let len = journal.metadata().map_err(Error::io(&path))?.len();
        if len <= state.index.applied {
            return Ok(());
        }
        match state.index.stop {
            Some(Stop::Incomplete | Stop::Damaged { last: true }) | Some(Stop::End) | None => {
                journal
                    .set_len(state.index.applied)
                    .map_err(Error::io(&path))?;
                journal.sync_data().map_err(Error::io(&path))?;
                state.index.stop = Some(Stop::End);
                Ok(())
            }
            Some(Stop::Damaged { last: false }) => Err(Error::corrupt(
                path.display().to_string(),
                format!(
                    "the entry at byte {} is damaged and more follow; \
                     restore the state from a copy (`splinter state repair --from`)",
                    state.index.applied
                ),
            )),
        }
    }

    /// Appends `objects` to the packs and one journal entry for each, in
    /// that order, with one sync of each. Called holding the lock.
    fn append(&self, state: &mut State, objects: &[(&Key, &[u8], u64)]) -> Result<()> {
        if objects.is_empty() {
            return Ok(());
        }
        let (mut pack, mut end) = state.index.append_point();
        let mut open: Option<(u32, File, bool)> = None;
        let mut entries = Vec::with_capacity(objects.len());
        let mut synced: Vec<(File, bool, PathBuf)> = Vec::new();
        for (key, bytes, modified_ns) in objects {
            let len = bytes.len() as u64;
            if end > 0 && end.saturating_add(len) > self.pack_file_bytes {
                pack += 1;
                end = 0;
            }
            if open.as_ref().map(|(n, _, _)| *n) != Some(pack) {
                if let Some((n, file, created)) = open.take() {
                    synced.push((file, created, self.pack_path(n)));
                }
                let (file, created) = self.open_for_append(pack, end)?;
                open = Some((pack, file, created));
            }
            let path = self.pack_path(pack);
            if let Some((_, file, _)) = open.as_ref() {
                file.write_all_at(bytes, end).map_err(Error::io(&path))?;
            }
            entries.push(Entry::Put {
                key: (*key).clone(),
                location: Location {
                    pack,
                    offset: end,
                    len,
                    modified_ns: *modified_ns,
                    digest: *blake3::hash(bytes).as_bytes(),
                },
            });
            end += len;
        }
        if let Some((n, file, created)) = open.take() {
            synced.push((file, created, self.pack_path(n)));
        }
        let mut new_pack = false;
        for (file, created, path) in &synced {
            file.sync_data().map_err(Error::io(path))?;
            new_pack |= *created;
        }
        if new_pack {
            sync_dir(&self.dir())?;
        }
        self.journal(state, entries)
    }

    /// Opens pack `pack` to append at `end`, which is where its last
    /// journalled object ends: bytes past it were written by an append that
    /// died before its entry, and are cut off. Returns whether the file is
    /// new.
    fn open_for_append(&self, pack: u32, end: u64) -> Result<(File, bool)> {
        let path = self.pack_path(pack);
        let existed = path.exists();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(Error::io(&path))?;
        let len = file.metadata().map_err(Error::io(&path))?.len();
        if len > end {
            file.set_len(end).map_err(Error::io(&path))?;
        } else if len < end {
            return Err(Error::corrupt(
                path.display().to_string(),
                format!("it holds {len} bytes, the journal names {end}"),
            ));
        }
        Ok((file, !existed))
    }

    /// Appends `entries` to the journal in one write, syncs it, and applies
    /// them. Called holding the lock.
    fn journal(&self, state: &mut State, entries: Vec<Entry>) -> Result<()> {
        let path = self.journal_path();
        let journal = state
            .journal
            .as_ref()
            .ok_or_else(|| Error::corrupt(path.display().to_string(), "it is not open"))?;
        let bytes: Vec<u8> = entries.iter().flat_map(Entry::frame).collect();
        journal
            .write_all_at(&bytes, state.index.applied)
            .map_err(Error::io(&path))?;
        journal.sync_data().map_err(Error::io(&path))?;
        state.index.applied += bytes.len() as u64;
        for entry in entries {
            state.index.apply(entry);
        }
        Ok(())
    }

    /// The location of `key`, looking at the journal again only when this
    /// handle does not know it: objects are immutable, so a known one is
    /// still what it was.
    fn locate(&self, key: &Key) -> Result<(Arc<File>, Location)> {
        let mut state = self.state();
        let location = match state.index.get(key) {
            Some(location) => location,
            None => {
                self.refresh(&mut state)?;
                // A file a collection has set aside is read where it lies, as
                // the loose layout reads it (`PosixBackend::open`).
                let aside = if key.kind().is_addressed() {
                    state.index.get(&key.set_aside()?)
                } else {
                    None
                };
                aside
                    .or_else(|| state.index.get(key))
                    .ok_or_else(|| Error::NotFound {
                        what: format!("{} `{}`", key.kind().dir(), key.name()),
                    })?
            }
        };
        let file = match state.readers.get(&location.pack) {
            Some(file) => Arc::clone(file),
            None => {
                let path = self.pack_path(location.pack);
                let file = Arc::new(File::open(&path).map_err(Error::io(&path))?);
                state.readers.insert(location.pack, Arc::clone(&file));
                file
            }
        };
        Ok((file, location))
    }

    /// The location of `key` as of the journal now.
    fn current(&self, key: &Key) -> Result<Option<Location>> {
        let mut state = self.state();
        self.refresh(&mut state)?;
        Ok(state.index.get(key))
    }

    /// Writes each object not already present, keeping the modification
    /// time given, in one append. Returns how many were written. For
    /// rewriting a store into this layout, whose objects keep their age.
    pub fn import(&self, objects: &[(Key, Vec<u8>, SystemTime)]) -> Result<usize> {
        if let Some((key, _, _)) = objects.iter().find(|(key, _, _)| !packs(key.kind())) {
            return Err(Error::invalid(
                "import",
                format!("{} are files, not packed objects", key.kind().dir()),
            ));
        }
        self.change(|this, state| {
            let fresh: Vec<(&Key, &[u8], u64)> = objects
                .iter()
                .filter(|(key, _, _)| state.index.get(key).is_none())
                .map(|(key, bytes, modified)| (key, bytes.as_slice(), ns_of(*modified)))
                .collect();
            this.append(state, &fresh)?;
            Ok(fresh.len())
        })
    }

    /// The files this backend keeps its packed objects in: the journal and
    /// the packs.
    pub fn pack_files(&self) -> Result<Vec<PathBuf>> {
        let dir = self.dir();
        let mut found = Vec::new();
        match fs::read_dir(&dir) {
            Ok(entries) => {
                for entry in entries {
                    found.push(entry.map_err(Error::io(&dir))?.path());
                }
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => {
                return Err(Error::Io {
                    path: dir,
                    source: e,
                })
            }
        }
        found.sort();
        Ok(found)
    }
}

fn sync_dir(dir: &Path) -> Result<()> {
    File::open(dir)
        .and_then(|d| d.sync_all())
        .map_err(Error::io(dir))
}

impl StorageBackend for PackedBackend {
    fn write_once(&self, key: &Key, bytes: &[u8]) -> Result<bool> {
        if !packs(key.kind()) {
            return self.files.write_once(key, bytes);
        }
        self.change(|this, state| {
            if state.index.get(key).is_some() {
                // As in the loose layout: writing what is there makes it young
                // again, so a collection in progress cannot take it.
                this.journal(
                    state,
                    vec![Entry::Touch {
                        key: key.clone(),
                        at_ns: now_ns(),
                    }],
                )?;
                return Ok(false);
            }
            this.append(state, &[(key, bytes, now_ns())])?;
            Ok(true)
        })
    }

    fn replace(&self, key: &Key, bytes: &[u8]) -> Result<()> {
        if !packs(key.kind()) {
            return self.files.replace(key, bytes);
        }
        self.change(|this, state| this.append(state, &[(key, bytes, now_ns())]))
    }

    fn rename(&self, from: &Key, to: &Key) -> Result<bool> {
        match (packs(from.kind()), packs(to.kind())) {
            (false, false) => self.files.rename(from, to),
            (true, true) => self.change(|this, state| {
                if state.index.get(from).is_none() {
                    return Ok(false);
                }
                this.journal(
                    state,
                    vec![Entry::Move {
                        from: from.clone(),
                        to: to.clone(),
                    }],
                )?;
                Ok(true)
            }),
            _ => Err(Error::invalid(
                "rename",
                format!(
                    "{} and {} are stored differently; an object cannot move between them",
                    from.kind().dir(),
                    to.kind().dir()
                ),
            )),
        }
    }

    fn read(&self, key: &Key) -> Result<Vec<u8>> {
        if !packs(key.kind()) {
            return self.files.read(key);
        }
        let (file, location) = self.locate(key)?;
        let len = usize::try_from(location.len).map_err(|_| {
            Error::corrupt(
                format!("{} `{}`", key.kind().dir(), key.name()),
                "too large",
            )
        })?;
        read_at(&file, &self.pack_path(location.pack), location.offset, len)
    }

    fn read_range(&self, key: &Key, offset: u64, len: usize) -> Result<Vec<u8>> {
        if !packs(key.kind()) {
            return self.files.read_range(key, offset, len);
        }
        let (file, location) = self.locate(key)?;
        if offset.saturating_add(len as u64) > location.len {
            return Err(Error::corrupt(
                format!("{} `{}`", key.kind().dir(), key.name()),
                format!("{len} bytes at {offset} are past the end"),
            ));
        }
        read_at(
            &file,
            &self.pack_path(location.pack),
            location.offset + offset,
            len,
        )
    }

    fn len(&self, key: &Key) -> Result<u64> {
        if !packs(key.kind()) {
            return self.files.len(key);
        }
        // As of now, so a removal by another writer shows, as it does for a
        // loose file; a file a collection has set aside is where it lies.
        let mut location = self.current(key)?;
        if location.is_none() && key.kind().is_addressed() {
            location = self.current(&key.set_aside()?)?;
        }
        let location = location.ok_or_else(|| Error::NotFound {
            what: format!("{} `{}`", key.kind().dir(), key.name()),
        })?;
        Ok(location.len)
    }

    fn exists(&self, key: &Key) -> Result<bool> {
        if !packs(key.kind()) {
            return self.files.exists(key);
        }
        Ok(self.current(key)?.is_some())
    }

    fn list(&self, kind: Kind) -> Result<Vec<Key>> {
        if !packs(kind) {
            return self.files.list(kind);
        }
        let mut state = self.state();
        self.refresh(&mut state)?;
        state
            .index
            .objects
            .get(&kind)
            .map(|names| names.keys().map(|name| Key::new(kind, name)).collect())
            .unwrap_or_else(|| Ok(Vec::new()))
    }

    fn remove(&self, key: &Key) -> Result<()> {
        if !packs(key.kind()) {
            return self.files.remove(key);
        }
        self.change(|this, state| {
            if state.index.get(key).is_none() {
                return Ok(());
            }
            this.journal(state, vec![Entry::Remove { key: key.clone() }])
        })
    }

    fn modified(&self, key: &Key) -> Result<SystemTime> {
        if !packs(key.kind()) {
            return self.files.modified(key);
        }
        let location = self.current(key)?.ok_or_else(|| Error::NotFound {
            what: format!("{} `{}`", key.kind().dir(), key.name()),
        })?;
        Ok(UNIX_EPOCH + Duration::from_nanos(location.modified_ns))
    }

    fn touch(&self, key: &Key) -> Result<()> {
        if !packs(key.kind()) {
            return self.files.touch(key);
        }
        self.change(|this, state| {
            if state.index.get(key).is_none() {
                return Ok(());
            }
            this.journal(
                state,
                vec![Entry::Touch {
                    key: key.clone(),
                    at_ns: now_ns(),
                }],
            )
        })
    }

    fn preferred_write_bytes(&self) -> usize {
        PREFERRED_WRITE_BYTES
    }

    fn layout(&self) -> Layout {
        Layout::Packed
    }
}

fn read_at(file: &File, path: &Path, offset: u64, len: usize) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; len];
    file.read_exact_at(&mut buf, offset).map_err(|e| {
        if e.kind() == io::ErrorKind::UnexpectedEof {
            Error::corrupt(
                path.display().to_string(),
                format!("{len} bytes at {offset} are past the end"),
            )
        } else {
            Error::Io {
                path: path.to_path_buf(),
                source: e,
            }
        }
    })?;
    Ok(buf)
}
