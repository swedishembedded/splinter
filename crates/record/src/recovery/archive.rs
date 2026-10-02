// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements durable, recoverable state for learning
// agents, for its clients. If your team needs expertise in backup and
// disaster recovery of training data, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The archive: a zstd-compressed tar led by an `ARCHIVE` member that lists
//! every file of the state with its size and digest and names the snapshot,
//! followed by the files in path order. Nothing in it depends on when or by
//! whom it was made.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use splinter_expdb::{Config, ContentId, Database};

use super::tar::{Reader, Writer};
use crate::artifacts::ArtifactStore;
use crate::error::StoreError;
use crate::workspace::Workspace;
use crate::StateRoot;

const MANIFEST: &str = "ARCHIVE";
const FORMAT: &str = "splinter-archive 1";
/// A manifest larger than this is not one this tool wrote.
const MANIFEST_LIMIT: u64 = 256 << 20;
/// Members of the database are read whole; none is larger than a pack.
const DATABASE_FILE_LIMIT: u64 = 4 << 30;
const COMPRESSION_LEVEL: i32 = 3;
/// The signals that are state rather than traffic: pointer history and the
/// loss ledger.
const SIGNAL_PREFIXES: [&str; 2] = ["pointer/", "lost/"];

/// What an archive covers.
#[derive(Clone, Debug, Default)]
pub struct ArchiveOptions {
    /// Leave the artifacts out: the database alone.
    pub no_artifacts: bool,
    /// An earlier archive: whatever it already holds is left out of this one,
    /// which then restores only together with it.
    pub since: Option<PathBuf>,
}

/// What an archive holds.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Archived {
    /// The snapshot the archive reproduces.
    pub snapshot: String,
    /// Files the state consists of.
    pub files: usize,
    /// Files whose bytes this archive carries.
    pub carried: usize,
    /// Of the files, how many are artifacts.
    pub artifacts: usize,
    /// The archive's size in bytes.
    pub bytes: u64,
}

/// What a restore put in place.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Restored {
    /// Files restored.
    pub files: usize,
    /// Of them, how many are artifacts.
    pub artifacts: usize,
    /// Whether the restored state passed a deep verification before it was
    /// put in place. A restore that does not verify is refused, so this is
    /// true whenever a restore returns.
    pub verified: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct FileEntry {
    path: String,
    size: u64,
    blake3: String,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    base: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct SignalEntry {
    name: String,
    note: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct Manifest {
    format: String,
    snapshot: String,
    files: Vec<FileEntry>,
    signals: Vec<SignalEntry>,
}

fn io_error(path: &Path) -> impl FnOnce(io::Error) -> StoreError + '_ {
    move |source| StoreError::Io {
        path: path.to_path_buf(),
        source,
    }
}

fn damaged(archive: &Path, error: io::Error) -> StoreError {
    StoreError::Recovery(format!(
        "{} is not a sound archive: {error}",
        archive.display()
    ))
}

type Members = Reader<zstd::stream::read::Decoder<'static, io::BufReader<fs::File>>>;

fn open(archive: &Path) -> Result<Members, StoreError> {
    let file = fs::File::open(archive).map_err(io_error(archive))?;
    let decoder = zstd::stream::read::Decoder::new(file).map_err(|e| damaged(archive, e))?;
    Ok(Reader::new(decoder))
}

/// The manifest an archive leads with, leaving its reader at the first file.
fn read_manifest(archive: &Path) -> Result<(Manifest, Members), StoreError> {
    let mut members = open(archive)?;
    let entry = members
        .next()
        .map_err(|e| damaged(archive, e))?
        .filter(|e| e.path == MANIFEST)
        .ok_or_else(|| StoreError::Recovery(format!("{} has no manifest", archive.display())))?;
    if entry.size > MANIFEST_LIMIT {
        return Err(StoreError::Recovery(format!(
            "{} has an implausible manifest",
            archive.display()
        )));
    }
    let mut text = Vec::new();
    members
        .data()
        .read_to_end(&mut text)
        .map_err(|e| damaged(archive, e))?;
    let manifest: Manifest = serde_json::from_slice(&text).map_err(|e| {
        StoreError::Recovery(format!("{}: unreadable manifest: {e}", archive.display()))
    })?;
    if manifest.format != FORMAT {
        return Err(StoreError::Recovery(format!(
            "{} is `{}`, not `{FORMAT}`",
            archive.display(),
            manifest.format
        )));
    }
    Ok((manifest, members))
}

/// The digest and size of the file at `path`.
fn describe(member: &str, path: &Path) -> Result<FileEntry, StoreError> {
    let mut file = fs::File::open(path).map_err(|e| match e.kind() {
        io::ErrorKind::NotFound => StoreError::Recovery(format!(
            "{member} is missing from the state; run `splinter state repair` first"
        )),
        _ => io_error(path)(e),
    })?;
    let mut hasher = blake3::Hasher::new();
    let size = io::copy(&mut file, &mut hasher).map_err(io_error(path))?;
    Ok(FileEntry {
        path: member.to_owned(),
        size,
        blake3: hasher.finalize().to_hex().to_string(),
        base: false,
    })
}

/// A member of `artifacts/` must be exactly what an artifact's path is.
fn artifact_member_ok(member: &str) -> bool {
    let Some(rest) = member.strip_prefix("artifacts/") else {
        return false;
    };
    let Some((fan, name)) = rest.split_once('/') else {
        return false;
    };
    let (hex, extension) = name.split_at(name.find('.').unwrap_or(name.len()));
    hex.len() == 64
        && hex
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        && hex.starts_with(fan)
        && extension.len() <= 33
        && extension
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

/// Writes `reader` to `dest` through a temporary file in the same directory and
/// puts it in place read-only if, and only if, its bytes hash to `expected_hex`.
/// Returns whether they did.
pub(super) fn place_artifact(
    dest: &Path,
    reader: impl Read,
    expected_hex: &str,
) -> Result<bool, StoreError> {
    let dir = dest
        .parent()
        .ok_or_else(|| StoreError::Recovery(format!("{} has no directory", dest.display())))?;
    fs::create_dir_all(dir).map_err(io_error(dir))?;
    let temporary = dir.join(format!(".tmp-restore-{}", std::process::id()));
    let outcome = (|| -> io::Result<bool> {
        let mut file = fs::File::create(&temporary)?;
        let mut hasher = blake3::Hasher::new();
        let mut reader = reader;
        let mut buffer = vec![0u8; 1 << 20];
        loop {
            let n = reader.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            hasher.update(&buffer[..n]);
            file.write_all(&buffer[..n])?;
        }
        if hasher.finalize().to_hex().as_str() != expected_hex {
            return Ok(false);
        }
        file.sync_all()?;
        let mut permissions = file.metadata()?.permissions();
        permissions.set_readonly(true);
        fs::set_permissions(&temporary, permissions)?;
        fs::rename(&temporary, dest)?;
        Ok(true)
    })();
    match outcome {
        Ok(true) => Ok(true),
        Ok(false) => {
            let _ = fs::remove_file(&temporary);
            Ok(false)
        }
        Err(e) => {
            let _ = fs::remove_file(&temporary);
            Err(io_error(dest)(e))
        }
    }
}

/// Reads a database member whole, at most `size` bytes.
pub(super) fn read_member(reader: impl Read, size: u64) -> io::Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(size.min(1 << 26) as usize);
    reader
        .take(size.min(DATABASE_FILE_LIMIT))
        .read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn dir_has_entries(path: &Path) -> bool {
    fs::read_dir(path).is_ok_and(|mut entries| entries.next().is_some())
}

/// Calls `visit` with every member of `archive` the caller wants, with its
/// data.
pub(super) fn scan(
    archive: &Path,
    mut visit: impl FnMut(&str, u64, &mut dyn Read) -> Result<(), StoreError>,
) -> Result<(), StoreError> {
    let mut members = open(archive)?;
    while let Some(entry) = members.next().map_err(|e| damaged(archive, e))? {
        if entry.path == MANIFEST {
            continue;
        }
        let mut data = members.data();
        visit(&entry.path, entry.size, &mut data)?;
    }
    Ok(())
}

impl Workspace {
    /// Packs the state into `out`: the database as of one snapshot, which is
    /// held while it is read so a collection cannot take its files, and the
    /// artifacts the database records. The archive is written beside `out` and
    /// renamed, so an archive that exists is whole.
    pub fn archive(&self, out: &Path, options: &ArchiveOptions) -> Result<Archived, StoreError> {
        self.commit()?;
        let db = self.database()?;
        let snapshot = db.snapshot()?;
        snapshot.persist()?;
        let holder = format!("archive-{}", std::process::id());
        snapshot.pin(&holder)?;
        let packed = self.pack(&db, &snapshot, out, options);
        let released = db.unpin(&holder);
        let packed = packed?;
        released?;
        Ok(packed)
    }

    fn pack(
        &self,
        db: &Database,
        snapshot: &splinter_expdb::manifest::Snapshot,
        out: &Path,
        options: &ArchiveOptions,
    ) -> Result<Archived, StoreError> {
        let root = self.root();
        let mut files: BTreeMap<String, (FileEntry, PathBuf)> = BTreeMap::new();
        for relative in snapshot.files()? {
            let member = format!("expdb/{relative}");
            let path = root.expdb().join(&relative);
            let entry = describe(&member, &path)?;
            files.insert(member, (entry, path));
        }
        let mut artifacts = 0;
        if !options.no_artifacts {
            let written_off: HashSet<String> = self
                .losses()?
                .into_iter()
                .filter(|l| l.kind == "artifact")
                .map(|l| l.id)
                .collect();
            for artifact in ArtifactStore::new(self, root).list()? {
                if written_off.contains(artifact.digest.as_str()) {
                    continue;
                }
                let member = artifact.relative_path();
                let path = root.path().join(&member);
                let entry = describe(&member, &path)?;
                if entry.blake3 != artifact.digest.hex() {
                    return Err(StoreError::Recovery(format!(
                        "artifact {} is corrupt on disk; run `splinter state verify --deep`",
                        artifact.digest
                    )));
                }
                files.insert(member, (entry, path));
                artifacts += 1;
            }
        }
        let mut signals = Vec::new();
        for prefix in SIGNAL_PREFIXES {
            for (name, note) in db.signals(prefix)? {
                signals.push(SignalEntry { name, note });
            }
        }

        let base: HashMap<String, String> = match &options.since {
            Some(earlier) => read_manifest(earlier)?
                .0
                .files
                .into_iter()
                .map(|f| (f.path, f.blake3))
                .collect(),
            None => HashMap::new(),
        };
        let total = files.len();
        let mut listed = Vec::with_capacity(total);
        let mut carry = Vec::new();
        for (member, (mut entry, path)) in files {
            entry.base = base.get(&member) == Some(&entry.blake3);
            if !entry.base {
                carry.push((member, entry.size, path));
            }
            listed.push(entry);
        }
        let manifest = serde_json::to_vec(&Manifest {
            format: FORMAT.into(),
            snapshot: snapshot.id().to_string(),
            files: listed,
            signals,
        })
        .map_err(|source| StoreError::Serialize {
            what: "archive manifest",
            source,
        })?;

        let partial = out.with_extension("partial");
        let written = (|| -> io::Result<u64> {
            let file = fs::File::create(&partial)?;
            let encoder =
                zstd::stream::write::Encoder::new(BufWriter::new(file), COMPRESSION_LEVEL)?;
            let mut tar = Writer::new(encoder);
            tar.member(MANIFEST, manifest.len() as u64, manifest.as_slice())?;
            for (member, size, path) in &carry {
                tar.member(member, *size, fs::File::open(path)?)?;
            }
            let file = tar
                .finish()?
                .finish()?
                .into_inner()
                .map_err(|e| e.into_error())?;
            file.sync_all()?;
            Ok(file.metadata()?.len())
        })();
        let bytes = match written {
            Ok(bytes) => bytes,
            Err(e) => {
                let _ = fs::remove_file(&partial);
                return Err(io_error(out)(e));
            }
        };
        fs::rename(&partial, out).map_err(io_error(out))?;
        Ok(Archived {
            snapshot: snapshot.id().to_string(),
            files: total,
            carried: carry.len(),
            artifacts,
            bytes,
        })
    }

    /// Unpacks `archives` (the one to restore, then any it was made
    /// incrementally after) into `root`, which must hold no state. Every
    /// member is checked against the manifest's digest as it is read, the
    /// state is assembled and verified deeply next to `root`, and only then
    /// moved into place: a restore that fails leaves `root` as it was.
    pub fn restore(root: &StateRoot, archives: &[PathBuf]) -> Result<Restored, StoreError> {
        let Some(newest) = archives.first() else {
            return Err(StoreError::Recovery("no archive to restore".into()));
        };
        if root.expdb().exists() || dir_has_entries(&root.artifacts()) {
            return Err(StoreError::Recovery(format!(
                "{} already holds state; restore into an empty root",
                root.path().display()
            )));
        }
        let (manifest, _) = read_manifest(newest)?;
        let stage = root.path().join(".restoring");
        let _ = fs::remove_dir_all(&stage);
        fs::create_dir_all(&stage).map_err(io_error(&stage))?;
        let staged = StateRoot::new(&stage);
        match assemble(&staged, &manifest, archives) {
            Ok(restored) => {
                let placed = (|| -> io::Result<()> {
                    let artifacts = root.artifacts();
                    if artifacts.exists() {
                        fs::remove_dir(&artifacts)?;
                    }
                    fs::rename(staged.artifacts(), &artifacts)?;
                    fs::rename(staged.expdb(), root.expdb())?;
                    fs::remove_dir_all(&stage)
                })();
                placed.map_err(io_error(root.path()))?;
                Ok(restored)
            }
            Err(e) => {
                let _ = fs::remove_dir_all(&stage);
                Err(e)
            }
        }
    }
}

fn assemble(
    staged: &StateRoot,
    manifest: &Manifest,
    archives: &[PathBuf],
) -> Result<Restored, StoreError> {
    fs::create_dir_all(staged.artifacts()).map_err(io_error(&staged.artifacts()))?;
    let db = Database::open(staged.expdb(), Config::default())?;
    let wanted: HashMap<&str, &FileEntry> = manifest
        .files
        .iter()
        .map(|f| (f.path.as_str(), f))
        .collect();
    let mut written: HashSet<String> = HashSet::new();
    for archive in archives {
        scan(archive, |member, size, data| {
            let Some(want) = wanted.get(member) else {
                return Ok(());
            };
            if written.contains(member) {
                return Ok(());
            }
            if size != want.size {
                return Err(StoreError::Recovery(format!(
                    "{member} in {} has {size} bytes, the manifest says {}",
                    archive.display(),
                    want.size
                )));
            }
            let sound = if let Some(relative) = member.strip_prefix("expdb/") {
                let bytes = read_member(data, size).map_err(io_error(archive))?;
                blake3::hash(&bytes).to_hex().as_str() == want.blake3
                    && db.fill(relative, &bytes)?
            } else if artifact_member_ok(member) {
                place_artifact(&staged.path().join(member), data, &want.blake3)?
            } else {
                false
            };
            if !sound {
                return Err(StoreError::Recovery(format!(
                    "{member} in {} does not match its digest; the archive is damaged",
                    archive.display()
                )));
            }
            written.insert(member.to_owned());
            Ok(())
        })?;
    }
    let missing: Vec<&str> = manifest
        .files
        .iter()
        .map(|f| f.path.as_str())
        .filter(|p| !written.contains(*p))
        .collect();
    if !missing.is_empty() {
        return Err(StoreError::Recovery(format!(
            "{} files are in none of the archives given (first: {}); an incremental archive restores only together with the archives it was made after",
            missing.len(),
            missing[0]
        )));
    }
    let head = ContentId::parse(&manifest.snapshot)?;
    db.adopt(head)?;
    for signal in &manifest.signals {
        db.signal(&signal.name, &signal.note)?;
    }
    drop(db);

    let verification = Workspace::at(staged).verify(true)?;
    if !verification.is_sound() {
        return Err(StoreError::Recovery(format!(
            "the restored state does not verify: {verification:?}"
        )));
    }
    Ok(Restored {
        files: manifest.files.len(),
        artifacts: manifest
            .files
            .iter()
            .filter(|f| f.path.starts_with("artifacts/"))
            .count(),
        verified: true,
    })
}
