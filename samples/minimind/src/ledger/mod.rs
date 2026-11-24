// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements resumable, idempotent training pipelines
// that pick up exactly where a killed run stopped, for its clients. If your
// team needs expertise in crash-safe multi-stage model training, you can
// procure our services by sending an email to info@swedishembedded.com.

//! A run directory's stage ledger: which units of work are done, under which
//! inputs, with which outputs.
//!
//! A *unit* is a stage (`pretrain`) or one item of a stage
//! (`explore/task-0003`). Its [`StageKey`] is the content address of
//! everything its result depends on ([`KeyParts`]): the unit's name, its slice
//! of the configuration, the digests of its inputs and the code revisions it
//! ran with. A key never names a time or a path, so the same inputs give the
//! same key in any run directory and on any day.
//!
//! The ledger is one append-only pack, `<run>/ledger.jsonl`. [`StageLedger::run`]
//! is the whole protocol:
//!
//! 1. a **commit** under the same key whose outputs still hash to their
//!    recorded digests is reused, and the work is not run;
//! 2. otherwise a **begin** line opens a new attempt, numbered past every
//!    earlier attempt of the unit, so an attempt id is never used twice;
//!    when the last attempt under this key began and never committed, the
//!    work is told which one it resumes;
//! 3. the work writes its outputs inside the run directory and names them;
//! 4. the outputs are hashed and the **commit** line is appended and synced.
//!    That line is the commit marker: a unit is done exactly when it exists.
//!
//! A process killed anywhere before step 4 leaves a begin without a commit
//! (or a torn last line, which is cut off), and the next run resumes. A
//! commit whose output no longer hashes to its digest is corrupt: refused
//! with the output named, or, when the caller asks for it, invalidated by an
//! **invalidate** line and recomputed. An output must be a file or a
//! directory below the run directory; a directory's digest covers every file
//! in it by relative path.
//!
//! One process holds a run directory at a time: [`StageLedger::open`] takes a
//! non-blocking exclusive lock on `<run>/ledger.lock` for the ledger's life,
//! and the operating system releases it when the process dies.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::os::unix::io::AsRawFd;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use splinter_sdk::vocabulary::digest::{digest_of, Digest};

mod lock;
use lock::{whole_lines, Locked};

/// The ledger pack's file name inside a run directory.
pub const LEDGER_FILE: &str = "ledger.jsonl";

/// The lock file that keeps a second process out of a run directory.
pub const LOCK_FILE: &str = "ledger.lock";

/// How often a held run directory is asked for before it is refused, and
/// how long apart: a two-second bound.
const LOCK_TRIES: u32 = 40;
const LOCK_RETRY: std::time::Duration = std::time::Duration::from_millis(50);

/// Why a ledger operation failed.
#[derive(Debug, thiserror::Error)]
pub enum LedgerError {
    /// Another process holds the run directory.
    #[error("run directory {path} is held by another process (its {LOCK_FILE} is locked)")]
    Busy {
        /// The run directory.
        path: String,
    },
    /// The ledger or an output cannot be read or written.
    #[error("{path}: {reason}")]
    Io {
        /// What was read or written.
        path: String,
        /// What went wrong.
        reason: String,
    },
    /// The ledger holds a line that is not a ledger record.
    #[error("stage ledger {path}: line {line} is not a ledger record: {reason}")]
    Malformed {
        /// The pack.
        path: String,
        /// The 1-based line number.
        line: usize,
        /// The parse error.
        reason: String,
    },
    /// A committed output no longer hashes to its recorded digest.
    #[error("{unit}: output {output} ({path}) is corrupt: recorded {expected}, found {found}")]
    Corrupt {
        /// The unit that produced it.
        unit: String,
        /// The output's name.
        output: String,
        /// Its path relative to the run directory.
        path: String,
        /// The digest the commit recorded.
        expected: String,
        /// What it hashes to now (`missing` when it is gone).
        found: String,
    },
    /// The work named an output that is not a file or directory below the
    /// run directory.
    #[error("{unit}: output {output} ({path}): {reason}")]
    Output {
        /// The unit.
        unit: String,
        /// The output's name.
        output: String,
        /// The path the work named.
        path: String,
        /// Why it is refused.
        reason: String,
    },
    /// The work itself failed; the attempt stays open and the next run
    /// resumes it.
    #[error("{unit} (attempt {attempt}): {reason}")]
    Stage {
        /// The unit.
        unit: String,
        /// The attempt that failed.
        attempt: u32,
        /// The work's error.
        reason: String,
    },
}

/// The content address of everything a unit's result depends on.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct StageKey(Digest);

impl StageKey {
    /// The key as text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl std::fmt::Display for StageKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// What a [`StageKey`] is derived from. Maps are ordered, so the order the
/// parts are given in never changes the key.
#[derive(Clone, Debug, Serialize)]
pub struct KeyParts {
    stage: String,
    config: serde_json::Value,
    inputs: BTreeMap<String, Digest>,
    revisions: BTreeMap<String, String>,
}

impl KeyParts {
    /// The parts of stage `stage`'s key, with no configuration, inputs or
    /// revisions yet.
    #[must_use]
    pub fn new(stage: &str) -> Self {
        Self {
            stage: stage.to_string(),
            config: serde_json::Value::Null,
            inputs: BTreeMap::new(),
            revisions: BTreeMap::new(),
        }
    }

    /// The stage's slice of the configuration: only what changes its result.
    #[must_use]
    pub fn config<T: Serialize + ?Sized>(mut self, config: &T) -> Self {
        self.config = serde_json::to_value(config).unwrap_or_else(|e| {
            unreachable!("a configuration slice is plain serialisable data: {e}")
        });
        self
    }

    /// An input, by the digest of what it is (usually an upstream output).
    #[must_use]
    pub fn input(mut self, name: &str, digest: &Digest) -> Self {
        self.inputs.insert(name.to_string(), digest.clone());
        self
    }

    /// The revision of code the result depends on (`splinter`, `brain`, ...).
    #[must_use]
    pub fn revision(mut self, name: &str, revision: &str) -> Self {
        self.revisions
            .insert(name.to_string(), revision.to_string());
        self
    }

    /// The key.
    #[must_use]
    pub fn key(&self) -> StageKey {
        StageKey(digest_of(self))
    }
}

/// One output of a committed unit.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Output {
    /// The output's name within the unit.
    pub name: String,
    /// Its path relative to the run directory.
    pub path: String,
    /// What it hashed to when committed.
    pub digest: Digest,
}

/// A unit that is done.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Commit {
    /// The unit.
    pub unit: String,
    /// The key it was done under.
    pub key: StageKey,
    /// The attempt that did it.
    pub attempt: u32,
    /// Its outputs, in the order the work named them.
    pub outputs: Vec<Output>,
    /// What the work reported (counts, losses, probe results).
    pub summary: serde_json::Value,
}

impl Commit {
    /// The digest of output `name`, if the unit has one by that name.
    #[must_use]
    pub fn output(&self, name: &str) -> Option<&Digest> {
        self.outputs
            .iter()
            .find(|o| o.name == name)
            .map(|o| &o.digest)
    }
}

/// Where a unit stands under one key.
#[derive(Clone, Debug, PartialEq)]
pub enum Standing {
    /// Done; its outputs are not checked by this answer.
    Committed(Commit),
    /// Its last attempt under this key began and never committed.
    Interrupted {
        /// That attempt.
        attempt: u32,
    },
    /// Never attempted under this key, or its commit was invalidated.
    Fresh,
}

/// What to do with a commit whose outputs no longer match.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OnCorrupt {
    /// Fail with [`LedgerError::Corrupt`], naming the output.
    Refuse,
    /// Invalidate the commit (recorded with the reason) and run the work again.
    Recompute,
}

/// The attempt a piece of work runs as.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Attempt {
    unit: String,
    number: u32,
    resumes: Option<u32>,
}

impl Attempt {
    /// The attempt's number; numbers of a unit only grow.
    #[must_use]
    pub fn number(&self) -> u32 {
        self.number
    }

    /// The interrupted attempt under the same key this one takes over, if any:
    /// work that can continue from durable partial state (a training job by
    /// its identity) does; anything else starts over.
    #[must_use]
    pub fn resumes(&self) -> Option<u32> {
        self.resumes
    }

    /// The attempt's durable id, `<unit>#<number>`.
    #[must_use]
    pub fn id(&self) -> String {
        format!("{}#{}", self.unit, self.number)
    }
}

/// What a piece of work did: its report and the outputs it wrote.
#[derive(Clone, Debug)]
pub struct StageWork {
    summary: serde_json::Value,
    outputs: Vec<(String, String)>,
}

impl StageWork {
    /// Work that reports `summary` and has no outputs yet.
    #[must_use]
    pub fn new(summary: serde_json::Value) -> Self {
        Self {
            summary,
            outputs: Vec::new(),
        }
    }

    /// Adds output `name`, a file or directory at `path` relative to the run
    /// directory.
    #[must_use]
    pub fn output(mut self, name: &str, path: &str) -> Self {
        self.outputs.push((name.to_string(), path.to_string()));
        self
    }
}

/// The result of [`StageLedger::run`].
#[derive(Clone, Debug, PartialEq)]
pub struct Ran {
    /// The commit the unit stands on.
    pub commit: Commit,
    /// True when an earlier commit was reused and no work ran.
    pub reused: bool,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "record", rename_all = "snake_case")]
enum Line {
    Begin {
        unit: String,
        key: StageKey,
        attempt: u32,
    },
    Commit(Commit),
    Invalidate {
        unit: String,
        key: StageKey,
        reason: String,
    },
}

/// The stage ledger of one run directory, held by this process.
pub struct StageLedger {
    dir: PathBuf,
    pack: PathBuf,
    _lock: File,
}

impl std::fmt::Debug for StageLedger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StageLedger")
            .field("dir", &self.dir)
            .finish()
    }
}

fn io_error(path: &Path, e: &std::io::Error) -> LedgerError {
    LedgerError::Io {
        path: path.display().to_string(),
        reason: e.to_string(),
    }
}

impl StageLedger {
    /// The ledger of run directory `dir`, created if absent, held by this
    /// process until dropped.
    ///
    /// # Errors
    /// The directory cannot be created, or another process holds it.
    pub fn open(dir: &Path) -> Result<Self, LedgerError> {
        std::fs::create_dir_all(dir).map_err(|e| io_error(dir, &e))?;
        let lock_path = dir.join(LOCK_FILE);
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&lock_path)
            .map_err(|e| io_error(&lock_path, &e))?;
        // A process forked by any thread of this one holds a copy of every
        // descriptor until it execs, so a lock just released can still look
        // held for a moment; another process holding the run never lets go.
        // The wait is bounded.
        let mut tries = 0;
        // SAFETY: `lock` is an open descriptor for the lifetime of the call.
        while unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            tries += 1;
            if tries >= LOCK_TRIES {
                return Err(LedgerError::Busy {
                    path: dir.display().to_string(),
                });
            }
            std::thread::sleep(LOCK_RETRY);
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            pack: dir.join(LEDGER_FILE),
            _lock: lock,
        })
    }

    /// The run directory.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn lines(&self) -> Result<Vec<Line>, LedgerError> {
        let bytes = match std::fs::read(&self.pack) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(io_error(&self.pack, &e)),
        };
        whole_lines(&bytes)
            .enumerate()
            .map(|(i, line)| {
                serde_json::from_slice(line).map_err(|e| LedgerError::Malformed {
                    path: self.pack.display().to_string(),
                    line: i + 1,
                    reason: e.to_string(),
                })
            })
            .collect()
    }

    fn append(&self, line: &Line) -> Result<(), LedgerError> {
        let mut text = serde_json::to_string(line)
            .unwrap_or_else(|e| unreachable!("a ledger line is plain data: {e}"));
        text.push('\n');
        let mut locked = Locked::open(&self.pack).map_err(|e| io_error(&self.pack, &e))?;
        locked.append(&text).map_err(|e| io_error(&self.pack, &e))
    }

    /// Where `unit` stands under `key`.
    ///
    /// # Errors
    /// The ledger cannot be read.
    pub fn standing(&self, unit: &str, key: &StageKey) -> Result<Standing, LedgerError> {
        let mut standing = Standing::Fresh;
        for line in self.lines()? {
            match line {
                Line::Begin {
                    unit: u,
                    key: k,
                    attempt,
                } if u == unit && &k == key => {
                    if !matches!(standing, Standing::Committed(_)) {
                        standing = Standing::Interrupted { attempt };
                    }
                }
                Line::Commit(c) if c.unit == unit && &c.key == key => {
                    standing = Standing::Committed(c);
                }
                Line::Invalidate {
                    unit: u, key: k, ..
                } if u == unit && &k == key => {
                    standing = Standing::Fresh;
                }
                _ => {}
            }
        }
        Ok(standing)
    }

    /// Every commit that stands (not invalidated), oldest first.
    ///
    /// # Errors
    /// The ledger cannot be read.
    pub fn commits(&self) -> Result<Vec<Commit>, LedgerError> {
        let mut commits: Vec<Commit> = Vec::new();
        for line in self.lines()? {
            match line {
                Line::Commit(c) => commits.push(c),
                Line::Invalidate { unit, key, .. } => {
                    commits.retain(|c| !(c.unit == unit && c.key == key));
                }
                Line::Begin { .. } => {}
            }
        }
        Ok(commits)
    }

    fn next_attempt(&self, unit: &str) -> Result<u32, LedgerError> {
        let last = self
            .lines()?
            .into_iter()
            .filter_map(|line| match line {
                Line::Begin {
                    unit: u, attempt, ..
                } if u == unit => Some(attempt),
                _ => None,
            })
            .max();
        Ok(last.map_or(1, |n| n + 1))
    }

    /// Checks that every output of `commit` still hashes to its digest.
    ///
    /// # Errors
    /// [`LedgerError::Corrupt`] naming the first output that does not.
    pub fn verify(&self, commit: &Commit) -> Result<(), LedgerError> {
        for output in &commit.outputs {
            let found = match self.digest_output(&commit.unit, &output.name, &output.path) {
                Ok(digest) => digest.to_string(),
                Err(LedgerError::Output { .. }) => "missing".to_string(),
                Err(other) => return Err(other),
            };
            if found != output.digest.as_str() {
                return Err(LedgerError::Corrupt {
                    unit: commit.unit.clone(),
                    output: output.name.clone(),
                    path: output.path.clone(),
                    expected: output.digest.to_string(),
                    found,
                });
            }
        }
        Ok(())
    }

    /// Records that the commit of `unit` under `key` no longer stands.
    ///
    /// # Errors
    /// The ledger cannot be written.
    pub fn invalidate(&self, unit: &str, key: &StageKey, reason: &str) -> Result<(), LedgerError> {
        self.append(&Line::Invalidate {
            unit: unit.to_string(),
            key: key.clone(),
            reason: reason.to_string(),
        })
    }

    /// Runs `work` as `unit` under `key` unless a sound commit of it exists;
    /// see the module documentation for the protocol.
    ///
    /// # Errors
    /// The work failed ([`LedgerError::Stage`]; the attempt stays open), named
    /// an unusable output, a reused commit is corrupt and `on_corrupt` is
    /// [`OnCorrupt::Refuse`], or the ledger cannot be read or written.
    pub fn run<F, E>(
        &self,
        unit: &str,
        key: &StageKey,
        on_corrupt: OnCorrupt,
        work: F,
    ) -> Result<Ran, LedgerError>
    where
        F: FnOnce(&Attempt) -> Result<StageWork, E>,
        E: std::fmt::Display,
    {
        let resumes = match self.standing(unit, key)? {
            Standing::Committed(commit) => match (self.verify(&commit), on_corrupt) {
                (Ok(()), _) => {
                    return Ok(Ran {
                        commit,
                        reused: true,
                    })
                }
                (Err(e @ LedgerError::Corrupt { .. }), OnCorrupt::Recompute) => {
                    self.invalidate(unit, key, &e.to_string())?;
                    None
                }
                (Err(e), _) => return Err(e),
            },
            Standing::Interrupted { attempt } => Some(attempt),
            Standing::Fresh => None,
        };
        let attempt = Attempt {
            unit: unit.to_string(),
            number: self.next_attempt(unit)?,
            resumes,
        };
        self.append(&Line::Begin {
            unit: unit.to_string(),
            key: key.clone(),
            attempt: attempt.number,
        })?;
        let done = work(&attempt).map_err(|e| LedgerError::Stage {
            unit: unit.to_string(),
            attempt: attempt.number,
            reason: e.to_string(),
        })?;
        let mut outputs = Vec::with_capacity(done.outputs.len());
        for (name, path) in done.outputs {
            let digest = self.digest_output(unit, &name, &path)?;
            outputs.push(Output { name, path, digest });
        }
        let commit = Commit {
            unit: unit.to_string(),
            key: key.clone(),
            attempt: attempt.number,
            outputs,
            summary: done.summary,
        };
        self.append(&Line::Commit(commit.clone()))?;
        Ok(Ran {
            commit,
            reused: false,
        })
    }

    fn digest_output(&self, unit: &str, name: &str, rel: &str) -> Result<Digest, LedgerError> {
        let refuse = |reason: &str| LedgerError::Output {
            unit: unit.to_string(),
            output: name.to_string(),
            path: rel.to_string(),
            reason: reason.to_string(),
        };
        let relative = Path::new(rel);
        if rel.is_empty()
            || !relative
                .components()
                .all(|c| matches!(c, Component::Normal(_)))
        {
            return Err(refuse("not a plain path below the run directory"));
        }
        let path = self.dir.join(relative);
        let meta = std::fs::symlink_metadata(&path).map_err(|_| refuse("does not exist"))?;
        if meta.is_file() {
            let file = File::open(&path).map_err(|e| io_error(&path, &e))?;
            Digest::of_reader(file).map_err(|e| io_error(&path, &e))
        } else if meta.is_dir() {
            let mut files = BTreeMap::new();
            self.digest_tree(&path, &path, &mut files, &refuse)?;
            Ok(digest_of(&files))
        } else {
            Err(refuse("is neither a file nor a directory"))
        }
    }

    fn digest_tree(
        &self,
        root: &Path,
        dir: &Path,
        files: &mut BTreeMap<String, Digest>,
        refuse: &dyn Fn(&str) -> LedgerError,
    ) -> Result<(), LedgerError> {
        let entries = std::fs::read_dir(dir).map_err(|e| io_error(dir, &e))?;
        for entry in entries {
            let entry = entry.map_err(|e| io_error(dir, &e))?;
            let path = entry.path();
            let kind = entry.file_type().map_err(|e| io_error(&path, &e))?;
            if kind.is_dir() {
                self.digest_tree(root, &path, files, refuse)?;
            } else if kind.is_file() {
                let file = File::open(&path).map_err(|e| io_error(&path, &e))?;
                let digest = Digest::of_reader(file).map_err(|e| io_error(&path, &e))?;
                let rel = path
                    .strip_prefix(root)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .into_owned();
                files.insert(rel, digest);
            } else {
                return Err(refuse(&format!(
                    "{} is neither a file nor a directory",
                    path.display()
                )));
            }
        }
        Ok(())
    }
}

/// Writes `bytes` to `path` so that a reader sees the old content or the new,
/// never a part: a sibling temporary file, synced, then renamed over it.
///
/// # Errors
/// The file or its temporary sibling cannot be written or renamed.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), LedgerError> {
    use std::io::Write;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| io_error(dir, &e))?;
    }
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".partial");
    let tmp = PathBuf::from(tmp);
    let mut file = File::create(&tmp).map_err(|e| io_error(&tmp, &e))?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|e| io_error(&tmp, &e))?;
    std::fs::rename(&tmp, path).map_err(|e| io_error(path, &e))
}
