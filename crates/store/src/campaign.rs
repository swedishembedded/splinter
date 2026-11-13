// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for long-running model-improvement
// campaigns that survive a crash and keep an honest account of what each
// experiment cost and found, for its clients. If your team needs expertise in
// resumable training orchestration or experiment ledgers you can procure our
// services by sending an email to info@swedishembedded.com.

//! The durable state of a continuous learning campaign, and its ledger, in
//! one file.
//!
//! A campaign keeps everything in a single append-only pack,
//! `<root>/campaigns/<name>.pack`: one JSON record per line, never a file per
//! state or per ledger entry (the project disk has a quota on the number of
//! files, and a file per record killed a registered run). Two kinds of line:
//!
//! * a **state** line holds a state document the controller defines, and is
//!   followed by a small **head** line naming its content address, its version
//!   and where the state line starts, written in the same append, so the
//!   current state is found from the tail of the file without reading the
//!   history;
//! * a **ledger** line holds one entry of the experiment ledger.
//!
//! Saving a state is a compare-and-swap done under an exclusive lock on the
//! pack: it names the state it replaces, and loses
//! ([`CampaignError::Conflict`]) when another controller advanced the
//! campaign meanwhile. A crash leaves the old head (a torn last line is cut
//! off when the pack is next opened for writing). Every earlier state stays in
//! the pack until it is compacted away; compaction keeps the ledger whole and
//! the newest few states, and is the only time the pack is rewritten.
//!
//! The ledger is the campaign's account of its experiments: one immutable
//! entry per arm event with the hypothesis, the digest of the configuration,
//! the GPU-hours, the cost, and the verdict. An unmeasured quantity is absent,
//! never `0`.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::io::AsRawFd;
use std::path::PathBuf;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use splinter_core::digest::{canonical_json, Digest};

use crate::StateRoot;

/// States kept when a pack is compacted.
pub const COMPACT_KEEP_STATES: usize = 4;

/// Size of a pack past which a save compacts it.
pub const COMPACT_AT_BYTES: u64 = 32 << 20;

/// Why a campaign operation failed.
#[derive(Debug, thiserror::Error)]
pub enum CampaignError {
    /// The name cannot be a campaign: empty, or it holds a character a file
    /// name keeps for its own structure.
    #[error("{0:?} is not a campaign name: use letters, digits, '-', '_' and '.'")]
    Name(String),
    /// Another controller advanced the campaign since this one loaded it.
    #[error("campaign {campaign} was advanced by another controller (expected {expected:?}, found {found:?}); reload it")]
    Conflict {
        /// The campaign.
        campaign: String,
        /// The state this controller loaded.
        expected: Option<String>,
        /// The state the campaign is at.
        found: Option<String>,
    },
    /// The pack cannot be read or written.
    #[error("campaign pack {path}: {reason}")]
    Io {
        /// The pack.
        path: String,
        /// What went wrong.
        reason: String,
    },
    /// The pack holds a record that is not what it should be.
    #[error("campaign pack {path}: {reason}")]
    Corrupt {
        /// The pack.
        path: String,
        /// What is wrong.
        reason: String,
    },
}

/// One line of a pack.
#[derive(Serialize, Deserialize)]
#[serde(tag = "record", rename_all = "snake_case")]
enum Line<S> {
    State { at: String, state: S },
    Head(Head),
    Ledger { entry: Box<LedgerEntry> },
}

/// Where the current state is and what it is.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Head {
    version: u64,
    digest: Digest,
    /// Byte offset of the state line.
    offset: u64,
    /// Byte length of the state line, newline excluded.
    len: u64,
    at: String,
}

/// One campaign's durable state and ledger.
pub struct CampaignStore {
    path: PathBuf,
    name: String,
}

/// A state as it was loaded: the value and the address a later save names as
/// the one it replaces.
#[derive(Clone, Debug, PartialEq)]
pub struct Loaded<S> {
    /// The state.
    pub state: S,
    /// Its address.
    pub digest: Digest,
    /// How many states the campaign has had, this one included.
    pub version: u64,
}

/// An exclusive lock on the pack, released when dropped.
struct Locked {
    file: File,
}

impl Locked {
    fn new(file: File) -> std::io::Result<Self> {
        // SAFETY: `file` is an open descriptor for the lifetime of the call.
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) } != 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(Self { file })
    }
}

impl Drop for Locked {
    fn drop(&mut self) {
        // SAFETY: the descriptor is still open; closing it would release the
        // lock as well, so a failure here has nothing left to undo.
        unsafe { libc::flock(self.file.as_raw_fd(), libc::LOCK_UN) };
    }
}

impl CampaignStore {
    /// The store of campaign `name` under `root`.
    ///
    /// # Errors
    /// The name is empty or holds anything but letters, digits, `-`, `_`, `.`.
    pub fn new(root: &StateRoot, name: &str) -> Result<Self, CampaignError> {
        let sound = !name.is_empty()
            && name != "."
            && name != ".."
            && name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
        if !sound {
            return Err(CampaignError::Name(name.to_string()));
        }
        Ok(Self {
            path: root.path().join("campaigns").join(format!("{name}.pack")),
            name: name.to_string(),
        })
    }

    /// The campaign's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    fn io(&self, e: &std::io::Error) -> CampaignError {
        CampaignError::Io {
            path: self.path.display().to_string(),
            reason: e.to_string(),
        }
    }

    fn corrupt(&self, reason: impl Into<String>) -> CampaignError {
        CampaignError::Corrupt {
            path: self.path.display().to_string(),
            reason: reason.into(),
        }
    }

    /// The pack opened for reading and writing, created if absent, and
    /// locked; a torn last line is cut off. A pack replaced by a compaction
    /// while this waited for the lock is opened again.
    fn lock(&self) -> Result<Locked, CampaignError> {
        use std::os::unix::fs::MetadataExt;
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| self.io(&e))?;
        }
        loop {
            let file = OpenOptions::new()
                .create(true)
                .read(true)
                .write(true)
                .truncate(false)
                .open(&self.path)
                .map_err(|e| self.io(&e))?;
            let mut locked = Locked::new(file).map_err(|e| self.io(&e))?;
            let held = locked.file.metadata().map_err(|e| self.io(&e))?;
            let current = std::fs::metadata(&self.path).map_err(|e| self.io(&e))?;
            if (held.dev(), held.ino()) != (current.dev(), current.ino()) {
                continue;
            }
            let len = held.len();
            if len > 0 {
                let mut last = [0u8; 1];
                locked
                    .file
                    .seek(SeekFrom::Start(len - 1))
                    .and_then(|_| locked.file.read_exact(&mut last))
                    .map_err(|e| self.io(&e))?;
                if last[0] != b'\n' {
                    let bytes = self.read_all(&mut locked.file)?;
                    let sound = bytes.iter().rposition(|b| *b == b'\n').map_or(0, |i| i + 1);
                    locked.file.set_len(sound as u64).map_err(|e| self.io(&e))?;
                }
            }
            return Ok(locked);
        }
    }

    fn read_all(&self, file: &mut File) -> Result<Vec<u8>, CampaignError> {
        let mut bytes = Vec::new();
        file.seek(SeekFrom::Start(0)).map_err(|e| self.io(&e))?;
        file.read_to_end(&mut bytes).map_err(|e| self.io(&e))?;
        Ok(bytes)
    }

    /// The last head of the pack, found from its tail.
    fn head(&self, file: &mut File) -> Result<Option<Head>, CampaignError> {
        let len = file.metadata().map_err(|e| self.io(&e))?.len();
        let mut window: u64 = 4096;
        loop {
            let start = len.saturating_sub(window);
            file.seek(SeekFrom::Start(start)).map_err(|e| self.io(&e))?;
            let mut buf = vec![0u8; (len - start) as usize];
            file.read_exact(&mut buf).map_err(|e| self.io(&e))?;
            // Whole lines only: the first line of a window that does not start
            // at the file's start may be a fragment.
            let text_start = if start == 0 {
                0
            } else {
                buf.iter()
                    .position(|b| *b == b'\n')
                    .map_or(buf.len(), |i| i + 1)
            };
            for line in buf[text_start..].split(|b| *b == b'\n').rev() {
                if line.starts_with(br#"{"record":"head""#) {
                    let parsed: Line<serde_json::Value> =
                        serde_json::from_slice(line).map_err(|e| self.corrupt(e.to_string()))?;
                    if let Line::Head(head) = parsed {
                        return Ok(Some(head));
                    }
                }
            }
            if start == 0 {
                return Ok(None);
            }
            window = window.saturating_mul(4);
        }
    }

    /// The campaign's current state; `None` for a campaign never saved.
    ///
    /// # Errors
    /// The pack cannot be read, the state it names no longer hashes to its
    /// address, or is not an `S`.
    pub fn load<S: DeserializeOwned + Serialize>(
        &self,
    ) -> Result<Option<Loaded<S>>, CampaignError> {
        let mut file = match File::open(&self.path) {
            Ok(file) => file,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(self.io(&e)),
        };
        let Some(head) = self.head(&mut file)? else {
            return Ok(None);
        };
        file.seek(SeekFrom::Start(head.offset))
            .map_err(|e| self.io(&e))?;
        let mut line = vec![0u8; head.len as usize];
        file.read_exact(&mut line).map_err(|e| self.io(&e))?;
        let parsed: Line<S> =
            serde_json::from_slice(&line).map_err(|e| self.corrupt(e.to_string()))?;
        let Line::State { state, .. } = parsed else {
            return Err(self.corrupt("the head names a line that is not a state"));
        };
        let found = digest_of_state(&state).map_err(|e| self.corrupt(e.to_string()))?;
        if found != head.digest {
            return Err(self.corrupt(format!(
                "the state at byte {} hashes to {found}, not the {} its head names",
                head.offset, head.digest
            )));
        }
        Ok(Some(Loaded {
            state,
            digest: head.digest,
            version: head.version,
        }))
    }

    /// Makes `state` the campaign's current state, provided it still is the
    /// one `replaces` names (`None` for a campaign never saved). Returns the
    /// new state's address. Saving the state the campaign is already at
    /// changes nothing.
    ///
    /// # Errors
    /// [`CampaignError::Conflict`] when another controller moved the campaign;
    /// an unreadable or unwritable pack.
    pub fn save<S: Serialize>(
        &self,
        replaces: Option<&Digest>,
        state: &S,
        at: &str,
    ) -> Result<Digest, CampaignError> {
        let digest = digest_of_state(state).map_err(|e| self.corrupt(e.to_string()))?;
        let mut locked = self.lock()?;
        let head = self.head(&mut locked.file)?;
        let found = head.as_ref().map(|h| h.digest.clone());
        if found.as_ref() == Some(&digest) {
            return Ok(digest);
        }
        if found.as_ref() != replaces {
            return Err(CampaignError::Conflict {
                campaign: self.name.clone(),
                expected: replaces.map(|d| d.as_str().to_string()),
                found: found.map(|d| d.as_str().to_string()),
            });
        }
        let end = locked.file.metadata().map_err(|e| self.io(&e))?.len();
        let state_line = serde_json::to_string(&Line::State {
            at: at.to_string(),
            state,
        })
        .map_err(|e| self.corrupt(e.to_string()))?;
        let head_line = serde_json::to_string(&Line::<()>::Head(Head {
            version: head.map_or(0, |h| h.version) + 1,
            digest: digest.clone(),
            offset: end,
            len: state_line.len() as u64,
            at: at.to_string(),
        }))
        .map_err(|e| self.corrupt(e.to_string()))?;
        // One write, so a crash leaves both lines or a torn tail that the
        // next writer cuts off.
        let mut both = state_line;
        both.push('\n');
        both.push_str(&head_line);
        both.push('\n');
        locked
            .file
            .seek(SeekFrom::End(0))
            .map_err(|e| self.io(&e))?;
        locked
            .file
            .write_all(both.as_bytes())
            .map_err(|e| self.io(&e))?;
        locked.file.sync_data().map_err(|e| self.io(&e))?;
        if end + both.len() as u64 > COMPACT_AT_BYTES {
            self.compact_locked(&mut locked, COMPACT_KEEP_STATES)?;
        }
        Ok(digest)
    }

    /// Appends `entry` to the ledger; returns its address.
    ///
    /// # Errors
    /// The entry names another campaign, or the pack cannot be written.
    pub fn record(&self, entry: &LedgerEntry) -> Result<Digest, CampaignError> {
        if entry.campaign != self.name {
            return Err(self.corrupt(format!(
                "ledger entry names campaign {:?}, this is {:?}",
                entry.campaign, self.name
            )));
        }
        let digest = digest_of_state(entry).map_err(|e| self.corrupt(e.to_string()))?;
        let mut locked = self.lock()?;
        let mut line = serde_json::to_string(&Line::<()>::Ledger {
            entry: Box::new(entry.clone()),
        })
        .map_err(|e| self.corrupt(e.to_string()))?;
        line.push('\n');
        locked
            .file
            .seek(SeekFrom::End(0))
            .map_err(|e| self.io(&e))?;
        locked
            .file
            .write_all(line.as_bytes())
            .map_err(|e| self.io(&e))?;
        locked.file.sync_data().map_err(|e| self.io(&e))?;
        Ok(digest)
    }

    /// The campaign's ledger entries, in the order they were written.
    ///
    /// # Errors
    /// The pack cannot be read or holds a ledger line that is not an entry.
    pub fn ledger(&self) -> Result<Vec<LedgerEntry>, CampaignError> {
        let bytes = match std::fs::read(&self.path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(self.io(&e)),
        };
        let mut entries = Vec::new();
        let complete = bytes.iter().rposition(|b| *b == b'\n').map_or(0, |i| i + 1);
        for line in bytes[..complete].split(|b| *b == b'\n') {
            if line.starts_with(br#"{"record":"ledger""#) {
                let parsed: Line<()> =
                    serde_json::from_slice(line).map_err(|e| self.corrupt(e.to_string()))?;
                if let Line::Ledger { entry } = parsed {
                    entries.push(*entry);
                }
            }
        }
        Ok(entries)
    }

    /// Rewrites the pack keeping every ledger entry and the newest `keep`
    /// states (the current one always).
    ///
    /// # Errors
    /// The pack cannot be read or rewritten; it is left as it was.
    pub fn compact(&self, keep: usize) -> Result<(), CampaignError> {
        let mut locked = self.lock()?;
        self.compact_locked(&mut locked, keep)
    }

    fn compact_locked(&self, locked: &mut Locked, keep: usize) -> Result<(), CampaignError> {
        let bytes = self.read_all(&mut locked.file)?;
        // Each state line is followed by its head line; pair them.
        let mut kept_ledger: Vec<&[u8]> = Vec::new();
        let mut pairs: Vec<(&[u8], &[u8])> = Vec::new();
        let mut pending_state: Option<&[u8]> = None;
        for line in bytes.split(|b| *b == b'\n').filter(|l| !l.is_empty()) {
            if line.starts_with(br#"{"record":"state""#) {
                pending_state = Some(line);
            } else if line.starts_with(br#"{"record":"head""#) {
                if let Some(state) = pending_state.take() {
                    pairs.push((state, line));
                }
            } else if line.starts_with(br#"{"record":"ledger""#) {
                kept_ledger.push(line);
            }
        }
        let skip = pairs.len().saturating_sub(keep.max(1));
        let mut out: Vec<u8> = Vec::new();
        for line in kept_ledger {
            out.extend_from_slice(line);
            out.push(b'\n');
        }
        for (state, head) in pairs.iter().skip(skip) {
            let parsed: Line<()> =
                serde_json::from_slice(head).map_err(|e| self.corrupt(e.to_string()))?;
            let Line::Head(mut head) = parsed else {
                continue;
            };
            head.offset = out.len() as u64;
            head.len = state.len() as u64;
            out.extend_from_slice(state);
            out.push(b'\n');
            out.extend_from_slice(
                serde_json::to_string(&Line::<()>::Head(head))
                    .map_err(|e| self.corrupt(e.to_string()))?
                    .as_bytes(),
            );
            out.push(b'\n');
        }
        let staged = self.path.with_extension("pack.compacting");
        std::fs::write(&staged, &out).map_err(|e| self.io(&e))?;
        std::fs::rename(&staged, &self.path).map_err(|e| self.io(&e))?;
        // The lock was on the replaced file: a writer that was waiting for it
        // notices the pack is another file and opens the new one ([`Self::lock`]).
        Ok(())
    }
}

fn digest_of_state<T: Serialize + ?Sized>(value: &T) -> Result<Digest, serde_json::Error> {
    canonical_json(value).map(|bytes| Digest::of(&bytes))
}

/// What an arm cost, in what is counted; each is absent when not measured.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Cost {
    /// GPU-hours: wall-clock hours times the cards held.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_hours: Option<f64>,
    /// Rollouts collected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rollouts: Option<u64>,
    /// Tokens generated by the policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generated_tokens: Option<u64>,
    /// Optimiser steps taken.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub train_steps: Option<u64>,
}

/// How an arm ended.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum Verdict {
    /// The arm has started and has not ended.
    Running,
    /// Refused before it ran: its exam could not detect the target effect, or
    /// the plan was invalid.
    Refused {
        /// Why.
        reason: String,
    },
    /// The candidate passed the dev gate and became the champion.
    Accepted {
        /// Whether the improvement is significant at the level this gate may
        /// spend; an accepted arm that is not is a point-estimate advance.
        significant: bool,
    },
    /// The candidate failed the dev gate and the champion stayed.
    RolledBack {
        /// Why.
        reason: String,
    },
    /// The arm was stopped by a signal bound before it finished.
    Halted {
        /// Why.
        reason: String,
    },
    /// The arm failed to run.
    Failed {
        /// Why.
        reason: String,
    },
}

/// Where the rewards of an arm came from: one per environment the arm drew
/// tasks from, so a verdict can say how far its numbers deserve trust.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RewardSource {
    /// The environment.
    pub environment: String,
    /// How its rewards are established.
    pub class: splinter_core::verifiability::VerifierClass,
    /// The verifier that produced them, with its version.
    pub verifier: String,
}

/// One account of one arm of a campaign.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LedgerEntry {
    /// The campaign.
    pub campaign: String,
    /// The arm, unique within the campaign.
    pub arm: String,
    /// Which gate of the campaign this arm's verdict is: the n-th, from 1,
    /// which sets the level it may spend.
    pub generation: u32,
    /// The strategy the arm ran (`policy-gradient`, `sft-repair`, ...).
    pub strategy: String,
    /// What the arm was expected to show, in one sentence, written before it
    /// ran.
    pub hypothesis: String,
    /// The digest of the configuration the arm ran under.
    pub config_digest: Digest,
    /// When this entry was written, UTC.
    pub at: String,
    /// What it cost; empty fields are unmeasured.
    pub cost: Cost,
    /// What the arm discovered, in the campaign's unit (tasks newly solved on
    /// the dev set); absent for an arm still running.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub discoveries: Option<f64>,
    /// How it ended.
    pub verdict: Verdict,
    /// The environments whose rewards the arm trained and was judged on, with
    /// how each reward is established; empty for tasks that come from no
    /// environment factory.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reward_sources: Vec<RewardSource>,
    /// What a verdict on these rewards must say beside its number (`proxy
    /// only`, `judge-graded`, `human evidence`); absent when every reward is
    /// an objective check.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verdict_label: Option<String>,
    /// The numbers behind the verdict (dev pass rates, informative fraction,
    /// KL), by name.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub measurements: BTreeMap<String, f64>,
}

impl LedgerEntry {
    /// The latest entry of each arm, in the order the arms first appeared.
    #[must_use]
    pub fn latest_per_arm(entries: &[Self]) -> Vec<&Self> {
        let mut order: Vec<&str> = Vec::new();
        let mut latest: BTreeMap<&str, &Self> = BTreeMap::new();
        for entry in entries {
            if !latest.contains_key(entry.arm.as_str()) {
                order.push(&entry.arm);
            }
            latest.insert(&entry.arm, entry);
        }
        order.into_iter().map(|arm| latest[arm]).collect()
    }
}
