// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements online reinforcement-learning runs whose
// every group can be followed while they run, for its clients. If your team
// needs expertise in observable asynchronous policy optimisation, you can
// procure our services by sending an email to info@swedishembedded.com.

//! `groups.jsonl`: the live, append-only log of an online run's groups.
//!
//! One JSON line each time a group reaches `sent`, `applied`, `committed`,
//! `filtered`, `stale` or `rolled_back` (the first time it reaches it), as
//! the transition is made durable: when, in which epoch, the learner's
//! version then, the group's task, version span, rewards and each rollout's
//! outcome (a limit that ended it, an abandoned rollout's failures).
//! Besides groups the log takes the run's own records ([`GroupLog::note`]),
//! such as a rollback's audit.
//!
//! **Crash-safe and resumable.** Each line is written whole and synced
//! before the transition is acted on; a line a crash cut short is dropped
//! when the log is opened again, and every transition the ledger holds that
//! the log lacks is written then ([`GroupLog::open`]), so the log is the
//! ledger's transitions, each once.

use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use splinter_core::factory::group::{GroupKey, GroupState, RolloutOutcome, RolloutState};
use splinter_store::factory::runstate::GroupEntry;

use super::control::Change;
use super::seams::{GroupLedger, LedgerError};

/// The log's file name.
pub const GROUPS_FILE: &str = "groups.jsonl";

/// The states a group is logged at.
const LOGGED: [GroupState; 6] = [
    GroupState::Sent,
    GroupState::Applied,
    GroupState::Committed,
    GroupState::Filtered,
    GroupState::Stale,
    GroupState::RolledBack,
];

/// What the log keeps of one rollout.
#[derive(Clone, Debug, Default)]
struct Rollout {
    state: Option<RolloutState>,
    failures: u32,
    outcome: Option<RolloutOutcome>,
}

/// What the log keeps of one group.
#[derive(Clone, Debug)]
struct Group {
    seq: u64,
    task: String,
    epoch: u64,
    created_ms: Option<u64>,
    rollouts: Vec<Rollout>,
}

/// The log of one run's groups.
pub struct GroupLog {
    path: PathBuf,
    file: File,
    written: HashSet<(String, GroupState)>,
    groups: HashMap<GroupKey, Group>,
    learner_version: u64,
}

fn io(path: &Path) -> impl Fn(std::io::Error) -> LedgerError + '_ {
    move |e| LedgerError(format!("{}: {e}", path.display()))
}

impl GroupLog {
    /// The log at `path`, opened for appending: a line a crash cut short is
    /// dropped, and every transition `entries` (the ledger) records that the
    /// log lacks is written now, at `now_ms`, marked `recovered`.
    ///
    /// # Errors
    /// The file cannot be read or written.
    pub fn open(path: &Path, entries: &[GroupEntry], now_ms: u64) -> Result<Self, LedgerError> {
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
            Err(e) => return Err(io(path)(e)),
        };
        // Whole lines only: a crash may have cut the last one.
        let whole = text.rfind('\n').map_or(0, |end| end + 1);
        if whole < text.len() {
            let file = OpenOptions::new()
                .write(true)
                .open(path)
                .map_err(io(path))?;
            file.set_len(u64::try_from(whole).unwrap_or(u64::MAX))
                .map_err(io(path))?;
            file.sync_data().map_err(io(path))?;
        }
        let mut written = HashSet::new();
        for line in text[..whole].lines().filter(|l| !l.trim().is_empty()) {
            let value: Value = serde_json::from_str(line)
                .map_err(|e| LedgerError(format!("{}: a line is not JSON: {e}", path.display())))?;
            if let (Some(group), Some(state)) = (value["group"].as_str(), value["state"].as_str()) {
                if let Ok(state) = state.parse::<GroupState>() {
                    written.insert((group.to_string(), state));
                }
            }
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(io(parent))?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(io(path))?;
        let mut log = Self {
            path: path.to_path_buf(),
            file,
            written,
            groups: HashMap::new(),
            learner_version: 0,
        };
        for entry in entries {
            let group = &entry.group;
            log.groups.insert(
                group.key.clone(),
                Group {
                    seq: group.seq,
                    task: group.task.clone(),
                    epoch: group.epoch,
                    created_ms: None,
                    rollouts: entry
                        .rollouts
                        .iter()
                        .map(|r| Rollout {
                            state: Some(r.state),
                            failures: r.failures,
                            outcome: r.outcome.clone(),
                        })
                        .collect(),
                },
            );
        }
        for entry in entries {
            let group = &entry.group;
            if LOGGED.contains(&group.state) {
                log.transition(
                    &group.key,
                    group.state,
                    group.reason.as_deref(),
                    group.staleness,
                    now_ms,
                    true,
                )?;
            }
        }
        Ok(log)
    }

    /// Where the log is.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The learner published `version`: what later lines name as the
    /// learner's version.
    pub fn learner_version(&mut self, version: u64) {
        self.learner_version = self.learner_version.max(version);
    }

    /// Appends one record of the run's own (not a group's), such as an
    /// audit, synced before this returns.
    ///
    /// # Errors
    /// The file cannot be written.
    pub fn note(&mut self, record: &Value) -> Result<(), LedgerError> {
        self.append(record)
    }

    fn append(&mut self, line: &Value) -> Result<(), LedgerError> {
        let mut text = line.to_string();
        text.push('\n');
        self.file
            .write_all(text.as_bytes())
            .and_then(|()| self.file.sync_data())
            .map_err(io(&self.path))
    }

    /// Records `change` in the log's picture of the run and appends the
    /// line of a transition it logs.
    ///
    /// # Errors
    /// The file cannot be written.
    pub fn record(&mut self, now_ms: u64, change: &Change) -> Result<(), LedgerError> {
        match change {
            Change::Create {
                key,
                seq,
                task,
                size,
                epoch,
            } => {
                self.groups.entry(key.clone()).or_insert_with(|| Group {
                    seq: *seq,
                    task: task.clone(),
                    epoch: *epoch,
                    created_ms: Some(now_ms),
                    rollouts: vec![Rollout::default(); usize::try_from(*size).unwrap_or(0)],
                });
            }
            Change::Start { key, index, .. } => {
                if let Some(r) = self.rollout(key, *index) {
                    r.state = Some(RolloutState::Running);
                }
            }
            Change::Finish {
                key,
                index,
                outcome,
                ..
            } => {
                if let Some(r) = self.rollout(key, *index) {
                    r.state = Some(RolloutState::Finished);
                    r.outcome = Some(outcome.clone());
                }
            }
            Change::Fail {
                key,
                index,
                abandon,
                ..
            } => {
                if let Some(r) = self.rollout(key, *index) {
                    r.failures += 1;
                    r.state = Some(if *abandon {
                        RolloutState::Abandoned
                    } else {
                        RolloutState::Pending
                    });
                }
            }
            Change::State {
                key,
                to,
                reason,
                staleness,
            } => {
                if LOGGED.contains(to) {
                    self.transition(key, *to, *reason, *staleness, now_ms, false)?;
                }
            }
        }
        Ok(())
    }

    fn rollout(&mut self, key: &GroupKey, index: u32) -> Option<&mut Rollout> {
        self.groups
            .get_mut(key)?
            .rollouts
            .get_mut(usize::try_from(index).ok()?)
    }

    fn transition(
        &mut self,
        key: &GroupKey,
        state: GroupState,
        reason: Option<&str>,
        staleness: Option<u64>,
        now_ms: u64,
        recovered: bool,
    ) -> Result<(), LedgerError> {
        if !self.written.insert((key.as_str().to_string(), state)) {
            return Ok(());
        }
        let line = match self.groups.get(key) {
            Some(group) => self.line(key, group, state, reason, staleness, now_ms, recovered),
            None => json!({"group": key, "state": state, "reason": reason, "at_ms": now_ms,
                           "recovered": recovered}),
        };
        self.append(&line)
    }

    #[allow(clippy::too_many_arguments)]
    fn line(
        &self,
        key: &GroupKey,
        group: &Group,
        state: GroupState,
        reason: Option<&str>,
        staleness: Option<u64>,
        now_ms: u64,
        recovered: bool,
    ) -> Value {
        let outcomes = || group.rollouts.iter().filter_map(|r| r.outcome.as_ref());
        let span = outcomes()
            .map(|o| (o.span.oldest(), o.span.newest()))
            .reduce(|(a, b), (c, d)| (a.min(c), b.max(d)));
        let rewards: Vec<Option<f64>> = outcomes().map(|o| o.reward).collect();
        let rollouts: Vec<Value> = group
            .rollouts
            .iter()
            .enumerate()
            .map(|(index, r)| {
                let o = r.outcome.as_ref();
                json!({
                    "index": index,
                    "state": r.state,
                    "failures": r.failures,
                    "reward": o.and_then(|o| o.reward),
                    "tokens": o.map(|o| o.tokens),
                    "span": o.map(|o| o.span),
                    "exact": o.map(|o| o.exact),
                    "truncation": o.and_then(|o| o.truncation),
                })
            })
            .collect();
        json!({
            "group": key,
            "seq": group.seq,
            "task": group.task,
            "epoch": group.epoch,
            "state": state,
            "reason": reason,
            "staleness": staleness,
            "learner_version": self.learner_version,
            "span": span.map(|(a, b)| [a, b]),
            "rewards": rewards,
            "created_ms": group.created_ms,
            "at_ms": now_ms,
            "recovered": recovered,
            "rollouts": rollouts,
        })
    }
}

/// A group ledger whose every transition is also logged
/// ([`GroupLog`]): made durable in the ledger first, then logged.
pub struct LoggedLedger<'a> {
    inner: &'a mut dyn GroupLedger,
    log: GroupLog,
}

impl<'a> LoggedLedger<'a> {
    /// `inner`, logged to `log`.
    #[must_use]
    pub fn new(inner: &'a mut dyn GroupLedger, log: GroupLog) -> Self {
        Self { inner, log }
    }

    /// The log, to note the run's own records in.
    pub fn log(&mut self) -> &mut GroupLog {
        &mut self.log
    }
}

impl GroupLedger for LoggedLedger<'_> {
    fn load(&mut self) -> Result<Vec<GroupEntry>, LedgerError> {
        self.inner.load()
    }

    fn apply(&mut self, now_ms: u64, change: &Change) -> Result<(), LedgerError> {
        self.inner.apply(now_ms, change)?;
        self.log.record(now_ms, change)
    }

    fn records(&mut self, key: &GroupKey) -> Result<Vec<(u32, String)>, LedgerError> {
        self.inner.records(key)
    }

    fn learner_version(&mut self, version: u64) {
        self.log.learner_version(version);
        self.inner.learner_version(version);
    }
}
