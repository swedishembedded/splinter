// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements self-improving coding agents whose every
// step is auditable, for its clients. If your team needs expertise in agent
// evaluation or locally operated coding agents, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The append-only event stream of a run.
//!
//! One JSON object per line in `events.jsonl`: a schema version, the run and
//! attempt it belongs to, an id that only grows, a timestamp, the id of the
//! event it is a child of, its type and its data. Every payload is redacted
//! before it is written. A payload larger than [`MAX_EVENT_DATA`] is stored
//! once, by content address, under `artifacts/`, and the event carries the
//! address, the size and a preview: the stream stays small and the evidence
//! stays complete. A gap in what the loop could observe is itself an event
//! (`events_dropped`), never a silent hole.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::{json, Value};
use splinter_sdk::vocabulary::clock::utc_now;
use splinter_sdk::vocabulary::digest::Digest;

use crate::redact::redact_value;

/// The version of the event line format.
pub const EVENT_SCHEMA: u32 = 1;

/// The most bytes of data an event carries inline.
pub const MAX_EVENT_DATA: usize = 4096;

/// How much of a stored payload its event previews.
const PREVIEW_CHARS: usize = 300;

/// The file the stream is appended to, inside a run's directory.
pub const EVENTS_FILE: &str = "events.jsonl";

/// The directory content-addressed artifacts are stored in.
pub const ARTIFACTS_DIR: &str = "artifacts";

/// One run's event stream, appended to from any thread.
pub struct Tracer {
    run: String,
    dir: PathBuf,
    state: Mutex<State>,
}

struct State {
    file: File,
    next_id: u64,
    attempt: u32,
}

/// One line of the stream, as written.
#[derive(Serialize)]
struct Line<'a> {
    v: u32,
    run: &'a str,
    attempt: u32,
    id: u64,
    ts: String,
    parent: Option<u64>,
    #[serde(rename = "type")]
    kind: &'a str,
    data: &'a Value,
}

impl Tracer {
    /// The stream of `run`, kept in `dir`; continues the ids of a stream
    /// that already exists there, so a resumed run's ids still only grow.
    pub fn open(dir: &Path, run: &str) -> Result<Self> {
        std::fs::create_dir_all(dir.join(ARTIFACTS_DIR))
            .with_context(|| format!("creating {}", dir.display()))?;
        let path = dir.join(EVENTS_FILE);
        let next_id = last_id(&path)?.map_or(1, |id| id + 1);
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("opening {}", path.display()))?;
        Ok(Self {
            run: run.to_string(),
            dir: dir.to_path_buf(),
            state: Mutex::new(State {
                file,
                next_id,
                attempt: 0,
            }),
        })
    }

    /// Events from now on belong to attempt `attempt` (0 is the run itself).
    pub fn set_attempt(&self, attempt: u32) {
        if let Ok(mut state) = self.state.lock() {
            state.attempt = attempt;
        }
    }

    /// Appends one event and returns its id.
    pub fn emit(&self, kind: &str, parent: Option<u64>, data: Value) -> Result<u64> {
        let data = self.bounded(&redact_value(&data))?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("the trace lock was poisoned"))?;
        let id = state.next_id;
        let line = Line {
            v: EVENT_SCHEMA,
            run: &self.run,
            attempt: state.attempt,
            id,
            ts: utc_now(),
            parent,
            kind,
            data: &data,
        };
        let mut text = serde_json::to_string(&line).context("encoding a trace event")?;
        text.push('\n');
        // One write of the whole line: an interrupted process leaves a
        // complete line or none.
        state
            .file
            .write_all(text.as_bytes())
            .context("appending to the trace")?;
        state.file.flush().context("flushing the trace")?;
        state.next_id += 1;
        Ok(id)
    }

    /// `data` itself when it is small enough to sit in an event, else a
    /// reference to the artifact it was stored as.
    fn bounded(&self, data: &Value) -> Result<Value> {
        let text = serde_json::to_string(data).context("encoding event data")?;
        if text.len() <= MAX_EVENT_DATA {
            return Ok(data.clone());
        }
        let digest = self.store_artifact(text.as_bytes())?;
        Ok(json!({
            "artifact": digest,
            "bytes": text.len(),
            "preview": text.chars().take(PREVIEW_CHARS).collect::<String>(),
        }))
    }

    /// Stores `bytes` under their content address, once, and returns the
    /// address. The caller redacts what it hands in.
    pub fn store_artifact(&self, bytes: &[u8]) -> Result<String> {
        let digest = Digest::sha256_of(bytes);
        let path = self.dir.join(ARTIFACTS_DIR).join(digest.hex());
        if !path.exists() {
            crate::store::write_atomic(&path, bytes)?;
        }
        Ok(digest.to_string())
    }
}

/// The id of the last complete line of the stream at `path`, if any.
fn last_id(path: &Path) -> Result<Option<u64>> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Ok(None);
    };
    Ok(text.lines().rev().find_map(|line| {
        serde_json::from_str::<Value>(line)
            .ok()?
            .get("id")?
            .as_u64()
    }))
}

/// Every event of the stream in `dir`, in order; a line that does not parse
/// is an error, since a damaged stream is evidence of a crash worth naming.
pub fn read_events(dir: &Path) -> Result<Vec<Value>> {
    let path = dir.join(EVENTS_FILE);
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    text.lines()
        .enumerate()
        .map(|(n, line)| {
            serde_json::from_str(line)
                .with_context(|| format!("{} line {} is not an event", path.display(), n + 1))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tracer(dir: &Path) -> Tracer {
        Tracer::open(dir, "run-1").unwrap()
    }

    #[test]
    fn ids_only_grow_and_survive_a_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let first = tracer(dir.path());
        let a = first.emit("a", None, json!({})).unwrap();
        let b = first.emit("b", Some(a), json!({})).unwrap();
        drop(first);
        let c = tracer(dir.path()).emit("c", None, json!({})).unwrap();
        assert!(a < b && b < c, "{a} {b} {c}");
        let events = read_events(dir.path()).unwrap();
        assert_eq!(events.len(), 3);
        assert_eq!(events[1]["parent"], a);
        assert_eq!(events[0]["v"], EVENT_SCHEMA);
    }

    #[test]
    fn a_large_payload_is_stored_once_and_referenced() {
        let dir = tempfile::tempdir().unwrap();
        let t = tracer(dir.path());
        let big = "x".repeat(MAX_EVENT_DATA * 3);
        t.emit("big", None, json!({"output": big})).unwrap();
        t.emit("big", None, json!({"output": big})).unwrap();
        let events = read_events(dir.path()).unwrap();
        let data = &events[0]["data"];
        assert!(data["bytes"].as_u64().unwrap() > MAX_EVENT_DATA as u64);
        let address = data["artifact"].as_str().unwrap();
        assert_eq!(events[1]["data"]["artifact"], address, "deduplicated");
        let stored = std::fs::read_dir(dir.path().join(ARTIFACTS_DIR))
            .unwrap()
            .count();
        assert_eq!(stored, 1);
        let line_len = std::fs::read_to_string(dir.path().join(EVENTS_FILE))
            .unwrap()
            .lines()
            .map(str::len)
            .max()
            .unwrap();
        assert!(line_len < MAX_EVENT_DATA, "the stream stays small");
    }

    #[test]
    fn secrets_never_reach_the_stream_or_an_artifact() {
        let dir = tempfile::tempdir().unwrap();
        let t = tracer(dir.path());
        let long = format!("{} api_key=hunter2hunter2", "y".repeat(MAX_EVENT_DATA));
        t.emit("note", None, json!({"text": "token=abcdef123456"}))
            .unwrap();
        t.emit("note", None, json!({"text": long})).unwrap();
        let mut all = std::fs::read_to_string(dir.path().join(EVENTS_FILE)).unwrap();
        for entry in std::fs::read_dir(dir.path().join(ARTIFACTS_DIR)).unwrap() {
            all.push_str(&std::fs::read_to_string(entry.unwrap().path()).unwrap());
        }
        assert!(!all.contains("abcdef123456") && !all.contains("hunter2hunter2"));
    }

    #[test]
    fn the_attempt_is_stamped_on_each_event() {
        let dir = tempfile::tempdir().unwrap();
        let t = tracer(dir.path());
        t.emit("run", None, json!({})).unwrap();
        t.set_attempt(2);
        t.emit("attempt", None, json!({})).unwrap();
        let events = read_events(dir.path()).unwrap();
        assert_eq!(events[0]["attempt"], 0);
        assert_eq!(events[1]["attempt"], 2);
    }
}
