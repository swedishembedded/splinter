// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements tamper-evident hand-over of agent benchmark
// results into training pipelines, for its clients. If your team needs
// expertise in turning benchmark runs into verified training data, you can
// procure our services by sending an email to info@swedishembedded.com.

//! A trial artifact exported by the benchmark harness, read and verified.
//!
//! An export directory holds `index.jsonl` (one row per trial, naming the
//! digest of the trial's manifest) and one immutable directory per trial.
//! A trial is read only when every claim the artifact makes about itself
//! holds: the manifest is the one the index names, it agrees with the index
//! row, and every file in the directory is exactly one the manifest lists,
//! with the digest it lists. Anything else is refused with the reason, and
//! nothing of a refused trial reaches the store.
//!
//! A trial that ran a harness (several conversations, each answered by a
//! model role) has no steps in its top-level trajectory; its record is the
//! `conversations/NNNN-<name>.json` files, each listed in the manifest with
//! its digest like every other file. Such a trial is read as one [`Unit`] per
//! conversation; any other trial is one unit, its trajectory.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufReader};
use std::path::{Component, Path};

use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest as _, Sha256};
use splinter_core::harness::{HarnessConversation, RecordedCalls, RoleSource};

use super::score::ScoreRecord;
use request_log::RequestLog;

mod request_log;

/// The manifest schema this reader understands.
pub const MANIFEST_SCHEMA: &str = "bench.trial-artifact/1";

/// The file name of the manifest in a trial directory.
const MANIFEST: &str = "manifest.json";

/// The file that is the trial's record.
const TRAJECTORY: &str = "trajectory.atif.json";

/// The file that holds every request the agent sent its model.
const REQUESTS: &str = "model-requests.jsonl";

/// The file that holds the verifier's own statement of its verdict.
const VERIFICATION: &str = "verification.json";

/// The directory of a harness trial's per-conversation trajectories.
const CONVERSATIONS: &str = "conversations";

/// The directory, beside a trajectory, of the pictures it shows.
const IMAGES: &str = "images";

/// One row of an export's `index.jsonl`.
#[derive(Clone, Debug, Deserialize)]
pub struct IndexRow {
    /// The trial's name.
    pub trial: String,
    /// The benchmark task's id.
    pub task: String,
    /// The reward the grade gave.
    pub reward: f64,
    /// `pass` or `fail`.
    pub verdict: String,
    /// The split the task belongs to, when the export states it.
    #[serde(default)]
    pub set: Option<String>,
    /// Whether the trial ran a released harness package, whose method is
    /// not the learner's to train on; absent in exports made before it.
    #[serde(default)]
    pub confidential: Option<bool>,
    /// The sha256 of the trial's `manifest.json`.
    pub manifest_sha256: String,
    /// The trial directory relative to the export.
    pub path: String,
    /// The capability ids the examiner's taxonomy labels the task with;
    /// `None` in exports made before the examiner named them.
    #[serde(default)]
    pub capabilities: Option<Vec<String>>,
}

/// The task a trial attempted, as the manifest names it.
#[derive(Clone, Debug, Deserialize)]
pub struct ManifestTask {
    /// The task's id.
    pub id: String,
    /// The task's version.
    pub version: String,
    /// The digest of the task's content.
    pub digest: String,
    /// The capability the task exercises.
    #[serde(default)]
    pub dimension: Option<String>,
    /// The split the task belongs to, when the export states it.
    #[serde(default)]
    pub set: Option<String>,
    /// The capability ids the examiner's taxonomy labels the task with;
    /// `None` in exports made before the examiner named them.
    #[serde(default)]
    pub capabilities: Option<Vec<String>>,
}

/// The agent that ran the trial.
#[derive(Clone, Debug, Deserialize)]
pub struct ManifestAgent {
    /// The agent's name.
    pub name: String,
    /// The agent's version.
    pub version: String,
    /// The digest of the agent build.
    #[serde(default)]
    pub sven_build_sha256: Option<String>,
}

/// The verifier that graded the trial.
#[derive(Clone, Debug, Deserialize)]
pub struct ManifestVerifier {
    /// The digest of the verifier.
    pub digest: String,
    /// How the checks combine into a reward.
    #[serde(default)]
    pub scoring: Option<String>,
}

/// The grade the verifier gave.
#[derive(Clone, Debug, Deserialize)]
pub struct ManifestGrade {
    /// The reward.
    pub reward: f64,
    /// `pass` or `fail`.
    pub verdict: String,
    /// The per-check metrics.
    #[serde(default)]
    pub metrics: Option<Value>,
}

/// The schema of the request log this reader reads (sven's
/// `--record-requests`): a `request` line with the whole conversation sent and
/// a `response` line with what came back, per model request.
pub const REQUEST_LOG_SCHEMA: &str = "sven.request-log/1";

/// The manifest's declaration of `model-requests.jsonl`.
#[derive(Clone, Debug, Deserialize)]
pub struct ManifestRequests {
    /// The log's format; [`REQUEST_LOG_SCHEMA`] is the one read.
    pub schema: String,
}

/// One model the trial's agent used.
#[derive(Clone, Debug, Deserialize)]
pub struct ManifestModel {
    /// The model's name as the server knew it.
    pub model: String,
}

/// The fields of `manifest.json` this reader uses; the rest are ignored.
#[derive(Clone, Debug, Deserialize)]
pub struct Manifest {
    /// Must be [`MANIFEST_SCHEMA`].
    pub schema: String,
    /// The trial's name.
    pub trial: String,
    /// The task attempted.
    pub task: ManifestTask,
    /// The repository snapshot the trial started from.
    #[serde(default)]
    pub snapshot: Option<String>,
    /// The container image and environment.
    #[serde(default)]
    pub image: Option<Value>,
    /// The agent.
    pub agent: ManifestAgent,
    /// The models, by role.
    #[serde(default)]
    pub models: BTreeMap<String, ManifestModel>,
    /// The verifier.
    pub verifier: ManifestVerifier,
    /// The grade.
    pub grade: ManifestGrade,
    /// Tokens the trial used.
    #[serde(default)]
    pub tokens: Option<Value>,
    /// Wall-clock seconds the trial took.
    #[serde(default)]
    pub wall_seconds: Option<f64>,
    /// The examiner's account of the request log: its schema, and how many
    /// requests it holds and how many were answered; absent in exports made
    /// before it declared one.
    #[serde(default)]
    pub requests: Option<ManifestRequests>,
    /// Whether the trial ran a released harness package.
    #[serde(default)]
    pub confidential: bool,
    /// The examiner's score record; absent in exports made before it.
    #[serde(default)]
    pub score: Option<ScoreRecord>,
    /// The sha256 of every file of the trial directory but the manifest.
    pub files: BTreeMap<String, String>,
}

/// What the model was given before the first turn, read from the first
/// request the agent sent it: the system prompt and the tool schemas, which
/// an ATIF trajectory of this agent does not carry.
#[derive(Clone, Debug, PartialEq)]
pub struct AgentContext {
    /// The system prompt of the first request.
    pub system: String,
    /// The tool schemas of the first request, in the order they were sent.
    pub tools: Vec<Value>,
    /// The sha256 of the `model-requests.jsonl` they were read from.
    pub source_sha256: String,
}

/// What a unit of a trial is.
#[derive(Debug)]
pub enum UnitKind {
    /// The trajectory of a trial that ran a single agent. The trajectory does
    /// not record its system prompt and tools; the context of the trial's
    /// first request does.
    Single(AgentContext),
    /// One conversation of a harness trial, whose trajectory records its
    /// system prompt and tools itself.
    Conversation(HarnessConversation),
}

/// One trajectory of a trial that becomes one experience.
#[derive(Debug)]
pub struct Unit {
    /// What it is.
    pub kind: UnitKind,
    /// The trajectory, parsed from the bytes that were verified.
    pub trajectory: atif::Trajectory,
    /// The sha256 of the trajectory file as exported.
    pub trajectory_sha256: String,
    /// The requests its model was sent and their responses, when the request
    /// log divides among its agent steps (a harness trial's: among its
    /// conversations) and holds every response.
    pub calls: Option<RecordedCalls>,
}

impl Unit {
    /// The harness conversation it is, if it is one.
    #[must_use]
    pub fn conversation(&self) -> Option<&HarnessConversation> {
        match &self.kind {
            UnitKind::Conversation(c) => Some(c),
            UnitKind::Single(_) => None,
        }
    }
}

/// A trial whose every file matched its digest.
#[derive(Debug)]
pub struct Artifact {
    /// The index row it was read for.
    pub row: IndexRow,
    /// The verified manifest.
    pub manifest: Manifest,
    /// What becomes experience: one unit, or one per conversation.
    pub units: Vec<Unit>,
    /// Why the verifier decided as it did, in its own words, when it says
    /// (`verification.json`'s `reason`): the one thing of what the verifier
    /// knows that a trial artifact carries.
    pub verifier_reason: Option<String>,
}

/// The files of a trial that are read, not only hashed.
struct Read {
    trajectory: atif::Trajectory,
    requests: Vec<u8>,
    conversations: Vec<(String, atif::Trajectory, String)>,
    verifier_reason: Option<String>,
}

/// Reads the rows of `directory`'s `index.jsonl`; a row that does not parse
/// is returned as the reason it was refused, with its line number.
pub fn read_index(directory: &Path) -> Result<Vec<Result<IndexRow, String>>, String> {
    let path = directory.join("index.jsonl");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(text
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(n, line)| {
            serde_json::from_str::<IndexRow>(line)
                .map_err(|e| format!("index.jsonl line {}: {e}", n + 1))
        })
        .collect())
}

fn hex(digest: impl AsRef<[u8]>) -> String {
    digest.as_ref().iter().map(|b| format!("{b:02x}")).collect()
}

/// The sha256 of `bytes`, in lower-case hex.
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex(Sha256::digest(bytes))
}

/// The sha256 of the file at `path`, read without holding it in memory.
pub(crate) fn sha256_file(path: &Path) -> io::Result<String> {
    let mut hasher = Sha256::new();
    io::copy(&mut BufReader::new(File::open(path)?), &mut hasher)?;
    Ok(hex(hasher.finalize()))
}

/// Whether `name` is one plain path component: no separator, no `..`.
fn is_plain_name(name: &str) -> bool {
    let mut parts = Path::new(name).components();
    matches!(
        (parts.next(), parts.next()),
        (Some(Component::Normal(_)), None)
    )
}

/// Reads and verifies the trial `row` names under `export`.
///
/// # Errors
///
/// The reason the artifact is refused: a path that leaves the export, a
/// missing or altered file, a file the manifest does not list, a manifest
/// that is not the indexed one or disagrees with the row, an unknown
/// schema, or a trajectory that is not ATIF.
pub fn read_trial(export: &Path, row: &IndexRow) -> Result<Artifact, String> {
    if !is_plain_name(&row.path) {
        return Err(format!(
            "the index path {:?} is not a directory directly under the export",
            row.path
        ));
    }
    let dir = export.join(&row.path);
    if !dir.is_dir() {
        return Err(format!(
            "{} is not a directory (partial export)",
            dir.display()
        ));
    }
    let manifest_bytes =
        std::fs::read(dir.join(MANIFEST)).map_err(|e| format!("{MANIFEST}: {e}"))?;
    let found = sha256_hex(&manifest_bytes);
    if !found.eq_ignore_ascii_case(&row.manifest_sha256) {
        return Err(format!(
            "{MANIFEST} is altered: the index names sha256 {}, the file is {found}",
            row.manifest_sha256
        ));
    }
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|e| format!("{MANIFEST} is not a trial manifest: {e}"))?;
    agree(row, &manifest)?;
    let read = verify_files(&dir, &manifest)?;
    let verifier_reason = read.verifier_reason.clone();
    let units = units_of(&manifest, read)?;
    Ok(Artifact {
        row: row.clone(),
        manifest,
        units,
        verifier_reason,
    })
}

/// The units of a verified trial: its trajectory, or its conversations.
fn units_of(manifest: &Manifest, read: Read) -> Result<Vec<Unit>, String> {
    let requests = &manifest.files[REQUESTS];
    if let Some(declared) = manifest
        .requests
        .as_ref()
        .filter(|r| r.schema != REQUEST_LOG_SCHEMA)
    {
        return Err(format!(
            "{REQUESTS} is declared {:?}, and only {REQUEST_LOG_SCHEMA:?} is read",
            declared.schema
        ));
    }
    if read.conversations.is_empty() {
        let trajectory_sha256 = manifest.files[TRAJECTORY].to_ascii_lowercase();
        let context = first_request(&read.requests, requests)?;
        // The requests the agent's model was sent, when the log holds one,
        // answered, per agent step; a request that showed a picture keeps
        // the trajectory's own record of it.
        let log = RequestLog::parse(&read.requests)?;
        let calls = log
            .attribute(&[&read.trajectory])
            .and_then(|a| a.into_iter().next())
            .filter(|a| !a.images)
            .and_then(|a| log.calls_of(&a))
            .map(|calls| RecordedCalls::record(requests, &calls));
        return Ok(vec![Unit {
            kind: UnitKind::Single(context),
            trajectory: read.trajectory,
            trajectory_sha256,
            calls,
        }]);
    }
    let named = read
        .trajectory
        .extra
        .as_ref()
        .and_then(|e| e.pointer("/harness/conversations"))
        .and_then(Value::as_array);
    let found: Vec<&str> = read.conversations.iter().map(|c| c.0.as_str()).collect();
    if let Some(named) = named {
        let named: Vec<&str> = named.iter().filter_map(Value::as_str).collect();
        if named != found {
            return Err(format!(
                "the trial's trajectory lists conversations {named:?}, the manifest holds \
                 {found:?} (partial artifact)"
            ));
        }
    }
    let log = RequestLog::parse(&read.requests)?;
    let trajectories: Vec<&atif::Trajectory> = read.conversations.iter().map(|c| &c.1).collect();
    let attributed = log.attribute(&trajectories);
    Ok(read
        .conversations
        .into_iter()
        .enumerate()
        .map(|(n, (name, trajectory, sha))| {
            let exported = trajectory
                .extra
                .as_ref()
                .and_then(|e| e.pointer("/harness/role"))
                .and_then(Value::as_str)
                .map(str::to_string);
            let (role, role_source, logged) = match (exported, &attributed) {
                (Some(role), _) => (
                    Some(role),
                    RoleSource::Export,
                    attributed.as_ref().map(|a| a[n].clone()),
                ),
                (None, Some(a)) => (a[n].role.clone(), a[n].source(), Some(a[n].clone())),
                (None, None) => (None, RoleSource::Unknown, None),
            };
            let calls = logged
                .as_ref()
                .and_then(|a| log.calls_of(a))
                .map(|calls| RecordedCalls::record(requests, &calls));
            Unit {
                kind: UnitKind::Conversation(HarnessConversation {
                    conversation: name,
                    role,
                    role_source,
                    // Without an attribution any conversation may have been
                    // shown the images the log holds.
                    shown_images: logged.map_or(log.any_image, |a| a.images),
                }),
                trajectory,
                trajectory_sha256: sha,
                calls,
            }
        })
        .collect())
}

/// The manifest and the index row must say the same about the trial.
fn agree(row: &IndexRow, manifest: &Manifest) -> Result<(), String> {
    if manifest.schema != MANIFEST_SCHEMA {
        return Err(format!(
            "manifest schema {:?} is not {MANIFEST_SCHEMA:?}",
            manifest.schema
        ));
    }
    let differs = |what: &str, index: &dyn std::fmt::Debug, manifest: &dyn std::fmt::Debug| {
        Err(format!(
            "the index and the manifest disagree on the {what}: {index:?} against {manifest:?}"
        ))
    };
    if manifest.trial != row.trial {
        return differs("trial", &row.trial, &manifest.trial);
    }
    if manifest.task.id != row.task {
        return differs("task", &row.task, &manifest.task.id);
    }
    if manifest.grade.verdict != row.verdict {
        return differs("verdict", &row.verdict, &manifest.grade.verdict);
    }
    if (manifest.grade.reward - row.reward).abs() > f64::EPSILON {
        return differs("reward", &row.reward, &manifest.grade.reward);
    }
    if manifest.task.set != row.set {
        return differs("set", &row.set, &manifest.task.set);
    }
    if row.confidential.is_some_and(|c| c != manifest.confidential) {
        return differs("confidentiality", &row.confidential, &manifest.confidential);
    }
    manifest
        .score
        .as_ref()
        .map_or(Ok(()), |score| score.agrees_with(manifest))
}

/// The names of the files in `dir`: a harness trial's conversation files as
/// `conversations/<file>`, and the pictures beside a trajectory as
/// `images/<file>` and `conversations/images/<file>`. Any other directory is a
/// name the manifest cannot list, hence refused by the caller.
fn present_files(dir: &Path) -> Result<Vec<String>, String> {
    let unreadable = |path: &Path, e: io::Error| format!("{}: {e}", path.display());
    let entries = |path: &Path| -> Result<Vec<(String, bool)>, String> {
        let mut found = Vec::new();
        for entry in std::fs::read_dir(path).map_err(|e| unreadable(path, e))? {
            let entry = entry.map_err(|e| unreadable(path, e))?;
            found.push((
                entry.file_name().to_string_lossy().into_owned(),
                entry.path().is_dir(),
            ));
        }
        Ok(found)
    };
    let mut present = Vec::new();
    for (name, is_dir) in entries(dir)? {
        match (name.as_str(), is_dir) {
            (CONVERSATIONS, true) => {
                for (file, inner_dir) in entries(&dir.join(CONVERSATIONS))? {
                    if file == IMAGES && inner_dir {
                        for (picture, _) in entries(&dir.join(CONVERSATIONS).join(IMAGES))? {
                            present.push(format!("{CONVERSATIONS}/{IMAGES}/{picture}"));
                        }
                    } else {
                        present.push(format!("{CONVERSATIONS}/{file}"));
                    }
                }
            }
            (IMAGES, true) => {
                for (picture, _) in entries(&dir.join(IMAGES))? {
                    present.push(format!("{IMAGES}/{picture}"));
                }
            }
            _ => present.push(name),
        }
    }
    Ok(present)
}

/// Whether `name` is a file a manifest may list: one plain file name, one
/// directly under `conversations/`, or a picture directly under `images/` or
/// `conversations/images/`.
fn is_listable(name: &str) -> bool {
    fn under<'a>(prefix: &str, name: &'a str) -> Option<&'a str> {
        name.strip_prefix(prefix)?.strip_prefix('/')
    }
    let file = under(CONVERSATIONS, name).unwrap_or(name);
    let file = under(IMAGES, file).unwrap_or(file);
    is_plain_name(file)
}

/// Whether manifest key `name` is a conversation's trajectory (not a picture).
fn is_conversation(name: &str) -> bool {
    name.strip_prefix(CONVERSATIONS)
        .and_then(|rest| rest.strip_prefix('/'))
        .is_some_and(is_plain_name)
}

/// The name of the conversation a manifest key `conversations/NNNN-<name>.json`
/// holds, when the trajectory does not state it.
fn conversation_file_name(key: &str) -> Option<&str> {
    let stem = key
        .strip_prefix(CONVERSATIONS)?
        .strip_prefix('/')?
        .strip_suffix(".json")?;
    let (index, name) = stem.split_once('-')?;
    (index.bytes().all(|b| b.is_ascii_digit()) && !name.is_empty()).then_some(name)
}

/// Checks every file of `dir` against `manifest.files`, both ways, and
/// returns the trajectories parsed from the verified bytes.
fn verify_files(dir: &Path, manifest: &Manifest) -> Result<Read, String> {
    for name in &present_files(dir)? {
        if name != MANIFEST && !manifest.files.contains_key(name) {
            return Err(format!(
                "{name} is in the directory but not in the manifest"
            ));
        }
    }
    let mut trajectory: Option<Vec<u8>> = None;
    let mut requests: Option<Vec<u8>> = None;
    let mut conversations: Vec<(String, atif::Trajectory, String)> = Vec::new();
    let mut verifier_reason: Option<String> = None;
    for (name, expected) in &manifest.files {
        if !is_listable(name) {
            return Err(format!(
                "the manifest lists {name:?}, which is not a plain file name"
            ));
        }
        let path = dir.join(name);
        let unreadable = |e: io::Error| format!("{name}: {e} (partial artifact)");
        // The trajectories and the requests are parsed from the very bytes
        // that were hashed.
        let parsed =
            name == TRAJECTORY || name == REQUESTS || name == VERIFICATION || is_conversation(name);
        let found = if parsed {
            let bytes = std::fs::read(&path).map_err(unreadable)?;
            let found = sha256_hex(&bytes);
            if found.eq_ignore_ascii_case(expected) {
                if name == TRAJECTORY {
                    trajectory = Some(bytes);
                } else if name == REQUESTS {
                    requests = Some(bytes);
                } else if name == VERIFICATION {
                    let statement: Value = serde_json::from_slice(&bytes)
                        .map_err(|e| format!("{VERIFICATION} is not JSON: {e}"))?;
                    verifier_reason = statement
                        .get("reason")
                        .and_then(Value::as_str)
                        .filter(|r| !r.trim().is_empty())
                        .map(str::to_string);
                } else {
                    let parsed: atif::Trajectory = serde_json::from_slice(&bytes)
                        .map_err(|e| format!("{name} is not ATIF: {e}"))?;
                    let named = parsed
                        .extra
                        .as_ref()
                        .and_then(|e| e.pointer("/harness/conversation"))
                        .and_then(Value::as_str)
                        .or_else(|| conversation_file_name(name))
                        .ok_or_else(|| format!("{name} does not say which conversation it is"))?
                        .to_string();
                    conversations.push((named, parsed, found.clone()));
                }
            }
            found
        } else {
            sha256_file(&path).map_err(unreadable)?
        };
        if !found.eq_ignore_ascii_case(expected) {
            return Err(format!(
                "{name} is altered: the manifest names sha256 {expected}, the file is {found}"
            ));
        }
    }
    let bytes = trajectory.ok_or_else(|| format!("the manifest lists no {TRAJECTORY}"))?;
    let trajectory =
        serde_json::from_slice(&bytes).map_err(|e| format!("{TRAJECTORY} is not ATIF: {e}"))?;
    let requests = requests.ok_or_else(|| {
        format!(
            "the manifest lists no {REQUESTS}: the system prompt and tools the model ran under \
             are unknown, and a record without them would not match what it saw"
        )
    })?;
    Ok(Read {
        trajectory,
        requests,
        conversations,
        verifier_reason,
    })
}

/// The context of the first request in the `model-requests.jsonl` `bytes`.
pub(crate) fn first_request(bytes: &[u8], sha256: &str) -> Result<AgentContext, String> {
    let line = bytes
        .split(|b| *b == b'\n')
        .find(|l| !l.is_empty())
        .ok_or_else(|| format!("{REQUESTS} is empty"))?;
    let row: Value = serde_json::from_slice(line).map_err(|e| format!("{REQUESTS} line 1: {e}"))?;
    let request = row
        .get("request")
        .ok_or_else(|| format!("{REQUESTS} line 1 holds no request"))?;
    let system = request
        .pointer("/messages/0")
        .filter(|m| m.get("role").and_then(Value::as_str) == Some("system"))
        .and_then(|m| m.get("content"))
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{REQUESTS}: the first request does not open with a system prompt"))?
        .to_string();
    let tools = request
        .get("tools")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    Ok(AgentContext {
        system,
        tools,
        source_sha256: sha256.to_ascii_lowercase(),
    })
}
