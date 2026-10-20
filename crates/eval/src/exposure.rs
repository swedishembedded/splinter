// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements audits that prove an agent never saw the
// material it is graded against, for its clients. If your team needs
// expertise in contamination-free evaluation of tool-using agents, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The exposure audit: did the solver see what only the grader may see?
//!
//! A task keeps material from its solver (hidden tests, a reference answer,
//! the location of the stored state). Where nothing physically stops a tool
//! from reaching it, the trajectory is the evidence. [`audit_exposure`]
//! reads what the solver was shown and what it sent to its tools, and fails
//! the experience when
//!
//! * a tool call or a tool output **names a hidden path** (a call whose own
//!   result is only a refusal or "no such file" read nothing and is recorded
//!   as a probe, `hidden_path_absent`, see below);
//! * a tool output (or any other text shown to the solver) contains a
//!   **shingle** of hidden content: a run of [`SHINGLE_TOKENS`] consecutive
//!   tokens that also occurs in hidden material and in no text the solver
//!   is allowed to see ([`HiddenMaterial::public_contents`]);
//! * a tool call or output holds one of the **forbidden strings**, such as
//!   the path of the state root;
//! * for a workspace task, a tool call **reads** an absolute path outside the
//!   episode's tree and that call's own output shows something came back
//!   (`outside_workspace`), or **searches the whole filesystem** from `/`
//!   (`outside_workspace_search`, whatever it returned).
//!
//! A read that was refused by the sandbox (`outside_workspace_denied`) or
//! found nothing there (`outside_workspace_absent`, which includes an output
//! that is only a failure report), a write to a scratch file
//! (`outside_workspace_write`) and a change of directory
//! (`outside_workspace_cd`) are recorded **probes**, not exposure: they are
//! listed under `outside_probes` in the evidence (of an abstention as well as
//! of a fail) so a caller can count and filter by it, and they do not fail
//! the audit. [`crate::exposure_paths`] says how a call's paths and results
//! are read.
//!
//! A clean trajectory is an **abstention**, not a pass: not having seen the
//! material does not make an answer right. The audit is a
//! [`Strength::Constraint`] verdict, so a fail outranks whatever a verifier
//! said about the answer. It works on any task with privileged material and
//! tools; nothing in it is specific to code. The evidence names the steps,
//! where in them and which listed item matched, never the matched text.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

use atif::{ContentSegment, MessageBody, StepOrigin, TraceStep};
use serde_json::json;
use splinter_core::annotation::{Producer, Strength};
use splinter_core::experience::{Experience, Task};

pub use crate::exposure_paths::WORKSPACE_TREE_KEY;
use crate::exposure_paths::{outputs_by_call, outside_workspace, returned_only_a_failure};
use crate::verifiers::{Finding, Verifier, VerifyError};

/// The producer name an exposure audit's annotations carry.
pub const EXPOSURE_PRODUCER: &str = "splinter-lab/exposure";

/// The audit's version: 2 reads parallel results by call id and tells a
/// read of a path from a write, a directory change and a failure report.
pub const VERSION: &str = "2";

/// How many consecutive tokens make a shingle. Long enough that boilerplate
/// every file shares (imports, a header) rarely fills one; short enough that
/// a few lines of a test quoted in a traceback do.
pub const SHINGLE_TOKENS: usize = 12;

/// Material that must not reach the solver.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HiddenMaterial {
    /// Paths (or any names) that must not appear in a call or an output.
    pub paths: Vec<String>,
    /// Texts whose content must not appear in what the solver was shown: the
    /// hidden files' contents. The task's own privileged items are audited
    /// in addition.
    pub contents: Vec<String>,
    /// Texts the solver may legitimately see (the visible tree). A shingle
    /// that also occurs here is shared boilerplate, not exposure.
    pub public_contents: Vec<String>,
    /// Strings that must not appear anywhere in a call or an output, such as
    /// the path of the state root.
    pub forbidden: Vec<String>,
    /// Whether a tool call that names an absolute path outside the episode's
    /// own tree and the system's directories is exposure
    /// (`outside_workspace`): for a workspace task, whose solver is confined
    /// to its tree and so has no business looking elsewhere.
    pub confine_to_workspace: bool,
}

/// Where in the trajectory a match was found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Place {
    ToolCall,
    ToolOutput,
    Shown,
}

impl Place {
    fn label(self) -> &'static str {
        match self {
            Self::ToolCall => "tool_call",
            Self::ToolOutput => "tool_output",
            Self::Shown => "message",
        }
    }
}

/// One text the solver was shown or sent, with where it was.
struct Seen {
    step: u64,
    place: Place,
    text: String,
    /// For a tool call, the output that call itself returned.
    result: Option<String>,
}

/// What a trajectory shows the solver or sends to its tools: each agent
/// step's tool calls, every observation, and the messages of the other
/// parties. What the agent itself says is not exposure.
fn seen(exp: &Experience) -> Vec<Seen> {
    let mut texts = Vec::new();
    for step in &exp.trajectory.steps {
        let outputs = outputs_by_call(step);
        for (call, output) in step.tool_calls.iter().flatten().zip(outputs) {
            texts.push(Seen {
                step: step.step_id,
                place: Place::ToolCall,
                text: format!("{} {}", call.function_name, call.arguments),
                result: Some(output),
            });
        }
        for result in step.observation.iter().flat_map(|o| &o.results) {
            if let Some(content) = &result.content {
                texts.push(Seen {
                    step: step.step_id,
                    place: Place::ToolOutput,
                    text: text_of(content),
                    result: None,
                });
            }
        }
        if step.source != StepOrigin::Agent {
            texts.push(shown(step));
        }
    }
    texts
}

fn shown(step: &TraceStep) -> Seen {
    Seen {
        step: step.step_id,
        place: Place::Shown,
        text: text_of(&step.message),
        result: None,
    }
}

fn text_of(body: &MessageBody) -> String {
    match body {
        MessageBody::Text(text) => text.clone(),
        MessageBody::Segments(segments) => segments
            .iter()
            .filter_map(|segment| match segment {
                ContentSegment::Text { text } => Some(text.as_str()),
                ContentSegment::Image { .. } => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

/// The tokens of `text` that identify content: its words, without those
/// that are only digits (a line-numbered listing numbers the lines of a file
/// it shows, which must not break a shingle).
fn tokens(text: &str) -> Vec<&str> {
    text.split_whitespace()
        .filter(|t| !t.bytes().all(|b| b.is_ascii_digit()))
        .collect()
}

/// The fewest tokens a text needs to be worth matching.
const MIN_TOKENS: usize = 4;

/// The shingles of one hidden text: their width in tokens (a text shorter
/// than a shingle is one shingle of its own length) and their hashes.
struct Secret {
    width: usize,
    hashes: HashSet<u64>,
    /// The hashes, at the same width, of the texts the solver may see.
    allowed: HashSet<u64>,
}

impl Secret {
    /// `None` when `text` is too short to mean anything.
    fn of(text: &str, public: &[Vec<&str>]) -> Option<Self> {
        let tokens = tokens(text);
        if tokens.len() < MIN_TOKENS {
            return None;
        }
        let width = SHINGLE_TOKENS.min(tokens.len());
        Some(Self {
            width,
            hashes: window_hashes(&tokens, width).collect(),
            allowed: public
                .iter()
                .flat_map(|t| window_hashes(t, width))
                .collect(),
        })
    }

    /// Whether `text` holds a shingle of this secret that the solver may
    /// not have seen elsewhere.
    fn leaks_into(&self, text: &[&str]) -> bool {
        window_hashes(text, self.width)
            .any(|h| self.hashes.contains(&h) && !self.allowed.contains(&h))
    }
}

fn window_hashes<'a>(tokens: &'a [&str], width: usize) -> impl Iterator<Item = u64> + 'a {
    tokens.windows(width).map(|window| {
        let mut hasher = DefaultHasher::new();
        window.hash(&mut hasher);
        hasher.finish()
    })
}

/// One match: where, and which listed item (never its text).
fn finding(seen: &Seen, kind: &str, item: usize) -> serde_json::Value {
    json!({ "step": seen.step, "in": seen.place.label(), "kind": kind, "item": item })
}

/// Audits `exp`, an experience of `task`, for exposure of `hidden` and of
/// the task's own privileged items. Fails with the matches as evidence, or
/// abstains when the trajectory is clean (or is not an experience of
/// `task`, which it cannot judge).
#[must_use]
pub fn audit_exposure(task: &Task, exp: &Experience, hidden: &HiddenMaterial) -> Finding {
    if task.task != exp.task {
        return Finding::abstain(
            "the experience is not of this task",
            json!({ "task": task.task.id, "experience": exp.task.id }),
        );
    }
    let texts = seen(exp);
    let public: Vec<Vec<&str>> = hidden.public_contents.iter().map(|c| tokens(c)).collect();
    let secrets: Vec<Option<Secret>> = hidden
        .contents
        .iter()
        .map(String::as_str)
        .chain(task.privileged.iter().map(|p| p.content.as_str()))
        .map(|text| Secret::of(text, &public))
        .collect();
    let mut matches = Vec::new();
    let mut probes = Vec::new();
    for text in &texts {
        let named = |list: &[String], kind: &str, out: &mut Vec<serde_json::Value>| {
            for (item, name) in list.iter().enumerate() {
                if !name.is_empty() && text.text.contains(name.as_str()) {
                    out.push(finding(text, kind, item));
                }
            }
        };
        // A call that named a hidden path and whose own result is only a
        // refusal or "no such file" read nothing: a recorded probe, not exposure.
        let failed_call = text.place == Place::ToolCall
            && text.result.as_deref().is_some_and(returned_only_a_failure);
        if failed_call {
            named(&hidden.paths, "hidden_path_absent", &mut probes);
        } else {
            named(&hidden.paths, "hidden_path", &mut matches);
        }
        named(&hidden.forbidden, "forbidden_string", &mut matches);
        // A call may quote what it was never shown only by guessing; the
        // shingle test is about what the solver was given to read.
        if text.place != Place::ToolCall {
            let shown = tokens(&text.text);
            for (item, secret) in secrets.iter().enumerate() {
                if secret.as_ref().is_some_and(|s| s.leaks_into(&shown)) {
                    matches.push(finding(text, "content_shingle", item));
                }
            }
        }
    }
    if hidden.confine_to_workspace {
        let recorded = exp
            .trajectory
            .agent
            .extra
            .as_ref()
            .and_then(|extra| extra.get(WORKSPACE_TREE_KEY))
            .and_then(|tree| tree.as_str());
        for (step, count, reach) in outside_workspace(&exp.trajectory.steps, recorded) {
            let hit = json!({
                "step": step, "in": Place::ToolCall.label(),
                "kind": reach.kind(), "item": count,
            });
            if reach.exposes() {
                matches.push(hit);
            } else {
                probes.push(hit);
            }
        }
    }
    if matches.is_empty() {
        return Finding::abstain(
            "no privileged material appears in the trajectory",
            json!({ "texts_audited": texts.len(), "outside_probes": probes }),
        );
    }
    Finding::decided(
        false,
        json!({
            "exposed": matches,
            "outside_probes": probes,
            "texts_audited": texts.len(),
        }),
    )
}

/// [`audit_exposure`] as a verifier, so the verdict reaches the store the
/// way every verdict does ([`crate::verifiers::annotation`]). Unlike the
/// other verifiers it reads the trajectory: that is what it audits.
#[derive(Clone, Debug)]
pub struct ExposureAudit {
    hidden: HiddenMaterial,
}

impl ExposureAudit {
    /// An audit for `hidden`.
    #[must_use]
    pub fn new(hidden: HiddenMaterial) -> Self {
        Self { hidden }
    }
}

impl Verifier for ExposureAudit {
    fn producer(&self) -> Producer {
        Producer {
            name: EXPOSURE_PRODUCER.into(),
            version: VERSION.into(),
        }
    }

    fn strength(&self) -> Strength {
        Strength::Constraint
    }

    fn verify(&self, task: &Task, exp: &Experience) -> Result<Finding, VerifyError> {
        Ok(audit_exposure(task, exp, &self.hidden))
    }
}
