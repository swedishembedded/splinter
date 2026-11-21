// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verification that keeps hinted agent
// traces honest before they become training data, for its clients. If your
// team needs expertise in self-generated rationales that carry no answer
// leakage, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The filters a hinted attempt passes before it is kept, and the sections
//! the hint and the notes take in a staged instruction.
//!
//! A hinted attempt is a trajectory of the model working a task through its
//! real tools with the task's reference solution appended to the
//! instruction between [`HINT_BEGIN`] and [`HINT_END`]. It is kept only when
//!
//! 1. **verified** ([`verified`]): the task's verifier gave it reward 1;
//! 2. **real observations** ([`read_before_write`]): the first agent step
//!    that calls a tool reads every file the task changes that has content
//!    to read (each of its non-empty lines is in that step's tool replies),
//!    and none of that step's replies confirms a write of such a file (an
//!    editor's `"FILE" ... written`). Which files change is the difference
//!    between the starting files and the task's target files
//!    (`tests/expected/`); a task without target files is undecidable and
//!    fails;
//! 3. **no answer present** ([`no_leak`], the record's `no_leak`): no text
//!    the model wrote (reasoning and messages, not tool arguments or
//!    replies) repeats the hint's own wording (a run of [`HINT_NGRAM`] words
//!    from one line of the hint section that the model was not also shown in
//!    the instruction, its system prompt or tool schemas, and that it did
//!    not itself send as a tool argument), or quotes a line of the task's
//!    files, starting or expected (at least [`MIN_QUOTED_CHARS`] characters,
//!    compared with runs of whitespace as one space), that neither the
//!    instruction nor an earlier step's tool reply showed; a run may add a
//!    word list ([`LEAK_TERMS`] is empty) whose mention (a word prefix,
//!    case-insensitive) is a leak too;
//! 4. **sufficient**: given the instruction and only the attempt's reasoning
//!    ([`reasoning_of`], in [`NOTES_BEGIN`]/[`NOTES_END`]), the model passes
//!    the verifier again in a fresh sandbox. That run is the stage's; this
//!    module stages its instruction.
//!
//! What is kept ([`kept_trajectory`]) is the attempt with its user turn
//! replaced by the task's own instruction, and nothing of the hint section
//! anywhere in it.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The line that opens the hint section of a staged instruction.
pub const HINT_BEGIN: &str = "<!-- splinter:hint:begin -->";

/// The line that closes it.
pub const HINT_END: &str = "<!-- splinter:hint:end -->";

/// The line that opens the notes section of a sufficiency run's
/// instruction.
pub const NOTES_BEGIN: &str = "<!-- splinter:notes:begin -->";

/// The line that closes it.
pub const NOTES_END: &str = "<!-- splinter:notes:end -->";

/// What the model is told about the hint, inside the hint section.
pub const HINT_PREAMBLE: &str = "A worked answer to this task follows, for your eyes only. \
Work the task yourself through the tool: read the files before you change them, and explain \
each step from what the files and the tool's replies show. Never mention, quote or refer to \
this section in anything you write.";

/// The private-scratch framing of the hint ([`HintFraming::PrivateScratch`]).
pub const PRIVATE_SCRATCH_PREAMBLE: &str = "Private scratch knowledge follows: an approach to this task \
that works. Use it only to steer your own work. Work as if you found the approach yourself: first \
observe the files through the tool, then act, and explain each step only from what the task, the files \
and the tool's replies show. Never mention, quote or allude to any hint, reference, provided answer, \
solution or solution file, in your reasoning or in anything you write.";

/// How the hint section frames the hint for the model: a prompt-template choice of a run, recorded in its key.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum HintFraming {
    /// A worked answer shown for the model's eyes only ([`HINT_PREAMBLE`]).
    #[default]
    WorkedAnswer,
    /// Private scratch knowledge, with the instruction to work as if unaided ([`PRIVATE_SCRATCH_PREAMBLE`]).
    PrivateScratch,
}

impl HintFraming {
    /// Every framing, so a residue check knows each preamble.
    pub const ALL: [HintFraming; 2] = [HintFraming::WorkedAnswer, HintFraming::PrivateScratch];

    /// The text the hint section opens with.
    #[must_use]
    pub fn preamble(self) -> &'static str {
        match self {
            HintFraming::WorkedAnswer => HINT_PREAMBLE,
            HintFraming::PrivateScratch => PRIVATE_SCRATCH_PREAMBLE,
        }
    }
}

/// What the model is told about the notes, inside the notes section.
pub const NOTES_PREAMBLE: &str = "Notes from an earlier attempt at this task follow.";

/// The words whose mention in the model's own text is a leak (prefixes,
/// case-insensitive) when a run names none: none. Merely naming a hint, a
/// solution or an expected result is not a leak; the default check is
/// content-only (the hint's wording, file lines nothing showed yet), and a
/// run opts into a word list with its own terms.
pub const LEAK_TERMS: &[&str] = &[];

/// The length, in words, of a run of the hint's wording that is a leak when
/// the model repeats it.
pub const HINT_NGRAM: usize = 5;

/// The shortest line of a task file, in characters without surrounding
/// whitespace, whose quotation is checked against what was shown: shorter
/// lines are too common to say where the model learned them.
pub const MIN_QUOTED_CHARS: usize = 8;

/// The most findings one verdict lists.
const MAX_FINDINGS: usize = 5;

/// How one filter decided.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Verdict {
    /// Whether the attempt passed it.
    pub passed: bool,
    /// Why it failed, when it did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl Verdict {
    /// A pass.
    #[must_use]
    pub fn pass() -> Self {
        Self {
            passed: true,
            reason: None,
        }
    }

    /// A fail, and why.
    #[must_use]
    pub fn fail(reason: impl Into<String>) -> Self {
        Self {
            passed: false,
            reason: Some(reason.into()),
        }
    }

    /// A pass when there are no findings, else a fail listing the first of
    /// them.
    fn of(findings: Vec<String>) -> Self {
        if findings.is_empty() {
            return Self::pass();
        }
        let more = findings.len().saturating_sub(MAX_FINDINGS);
        let mut reason = findings
            .into_iter()
            .take(MAX_FINDINGS)
            .collect::<Vec<_>>()
            .join("; ");
        if more > 0 {
            reason.push_str(&format!("; and {more} more"));
        }
        Self::fail(reason)
    }
}

/// The task's files the filters read: its starting files and, when the task
/// states them, its target files, each by path relative to the workspace.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TaskFiles {
    /// The files the agent starts from.
    pub starter: BTreeMap<String, String>,
    /// The files the verifier requires; `None` when the task does not state
    /// them.
    pub expected: Option<BTreeMap<String, String>>,
}

impl TaskFiles {
    /// The files the task changes (created, edited or removed), in path
    /// order; `None` when the task states no target files.
    #[must_use]
    pub fn changed(&self) -> Option<Vec<&str>> {
        let expected = self.expected.as_ref()?;
        let paths: BTreeSet<&str> = self
            .starter
            .keys()
            .chain(expected.keys())
            .map(String::as_str)
            .collect();
        Some(
            paths
                .into_iter()
                .filter(|path| self.starter.get(*path) != expected.get(*path))
                .collect(),
        )
    }

    /// Every line of every file, starting and target, long enough to say
    /// where the model learned it, with its runs of whitespace as one space
    /// ([`spaced`]), and the file it is from.
    fn quotable_lines(&self) -> BTreeMap<String, &str> {
        let mut lines = BTreeMap::new();
        let target = self.expected.iter().flat_map(BTreeMap::iter);
        for (path, content) in self.starter.iter().chain(target) {
            for line in content.lines().map(spaced) {
                if line.chars().count() >= MIN_QUOTED_CHARS {
                    lines.entry(line).or_insert(path.as_str());
                }
            }
        }
        lines
    }
}

/// A code fence longer than any backtick run in `text`.
fn fence_for(text: &str) -> String {
    let longest = text.split(|c| c != '`').map(str::len).max().unwrap_or(0);
    "`".repeat((longest + 1).max(3))
}

/// `instruction`, then a section between `begin` and `end` holding
/// `preamble` and `body` fenced.
fn with_section(instruction: &str, begin: &str, end: &str, preamble: &str, body: &str) -> String {
    let fence = fence_for(body);
    format!(
        "{}\n\n{begin}\n{preamble}\n\n{fence}\n{}\n{fence}\n{end}\n",
        instruction.trim_end(),
        body.trim_end()
    )
}

/// The instruction a hinted attempt is given: `instruction`, then the hint
/// section holding `framing`'s preamble and `hint` in a code fence longer than
/// any backtick run in it.
#[must_use]
pub fn hinted_instruction(instruction: &str, hint: &str, framing: HintFraming) -> String {
    with_section(instruction, HINT_BEGIN, HINT_END, framing.preamble(), hint)
}

/// The instruction a sufficiency run is given: `instruction`, then the
/// notes section holding [`NOTES_PREAMBLE`] and `notes`.
#[must_use]
pub fn notes_instruction(instruction: &str, notes: &str) -> String {
    with_section(instruction, NOTES_BEGIN, NOTES_END, NOTES_PREAMBLE, notes)
}

/// `text` without its one hint section.
///
/// # Errors
/// When `text` holds no hint section, more than one, or a malformed one.
pub fn strip_hint(text: &str) -> Result<String, String> {
    let begins = text.matches(HINT_BEGIN).count();
    let ends = text.matches(HINT_END).count();
    if begins != 1 || ends != 1 {
        return Err(format!(
            "the text holds {begins} hint openings and {ends} closings; exactly one section was \
             staged"
        ));
    }
    let (Some(begin), Some(end)) = (text.find(HINT_BEGIN), text.find(HINT_END)) else {
        return Err("the hint section is not in the text".into());
    };
    if end < begin {
        return Err("the hint section closes before it opens".into());
    }
    let after = text[end + HINT_END.len()..].trim();
    let before = text[..begin].trim_end();
    Ok(if after.is_empty() {
        before.to_string()
    } else {
        format!("{before}\n\n{after}")
    })
}

/// Filter 1: the verifier's reward, when it gave one.
#[must_use]
pub fn verified(reward: Option<f64>) -> Verdict {
    match reward {
        Some(r) if (r - 1.0).abs() < f64::EPSILON => Verdict::pass(),
        Some(r) => Verdict::fail(format!("the verifier gave reward {r}")),
        None => Verdict::fail("the verifier gave no reward"),
    }
}

/// The agent steps of `trajectory`, in order.
fn agent_steps(trajectory: &atif::Trajectory) -> impl Iterator<Item = &atif::TraceStep> {
    trajectory
        .steps
        .iter()
        .filter(|s| s.source == atif::StepOrigin::Agent)
}

/// The text of a message body: its text, or its text segments.
fn body_text(body: &atif::MessageBody) -> String {
    if let Some(text) = body.as_text() {
        return text.to_string();
    }
    body.as_segments()
        .unwrap_or_default()
        .iter()
        .filter_map(|segment| match segment {
            atif::ContentSegment::Text { text } => Some(text.as_str()),
            atif::ContentSegment::Image { .. } => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The text of `trajectory`'s first user turn: the instruction the attempt
/// was given.
#[must_use]
pub fn user_turn(trajectory: &atif::Trajectory) -> Option<String> {
    trajectory
        .steps
        .iter()
        .find(|s| s.source == atif::StepOrigin::User)
        .map(|s| body_text(&s.message))
}

/// What the model wrote in `step`: its reasoning and its message.
fn model_text(step: &atif::TraceStep) -> String {
    let reasoning = step.reasoning_content.as_deref().unwrap_or_default();
    format!("{reasoning}\n{}", body_text(&step.message))
}

/// The tool replies `step` got.
fn replies(step: &atif::TraceStep) -> String {
    step.observation
        .iter()
        .flat_map(|o| &o.results)
        .filter_map(|r| r.content.as_ref().map(body_text))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Every string leaf of `value`.
fn strings(value: &Value, into: &mut Vec<String>) {
    match value {
        Value::String(s) => into.push(s.clone()),
        Value::Array(items) => items.iter().for_each(|v| strings(v, into)),
        Value::Object(map) => map.values().for_each(|v| strings(v, into)),
        _ => {}
    }
}

/// The text of the tool arguments `step` sent.
fn call_text(step: &atif::TraceStep) -> String {
    let mut leaves = Vec::new();
    for call in step.tool_calls.iter().flatten() {
        strings(&call.arguments, &mut leaves);
    }
    leaves.join("\n")
}

/// The step's id, as findings name it.
fn step_name(step: &atif::TraceStep) -> String {
    format!("step {}", step.step_id)
}

/// Filter 2: the first agent step that calls a tool reads every changed
/// file with content and writes none of them, except a file `instruction`
/// already shows (every non-empty line of its starting content is in it),
/// which the model has observed before any call; see the module
/// documentation.
#[must_use]
pub fn read_before_write(
    trajectory: &atif::Trajectory,
    files: &TaskFiles,
    instruction: &str,
) -> Verdict {
    let Some(changed) = files.changed() else {
        return Verdict::fail(
            "the task states no target files (no tests/expected/), so which files it changes \
             is unknown",
        );
    };
    let Some(first) =
        agent_steps(trajectory).find(|s| s.tool_calls.as_ref().is_some_and(|c| !c.is_empty()))
    else {
        return Verdict::fail("the attempt calls no tool");
    };
    let reply = replies(first);
    let mut findings = Vec::new();
    for path in changed {
        let Some(start) = files.starter.get(path).filter(|c| !c.trim().is_empty()) else {
            continue;
        };
        let shown = start
            .lines()
            .map(str::trim_end)
            .filter(|line| !line.trim().is_empty())
            .all(|line| instruction.contains(line));
        if shown {
            continue;
        }
        if let Some(unseen) = start
            .lines()
            .map(str::trim_end)
            .find(|line| !line.trim().is_empty() && !reply.contains(line))
        {
            findings.push(format!(
                "the first tool call ({}) does not show {path} before it is changed: its line \
                 {unseen:?} is not in the reply",
                step_name(first)
            ));
        }
        if writes(&reply, path) {
            findings.push(format!(
                "the first tool call ({}) writes {path} before any reply showed it",
                step_name(first)
            ));
        }
    }
    Verdict::of(findings)
}

/// Whether `reply` confirms a write of `path` the way an editor does
/// (`"FILE" ... written`), by its path or its file name.
fn writes(reply: &str, path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path);
    let quoted = [format!("\"{path}\""), format!("\"{name}\"")];
    reply
        .lines()
        .any(|line| line.contains("written") && quoted.iter().any(|q| line.contains(q.as_str())))
}

/// What the leak filter compares the model's text with.
pub struct LeakCheck<'a> {
    /// The leak terms.
    pub terms: &'a [String],
    /// The hint section as staged (preamble and hint).
    pub hint: &'a str,
    /// The task's instruction, without the hint.
    pub instruction: &'a str,
    /// What else the model was shown that is not the hint: its system prompt
    /// and tool schemas.
    pub shown: &'a str,
    /// The task's files.
    pub files: &'a TaskFiles,
}

/// `text` with every run of whitespace one space and none around it: a line
/// reproduced with other spacing or wrapping is still that line.
fn spaced(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The words of `text`, lowercased, each without the punctuation around it.
fn words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

/// The runs of [`HINT_NGRAM`] consecutive words of `text`.
fn ngrams(text: &str) -> HashSet<String> {
    words(text)
        .windows(HINT_NGRAM)
        .map(|w| w.join(" "))
        .collect()
}

/// Whether `text` (lowercased) mentions `term` (lowercased): an occurrence
/// that does not continue a word.
fn mentions(text: &str, term: &str) -> bool {
    text.match_indices(term).any(|(at, _)| {
        text[..at]
            .chars()
            .next_back()
            .is_none_or(|c| !c.is_alphanumeric())
    })
}

/// The leak terms of `terms` the model's own text in `trajectory` mentions,
/// by the rule [`no_leak`] applies. Run over attempts that never saw a hint,
/// it is the rate at which the term check rejects unhinted work.
#[must_use]
pub fn mentioned_terms<'t>(trajectory: &atif::Trajectory, terms: &'t [String]) -> Vec<&'t str> {
    let text = agent_steps(trajectory)
        .map(model_text)
        .collect::<Vec<_>>()
        .join("\n")
        .to_lowercase();
    terms
        .iter()
        .filter(|t| mentions(&text, &t.to_lowercase()))
        .map(String::as_str)
        .collect()
}

/// Filter 3, no answer present: the model's own text neither repeats the
/// hint's wording nor quotes file content nothing showed it, nor mentions a
/// term of the run's word list; see the module documentation.
#[must_use]
pub fn no_leak(trajectory: &atif::Trajectory, check: &LeakCheck<'_>) -> Verdict {
    let terms: Vec<String> = check.terms.iter().map(|t| t.to_lowercase()).collect();
    // The hint's wording: runs within one line of it that the model had
    // another way to know.
    // Known from the start: what the model was shown besides the hint. The model's own tool arguments become known
    // only to the steps after the one that sent them, so wording written first and sent later is still the hint's.
    let mut known = ngrams(check.instruction);
    known.extend(ngrams(check.shown));
    let hint_runs: HashSet<String> = check.hint.lines().flat_map(ngrams).collect();
    let quotable = check.files.quotable_lines();
    let mut seen = spaced(check.instruction);
    let commands = hint_commands(
        check.hint,
        &spaced(&format!("{}\n{}", check.instruction, check.shown)),
    );
    let mut findings = Vec::new();
    for step in agent_steps(trajectory) {
        let text = model_text(step);
        let lower = text.to_lowercase();
        for term in terms.iter().filter(|t| mentions(&lower, t)) {
            findings.push(format!("{} mentions {term:?}", step_name(step)));
        }
        let mut repeated: Vec<String> = ngrams(&text)
            .into_iter()
            .filter(|n| hint_runs.contains(n) && !known.contains(n))
            .collect();
        repeated.sort();
        if let Some(run) = repeated.first() {
            findings.push(format!(
                "{} repeats the hint's wording {run:?}",
                step_name(step)
            ));
        }
        let said = spaced(&text);
        for (line, path) in &quotable {
            if said.contains(line.as_str()) && !seen.contains(line.as_str()) {
                findings.push(format!(
                    "{} quotes {line:?} of {path}, which neither the instruction nor an earlier \
                     tool reply showed",
                    step_name(step)
                ));
            }
        }
        for command in commands
            .iter()
            .filter(|c| said.contains(c.as_str()) && !seen.contains(c.as_str()))
        {
            findings.push(format!(
                "{} writes the hint's command {command:?}, which no tool reply had shown",
                step_name(step)
            ));
        }
        seen.push(' ');
        seen.push_str(&spaced(&replies(step)));
        known.extend(ngrams(&call_text(step)));
        known.extend(ngrams(&replies(step)));
    }
    Verdict::of(findings)
}

/// The shortest hint command, or word of one, that counts as the answer: shorter ones (`dd`, `1`) are too common in
/// ordinary text to say where the model learned them.
const MIN_COMMAND_CHARS: usize = 6;

/// The commands of the hint's code (the lines of its fenced body, comments left out) and the words of each, at least
/// [`MIN_COMMAND_CHARS`] long (a word only when it is not all letters), spaced as [`spaced`] spaces them, leaving out what `known` (the instruction, system
/// prompt and tool schemas, spaced) already holds.
fn hint_commands(hint: &str, known: &str) -> BTreeSet<String> {
    let mut fences = hint
        .lines()
        .enumerate()
        .filter(|(_, l)| l.trim_start().starts_with("```"));
    let (Some((open, _)), Some((close, _))) = (fences.next(), fences.last()) else {
        return BTreeSet::new();
    };
    hint.lines()
        .skip(open + 1)
        .take(close.saturating_sub(open + 1))
        .map(spaced)
        .filter(|line| !line.starts_with('#'))
        .flat_map(|line| {
            // A word of a command counts only when it is not a plain word: `f{ci{new` does, `execute` does not.
            let words: Vec<String> = line
                .split(' ')
                .filter(|w| !w.chars().all(char::is_alphabetic))
                .map(str::to_string)
                .collect();
            std::iter::once(line).chain(words)
        })
        .filter(|piece| {
            piece.chars().count() >= MIN_COMMAND_CHARS && !known.contains(piece.as_str())
        })
        .collect()
}

/// The reasoning of an attempt, as a sufficiency run is shown it: each agent
/// step's reasoning and message, in order, without its tool calls or their
/// replies.
#[must_use]
pub fn reasoning_of(trajectory: &atif::Trajectory) -> String {
    agent_steps(trajectory)
        .map(|step| model_text(step).trim().to_string())
        .filter(|text| !text.is_empty())
        .enumerate()
        .map(|(n, text)| format!("Step {}:\n{text}", n + 1))
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// The trajectory a hinted attempt is kept as: its user turn the task's own
/// `instruction`, the hint section nowhere in it.
///
/// # Errors
/// When the user turn is not `instruction` once its hint section is
/// removed, or any part of the hint section is still in the trajectory.
pub fn kept_trajectory(
    hinted: &atif::Trajectory,
    instruction: &str,
) -> Result<atif::Trajectory, String> {
    let mut kept = hinted.clone();
    let user = kept
        .steps
        .iter_mut()
        .find(|s| s.source == atif::StepOrigin::User)
        .ok_or("the attempt has no user turn")?;
    let stripped = strip_hint(&body_text(&user.message))?;
    if stripped.trim_end() != instruction.trim_end() {
        return Err(
            "the user turn is not the task's instruction once the hint section is removed".into(),
        );
    }
    user.message = atif::MessageBody::text(instruction);
    match hint_residue(&kept) {
        Some(residue) => Err(residue),
        None => Ok(kept),
    }
}

/// Why `trajectory` still holds part of a hint section, when it does: one of
/// its markers or its preamble, anywhere in it.
#[must_use]
pub fn hint_residue(trajectory: &atif::Trajectory) -> Option<String> {
    let mut leaves = Vec::new();
    strings(&serde_json::to_value(trajectory).ok()?, &mut leaves);
    [HINT_BEGIN, HINT_END]
        .into_iter()
        .chain(HintFraming::ALL.map(HintFraming::preamble))
        .find(|part| leaves.iter().any(|leaf| leaf.contains(part)))
        .map(|part| format!("the trajectory still holds the hint section's {part:?}"))
}

#[cfg(test)]
#[path = "filters_tests.rs"]
mod tests;
