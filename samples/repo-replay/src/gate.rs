// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The deterministic gate: did the agent reproduce the commit, and did it do
//! so the way a student should learn to?
//!
//! Only an attempt that passes every rule is a positive; the rest are kept as
//! negatives with every reason that applied, each in a class a report counts.
//! The gate reads the final tree and the recorded trajectory, never the
//! agent's account of itself.
//!
//! # Rules
//!
//! Exact (the result):
//! * `tree_mismatch` - a path of the commit's tree is missing or differs in
//!   bytes or mode (generated and vendored paths excepted).
//! * `junk` - a path exists that the commit's tree does not have.
//!
//! Process (how it was reached):
//! * `missing_edit` - a changed file got fewer than
//!   [`GateConfig::min_edits_per_file`] edit calls (`edit_file`/`write_file`),
//!   or a deleted or moved file got no `rm`/`mv` call naming it.
//! * `forbidden_shell` - the diff applied wholesale (`git apply`, `patch`),
//!   history reached (`git checkout`, `fetch`, `show <rev>` ...), a file
//!   rewritten from the shell (here-document, redirection, `tee`, `sed -i`,
//!   an interpreter's inline write), the network, or a path of the source
//!   repository or the run's state.
//! * `outside_tree` - a shell command naming a path above or outside the
//!   attempt tree.
//! * `unread_edit` - a file that existed was edited before any call showed
//!   its content (a read, or a search in that file with context lines).
//! * `unmotivated_location` - a file was edited that neither the plan named
//!   nor an earlier search or listing showed.
//! * `leaked_target` - an identifier that only the diff introduces appeared
//!   in a call that is not an edit (a search for a name nobody could know).
//! * `unjustified_content` - an edit wrote a name, a number or a string that
//!   neither the plan nor any earlier tool result holds (see
//!   [`crate::justify`]): the plan was under-specified.
//! * `diff_offset_read` - a `read_file` with an explicit line offset whose
//!   range begins at or just before (within 15 lines of) where a hunk of the diff starts, before any search result
//!   named that file: the agent knew the place from the diff's positions.
//! * `diff_reference` - the agent's words refer to the change it was given
//!   ("the intended change", "matches the diff"), the system prompt or its
//!   instructions. Talking about its own `git diff` is not one.
//! * `tool_calls_out_of_range` - fewer or more calls than the bounds.
//! * `run_incomplete` - the run was stopped (budget, repetition, deadline)
//!   or never concluded.

use std::collections::HashSet;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::git::{Change, Kind};
use crate::justify::{unjustified, EDIT_TOOLS};
use crate::trace::{Call, Trace};
use crate::tree::{dashed_names, identifiers, Residual};

/// The bounds of the process rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GateConfig {
    /// Edit calls a changed file needs.
    pub min_edits_per_file: usize,
    /// Fewest tool calls an attempt may make.
    pub min_tool_calls: usize,
    /// Most tool calls an attempt may make.
    pub max_tool_calls: usize,
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            min_edits_per_file: 1,
            min_tool_calls: 1,
            max_tool_calls: 60,
        }
    }
}

/// What one attempt is judged on.
pub struct GateInput<'a> {
    /// The commit's changes, generated paths already removed.
    pub changes: &'a [Change],
    /// The task the agent was given.
    pub plan: &'a str,
    /// Identifier-shaped words the parent tree already contains.
    pub parent_words: &'a HashSet<String>,
    /// The commit's diff as git wrote it (unmasked): where its hunks start.
    pub diff: &'a str,
    /// The text of the diff's added lines.
    pub added_text: &'a str,
    /// What the agent did.
    pub trace: &'a Trace,
    /// Where the final tree differs from the commit's.
    pub residual: &'a [Residual],
    /// Paths of the source repository, mirror and state the shell must not name.
    pub forbidden_locations: &'a [String],
    /// The attempt tree's absolute path, spelled every way the agent may have
    /// been told it (as laid out, and with symbolic links resolved).
    pub attempt_roots: &'a [String],
    /// Why the run stopped early, if it did not conclude by itself.
    pub stopped: Option<&'a str>,
    /// The user's follow-up messages (corrections) and the number of calls the
    /// agent had made when each came: what they say is something it was told.
    pub guidance: &'a [(usize, String)],
}

/// One reason an attempt failed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reason {
    /// The class a report counts it under (see the module documentation).
    pub class: String,
    /// What exactly was found.
    pub detail: String,
}

/// The gate's decision.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Verdict {
    /// Whether every rule held.
    pub passed: bool,
    /// Every reason that applied; empty exactly when `passed`.
    pub reasons: Vec<Reason>,
}

impl Verdict {
    /// Whether a reason of `class` applied.
    #[must_use]
    pub fn has(&self, class: &str) -> bool {
        self.reasons.iter().any(|r| r.class == class)
    }

    /// The classes that applied, once each, in order.
    #[must_use]
    pub fn classes(&self) -> Vec<&str> {
        let mut seen = Vec::new();
        for r in &self.reasons {
            if !seen.contains(&r.class.as_str()) {
                seen.push(r.class.as_str());
            }
        }
        seen
    }
}

fn re(pattern: &str) -> Regex {
    Regex::new(pattern).unwrap_or_else(|_| unreachable!("a fixed pattern"))
}

/// Top-level directories a command may name: the system's programs and libraries.
const SYSTEM_DIRS: &[&str] = &["usr", "bin", "lib", "lib64", "opt", "etc"];

/// Whether `path` is the null device.
fn is_null_device(path: &str) -> bool {
    path.strip_prefix('/') == Some("dev/null")
}

fn norm(path: &str, roots: &[String]) -> String {
    let path = roots
        .iter()
        .find_map(|root| path.strip_prefix(root.as_str()))
        .map_or(path, |p| p.trim_start_matches('/'));
    path.trim_start_matches("./").to_string()
}

fn base(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

fn parent_dir(path: &str) -> Option<&str> {
    path.rsplit_once('/').map(|(d, _)| d)
}

/// Whether `command` reads files (shows their content or lists them).
fn reads(command: &str) -> bool {
    re(r"\b(?:cat|head|tail|sed|less|more|nl|bat|awk|grep|rg|ag)\b").is_match(command)
}

/// Whether `command` lists or searches.
fn lists(command: &str) -> bool {
    re(r"\b(?:ls|find|tree|grep|rg|ag|fd|locate)\b|git\s+(?:ls-files|grep)").is_match(command)
}

fn mentions(text: &str, path: &str) -> bool {
    let b = base(path);
    text.contains(path) || (b.len() >= 3 && text.contains(b))
}

/// Judges one attempt.
#[must_use]
pub fn evaluate(input: &GateInput<'_>, config: &GateConfig) -> Verdict {
    let mut reasons = Vec::new();
    let mut add = |class: &str, detail: String| {
        reasons.push(Reason {
            class: class.into(),
            detail,
        })
    };

    for r in input.residual {
        match r {
            Residual::Extra { path } => add(
                "junk",
                format!("{path} exists and the commit has no such file"),
            ),
            Residual::Stale { path } => add(
                "tree_mismatch",
                format!("{path} is still present; the commit removes it"),
            ),
            Residual::Missing { path } => add("tree_mismatch", format!("{path} is missing")),
            Residual::Differs {
                path,
                first_difference,
            } => {
                add(
                    "tree_mismatch",
                    format!("{path} differs: {first_difference}"),
                );
            }
        }
    }
    if let Some(why) = input.stopped {
        add("run_incomplete", why.to_string());
    }
    let calls = &input.trace.calls;
    if calls.len() < config.min_tool_calls || calls.len() > config.max_tool_calls {
        add(
            "tool_calls_out_of_range",
            format!(
                "{} calls; allowed {} to {}",
                calls.len(),
                config.min_tool_calls,
                config.max_tool_calls
            ),
        );
    }
    for call in calls.iter().filter(|c| c.tool == "shell") {
        if let Some(command) = call.command() {
            shell_rules(command, input, &mut add);
        }
    }
    edit_rules(input, config, &mut add);
    leak_rules(input, &mut add);
    for found in unjustified(input.plan, calls, input.guidance) {
        add(
            "unjustified_content",
            format!(
                "{} was written by {} at step {} but neither the plan nor anything the agent had seen holds it",
                found.text, found.tool, found.step
            ),
        );
    }
    offset_rules(input, &mut add);
    for text in &input.trace.said {
        if let Some(found) = refers_to_given_change(text) {
            add("diff_reference", format!("the agent said {found:?}"));
        }
    }
    Verdict {
        passed: reasons.is_empty(),
        reasons,
    }
}

/// The words by which `text` refers to the change the agent was shown or to
/// its instructions ("the intended change", "matches the diff", "system
/// prompt"), if it does. Talking about its own `git diff` is not a reference.
#[must_use]
pub fn refers_to_given_change(text: &str) -> Option<String> {
    let said = re(
        r"(?i)\b(?:intended|expected|target|original|given|provided|above|masked) (?:diff|patch|change)\b|\b(?:against|matches|match|matching) the (?:diff|patch|change)\b|system prompt|my instructions|these instructions|target commit|(?:was|been) (?:told|instructed|given)|reproduce (?:exactly )?(?:the|these) changes",
    );
    said.find(text).map(|m| m.as_str().to_string())
}

fn shell_rules(command: &str, input: &GateInput<'_>, add: &mut impl FnMut(&str, String)) {
    // What the command runs, with the text of quoted arguments taken out: a
    // grep pattern naming `patch` or `<<` is not that command. A command that
    // hands a quoted script to a shell or interpreter is read as written.
    let quoted = re(r#""[^"]*"|'[^']*'"#);
    let runs_a_script =
        re(r"\b(?:(?:ba|z|da)?sh|eval|python3?|node|ruby|perl)\b\s+(?:-\w+\s+)*-[ce]\b")
            .is_match(command);
    let visible = if runs_a_script {
        command.to_string()
    } else {
        quoted.replace_all(command, " ").into_owned()
    };
    let command_start = r"(?:^|[;&|(]\s*|\bxargs\s+|\bsudo\s+)(?:\S*/)?";
    // (name, pattern, whether it is read on the visible text or the command as written)
    let forbidden: [(&str, Regex, bool); 8] = [
        (
            "git apply",
            re(r"\bgit\s+(?:-\S+\s+)*(?:apply|am)\b"),
            false,
        ),
        ("patch", re(&format!(r"{command_start}patch(?:\s|$)")), true),
        (
            "a git command that changes or fetches files or history",
            re(
                r"\bgit\s+(?:-\S+\s+)*(?:checkout|switch|restore|fetch|pull|clone|merge|rebase|cherry-pick|reset|remote|worktree|format-patch|revert|bundle|submodule)\b",
            ),
            false,
        ),
        (
            "an in-place edit",
            re(r"\b(?:sed|perl|ruby)\b[^|;&]*\s-[a-zA-Z]*i|\bawk\b.*-i\s*inplace"),
            false,
        ),
        ("a here-document", re(r"<<"), true),
        ("tee", re(&format!(r"{command_start}tee\b")), true),
        (
            "an interpreter writing a file",
            re(
                r#"\b(?:python3?|node|ruby|perl)\b\s+-[ce]\b.*(?:write|open\s*\([^)]*,\s*['"][^'"]*[wax+]|mode\s*=\s*['"][^'"]*[wax+])"#,
            ),
            false,
        ),
        (
            "the network",
            re(&format!(r"{command_start}(?:curl|wget|ssh|scp|nc)\b")),
            true,
        ),
    ];
    for (name, pattern, on_visible) in &forbidden {
        let text = if *on_visible {
            visible.as_str()
        } else {
            command
        };
        if pattern.is_match(text) {
            add("forbidden_shell", format!("{name}: {command}"));
        }
    }
    // Quoted text (a grep pattern with `=>`) is not a redirection.
    let unquoted = re(r#""[^"]*"|'[^']*'"#).replace_all(command, "");
    for caps in re(r"(\d?>{1,2})\s*([^\s;|&)]+|&\d)").captures_iter(&unquoted) {
        let target = &caps[2];
        if !(target.starts_with('&') || is_null_device(target)) {
            add(
                "forbidden_shell",
                format!("a redirection writing {target}: {command}"),
            );
        }
    }
    for place in input.forbidden_locations {
        if !place.is_empty() && command.contains(place.as_str()) {
            add("forbidden_shell", format!("names {place}: {command}"));
        }
    }
    if command.contains("../") || command.ends_with("..") {
        add("outside_tree", format!("goes above the tree: {command}"));
    }
    for caps in re(r#"(?:^|[\s=:'"(])(/[A-Za-z0-9_.\-/]+)"#).captures_iter(command) {
        let path = &caps[1];
        let top = path
            .trim_start_matches('/')
            .split('/')
            .next()
            .unwrap_or_default();
        let ok = input
            .attempt_roots
            .iter()
            .any(|r| path.starts_with(r.as_str()))
            || is_null_device(path)
            || SYSTEM_DIRS.contains(&top);
        if !ok {
            add("outside_tree", format!("names {path}: {command}"));
        }
    }
}

/// Paths an edit-tool call writes, normalised.
fn edit_target(call: &Call, root: &[String]) -> Option<String> {
    EDIT_TOOLS
        .contains(&call.tool.as_str())
        .then(|| call.arg("path").map(|p| norm(p, root)))
        .flatten()
}

/// Whether `call` shows the content of `path` (or its former path).
fn shows(call: &Call, paths: &[&str], root: &[String]) -> bool {
    match call.tool.as_str() {
        "read_file" => call
            .arg("path")
            .is_some_and(|p| paths.contains(&norm(p, root).as_str())),
        // A search in that file that prints its surroundings shows the file.
        "grep" => {
            call.args
                .get("context_lines")
                .and_then(serde_json::Value::as_u64)
                .is_some_and(|n| n > 0)
                && call
                    .arg("path")
                    .is_some_and(|p| paths.contains(&norm(p, root).as_str()))
                && !call.result.trim().is_empty()
        }
        "shell" => call.command().is_some_and(|c| {
            reads(c) && paths.iter().any(|p| c.contains(p) || c.contains(base(p)))
        }),
        _ => false,
    }
}

fn is_removal(command: &str) -> bool {
    re(r"\b(?:rm|git\s+rm|mv|git\s+mv)\b").is_match(command)
}

fn edit_rules(input: &GateInput<'_>, config: &GateConfig, add: &mut impl FnMut(&str, String)) {
    let root = input.attempt_roots;
    let calls = &input.trace.calls;
    let plan = input.plan.to_lowercase();

    for change in input.changes {
        let edits = calls
            .iter()
            .filter(|c| edit_target(c, root).as_deref() == Some(&change.path))
            .count();
        let moves = |p: &str| {
            calls.iter().any(|c| {
                c.tool == "shell"
                    && c.command()
                        .is_some_and(|cmd| is_removal(cmd) && cmd.contains(p))
            })
        };
        match change.kind {
            Kind::Added | Kind::Modified => {
                if edits < config.min_edits_per_file {
                    add(
                        "missing_edit",
                        format!("{} got {edits} edit call(s)", change.path),
                    );
                }
            }
            Kind::Deleted => {
                if !moves(&change.path) && edits == 0 {
                    add(
                        "missing_edit",
                        format!("{} was never removed with rm or git rm", change.path),
                    );
                }
            }
            Kind::Renamed => {
                let old = change.old_path.as_deref().unwrap_or_default();
                if !(moves(old) && moves(&change.path)) {
                    add(
                        "missing_edit",
                        format!("{old} was never moved to {} with mv or git mv", change.path),
                    );
                }
                if change.lines() > 0 && edits < config.min_edits_per_file {
                    add(
                        "missing_edit",
                        format!(
                            "{} was moved but its content changes were not edited",
                            change.path
                        ),
                    );
                }
            }
        }
    }

    // Everything the agent edited or removed, whether the commit changes it or not.
    let created: HashSet<&str> = input
        .changes
        .iter()
        .filter(|c| c.kind == Kind::Added)
        .map(|c| c.path.as_str())
        .collect();
    for (index, call) in calls.iter().enumerate() {
        let before = &calls[..index];
        let prior_results = |pred: &dyn Fn(&Call) -> bool| -> String {
            before
                .iter()
                .filter(|c| pred(c))
                .map(|c| c.result.as_str())
                .collect::<Vec<_>>()
                .join("\n")
        };
        let found = prior_results(&|c| match c.tool.as_str() {
            "find_file" | "grep" => true,
            "shell" => c.command().is_some_and(lists),
            _ => false,
        });
        let motivated =
            |path: &str| mentions(&plan, &path.to_lowercase()) || mentions(&found, path);

        if let Some(path) = edit_target(call, root) {
            let former = input
                .changes
                .iter()
                .find(|c| c.path == path)
                .and_then(|c| c.old_path.as_deref())
                .map(String::from);
            let mut names = vec![path.as_str()];
            names.extend(former.as_deref());
            let new_file = created.contains(path.as_str());
            if !new_file
                && !before.iter().any(|c| {
                    shows(c, &names, root) || edit_target(c, root).as_deref() == Some(&path)
                })
            {
                add(
                    "unread_edit",
                    format!(
                        "{path} was edited at step {} before its content was shown",
                        call.step
                    ),
                );
            }
            let located = if new_file {
                motivated(&path) || parent_dir(&path).is_some_and(motivated)
            } else {
                motivated(&path) || former.as_deref().is_some_and(motivated)
            };
            if !located {
                add(
                    "unmotivated_location",
                    format!(
                        "{path} was edited at step {} but neither the plan nor a search showed it",
                        call.step
                    ),
                );
            }
        } else if call.tool == "shell" {
            let Some(command) = call.command().filter(|c| is_removal(c)) else {
                continue;
            };
            for change in input
                .changes
                .iter()
                .filter(|c| matches!(c.kind, Kind::Deleted | Kind::Renamed))
            {
                let source = change.old_path.as_deref().unwrap_or(&change.path);
                if command.contains(source)
                    && !(motivated(source) || before.iter().any(|c| shows(c, &[source], root)))
                {
                    add(
                        "unmotivated_location",
                        format!(
                            "{source} was removed or moved at step {} but nothing pointed to it",
                            call.step
                        ),
                    );
                }
            }
        }
    }
}

fn leak_rules(input: &GateInput<'_>, add: &mut impl FnMut(&str, String)) {
    let plan_names = dashed_names(input.plan);
    let plan_words: HashSet<&str> = identifiers(input.plan)
        .chain(plan_names.iter().map(String::as_str))
        .collect();
    let added_names = dashed_names(input.added_text);
    let only_diff: HashSet<&str> = identifiers(input.added_text)
        .chain(added_names.iter().map(String::as_str))
        .filter(|w| !input.parent_words.contains(*w) && !plan_words.contains(w))
        .collect();
    if only_diff.is_empty() {
        return;
    }
    let path_words: HashSet<&str> = input
        .changes
        .iter()
        .flat_map(|c| identifiers(&c.path))
        .collect();
    for call in &input.trace.calls {
        // An edit writes the new names; the todo list is the agent's own plan of
        // what it will write, in its own words. Neither is a search.
        if EDIT_TOOLS.contains(&call.tool.as_str()) || call.tool == "todo" {
            continue;
        }
        let is_move = call.command().is_some_and(is_removal);
        let text = call.args_text();
        let leaked = identifiers(&text)
            .find(|w| only_diff.contains(w) && !(is_move && path_words.contains(w)));
        if let Some(word) = leaked {
            add(
                "leaked_target",
                format!(
                    "{word} appears only in the change but was used in a {} call at step {}",
                    call.tool, call.step
                ),
            );
        }
    }
}

/// The files and original line numbers where the hunks of `diff` start.
fn hunk_starts(diff: &str) -> Vec<(String, u64)> {
    let header = re(r"^diff --git a/(.+) b/(.+)$");
    let hunk = re(r"^@@ -(\d+)");
    let mut out = Vec::new();
    let mut names: Vec<String> = Vec::new();
    for line in diff.lines() {
        if let Some(c) = header.captures(line) {
            names = vec![c[1].to_string(), c[2].to_string()];
        } else if let Some(c) = hunk.captures(line) {
            let start = c[1].parse().unwrap_or(0);
            out.extend(names.iter().map(|n| (n.clone(), start)));
        }
    }
    out
}

/// How many lines before a hunk's start a read may begin and still count as
/// aimed at the hunk. A read paging through a file lands on hunk starts by
/// chance only when it begins within this window of one.
const READ_AHEAD: u64 = 15;

/// A read at the place the diff says, before a search found the file.
fn offset_rules(input: &GateInput<'_>, add: &mut impl FnMut(&str, String)) {
    let hunks = hunk_starts(input.diff);
    let calls = &input.trace.calls;
    for (index, call) in calls.iter().enumerate() {
        if call.tool != "read_file" {
            continue;
        }
        let Some(offset) = call
            .args
            .get("offset")
            .and_then(serde_json::Value::as_u64)
            .filter(|o| *o > 1)
        else {
            continue;
        };
        let Some(path) = call.arg("path").map(|p| norm(p, input.attempt_roots)) else {
            continue;
        };
        let covers = hunks.iter().any(|(p, start)| {
            *p == path && start.saturating_sub(READ_AHEAD) <= offset && offset <= *start
        });
        let searched = calls[..index].iter().any(|c| {
            let search = match c.tool.as_str() {
                "find_file" | "grep" => true,
                "shell" => c.command().is_some_and(lists),
                _ => false,
            };
            search && mentions(&c.result, &path)
        });
        if covers && !searched {
            add(
                "diff_offset_read",
                format!("{path} was read from line {offset} at step {}, where a hunk starts, before any search named it", call.step),
            );
        }
    }
}
