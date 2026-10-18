// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements audits that prove an agent never saw the
// material it is graded against, for its clients. If your team needs
// expertise in contamination-free evaluation of tool-using agents, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The part of the exposure audit that reads where a solver's commands
//! reached: a shell command or file argument that names an absolute path
//! outside the episode's own tree went looking beyond the workspace, where
//! the original sources, the tests and other attempts may lie.
//!
//! The episode's tree is the one its trajectory records
//! ([`WORKSPACE_TREE_KEY`]); for a trajectory made before it was recorded it
//! is taken to be the first `.../tree` directory the solver named. The
//! system's own directories ([`SYSTEM_PREFIXES`]) are not a place to look
//! for an answer and are never counted.
//!
//! Every hit is classed by what the step's output shows ([`Reach`]): the
//! sandbox denied the access, the path did not exist or nothing came back, or
//! something was returned. The classes are reported separately and none is
//! dropped; whether a denied attempt still excludes a trajectory is the
//! caller's rule, not this reader's.

use atif::TraceStep;
use serde_json::Value;

/// The key of the trajectory's agent `extra` that names the episode's tree.
pub const WORKSPACE_TREE_KEY: &str = "workspace_tree";

/// Directories every command reads from.
const SYSTEM_PREFIXES: [&str; 9] = [
    "/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/dev", "/proc", "/opt",
];

/// The argument names that carry a command or a path.
const PATH_ARGUMENTS: [&str; 9] = [
    "command",
    "shell_command",
    "cmd",
    "path",
    "file_path",
    "filename",
    "dir",
    "directory",
    "cwd",
];

/// The absolute paths in `text` (after a space, quote, `=`, `(` or `:`),
/// not `//` (a URL) and not `/*`, `/` alone or a division.
fn absolute_paths(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let mut found = Vec::new();
    for (at, byte) in bytes.iter().enumerate() {
        let opens = *byte == b'/'
            && (at == 0
                || matches!(
                    bytes[at - 1],
                    b' ' | b'\t' | b'\n' | b'"' | b'\'' | b'=' | b'(' | b':' | b'`'
                ))
            && bytes
                .get(at + 1)
                .is_some_and(|n| n.is_ascii_alphanumeric() || matches!(n, b'_' | b'.' | b'-'));
        if !opens {
            if *byte == b'/' && is_search_root(text, at) {
                found.push("/");
            }
            continue;
        }
        let end = text[at..]
            .find(|c: char| c.is_whitespace() || "\"'`;|&)<>\\,".contains(c))
            .map_or(text.len(), |n| at + n);
        found.push(&text[at..end]);
    }
    found
}

/// Whether the `/` at `at` stands alone as the argument of a command that
/// searches or enters a directory (`find / -name x`, `ls /`): the whole
/// filesystem. A lone `/` elsewhere is a division.
fn is_search_root(text: &str, at: usize) -> bool {
    const SEARCHERS: [&str; 10] = [
        "find", "ls", "cd", "tree", "du", "rg", "grep", "-C", "-r", "-R",
    ];
    let alone = text[at + 1..]
        .chars()
        .next()
        .is_none_or(char::is_whitespace);
    let before = text[..at].trim_end().rsplit(char::is_whitespace).next();
    alone
        && text[..at].ends_with(char::is_whitespace)
        && before.is_some_and(|w| SEARCHERS.contains(&w))
}

fn is_system(path: &str) -> bool {
    path == "/etc"
        || path.starts_with("/etc/")
        || SYSTEM_PREFIXES
            .iter()
            .any(|p| path == *p || path.strip_prefix(p).is_some_and(|r| r.starts_with('/')))
}

/// The `.../tree` directory `path` lies in, if it names one.
fn tree_of(path: &str) -> Option<&str> {
    let at = path.find("/tree")?;
    let end = at + "/tree".len();
    (path.len() == end || path[end..].starts_with('/')).then(|| &path[..end])
}

/// The strings of a tool call's command and path arguments.
fn call_texts(arguments: &Value) -> Vec<&str> {
    let Value::Object(map) = arguments else {
        return Vec::new();
    };
    PATH_ARGUMENTS
        .iter()
        .filter_map(|name| map.get(*name)?.as_str())
        .collect()
}

/// What came of a step that named a path outside the tree, as its output
/// shows it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Reach {
    /// Every line of the output is a refusal of the sandbox.
    Denied,
    /// The output is empty or every line says the path does not exist.
    Absent,
    /// The output holds something else: an access that may have succeeded.
    Reached,
}

impl Reach {
    /// The kind a hit of this class is reported under.
    pub(crate) fn kind(self) -> &'static str {
        match self {
            Self::Denied => "outside_workspace_denied",
            Self::Absent => "outside_workspace_absent",
            Self::Reached => "outside_workspace",
        }
    }
}

const DENIALS: [&str; 3] = [
    "permission denied",
    "operation not permitted",
    "outside the project root",
];
const ABSENCES: [&str; 2] = ["no such file or directory", "does not exist"];

/// The class of a step's output: a refusal only when every non-blank line is
/// one, so output that mixes a refusal with a result counts as reached.
fn reach_of(output: &str) -> Reach {
    let mut refused = false;
    let mut absent = false;
    for line in output.lines().map(str::trim).filter(|l| !l.is_empty()) {
        let lower = line.to_ascii_lowercase();
        if DENIALS.iter().any(|d| lower.contains(d)) {
            refused = true;
        } else if ABSENCES.iter().any(|a| lower.contains(a)) {
            absent = true;
        } else {
            return Reach::Reached;
        }
    }
    match (refused, absent) {
        (true, _) => Reach::Denied,
        (false, _) => Reach::Absent,
    }
}

/// The text a step's observation holds.
fn output_of(step: &TraceStep) -> String {
    step.observation
        .iter()
        .flat_map(|o| &o.results)
        .filter_map(|r| r.content.as_ref())
        .map(|c| match c {
            atif::MessageBody::Text(t) => t.clone(),
            atif::MessageBody::Segments(segments) => segments
                .iter()
                .filter_map(|s| match s {
                    atif::ContentSegment::Text { text } => Some(text.as_str()),
                    atif::ContentSegment::Image { .. } => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The steps, with the count of paths in each and what came of them, whose
/// tool calls name an absolute path outside `own_tree` and outside the
/// system.
pub(crate) fn outside_workspace(
    steps: &[TraceStep],
    recorded_tree: Option<&str>,
) -> Vec<(u64, usize, Reach)> {
    let named: Vec<(u64, Vec<&str>)> = steps
        .iter()
        .map(|step| {
            let paths = step
                .tool_calls
                .iter()
                .flatten()
                .flat_map(|call| call_texts(&call.arguments))
                .flat_map(absolute_paths)
                .collect();
            (step.step_id, paths)
        })
        .collect();
    let own = recorded_tree.map(str::to_string).or_else(|| {
        named
            .iter()
            .flat_map(|(_, paths)| paths)
            .find_map(|p| tree_of(p))
            .map(str::to_string)
    });
    let outputs: std::collections::HashMap<u64, Reach> = steps
        .iter()
        .map(|step| (step.step_id, reach_of(&output_of(step))))
        .collect();
    named
        .into_iter()
        .filter_map(|(step, paths)| {
            let outside = paths
                .iter()
                .filter(|p| !is_system(p))
                .filter(|p| {
                    !own.as_deref().is_some_and(|tree| {
                        p.strip_prefix(tree)
                            .is_some_and(|r| r.is_empty() || r.starts_with('/'))
                    })
                })
                .count();
            (outside > 0).then(|| (step, outside, outputs[&step]))
        })
        .collect()
}
