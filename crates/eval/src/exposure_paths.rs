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
//! Every hit is classed by what its own call's output shows ([`Reach`]): the
//! sandbox denied the access, the path did not exist or nothing came back, or
//! something was returned. A step that made several calls at once is read call
//! by call (a result belongs to the call at its position; only a step whose
//! results do not pair with its calls is read as a whole), and the shell
//! tool's own `[exit code N]` line is not a result. The classes are reported
//! separately and none is dropped; whether a denied attempt still excludes a
//! trajectory is the caller's rule, not this reader's.

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
    /// Whether a hit of this class is exposure: only an access that returned
    /// something may have read material the solver was not given. A refused
    /// or empty probe is recorded and does not fail the audit.
    pub(crate) fn exposes(self) -> bool {
        self == Self::Reached
    }

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
    for line in output
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !is_exit_code_line(l))
    {
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

/// The text of one observation result.
fn text_of(result: &atif::ObservationEntry) -> String {
    match result.content.as_ref() {
        None => String::new(),
        Some(atif::MessageBody::Text(t)) => t.clone(),
        Some(atif::MessageBody::Segments(segments)) => segments
            .iter()
            .filter_map(|s| match s {
                atif::ContentSegment::Text { text } => Some(text.as_str()),
                atif::ContentSegment::Image { .. } => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

/// Whether `output` says the access failed (refused, or no such path) and
/// returned nothing else: a call with such a result read nothing.
pub(crate) fn returned_only_a_failure(output: &str) -> bool {
    output
        .lines()
        .any(|l| !l.trim().is_empty() && !is_exit_code_line(l.trim()))
        && reach_of(output) != Reach::Reached
}

/// The shell tool's own trailer, `[exit code N]`: it says how the command
/// ended, it is not something the command returned.
fn is_exit_code_line(line: &str) -> bool {
    line.strip_prefix("[exit code ")
        .and_then(|rest| rest.strip_suffix(']'))
        .is_some_and(|code| {
            let digits = code.strip_prefix('-').unwrap_or(code);
            !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())
        })
}

/// The output each call of `step` returned, by position, when the step's
/// results pair with its calls one to one; else the whole step's output for
/// every call, which is the most that can be said of it.
pub(crate) fn outputs_by_call(step: &TraceStep) -> Vec<String> {
    let calls = step.tool_calls.as_deref().map_or(0, <[_]>::len);
    let results: Vec<String> = step
        .observation
        .iter()
        .flat_map(|o| &o.results)
        .map(text_of)
        .collect();
    if results.len() == calls {
        results
    } else {
        vec![results.join("\n"); calls]
    }
}

/// The calls, with the count of paths in each and what came of them (what
/// that call's own output shows), that name an absolute path outside
/// `own_tree` and outside the system, as `(step, outside paths, reach)`.
pub(crate) fn outside_workspace(
    steps: &[TraceStep],
    recorded_tree: Option<&str>,
) -> Vec<(u64, usize, Reach)> {
    let own = recorded_tree.map(str::to_string).or_else(|| {
        steps
            .iter()
            .flat_map(|step| step.tool_calls.iter().flatten())
            .flat_map(|call| call_texts(&call.arguments))
            .flat_map(absolute_paths)
            .find_map(tree_of)
            .map(str::to_string)
    });
    let inside_own = |path: &str| {
        own.as_deref().is_some_and(|tree| {
            path.strip_prefix(tree)
                .is_some_and(|r| r.is_empty() || r.starts_with('/'))
        })
    };
    let mut hits = Vec::new();
    for step in steps {
        let outputs = outputs_by_call(step);
        for (call, output) in step.tool_calls.iter().flatten().zip(&outputs) {
            let outside = call_texts(&call.arguments)
                .into_iter()
                .flat_map(absolute_paths)
                .filter(|p| !is_system(p) && !inside_own(p))
                .count();
            if outside > 0 {
                hits.push((step.step_id, outside, reach_of(output)));
            }
        }
    }
    hits
}
