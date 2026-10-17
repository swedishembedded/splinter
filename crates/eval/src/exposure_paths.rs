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

/// The steps, with the count of paths in each, whose tool calls name an
/// absolute path outside `own_tree` and outside the system.
pub(crate) fn outside_workspace(
    steps: &[TraceStep],
    recorded_tree: Option<&str>,
) -> Vec<(u64, usize)> {
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
            (outside > 0).then_some((step, outside))
        })
        .collect()
}
