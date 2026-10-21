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
//! What a call does with a path decides whether naming it is looking:
//!
//! * a **read** (any argument that is not one of the cases below) is classed
//!   by what its own call's output shows ([`Reach`]): the sandbox denied the
//!   access, the path did not exist or nothing came back but a failure
//!   report, or something was returned;
//! * a **write** (a redirect target, `tee`, the write tool's path) reads
//!   nothing, and the file is the solver's own scratch from then on: reading
//!   it back is not looking elsewhere;
//! * a **directory change** (`cd`) reads nothing either; what a command reads
//!   after it is judged by the paths it names, and hidden material by the
//!   hidden-path and content checks of the audit. A change into another
//!   attempt's tree (a `.../tree` directory that is not the episode's own) is
//!   a read;
//! * a path inside a here-document is program text: it counts only as the
//!   argument of a call (`open("/x")`), not as data (`{"root": "/x"}`);
//! * a **search of the whole filesystem** (`find /`, `grep -r ... /`) is the
//!   behaviour the audit exists to keep out of training data, and is reported
//!   whatever it returned.
//!
//! A step that made several calls at once is read call by call: a result
//! belongs to the call whose id it names (results arrive in the order the
//! calls finished, not the order they were made); only a step whose results
//! cannot be paired is read as a whole. The shell tool's own `[exit code N]`
//! line is not a result. The classes are reported separately and none is
//! dropped; whether a class excludes a trajectory is [`Reach::exposes`].

use std::collections::HashSet;

use atif::{ToolInvocation, TraceStep};
use serde_json::Value;
use splinter_core::tool_calls::outputs_by_call;

pub use splinter_core::prompt::WORKSPACE_TREE_KEY;

/// Directories every command reads from.
const SYSTEM_PREFIXES: [&str; 9] = [
    "/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/dev", "/proc", "/opt",
];

/// The arguments that carry a shell command.
const COMMAND_ARGUMENTS: [&str; 3] = ["command", "shell_command", "cmd"];

/// The arguments that carry a path.
const PATH_ARGUMENTS: [&str; 6] = ["path", "file_path", "filename", "dir", "directory", "cwd"];

/// The tool that writes a whole file: its path argument is a write target.
const WRITE_TOOL: &str = "write_file";

/// What a call does with an absolute path it names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Use {
    /// It may read it.
    Read,
    /// It creates or overwrites it.
    Write,
    /// It makes it the working directory.
    Enter,
}

/// An absolute path a call names, and what the call does with it.
struct Named<'a> {
    path: &'a str,
    usage: Use,
}

/// The absolute paths in `text` and where each starts (after a space, quote,
/// `=`, `(` or `:`), not `//` (a URL) and not `/*`, `/` alone or a division.
fn absolute_paths(text: &str) -> Vec<(usize, &str)> {
    let bytes = text.as_bytes();
    let mut found = Vec::new();
    for (at, byte) in bytes.iter().enumerate() {
        let opens = *byte == b'/'
            && (at == 0
                || matches!(
                    bytes[at - 1],
                    b' ' | b'\t' | b'\n' | b'=' | b'(' | b':' | b'`'
                )
                || (matches!(bytes[at - 1], b'"' | b'\'') && opens_quote(bytes, at - 1)))
            && bytes
                .get(at + 1)
                .is_some_and(|n| n.is_ascii_alphanumeric() || matches!(n, b'_' | b'.' | b'-'));
        if !opens {
            if *byte == b'/' && is_search_root(text, at) {
                found.push((at, "/"));
            }
            continue;
        }
        let end = text[at..]
            .find(|c: char| c.is_whitespace() || "\"'`;|&)<>\\,".contains(c))
            .map_or(text.len(), |n| at + n);
        found.push((at, &text[at..end]));
    }
    found
}

/// Whether the quote at `at` opens a string rather than closes one: what
/// closes a quote follows the text it quotes (`'$(pwd)'/src`), what opens one
/// follows a separator.
fn opens_quote(bytes: &[u8], at: usize) -> bool {
    at == 0
        || matches!(
            bytes[at - 1],
            b' ' | b'\t'
                | b'\n'
                | b'='
                | b'('
                | b','
                | b':'
                | b'['
                | b'{'
                | b';'
                | b'|'
                | b'&'
                | b'<'
                | b'>'
                | b'`'
        )
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

/// A shell script split into the lines the shell runs and the bodies of its
/// here-documents, which are text handed to a program.
struct Script<'a> {
    code: Vec<&'a str>,
    data: Vec<&'a str>,
}

impl<'a> Script<'a> {
    fn of(text: &'a str) -> Self {
        let mut script = Self {
            code: Vec::new(),
            data: Vec::new(),
        };
        let mut closing: Option<&str> = None;
        for line in text.lines() {
            match closing {
                Some(delimiter) if line.trim() == delimiter => closing = None,
                Some(_) => script.data.push(line),
                None => {
                    script.code.push(line);
                    closing = heredoc_delimiter(line);
                }
            }
        }
        script
    }
}

/// The word that ends the here-document `line` opens (`<<'EOF'`, `<<-EOF`),
/// not a here-string (`<<<`).
fn heredoc_delimiter(line: &str) -> Option<&str> {
    let after = &line[line.find("<<")? + 2..];
    if after.starts_with('<') {
        return None;
    }
    let word = after
        .trim_start_matches('-')
        .trim_start()
        .trim_start_matches(['\'', '"']);
    let end = word
        .find(|c: char| !(c.is_alphanumeric() || c == '_'))
        .unwrap_or(word.len());
    (end > 0).then(|| &word[..end])
}

/// What the shell does with the path that starts at `at` in `line`.
fn shell_use(line: &str, at: usize) -> Use {
    let before = line[..at].trim_end();
    if before.ends_with('>') {
        return Use::Write;
    }
    match before
        .split_whitespace()
        .rev()
        .find(|word| !word.starts_with('-'))
    {
        Some("tee") => Use::Write,
        Some("cd" | "pushd") => Use::Enter,
        _ => Use::Read,
    }
}

/// The paths a here-document body names as the argument of a call: program
/// text that opens `"/x"` looks there, a table that holds `"/x"` does not.
fn called_with(line: &str) -> Vec<&str> {
    absolute_paths(line)
        .into_iter()
        .filter(|(at, _)| line[..at.saturating_sub(1)].trim_end().ends_with('('))
        .map(|(_, path)| path)
        .collect()
}

/// The absolute paths a call names and what it does with each.
fn named_paths(call: &ToolInvocation) -> Vec<Named<'_>> {
    let Value::Object(map) = &call.arguments else {
        return Vec::new();
    };
    let mut named = Vec::new();
    for text in COMMAND_ARGUMENTS
        .iter()
        .filter_map(|n| map.get(*n)?.as_str())
    {
        let script = Script::of(text);
        for line in &script.code {
            named.extend(absolute_paths(line).into_iter().map(|(at, path)| Named {
                path,
                usage: shell_use(line, at),
            }));
        }
        for line in &script.data {
            named.extend(called_with(line).into_iter().map(|path| Named {
                path,
                usage: Use::Read,
            }));
        }
    }
    let usage = if call.function_name == WRITE_TOOL {
        Use::Write
    } else {
        Use::Read
    };
    for text in PATH_ARGUMENTS.iter().filter_map(|n| map.get(*n)?.as_str()) {
        named.extend(
            absolute_paths(text)
                .into_iter()
                .map(|(_, path)| Named { path, usage }),
        );
    }
    named
}

/// The commands that walk a directory tree from the root they are given.
const WALKERS: [&str; 8] = ["find", "fd", "locate", "tree", "du", "rg", "grep", "egrep"];

/// Whether a command of `call` searches the whole filesystem from `/`:
/// `find / -name x`, `grep -rl x /`. Program text is not a command.
fn searches_filesystem_root(call: &ToolInvocation) -> bool {
    let Value::Object(map) = &call.arguments else {
        return false;
    };
    COMMAND_ARGUMENTS
        .iter()
        .filter_map(|n| map.get(*n)?.as_str())
        .flat_map(|text| Script::of(text).code)
        .flat_map(|line| line.split([';', '|', '&']))
        .any(|segment| {
            let words: Vec<&str> = segment.split_whitespace().collect();
            let walks = words.first().is_some_and(|w| WALKERS.contains(w))
                && (!matches!(words[0], "grep" | "egrep")
                    || words.iter().any(|w| {
                        *w == "--recursive"
                            || (w.starts_with('-')
                                && !w.starts_with("--")
                                && w.contains(['r', 'R']))
                    }));
            walks && words.iter().skip(1).any(|w| *w == "/")
        })
}

/// What came of a step that named a path outside the tree, as its output
/// shows it, or what the call did instead of reading.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Reach {
    /// Every line of the output is a refusal of the sandbox.
    Denied,
    /// The output is empty, says the path does not exist, or is only a
    /// failure report (a traceback).
    Absent,
    /// The output holds something else: an access that may have succeeded.
    Reached,
    /// The call only wrote a file there, which reads nothing.
    Written,
    /// The call only changed its working directory there, which reads nothing.
    Entered,
    /// The call searched the whole filesystem, whatever came back.
    Searched,
}

impl Reach {
    /// Whether a hit of this class is exposure: an access that returned
    /// something may have read material the solver was not given, and a
    /// search of the whole filesystem is the behaviour the audit keeps out of
    /// training data. A refused or empty probe, a write and a directory change
    /// are recorded and do not fail the audit.
    pub(crate) fn exposes(self) -> bool {
        matches!(self, Self::Reached | Self::Searched)
    }

    /// The kind a hit of this class is reported under.
    pub(crate) fn kind(self) -> &'static str {
        match self {
            Self::Denied => "outside_workspace_denied",
            Self::Absent => "outside_workspace_absent",
            Self::Reached => "outside_workspace",
            Self::Written => "outside_workspace_write",
            Self::Entered => "outside_workspace_cd",
            Self::Searched => "outside_workspace_search",
        }
    }
}

const DENIALS: [&str; 3] = [
    "permission denied",
    "operation not permitted",
    "outside the project root",
];
const ABSENCES: [&str; 3] = [
    "no such file or directory",
    "does not exist",
    "command not found",
];

/// Whether `line` belongs to an interpreter's report of a failure rather
/// than to anything a command returned: a Python traceback (its header, its
/// frames with the source line and markers under them, its final exception
/// line) or a Lua one (`lua: ...`, `stack traceback:`, its frames and the
/// list of files a missing module was looked for in).
fn is_failure_report(line: &str, previous: Option<&str>, next: Option<&str>) -> bool {
    let text = line.trim();
    let frame = |l: &str| l.trim().starts_with("File \"") && l.contains(", line ");
    let markers = |l: &str| !l.trim().is_empty() && l.trim().chars().all(|c| "~^".contains(c));
    let exception = text.split_once(": ").is_some_and(|(name, _)| {
        ["Error", "Exception", "Exit", "Interrupt"]
            .iter()
            .any(|suffix| name.ends_with(suffix))
            && name
                .chars()
                .all(|c| c.is_alphanumeric() || c == '_' || c == '.')
    });
    let lua_frame = line.starts_with('\t')
        && (text.starts_with("no field")
            || text.starts_with("no file")
            || text.starts_with("[C]:")
            || text.contains(": in "));
    let echo = line.starts_with(' ') && (previous.is_some_and(frame) || next.is_some_and(markers));
    text.starts_with("Traceback (most recent call last)")
        || text.starts_with("stack traceback:")
        || text.starts_with("lua: ")
        || frame(line)
        || markers(line)
        || exception
        || lua_frame
        || echo
}

/// The class of a step's output: a refusal only when every line that is not a
/// failure report is one, so output that mixes a refusal with a result
/// counts as reached.
fn reach_of(output: &str) -> Reach {
    let lines: Vec<&str> = output.lines().collect();
    let mut refused = false;
    for (at, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || is_exit_code_line(trimmed) {
            continue;
        }
        let lower = trimmed.to_ascii_lowercase();
        if DENIALS.iter().any(|d| lower.contains(d)) {
            refused = true;
        } else if ABSENCES.iter().any(|a| lower.contains(a))
            || is_failure_report(
                line,
                at.checked_sub(1).map(|p| lines[p]),
                lines.get(at + 1).copied(),
            )
        {
            continue;
        } else {
            return Reach::Reached;
        }
    }
    if refused {
        Reach::Denied
    } else {
        Reach::Absent
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
            .flat_map(named_paths)
            .find_map(|named| tree_of(named.path).map(str::to_string))
    });
    let inside_own = |path: &str| {
        own.as_deref().is_some_and(|tree| {
            path.strip_prefix(tree)
                .is_some_and(|r| r.is_empty() || r.starts_with('/'))
        })
    };
    // Files the solver wrote: its own scratch from then on.
    let mut scratch: HashSet<String> = HashSet::new();
    let mut hits = Vec::new();
    for step in steps {
        let outputs = outputs_by_call(step);
        for (call, output) in step.tool_calls.iter().flatten().zip(&outputs) {
            let outside: Vec<Named> = named_paths(call)
                .into_iter()
                .filter(|n| !is_system(n.path) && !inside_own(n.path))
                .collect();
            // A file written by this very call is read back as its own.
            scratch.extend(
                outside
                    .iter()
                    .filter(|n| n.usage == Use::Write)
                    .map(|n| n.path.to_string()),
            );
            let count = |wanted: Use| {
                outside
                    .iter()
                    .filter(|n| match n.usage {
                        // Another attempt's tree is a place to read.
                        Use::Enter if tree_of(n.path).is_some() => wanted == Use::Read,
                        usage => usage == wanted,
                    })
                    .filter(|n| wanted == Use::Write || !scratch.contains(n.path))
                    .count()
            };
            let (reads, entered, written) =
                (count(Use::Read), count(Use::Enter), count(Use::Write));
            let hit = if searches_filesystem_root(call) {
                Some((1, Reach::Searched))
            } else if reads > 0 {
                Some((reads, reach_of(output)))
            } else if entered > 0 {
                Some((entered, Reach::Entered))
            } else if written > 0 {
                Some((written, Reach::Written))
            } else {
                None
            };
            if let Some((paths, reach)) = hit {
                hits.push((step.step_id, paths, reach));
            }
        }
    }
    hits
}
