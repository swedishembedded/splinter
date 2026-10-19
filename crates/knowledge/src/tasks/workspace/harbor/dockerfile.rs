// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements interchange of agent benchmark tasks between
// task formats for its clients. If your team needs expertise in importing
// containerised benchmark tasks into reproducible, verifiable task sets, you
// can procure our services by sending an email to info@swedishembedded.com.

//! The part of a Harbor task's `environment/Dockerfile` the host can honour.
//!
//! A task whose environment is a repository checkout needs only `FROM`,
//! `WORKDIR`, `COPY` and a few declarations. Anything that executes during
//! the build (`RUN`), fetches (`ADD`), builds in stages or names an image
//! the host cannot unpack is a container build, and a task that needs one is
//! refused with that reason rather than imported with a different
//! environment than it declares.

use std::path::{Component, Path, PathBuf};

/// One `COPY` instruction, resolved against the `WORKDIR` in force.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Copy {
    /// Sources, relative to the environment directory.
    pub(super) sources: Vec<String>,
    /// The destination, absolute and lexically normalised.
    pub(super) destination: String,
    /// Whether the destination was written as a directory (a trailing `/`).
    pub(super) into_directory: bool,
}

/// What the host honours of a Dockerfile.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct Plan {
    /// The last `WORKDIR`, absolute.
    pub(super) workdir: Option<String>,
    /// The copies, in order.
    pub(super) copies: Vec<Copy>,
}

/// The instructions that change nothing a repository checkout is made of.
const HARMLESS: [&str; 8] = [
    "ARG",
    "ENV",
    "LABEL",
    "USER",
    "CMD",
    "ENTRYPOINT",
    "EXPOSE",
    "HEALTHCHECK",
];

/// The logical lines of `text`: comments dropped, `\` continuations joined.
fn lines(text: &str) -> Vec<String> {
    let mut logical = Vec::new();
    let mut open = String::new();
    for raw in text.lines() {
        let line = raw.trim();
        if open.is_empty() && (line.is_empty() || line.starts_with('#')) {
            continue;
        }
        if line.starts_with('#') {
            continue;
        }
        match line.strip_suffix('\\') {
            Some(head) => {
                open.push_str(head.trim_end());
                open.push(' ');
            }
            None => {
                open.push_str(line);
                logical.push(std::mem::take(&mut open));
            }
        }
    }
    if !open.trim().is_empty() {
        logical.push(open);
    }
    logical
}

/// `path` with `.` and `..` resolved lexically; `None` when it climbs out of
/// the root.
fn normalise(path: &str) -> Option<String> {
    let mut parts: Vec<&str> = Vec::new();
    for part in path.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                parts.pop()?;
            }
            other => parts.push(other),
        }
    }
    Some(format!("/{}", parts.join("/")))
}

/// Whether `path` is relative and stays under the directory it is joined to.
pub(super) fn stays_inside(path: &str) -> bool {
    Path::new(path)
        .components()
        .all(|c| matches!(c, Component::Normal(_) | Component::CurDir))
        && !path.starts_with('/')
}

/// The plan `text` describes, or why the host cannot honour it.
pub(super) fn parse(text: &str, configured_workdir: Option<&str>) -> Result<Plan, String> {
    let mut plan = Plan {
        workdir: configured_workdir.map(str::to_string),
        copies: Vec::new(),
    };
    let mut stages = 0;
    for line in lines(text) {
        let (instruction, rest) = line
            .split_once(char::is_whitespace)
            .map_or((line.as_str(), ""), |(i, r)| (i, r.trim()));
        let instruction = instruction.to_ascii_uppercase();
        match instruction.as_str() {
            "FROM" => {
                stages += 1;
                if stages > 1 {
                    return Err("the Dockerfile builds in stages".into());
                }
            }
            "WORKDIR" => {
                let current = plan.workdir.clone().unwrap_or_else(|| "/".into());
                let joined = if rest.starts_with('/') {
                    rest.to_string()
                } else {
                    format!("{current}/{rest}")
                };
                if rest.contains('$') {
                    return Err("WORKDIR uses a variable".into());
                }
                plan.workdir =
                    Some(normalise(&joined).ok_or_else(|| "WORKDIR leaves the root".to_string())?);
            }
            "COPY" => {
                if rest.starts_with("--") {
                    return Err("COPY has flags (--from, --chown, ...)".into());
                }
                if rest.starts_with('[') {
                    return Err("COPY uses the JSON form".into());
                }
                if rest.contains('$') {
                    return Err("COPY uses a variable".into());
                }
                let mut words: Vec<&str> = rest.split_whitespace().collect();
                if words.len() < 2 {
                    return Err("COPY needs a source and a destination".into());
                }
                let destination = words.pop().unwrap_or_default();
                if words.iter().any(|s| s.contains(['*', '?', '['])) {
                    return Err("COPY uses a wildcard source".into());
                }
                if let Some(bad) = words.iter().find(|s| !stays_inside(s)) {
                    return Err(format!("COPY source {bad:?} is not inside the environment"));
                }
                let base = plan.workdir.clone().unwrap_or_else(|| "/".into());
                let absolute = if destination.starts_with('/') {
                    destination.to_string()
                } else {
                    format!("{base}/{destination}")
                };
                plan.copies.push(Copy {
                    sources: words.iter().map(|s| (*s).to_string()).collect(),
                    destination: normalise(&absolute)
                        .ok_or_else(|| "a COPY destination leaves the root".to_string())?,
                    into_directory: destination.ends_with('/') || words.len() > 1,
                });
            }
            "RUN" => {
                return Err(
                    "the Dockerfile has RUN: the environment needs a container build".into(),
                )
            }
            "ADD" => return Err("the Dockerfile has ADD".into()),
            other if HARMLESS.contains(&other) => {}
            other => {
                return Err(format!(
                    "the Dockerfile has {other}, which the host cannot honour"
                ))
            }
        }
    }
    if stages == 0 {
        return Err("the Dockerfile has no FROM".into());
    }
    Ok(plan)
}

/// Where a destination lands in the tree: its path relative to `workdir`.
pub(super) fn under(workdir: &str, destination: &str) -> Result<PathBuf, String> {
    let root = workdir.trim_end_matches('/');
    let rest = if root.is_empty() {
        destination.trim_start_matches('/')
    } else {
        destination
            .strip_prefix(root)
            .filter(|r| r.is_empty() || r.starts_with('/'))
            .ok_or_else(|| format!("COPY destination {destination} is outside WORKDIR {workdir}"))?
            .trim_start_matches('/')
    };
    Ok(PathBuf::from(rest))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_repository_checkout_is_a_from_a_workdir_and_a_copy() {
        let plan = parse(
            "# env\nFROM ubuntu:24.04\nENV A=1\nWORKDIR /app\nCOPY repo/ /app/\nCOPY data.txt notes/\n",
            None,
        )
        .unwrap();
        assert_eq!(plan.workdir.as_deref(), Some("/app"));
        assert_eq!(plan.copies.len(), 2);
        assert_eq!(plan.copies[0].destination, "/app");
        assert_eq!(plan.copies[1].destination, "/app/notes");
        assert!(plan.copies[1].into_directory);
    }

    #[test]
    fn what_runs_fetches_or_stages_a_build_is_refused_with_its_name() {
        for (text, needle) in [
            ("FROM a\nRUN apt-get install x\n", "RUN"),
            ("FROM a\nADD https://x /y\n", "ADD"),
            ("FROM a\nFROM b\n", "stages"),
            ("FROM a\nCOPY --from=b /x /y\n", "flags"),
            ("FROM a\nCOPY *.py /app/\n", "wildcard"),
            ("FROM a\nCOPY ../x /app/\n", "inside"),
            ("FROM a\nWORKDIR /app\nCOPY . /etc/\n", "x"),
        ] {
            let result = parse(text, None).and_then(|plan| {
                let workdir = plan.workdir.clone().unwrap_or_default();
                for copy in &plan.copies {
                    under(&workdir, &copy.destination)?;
                }
                Ok(plan)
            });
            let why = result.expect_err(text);
            if needle != "x" {
                assert!(why.contains(needle), "{text}: {why}");
            }
        }
    }
}
