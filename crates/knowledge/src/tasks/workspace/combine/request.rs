// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements executable evaluation of code changes
// against a project's own test suite, for its clients. If your team needs
// expertise in test-driven benchmarks for code-editing agents, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The request of a combined task: the requests of its members read apart
//! and written as one.
//!
//! A single task's request lists the symptoms its hidden tests name, the
//! order to fix them in and the command that runs the tests in the tree. The
//! combined request lists every member's symptoms, says that several
//! independent faults cause them, and runs the tests with the command the
//! combined tree can run: the arguments every member's request shows, so a
//! test module any member hid is not named. It names no file the members did
//! not name, and the members name none.

use std::collections::BTreeSet;

use super::super::instruction::{symptoms_of, Request};

/// Opens the list of symptoms in a request of the original wording.
const LIST_OPENS: &str = "each names a symptom:\n\n";

/// Precedes the test command in a request of the original wording.
const COMMAND_OPENS: &str = "they are run with `";

/// The prefix of the line that counts the symptoms a request leaves unlisted.
const MORE_PREFIX: &str = "- and ";

/// The suffix of that line.
const MORE_SUFFIX: &str = " more checks of the same behaviour";

/// What one request says: the symptoms it lists, how many it leaves
/// unlisted, and the arguments of the command it runs the tests with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Symptoms {
    pub(super) lines: BTreeSet<String>,
    pub(super) unlisted: usize,
    pub(super) command: Vec<String>,
}

impl Symptoms {
    /// What `request` says; `None` when it is not worded as the original
    /// wording of a single task's request is, which is the only wording
    /// that can be taken apart and put together again.
    pub(super) fn read(request: &str) -> Option<Self> {
        if !request.contains(LIST_OPENS) {
            return None;
        }
        let mut lines = BTreeSet::new();
        let mut unlisted = 0;
        for line in symptoms_of(request).lines().filter(|l| l.starts_with("- ")) {
            match more(line) {
                Some(count) => unlisted += count,
                None => {
                    lines.insert(line.to_string());
                }
            }
        }
        let (_, after) = request.rsplit_once(COMMAND_OPENS)?;
        let (command, _) = after.split_once('`')?;
        Some(Self {
            lines,
            unlisted,
            command: command.split_whitespace().map(String::from).collect(),
        })
    }
}

/// How many symptoms a "and N more checks" line leaves unlisted.
fn more(line: &str) -> Option<usize> {
    line.strip_prefix(MORE_PREFIX)?
        .strip_suffix(MORE_SUFFIX)?
        .parse()
        .ok()
}

/// The request for the faults `parts` are the requests of.
pub(super) fn combined(parts: &[&Symptoms]) -> Request {
    let lines: BTreeSet<&String> = parts.iter().flat_map(|p| &p.lines).collect();
    let unlisted: usize = parts.iter().map(|p| p.unlisted).sum();
    let mut listed: Vec<String> = lines.iter().map(|line| (*line).clone()).collect();
    if unlisted > 0 {
        listed.push(format!("{MORE_PREFIX}{unlisted}{MORE_SUFFIX}"));
    }
    let command = shared_command(parts).join(" ");
    let text = format!(
        "{} independent faults are in this repository, in different places, and each breaks \
         behaviour of its own. The tests that check it are not in the tree you were given; they \
         fail on it, and {LIST_OPENS}{}\n\nFind the causes in the repository's code and fix them \
         so that this behaviour works again: fixing one fault does not fix the others. The tests \
         that are in the tree must keep passing and are not to be edited; they are run with \
         `{command}`.",
        parts.len(),
        listed.join("\n"),
    );
    // The values a symptom quotes are the one disclosure a request makes on
    // purpose; they sit between backticks.
    let shown = lines
        .iter()
        .flat_map(|line| line.split('`').skip(1).step_by(2))
        .map(str::to_string)
        .collect();
    Request { text, shown }
}

/// The arguments of the first command that every command shows, in order.
fn shared_command(parts: &[&Symptoms]) -> Vec<String> {
    let Some((first, rest)) = parts.split_first() else {
        return Vec::new();
    };
    first
        .command
        .iter()
        .filter(|arg| rest.iter().all(|p| p.command.contains(arg)))
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn single(symptoms: &[&str], more: usize, modules: &[&str]) -> String {
        let mut lines: Vec<String> = symptoms.iter().map(|s| format!("- {s}")).collect();
        if more > 0 {
            lines.push(format!("{MORE_PREFIX}{more}{MORE_SUFFIX}"));
        }
        let command = ["python3", "-m", "unittest"]
            .into_iter()
            .chain(modules.iter().copied())
            .collect::<Vec<_>>()
            .join(" ");
        format!(
            "Some behaviour of this repository is broken. The tests that check it are not in the \
             tree you were given; they fail on it, and {LIST_OPENS}{}\n\nFind the cause in the \
             repository's code and fix it so that this behaviour works again. The tests that are \
             in the tree must keep passing and are not to be edited; they are run with \
             `{command}`.",
            lines.join("\n")
        )
    }

    #[test]
    fn a_request_is_read_apart_into_its_symptoms_its_unlisted_count_and_its_command() {
        let read = Symptoms::read(&single(&["a b c", "d e f"], 3, &["tests.keep"])).unwrap();
        assert_eq!(read.lines.len(), 2);
        assert_eq!(read.unlisted, 3);
        assert_eq!(read.command, ["python3", "-m", "unittest", "tests.keep"]);
        assert_eq!(Symptoms::read("Fix the bug."), None);
    }

    #[test]
    fn the_combined_request_lists_every_symptom_and_runs_only_the_modules_every_member_kept() {
        let one = Symptoms::read(&single(&["a b c"], 2, &["tests.one", "tests.shared"])).unwrap();
        let other = Symptoms::read(&single(&["a b c", "d e f"], 1, &["tests.shared"])).unwrap();
        let request = combined(&[&one, &other]);
        assert!(
            request.text.starts_with("2 independent faults"),
            "{}",
            request.text
        );
        assert!(request
            .text
            .contains("- a b c\n- d e f\n- and 3 more checks"));
        assert!(request
            .text
            .ends_with("`python3 -m unittest tests.shared`."));
        // The combined request reads back as a request of the same shape.
        let back = Symptoms::read(&request.text).unwrap();
        assert_eq!(back.lines.len(), 2);
        assert_eq!(back.unlisted, 3);
    }

    #[test]
    fn the_values_a_symptom_quotes_are_the_ones_the_request_shows_on_purpose() {
        let one = Symptoms::read(&single(
            &["cast: two values that must be equal are `1` and `2`"],
            0,
            &[],
        ))
        .unwrap();
        let request = combined(&[&one, &one]);
        assert_eq!(request.shown, ["1", "2"]);
    }
}
