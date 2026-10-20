// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Where a piece of new content came from: the task, or something the agent
//! had seen.
//!
//! An edit that writes a name, a number or a string neither the plan nor any
//! earlier tool result holds invents it, or knew it from somewhere it should
//! not have: a student trained on that record learns to invent. The attempt
//! then fails as `unjustified_content` and the plan is to blame - it was
//! under-specified.
//!
//! What counts as content:
//! * identifier-shaped words of four or more characters
//!   ([`crate::tree::identifiers`]), compared without regard to letter case;
//! * numbers of two or more digits, and hexadecimal numbers;
//! * double-quoted string literals with at least three letters, which must
//!   appear verbatim (their words are then covered by the same text).
//!
//! What counts as support: the plan, and the result of every earlier call that
//! is not an edit, the todo list or a write (a list the agent wrote itself, or
//! the echo of its own edit, shows nothing the agent was told), and the new
//! content of the agent's own earlier edits (judged when it was made).

use std::collections::HashSet;

use regex::Regex;

use crate::trace::Call;
use crate::tree::{dashed_names, identifiers};

/// Tools whose arguments write file content.
pub const EDIT_TOOLS: &[&str] = &["edit_file", "write_file"];

/// Tools whose results show the agent nothing new.
const ECHO_TOOLS: &[&str] = &["edit_file", "write_file", "todo"];

/// What new content was found without support.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unjustified {
    /// The agent step that wrote it.
    pub step: u64,
    /// The tool that wrote it.
    pub tool: String,
    /// The name, number or string.
    pub text: String,
}

fn re(pattern: &str) -> Regex {
    Regex::new(pattern).unwrap_or_else(|_| unreachable!("a fixed pattern"))
}

fn squeeze(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The numbers of `text` that count as content (two or more digits, or hexadecimal).
pub fn numbers(text: &str) -> Vec<String> {
    re(r"\b0[xX][0-9A-Fa-f_]+\b|\b\d[\d_]*(?:\.\d+)?\b")
        .find_iter(text)
        .map(|m| m.as_str().to_lowercase())
        .filter(|n| n.chars().filter(char::is_ascii_alphanumeric).count() >= 2)
        .collect()
}

/// The strings of `text` that count as content, and `text` without them.
fn strings(text: &str) -> (Vec<String>, String) {
    let quoted = re(r#""([^"\n\\]*(?:\\.[^"\n\\]*)*)""#);
    let mut found = Vec::new();
    for caps in quoted.captures_iter(text) {
        let inner = squeeze(&caps[1]);
        if inner.chars().filter(|c| c.is_alphabetic()).count() >= 3 {
            found.push(inner);
        }
    }
    (found, quoted.replace_all(text, " ").into_owned())
}

/// What the agent was told, grown as the run goes on.
#[derive(Clone, Debug, Default)]
pub struct Support {
    words: HashSet<String>,
    verbatim: String,
}

impl Support {
    /// Support from `text` (a plan, or a tool result).
    #[must_use]
    pub fn of(text: &str) -> Self {
        let mut support = Self::default();
        support.add(text);
        support
    }

    /// Adds what `text` shows.
    pub fn add(&mut self, text: &str) {
        self.words.extend(identifiers(text).map(str::to_lowercase));
        self.words
            .extend(dashed_names(text).into_iter().map(|n| n.to_lowercase()));
        self.words.extend(numbers(text));
        self.verbatim.push_str(&squeeze(text));
        self.verbatim.push('\n');
    }

    /// The names and numbers of `text` that nothing seen supports (prose: no
    /// string literals are taken out of it).
    #[must_use]
    pub fn missing_words(&self, text: &str) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for word in identifiers(text).map(str::to_string).chain(numbers(text)) {
            if !self.words.contains(&word.to_lowercase()) && !out.contains(&word) {
                out.push(word);
            }
        }
        out
    }

    /// Whether `word` (a name or a number) is supported.
    #[must_use]
    pub fn has(&self, word: &str) -> bool {
        self.words.contains(&word.to_lowercase())
    }

    /// The strings, names and numbers of `added` (the new text of an edit)
    /// that nothing seen supports.
    #[must_use]
    pub fn missing(&self, added: &str) -> Vec<String> {
        let (strings, rest) = strings(added);
        let mut out: Vec<String> = strings
            .into_iter()
            .filter(|s| !self.verbatim.contains(s.as_str()))
            .map(|s| format!("{s:?}"))
            .collect();
        out.extend(self.missing_words(&rest));
        out
    }
}

/// Whether `word` is spelled like a name in code (`snake_case`, `camelCase`,
/// or with digits) and not like a word of prose.
#[must_use]
pub fn is_code_shaped(word: &str) -> bool {
    word.contains('_')
        || word.chars().any(|c| c.is_ascii_digit())
        || word
            .chars()
            .zip(word.chars().skip(1))
            .any(|(a, b)| a.is_lowercase() && b.is_uppercase())
}

/// The names and numbers of prose `text` that stand for code: words spelled
/// like code, anything inside backticks, and numbers.
#[must_use]
pub fn code_names(text: &str) -> Vec<String> {
    let ticks = re(r"`([^`\n]+)`");
    let mut out: Vec<String> = Vec::new();
    let mut push = |w: &str| {
        if !out.iter().any(|o| o == w) {
            out.push(w.to_string());
        }
    };
    for span in ticks.captures_iter(text) {
        identifiers(&span[1]).for_each(&mut push);
    }
    identifiers(text)
        .filter(|w| is_code_shaped(w))
        .for_each(&mut push);
    numbers(text).iter().for_each(|n| push(n));
    out
}

/// The lines an edit call adds: the `+` lines of an `edit_file` diff, or the
/// whole text of a `write_file`.
#[must_use]
pub fn added_text(call: &Call) -> Option<String> {
    match call.tool.as_str() {
        "edit_file" => call.arg("diff").map(|d| {
            d.lines()
                .filter(|l| !l.starts_with("+++"))
                .filter_map(|l| l.strip_prefix('+'))
                .collect::<Vec<_>>()
                .join("\n")
        }),
        "write_file" => call.arg("text").map(String::from),
        _ => None,
    }
}

/// What the run so far has told the agent: the plan, every earlier result that
/// shows something, the content of its own edits and the user's messages.
#[must_use]
pub fn support_of_run(plan: &str, calls: &[Call], guidance: &[(usize, String)]) -> Support {
    let mut support = Support::of(plan);
    for (index, call) in calls.iter().enumerate() {
        for (_, message) in guidance.iter().filter(|(at, _)| *at == index) {
            support.add(message);
        }
        if let Some(added) = added_text(call) {
            support.add(&added);
        }
        if !ECHO_TOOLS.contains(&call.tool.as_str()) {
            support.add(&call.result);
        }
    }
    for (_, message) in guidance.iter().filter(|(at, _)| *at >= calls.len()) {
        support.add(message);
    }
    support
}

/// Every piece of new content in `calls` that neither `plan`, an earlier
/// result nor an earlier message of the user (`guidance`: the message and the
/// number of calls made when it came) supports, in order.
#[must_use]
pub fn unjustified(plan: &str, calls: &[Call], guidance: &[(usize, String)]) -> Vec<Unjustified> {
    let mut support = Support::of(plan);
    let mut out = Vec::new();
    for (index, call) in calls.iter().enumerate() {
        for (_, message) in guidance.iter().filter(|(at, _)| *at == index) {
            support.add(message);
        }
        if let Some(added) = added_text(call) {
            for text in support.missing(&added) {
                out.push(Unjustified {
                    step: call.step,
                    tool: call.tool.clone(),
                    text,
                });
            }
            // Judged once: the same name used again is not reported again.
            support.add(&added);
        }
        if !ECHO_TOOLS.contains(&call.tool.as_str()) {
            support.add(&call.result);
        }
    }
    out
}
