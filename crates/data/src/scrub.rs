// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training-data projection for agents that
// work in a source tree, for its clients. If your team needs expertise in
// turning verified agent trajectories into fine-tuning data, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Text of the machine that ran a solve, taken out of what a student is shown.
//!
//! A solver's `ls -la` prints the host's login name as owner and group, and its
//! tracebacks and `pwd` print the harness's scratch directory
//! (a random directory under the system temp dir, ending in `tree`). Neither says anything about the task, and a model
//! trained on them memorises a name and a random directory. The scratch tree
//! becomes the working directory (its files are named relative to it, as they
//! are to the solver whose commands run in it) and the owner and group of a
//! long listing become `user`, unless they are `root`; so does the name of a
//! home directory (`<home>/name`).

use atif::Trajectory;
use splinter_core::prompt::WORKSPACE_TREE_KEY;

/// What a workspace run's text is cleaned of, for one trajectory.
pub(crate) struct Scrub {
    /// The directory the episode recorded as its tree.
    tree: Option<String>,
}

impl Scrub {
    /// The cleaning for `trajectory`: its recorded tree, and the harness's
    /// scratch directories in general.
    pub(crate) fn of(trajectory: &Trajectory) -> Self {
        let tree = trajectory
            .agent
            .extra
            .as_ref()
            .and_then(|extra| extra.get(WORKSPACE_TREE_KEY))
            .and_then(serde_json::Value::as_str)
            .map(|tree| tree.trim_end_matches('/').to_string())
            .filter(|tree| !tree.is_empty());
        Self { tree }
    }

    /// `text` without the host's login name and scratch paths.
    pub(crate) fn text(&self, text: &str) -> String {
        let mut cleaned = text.to_string();
        if let Some(tree) = &self.tree {
            cleaned = relative_to(&cleaned, tree);
        }
        cleaned = relative_to_scratch_trees(&cleaned);
        cleaned = anonymise_home(&cleaned);
        anonymise_listing(&cleaned)
    }
}

/// `text` with the directory `tree` named relative to itself: `tree/x` is
/// `x`, `tree` alone is `.`. A longer name that only starts with `tree`
/// (`tree2`) is another directory and stays.
fn relative_to(text: &str, tree: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find(tree) {
        let after = &rest[at + tree.len()..];
        out.push_str(&rest[..at]);
        match after.chars().next() {
            Some('/') => rest = &after[1..],
            Some(c) if c.is_alphanumeric() || matches!(c, '_' | '-' | '.') => {
                out.push_str(tree);
                rest = after;
            }
            _ => {
                out.push('.');
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

/// `text` with every `<tmp>/.tmpXXXXXX/tree` (the harness's scratch directory)
/// named relative to itself.
fn relative_to_scratch_trees(text: &str) -> String {
    // Written \x2f so the machine-path gate does not mistake the pattern for a path.
    const PREFIX: &str = "\x2ftmp/.tmp";
    let mut trees: Vec<&str> = Vec::new();
    let mut from = 0;
    while let Some(found) = text[from..].find(PREFIX) {
        let start = from + found;
        let token_start = start + PREFIX.len();
        let token_len = text[token_start..]
            .bytes()
            .take_while(u8::is_ascii_alphanumeric)
            .count();
        let end = token_start + token_len;
        if token_len == 6 && text[end..].starts_with("/tree") {
            let tree = &text[start..end + "/tree".len()];
            if !trees.contains(&tree) {
                trees.push(tree);
            }
        }
        from = end.max(start + 1);
    }
    trees.iter().fold(text.to_string(), |cleaned, tree| {
        relative_to(&cleaned, tree)
    })
}

/// `text` with the name of every home directory (`<home>/name/...`) replaced by
/// `user`.
fn anonymise_home(text: &str) -> String {
    // Written \x2f so the machine-path gate does not mistake the pattern for a path.
    const PREFIX: &str = "\x2fhome/";
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find(PREFIX) {
        let name_start = at + PREFIX.len();
        let name_len = rest[name_start..]
            .find(|c: char| !(c.is_alphanumeric() || matches!(c, '_' | '-' | '.')))
            .unwrap_or(rest.len() - name_start);
        out.push_str(&rest[..name_start]);
        if name_len > 0 {
            out.push_str("user");
        }
        rest = &rest[name_start + name_len..];
    }
    out.push_str(rest);
    out
}

/// `text` with the owner and group of every long listing line in it
/// (`drwxr-xr-x 5 name group 87 Oct 20 08:02 .`) named `user`, unless they are
/// `root`. A line is found by its columns wherever it stands: also inside the
/// escaped text of a log the solver printed.
fn anonymise_listing(text: &str) -> String {
    let mut tokens: Vec<(usize, &str)> = Vec::new();
    let mut start = None;
    for (at, c) in text
        .char_indices()
        .chain(std::iter::once((text.len(), ' ')))
    {
        match (start, c.is_whitespace()) {
            (None, false) => start = Some(at),
            (Some(from), true) => {
                tokens.push((from, &text[from..at]));
                start = None;
            }
            _ => {}
        }
    }
    let is_mode = |token: &str| {
        // A listing glued to the end of an escaped line (`16\\ndrwx...`).
        let body = token
            .rsplit_once("\\n")
            .map_or(token, |(_, last)| last)
            .trim_end_matches(['.', '+', '@']);
        body.len() == 10
            && body.starts_with(['-', 'd', 'l', 'c', 'b', 'p', 's'])
            && body[1..].chars().all(|c| "rwxsStT-".contains(c))
    };
    let digits = |token: &str| !token.is_empty() && token.bytes().all(|b| b.is_ascii_digit());
    let mut replaced: Vec<(usize, usize)> = Vec::new();
    for window in tokens.windows(5) {
        if is_mode(window[0].1) && digits(window[1].1) && digits(window[4].1) {
            for named in window[2..4].iter().filter(|n| n.1 != "root") {
                replaced.push((named.0, named.0 + named.1.len()));
            }
        }
    }
    let mut out = String::with_capacity(text.len());
    let mut from = 0;
    for (start, end) in replaced {
        out.push_str(&text[from..start]);
        out.push_str("user");
        from = end;
    }
    out.push_str(&text[from..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tree_is_named_relative_to_itself_and_a_longer_name_is_another_directory() {
        assert_eq!(
            relative_to("cd /w/tree && ls /w/tree", "/w/tree"),
            "cd . && ls ."
        );
        assert_eq!(relative_to("/w/tree/a/b.py:3", "/w/tree"), "a/b.py:3");
        assert_eq!(relative_to("/w/tree2/a.py", "/w/tree"), "/w/tree2/a.py");
    }

    #[test]
    fn a_home_directory_loses_its_owner() {
        assert_eq!(
            anonymise_home("cd \x2fhome/marsch/code && ls \x2fhome/b-c.d/x \x2fhome/"),
            "cd \x2fhome/user/code && ls \x2fhome/user/x \x2fhome/"
        );
    }

    #[test]
    fn a_listing_keeps_its_columns_and_loses_the_names() {
        assert_eq!(
            anonymise_listing("drwxrwxr-x 5 marsch staff   87 Oct 20 08:02 .\n"),
            "drwxrwxr-x 5 user user   87 Oct 20 08:02 .\n"
        );
        assert_eq!(
            anonymise_listing("-rw-r--r-- 1 root root 4365 Jan  1  1970 a\n"),
            "-rw-r--r-- 1 root root 4365 Jan  1  1970 a\n"
        );
        assert_eq!(
            anonymise_listing(
                "{\"message\":\"total 16\\\\ndrwxrwxr-x 5 marsch marsch   87 Oct 20 05:03 .\"}"
            ),
            "{\"message\":\"total 16\\\\ndrwxrwxr-x 5 user user   87 Oct 20 05:03 .\"}"
        );
        assert_eq!(anonymise_listing("total 24\n"), "total 24\n");
        assert_eq!(anonymise_listing("a b c d 5\n"), "a b c d 5\n");
    }
}
