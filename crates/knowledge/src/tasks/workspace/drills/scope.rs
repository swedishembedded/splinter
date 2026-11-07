// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements comprehension drills made by code from a
// codebase's own functions and tests, for its clients. If your team needs
// expertise in turning a repository into verified training tasks you can
// procure our services by sending an email to info@swedishembedded.com.

//! Where a repair task touches the code: the lines of the clean tree that a
//! reference change (broken tree to clean tree) changes, so that drills can
//! be made for exactly the functions that contain them.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::tasks::workspace::WorkspaceError;

/// The lines of one file a unified diff changes, numbered in the file the
/// diff produces (its `+` side), and the text it expects at the lines it
/// shows, to be held against the tree the lines are read in.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Touched {
    /// The lines the change adds, and for a change that only removes lines the
    /// two lines either side of the removal.
    pub lines: BTreeSet<usize>,
    /// The line number and text of every line the diff shows on its `+`
    /// side (context and additions).
    pub shown: Vec<(usize, String)>,
}

impl Touched {
    /// Refuses, naming the first line that differs, when `file` of the tree at
    /// `tree` does not hold what the diff shows on its `+` side: the change
    /// was made for another revision than the tree.
    pub(super) fn check_against(&self, tree: &Path, file: &str) -> Result<(), WorkspaceError> {
        let text =
            std::fs::read_to_string(tree.join(file)).map_err(|source| WorkspaceError::Io {
                path: tree.join(file),
                source,
            })?;
        let lines: Vec<&str> = text.split('\n').collect();
        for (number, expected) in &self.shown {
            let held = lines.get(number - 1).copied().unwrap_or("");
            if held.trim_end_matches('\r') != expected.trim_end_matches('\r') {
                return Err(WorkspaceError::Parameter {
                    name: "for_tasks",
                    reason: format!(
                        "the reference change shows line {number} of {file} as {expected:?}, but \
                         the revision has {held:?}: drill the revision the tasks were made from"
                    ),
                });
            }
        }
        Ok(())
    }
}

/// The lines `patch`, a unified diff, changes in each file it changes (a file
/// the diff adds or deletes is left out: it has no function to find).
#[must_use]
pub fn touched_lines(patch: &str) -> BTreeMap<String, Touched> {
    let mut found: BTreeMap<String, Touched> = BTreeMap::new();
    let mut file: Option<String> = None;
    let mut next = 0usize;
    // Lines the current hunk still has on its old and new side. A line that
    // starts `--- ` inside a hunk is a removed line (a Lua comment), not a
    // header, so the counts of the hunk's header say where it ends.
    let (mut old_left, mut new_left) = (0usize, 0usize);
    for line in patch.lines() {
        if old_left == 0 && new_left == 0 {
            if let Some(rest) = line.strip_prefix("+++ ") {
                let path = rest.split('\t').next().unwrap_or(rest);
                file = (path != "/dev/null")
                    .then(|| path.strip_prefix("b/").unwrap_or(path).to_string());
            } else if let Some(header) = line.strip_prefix("@@ ") {
                let mut words = header.split_whitespace();
                let range = |sign: char, word: Option<&str>| -> (usize, usize) {
                    let body = word.and_then(|w| w.strip_prefix(sign)).unwrap_or("");
                    let (start, count) = body.split_once(',').unwrap_or((body, "1"));
                    (start.parse().unwrap_or(0), count.parse().unwrap_or(0))
                };
                old_left = range('-', words.next()).1;
                (next, new_left) = range('+', words.next());
            }
            continue;
        }
        let Some(name) = &file else {
            // A hunk of a file the diff adds or deletes: skipped to its end.
            match line.chars().next() {
                Some('+') => new_left = new_left.saturating_sub(1),
                Some('-') => old_left = old_left.saturating_sub(1),
                Some('\\') => {}
                _ => {
                    old_left = old_left.saturating_sub(1);
                    new_left = new_left.saturating_sub(1);
                }
            }
            continue;
        };
        let entry = found.entry(name.clone()).or_default();
        match line.chars().next() {
            Some('+') => {
                entry.lines.insert(next);
                entry.shown.push((next, line[1..].to_string()));
                next += 1;
                new_left = new_left.saturating_sub(1);
            }
            Some('-') => {
                entry.lines.insert(next);
                entry.lines.insert(next.saturating_sub(1).max(1));
                old_left = old_left.saturating_sub(1);
            }
            Some('\\') => {}
            _ => {
                entry
                    .shown
                    .push((next, line.get(1..).unwrap_or("").to_string()));
                next += 1;
                old_left = old_left.saturating_sub(1);
                new_left = new_left.saturating_sub(1);
            }
        }
    }
    found.retain(|_, touched| !touched.lines.is_empty());
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    const PATCH: &str = "\
diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -3,5 +3,5 @@ def f(x):
     a = 1
-    b = 3
+    b = 2
     return a + b
\x20
\x20
";  // \x20: the diff's context lines of an empty source line are one space

    #[test]
    fn a_change_is_numbered_in_the_file_it_produces() {
        let touched = touched_lines(PATCH);
        let one = &touched["pkg/mod.py"];
        // The removed line sits at line 4 (and just after 3); the added one is line 4.
        assert_eq!(one.lines, BTreeSet::from([3, 4]));
        assert_eq!(one.shown[0], (3, "    a = 1".to_string()));
        assert!(one.shown.contains(&(4, "    b = 2".to_string())));
    }

    #[test]
    fn a_removed_lua_comment_is_a_removed_line_not_a_header() {
        let patch =
            "--- a/m.lua\n+++ b/m.lua\n@@ -1,3 +1,2 @@\n local a = 1\n--- a note\n local b = 2\n";
        let touched = touched_lines(patch);
        assert_eq!(touched["m.lua"].lines, BTreeSet::from([2, 1]));
    }

    #[test]
    fn a_file_the_diff_deletes_is_left_out() {
        let deleted = "--- a/old.py\n+++ /dev/null\n@@ -1,2 +0,0 @@\n-x = 1\n-y = 2\n";
        assert!(touched_lines(deleted).is_empty());
    }
}
