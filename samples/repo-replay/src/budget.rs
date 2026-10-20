// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! How many tool calls an attempt may make, by the size of the commit.
//!
//! A one-line change and a twelve-file change need different amounts of
//! looking and checking, so the budget grows with the files and hunks the
//! commit touches: `base + per_file * files + per_hunk * hunks`, at most
//! `cap`. The agent is told its budget; the run is stopped at it. Wall time
//! never bounds an attempt (queueing on a shared server is not the agent's
//! doing and is recorded separately).

use serde::{Deserialize, Serialize};

/// The parameters of the budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Budget {
    /// Calls any attempt gets: the todo list, the review, a check.
    pub base: usize,
    /// Calls each changed file adds: locate, read, edit, re-read.
    pub per_file: usize,
    /// Calls each hunk adds.
    pub per_hunk: usize,
    /// The most any attempt gets.
    pub cap: usize,
    /// Calls each correction round adds: the fix and its re-reading.
    pub per_correction: usize,
}

impl Default for Budget {
    fn default() -> Self {
        Self {
            base: 10,
            per_file: 8,
            per_hunk: 3,
            cap: 120,
            per_correction: 12,
        }
    }
}

impl Budget {
    /// The calls for a commit changing `files` files in `hunks` hunks.
    #[must_use]
    pub fn calls(&self, files: usize, hunks: usize) -> usize {
        (self.base + self.per_file * files + self.per_hunk * hunks).min(self.cap)
    }
}

/// The hunks of a unified diff.
#[must_use]
pub fn hunks_of(diff: &str) -> usize {
    diff.lines().filter(|l| l.starts_with("@@")).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_budget_grows_with_the_commit_and_stops_at_the_cap() {
        let b = Budget::default();
        assert_eq!(b.calls(1, 1), 21);
        assert!(b.calls(3, 6) > b.calls(1, 1));
        assert_eq!(b.calls(50, 200), b.cap);
        assert_eq!(
            hunks_of("diff --git a/x b/x\n@@ -1 +1 @@\n-a\n+b\n@@ -9 +9 @@\n"),
            2
        );
    }
}
