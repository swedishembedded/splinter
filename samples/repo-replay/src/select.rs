// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Which commits of a history can be replayed, and why the others cannot.
//!
//! Selection is a pure function of the history and the [`SelectConfig`], so
//! the same repository and bounds give the same list. Every commit that is
//! not replayed is counted under one [`Skip`] class.

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::attempt::CommitPlan;
use crate::generated::{has_generated_header, is_generated_path};
use crate::git::{Change, CommitMeta, Git, Kind};
use crate::scrub::Scrubber;

/// The bounds of selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelectConfig {
    /// Most changed lines (added plus removed) of a replayable commit.
    pub max_changed_lines: usize,
    /// Most changed files of a replayable commit.
    pub max_files: usize,
    /// Largest diff, in tokens ([`estimate_tokens`]), a replay's system
    /// prompt may carry: the diff is in the context of every turn. Set from
    /// the served model's context with [`SelectConfig::with_context_share`].
    pub max_diff_tokens: usize,
    /// Share of changed lines, in percent, above which generated content
    /// makes a commit not worth replaying.
    pub max_generated_percent: usize,
}

/// The context assumed when a caller does not know the model's: the
/// smallest window the 27B is served with here.
pub const DEFAULT_CONTEXT_TOKENS: usize = 32768;

/// The share of the model's context the diff may take, in percent.
pub const DEFAULT_CONTEXT_SHARE_PERCENT: usize = 25;

/// Tokens of `text` as a conservative estimate: one token per three
/// characters. English prose and code run at about four characters per token
/// on the tokenizers of the models served here, so this overcounts by about a
/// third and skips a commit rather than overflowing a context. A served model
/// exposes no tokenizer through its API; a local one's could replace this.
#[must_use]
pub fn estimate_tokens(text: &str) -> usize {
    text.len().div_ceil(3)
}

impl SelectConfig {
    /// This configuration with the diff limited to `percent` of a model's
    /// `context_tokens`.
    #[must_use]
    pub fn with_context_share(self, context_tokens: usize, percent: usize) -> Self {
        Self {
            max_diff_tokens: context_tokens * percent / 100,
            ..self
        }
    }
}

impl Default for SelectConfig {
    fn default() -> Self {
        Self {
            max_changed_lines: 100,
            max_files: 3,
            max_diff_tokens: DEFAULT_CONTEXT_TOKENS * DEFAULT_CONTEXT_SHARE_PERCENT / 100,
            max_generated_percent: 50,
        }
    }
}

/// Why a commit is not replayed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Skip {
    /// More than one parent.
    Merge,
    /// No parent.
    Root,
    /// It changes a binary file.
    Binary,
    /// It changes a submodule or a symbolic link.
    SpecialEntry,
    /// Too many lines or files.
    Oversize,
    /// Mostly generated or vendored content, or nothing else.
    MostlyGenerated,
    /// The diff takes more than its share of the model's context.
    DiffOverContextShare,
    /// The diff holds a secret-shaped string.
    SecretInDiff,
    /// git could not be read for the commit.
    Unreadable,
}

impl Skip {
    /// The class a report counts it under.
    #[must_use]
    pub fn class(self) -> &'static str {
        match self {
            Self::Merge => "merge",
            Self::Root => "root",
            Self::Binary => "binary",
            Self::SpecialEntry => "special_entry",
            Self::Oversize => "oversize",
            Self::MostlyGenerated => "mostly_generated",
            Self::DiffOverContextShare => "diff_over_context_share",
            Self::SecretInDiff => "secret_in_diff",
            Self::Unreadable => "unreadable",
        }
    }
}

/// A commit that was looked at: replayable, or why not.
#[derive(Debug)]
pub enum Selected {
    /// Ready to replay.
    Replay(Box<CommitPlan>),
    /// Not replayed, and why.
    Skipped(CommitMeta, Skip),
}

/// Looks at `meta`: its changes, its diff and both trees.
pub fn prepare(
    git: &Git,
    meta: &CommitMeta,
    scrubber: &Scrubber,
    config: &SelectConfig,
) -> Result<Selected> {
    let skip = |why| Ok(Selected::Skipped(meta.clone(), why));
    let parent = match meta.parents.as_slice() {
        [] => return skip(Skip::Root),
        [one] => one.clone(),
        _ => return skip(Skip::Merge),
    };
    let all = git.changes(&parent, &meta.sha)?;
    if all.iter().any(|c| c.special) {
        return skip(Skip::SpecialEntry);
    }
    let generated = |c: &Change| {
        is_generated_path(&c.path)
            || (c.kind != Kind::Deleted
                && has_generated_header(&git.head_of_blob(&meta.sha, &c.path, 400)))
    };
    let (generated_changes, changes): (Vec<Change>, Vec<Change>) =
        all.into_iter().partition(|c| generated(c));
    let lines = |cs: &[Change]| cs.iter().map(Change::lines).sum::<usize>();
    let generated_lines = lines(&generated_changes);
    let total = generated_lines + lines(&changes);
    if changes.is_empty()
        || (total > 0 && generated_lines * 100 / total > config.max_generated_percent)
    {
        return skip(Skip::MostlyGenerated);
    }
    if changes.iter().any(Change::binary) {
        return skip(Skip::Binary);
    }
    if lines(&changes) > config.max_changed_lines || changes.len() > config.max_files {
        return skip(Skip::Oversize);
    }
    let paths: Vec<String> = changes
        .iter()
        .flat_map(|c| std::iter::once(c.path.clone()).chain(c.old_path.clone()))
        .collect();
    let diff = git.diff(&parent, &meta.sha, &paths)?;
    if estimate_tokens(&diff) > config.max_diff_tokens {
        return skip(Skip::DiffOverContextShare);
    }
    if scrubber.finds_secret(&diff) {
        return skip(Skip::SecretInDiff);
    }

    Ok(Selected::Replay(Box::new(CommitPlan {
        sha: meta.sha.clone(),
        parent,
        message: scrubber.scrub(&meta.message),
        changes,
        diff,
    })))
}
