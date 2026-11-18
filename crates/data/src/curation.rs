// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements curation of agent demonstrations so that a
// student model does not learn its teacher's unsafe exploration, for its
// clients. If your team needs expertise in safe distillation of coding agents,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! Curation of demonstrations against learned unsafe exploration.
//!
//! A student imitates what its teacher did, including how it looked around. A
//! teacher that read the repository's history (`git log --oneline -20`,
//! `git show --stat HEAD`) taught a 4B student to dig through it: trained on
//! such demonstrations, the student ran `git log --all`, `git reflog` and read
//! `.git/refs` and `.git/objects`, which a benchmark's leak audit fails as a
//! security violation (a sibling task's state, or the answer, can be in the
//! history beyond the checked-out commit), while the base model never did.
//!
//! The policy [`NO_REPOSITORY_HISTORY`] flags an action whose tool call reads
//! the repository's history or its raw storage ([`reads_repository_history`]):
//!
//! - a git command that walks commits (`log`, `rev-list`, `reflog`,
//!   `shortlog`, `whatchanged`, `show-branch`), shows a commit (`show`),
//!   annotates lines with their commits (`blame`, `annotate`), reads the
//!   object store directly (`cat-file`, `fsck`, `count-objects`), or names a
//!   commit before `HEAD` (`HEAD~1`, `HEAD^`, `@{1}`), git's global options
//!   (`-C <dir>`, `--no-pager`) allowed before the command;
//! - a path into git's storage: `.git/objects`, `.git/refs`, `.git/logs`,
//!   `.git/packed-refs`, `.git/HEAD` and the other `*HEAD` files.
//!
//! The working tree and its state against `HEAD` stay readable (`git status`,
//! `git diff`, `git diff HEAD`, `git ls-files`, `git rev-parse`), and so does
//! a command that only excludes `.git` from a listing (`--ignore=.git`,
//! `-not -path './.git/*'`). Every string in a call's arguments is checked,
//! as the leak audit checks them.
//!
//! What happens to a flagged action is the [`HistoryCuration`]: dropped (it is
//! not supervised, and stays in the context of what follows), or the
//! demonstration truncated before it (nothing from it on is supervised).

use serde::{Deserialize, Serialize};

pub use splinter_core::repository_history::reads_repository_history;

/// The name of the curation policy that flags actions reading the
/// repository's history, as a dataset's manifest records it.
pub const NO_REPOSITORY_HISTORY: &str = "no-repository-history";

/// What a view does with an action that reads the repository's history
/// ([`reads_repository_history`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum HistoryCuration {
    /// Nothing: such actions are learned like any other.
    Keep,
    /// The action is not supervised, counted as
    /// [`Exclusion::RepositoryHistory`](crate::Exclusion::RepositoryHistory);
    /// it stays in the context of the actions after it.
    #[default]
    DropStep,
    /// The demonstration ends before the first such action: it is counted as
    /// [`Exclusion::RepositoryHistory`](crate::Exclusion::RepositoryHistory)
    /// and every action after it as
    /// [`Exclusion::AfterRepositoryHistory`](crate::Exclusion::AfterRepositoryHistory).
    Truncate,
}

impl HistoryCuration {
    /// Every curation, by its name.
    pub const ALL: [HistoryCuration; 3] = [
        HistoryCuration::Keep,
        HistoryCuration::DropStep,
        HistoryCuration::Truncate,
    ];

    /// The curation's name: `keep`, `drop-step` or `truncate`.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            HistoryCuration::Keep => "keep",
            HistoryCuration::DropStep => "drop-step",
            HistoryCuration::Truncate => "truncate",
        }
    }

    /// The curation `name` names, refused with the names there are.
    ///
    /// # Errors
    /// No curation has that name.
    pub fn parse(name: &str) -> Result<Self, String> {
        Self::ALL
            .into_iter()
            .find(|c| c.name() == name)
            .ok_or_else(|| format!("{name:?} is not keep, drop-step or truncate"))
    }

    /// The curation as a dataset's manifest records it; `None` for
    /// [`HistoryCuration::Keep`], which curates nothing.
    #[must_use]
    pub fn recorded(self) -> Option<Curation> {
        (self != HistoryCuration::Keep).then(|| Curation {
            policy: NO_REPOSITORY_HISTORY.into(),
            mode: self,
        })
    }
}

/// A curation policy a dataset was built under, as its manifest records it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Curation {
    /// The policy's name: [`NO_REPOSITORY_HISTORY`].
    pub policy: String,
    /// What was done with the actions it flagged.
    pub mode: HistoryCuration,
}
