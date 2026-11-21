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

/// How the passes of each task are curated before they become
/// demonstrations: in this order, a pass without a final answer is left out,
/// passes that made the same tool calls are one demonstration, and at most a
/// number of passes per task are kept. The default curates nothing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PassCuration {
    /// Leave out a pass whose last agent step calls a tool or says nothing:
    /// it teaches acting without concluding
    /// ([`Exclusion::NoFinalAnswer`](crate::Exclusion::NoFinalAnswer)).
    #[serde(default)]
    pub final_answer: bool,
    /// Keep one of the passes of a task whose sequences of tool calls (name
    /// and arguments, keys sorted and runs of whitespace one space) are the
    /// same, the one with the least reasoning
    /// ([`Exclusion::DuplicatePass`](crate::Exclusion::DuplicatePass)).
    #[serde(default)]
    pub distinct_calls: bool,
    /// Keep at most this many passes of a task, the least reasoning per
    /// action first ([`Exclusion::OverTaskCap`](crate::Exclusion::OverTaskCap)).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub per_task: Option<usize>,
}

/// One pass the curation weighs.
#[derive(Clone, Debug)]
pub struct Pass<'a> {
    /// The experience.
    pub id: &'a splinter_core::experience::ExperienceId,
    /// Its task, by the task's id.
    pub task: String,
    /// What it did.
    pub trajectory: &'a atif::Trajectory,
}

/// What curation reads of a pass.
struct Profile {
    calls: Vec<(String, String)>,
    reasoning: usize,
    actions: usize,
    final_answer: bool,
}

fn spaced(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn message_text(body: &atif::MessageBody) -> String {
    if let Some(text) = body.as_text() {
        return text.to_string();
    }
    body.as_segments()
        .unwrap_or_default()
        .iter()
        .filter_map(|segment| match segment {
            atif::ContentSegment::Text { text } => Some(text.as_str()),
            atif::ContentSegment::Image { .. } => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn profile(trajectory: &atif::Trajectory) -> Profile {
    let agent: Vec<&atif::TraceStep> = trajectory
        .steps
        .iter()
        .filter(|s| s.source == atif::StepOrigin::Agent)
        .collect();
    let calls: Vec<(String, String)> = agent
        .iter()
        .flat_map(|s| s.tool_calls.iter().flatten())
        .map(|c| {
            let arguments = splinter_core::digest::canonical_json(&c.arguments)
                .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
                .unwrap_or_else(|_| c.arguments.to_string());
            (c.function_name.clone(), spaced(&arguments))
        })
        .collect();
    let final_answer = agent.last().is_some_and(|last| {
        last.tool_calls.as_ref().is_none_or(Vec::is_empty)
            && !message_text(&last.message).trim().is_empty()
    });
    Profile {
        reasoning: agent
            .iter()
            .map(|s| {
                s.reasoning_content
                    .as_deref()
                    .map_or(0, |r| r.chars().count())
            })
            .sum(),
        actions: calls.len(),
        calls,
        final_answer,
    }
}

impl PassCuration {
    /// Whether it curates anything.
    #[must_use]
    pub fn curates(&self) -> bool {
        self.final_answer || self.distinct_calls || self.per_task.is_some()
    }

    /// The passes of `passes` it leaves out, each with why. Ties are broken by
    /// the experience's id, so the same passes are always kept.
    #[must_use]
    pub fn left_out(
        &self,
        passes: &[Pass<'_>],
    ) -> std::collections::BTreeMap<splinter_core::experience::ExperienceId, crate::Exclusion> {
        use crate::Exclusion;
        let mut out = std::collections::BTreeMap::new();
        let mut by_task: std::collections::BTreeMap<&str, Vec<(&Pass<'_>, Profile)>> =
            std::collections::BTreeMap::new();
        for pass in passes {
            let profile = profile(pass.trajectory);
            if self.final_answer && !profile.final_answer {
                out.insert(pass.id.clone(), Exclusion::NoFinalAnswer);
                continue;
            }
            by_task
                .entry(pass.task.as_str())
                .or_default()
                .push((pass, profile));
        }
        for (_, mut task) in by_task {
            task.sort_by(|(a, pa), (b, pb)| pa.reasoning.cmp(&pb.reasoning).then(a.id.cmp(b.id)));
            if self.distinct_calls {
                let mut seen = std::collections::BTreeSet::new();
                task.retain(|(pass, profile)| {
                    let first = seen.insert(profile.calls.clone());
                    if !first {
                        out.insert(pass.id.clone(), Exclusion::DuplicatePass);
                    }
                    first
                });
            }
            if let Some(cap) = self.per_task {
                let per_action = |p: &Profile| p.reasoning as f64 / p.actions.max(1) as f64;
                task.sort_by(|(a, pa), (b, pb)| {
                    per_action(pa)
                        .total_cmp(&per_action(pb))
                        .then(a.id.cmp(b.id))
                });
                for (pass, _) in task.iter().skip(cap) {
                    out.insert(pass.id.clone(), Exclusion::OverTaskCap);
                }
            }
        }
        out
    }
}
