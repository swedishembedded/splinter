// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements executable evaluation of code changes
// against a project's own test suite, for its clients. If your team needs
// expertise in test-driven benchmarks for code-editing agents, you can
// procure our services by sending an email to info@swedishembedded.com.

//! A single-fault task read as a part of a combined one.

use std::collections::BTreeSet;

use splinter_core::digest::Digest;
use splinter_core::experience::{Environment, PrivilegedKind, Task};
use splinter_core::family::COMPOSITE_SEPARATOR;
use splinter_eval::verifiers::workspace::WorkspaceCheck;
use splinter_sandbox::TreeSnapshot;

use super::request::Symptoms;
use super::CombineReason;

/// What a combined task needs of one admitted task.
pub(super) struct Single {
    /// The task's address.
    pub(super) id: Digest,
    /// The file its fault was seeded in, as the task declares it.
    pub(super) family: String,
    /// What grades it.
    pub(super) check: WorkspaceCheck,
    /// The change that puts its fault right, as a diff.
    pub(super) reference: String,
    /// The files that change touches.
    pub(super) files: BTreeSet<String>,
    /// The tree it is solved in.
    pub(super) snapshot: TreeSnapshot,
    /// What its request says.
    pub(super) symptoms: Symptoms,
}

impl Single {
    /// `task` as a part of a combined one, or why it cannot be.
    pub(super) fn read(task: &Task) -> Result<Self, CombineReason> {
        if task.environment.kind != Environment::WORKSPACE {
            return Err(CombineReason::NotAWorkspaceTask);
        }
        let check = match WorkspaceCheck::of_task(task) {
            Ok(Some((_, check))) => check,
            _ => return Err(CombineReason::NoCheck),
        };
        let reference = task
            .privileged
            .iter()
            .find(|item| item.kind == PrivilegedKind::Reference)
            .map(|item| with_final_newline(&item.content))
            .ok_or(CombineReason::NoReference)?;
        let files = changed_files(&reference);
        if files.is_empty() {
            return Err(CombineReason::NoReference);
        }
        let family = task
            .concepts
            .first()
            .filter(|family| !family.is_empty() && !family.contains(COMPOSITE_SEPARATOR))
            .cloned()
            .ok_or(CombineReason::NoFamily)?;
        Ok(Self {
            id: task.task.id.clone(),
            family,
            check,
            reference,
            files,
            snapshot: snapshot_of(&task.environment).ok_or(CombineReason::NoSnapshot)?,
            symptoms: Symptoms::read(&task.instruction).ok_or(CombineReason::UnsupportedWording)?,
        })
    }
}

/// `patch` ending in a newline, as `git apply` requires.
pub(super) fn with_final_newline(patch: &str) -> String {
    if patch.ends_with('\n') {
        patch.to_string()
    } else {
        format!("{patch}\n")
    }
}

/// The tree a workspace environment names.
fn snapshot_of(environment: &Environment) -> Option<TreeSnapshot> {
    let text = |key: &str| environment.spec.get(key)?.as_str();
    Some(TreeSnapshot {
        digest: Digest::parse(text("snapshot")?).ok()?,
        tree_id: text("tree_id")?.to_string(),
    })
}

/// The files a diff changes: those its `+++` and `---` headers name.
fn changed_files(diff: &str) -> BTreeSet<String> {
    diff.lines()
        .filter_map(|line| {
            line.strip_prefix("+++ b/")
                .or_else(|| line.strip_prefix("--- a/"))
        })
        .map(str::to_string)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_files_a_diff_changes_are_the_ones_its_headers_name() {
        let diff = "diff --git a/src/a.py b/src/a.py\n--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-x\n+y\n\
                    diff --git a/old.py b/old.py\ndeleted file mode 100644\n--- a/old.py\n+++ /dev/null\n";
        let files: Vec<String> = changed_files(diff).into_iter().collect();
        assert_eq!(files, ["old.py", "src/a.py"]);
    }

    #[test]
    fn a_diff_is_given_the_final_newline_git_apply_asks_for() {
        assert_eq!(with_final_newline("a"), "a\n");
        assert_eq!(with_final_newline("a\n"), "a\n");
    }
}
