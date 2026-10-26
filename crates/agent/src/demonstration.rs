// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that learn tasks they cannot yet
// solve from verified demonstrations, for its clients. If your team needs
// expertise in turning reference solutions into training trajectories you
// can procure our services by sending an email to info@swedishembedded.com.

//! A reference change replayed as the actions of an agent.
//!
//! A task that is a change to a repository may carry the change that solves
//! it (the inverse of the fault that was seeded, as a git patch). That patch
//! is not a trajectory: a student is trained on what an agent did, action by
//! action, with the tools it was offered. [`Demonstration`] turns the patch
//! into those actions - read each file the patch changes, then edit it by
//! the patch's own hunks - and [`Demonstration::provider`] is a model that
//! says them in order. Run through [`crate::work::solve_in_workspace`] like
//! any other model, the demonstration executes in a fresh copy of the
//! task's snapshot with sven's real tools, so what it records is what those
//! tools returned, and the patch read back from git is graded by the same
//! independent verifier as a student's: the teacher never grades itself.
//!
//! The actions name the file the patch changes without searching for it, so
//! a demonstration teaches what to do once the place is known, not how to
//! find it. A patch that creates, deletes, renames or is binary is refused:
//! sven's edit tool only changes files that exist.

use std::sync::Arc;

use serde_json::json;
use sven_sdk::model::{
    CompletionRequest, MessageContent, ModelProvider, ResponseEvent, ResponseStream,
};

/// The identity a demonstration is recorded under as the solver.
pub const DEMONSTRATOR: &str = "splinter/reference-demonstration";

/// Why a patch cannot be demonstrated.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DemonstrationError {
    /// The patch changes no file.
    #[error("the reference patch changes no file")]
    Empty,
    /// The patch does something sven's edit tool cannot: create, delete,
    /// rename, a binary file or a path git quotes.
    #[error("the reference patch cannot be replayed with the edit tool: {0}")]
    Unsupported(String),
}

/// One tool call of a demonstration.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Action {
    tool: &'static str,
    arguments: serde_json::Value,
}

/// The actions that make a reference patch, and what is said at the end.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Demonstration {
    actions: Vec<Action>,
    closing: String,
}

impl Demonstration {
    /// The demonstration of `patch`, a git unified diff over existing
    /// files: for each file, `read_file` then `edit_file` with the file's
    /// hunks.
    pub fn from_patch(patch: &str) -> Result<Self, DemonstrationError> {
        let mut actions = Vec::new();
        let mut files = Vec::new();
        for section in sections(patch) {
            let file = File::parse(section)?;
            actions.push(Action {
                tool: "read_file",
                arguments: json!({"path": file.path}),
            });
            actions.push(Action {
                tool: "edit_file",
                arguments: json!({"path": file.path, "diff": file.hunks}),
            });
            files.push(file.path);
        }
        if files.is_empty() {
            return Err(DemonstrationError::Empty);
        }
        Ok(Self {
            actions,
            closing: format!(
                "I changed {} to fix the failing behaviour.",
                files.join(", ")
            ),
        })
    }

    /// The demonstration of a task whose answer is `answer` itself: no tool
    /// call, the answer said as the reply - a closed-book task's reference
    /// (an output the runtime computed, a reference program).
    #[must_use]
    pub fn answer(answer: impl Into<String>) -> Self {
        Self {
            actions: Vec::new(),
            closing: answer.into(),
        }
    }

    /// The tool calls in the order they are made: `(tool, arguments)`.
    pub fn calls(&self) -> impl Iterator<Item = (&'static str, &serde_json::Value)> {
        self.actions.iter().map(|a| (a.tool, &a.arguments))
    }

    /// A model that makes these calls one per turn, then says its closing
    /// words, however many turns follow.
    #[must_use]
    pub fn provider(self) -> Arc<dyn ModelProvider> {
        Arc::new(Demonstrator { script: self })
    }
}

/// The part of a diff for one file, each starting at its `diff --git` line.
fn sections(patch: &str) -> Vec<&str> {
    let starts: Vec<usize> = patch
        .match_indices("diff --git ")
        .map(|(at, _)| at)
        .filter(|&at| at == 0 || patch.as_bytes()[at - 1] == b'\n')
        .collect();
    starts
        .iter()
        .enumerate()
        .map(|(i, &start)| &patch[start..starts.get(i + 1).copied().unwrap_or(patch.len())])
        .collect()
}

struct File {
    path: String,
    hunks: String,
}

impl File {
    fn parse(section: &str) -> Result<Self, DemonstrationError> {
        let unsupported = |what: &str| Err(DemonstrationError::Unsupported(what.to_string()));
        let header = section.lines().next().unwrap_or_default();
        for marker in [
            "new file mode",
            "deleted file mode",
            "rename ",
            "copy ",
            "Binary files",
            "GIT binary patch",
        ] {
            if section.lines().any(|l| l.starts_with(marker)) {
                return unsupported(marker.trim());
            }
        }
        let path = section
            .lines()
            .find_map(|l| l.strip_prefix("+++ b/"))
            .map(str::to_string);
        let Some(path) = path else {
            return unsupported(&format!("no `+++ b/` path in {header:?}"));
        };
        if path.starts_with('"') {
            return unsupported("a quoted path");
        }
        let Some(first_hunk) = section.find("\n@@") else {
            return unsupported(&format!("no hunk for {path}"));
        };
        Ok(Self {
            path,
            hunks: section[first_hunk + 1..].to_string(),
        })
    }
}

/// The model of a [`Demonstration`]: stateless, it reads how many tool
/// results the conversation holds to know which call is next.
struct Demonstrator {
    script: Demonstration,
}

#[async_trait::async_trait]
impl ModelProvider for Demonstrator {
    fn name(&self) -> &str {
        "splinter"
    }

    fn model_name(&self) -> &str {
        "reference-demonstration"
    }

    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let done = req
            .messages
            .iter()
            .filter(|m| matches!(m.content, MessageContent::ToolResult { .. }))
            .count();
        let events = match self.script.actions.get(done) {
            Some(action) => vec![
                ResponseEvent::ToolCall {
                    index: 0,
                    id: format!("demonstration-{done}"),
                    name: action.tool.to_string(),
                    arguments: action.arguments.to_string(),
                },
                ResponseEvent::Done,
            ],
            None => vec![
                ResponseEvent::TextDelta(self.script.closing.clone()),
                ResponseEvent::Done,
            ],
        };
        Ok(Box::pin(futures::stream::iter(
            events.into_iter().map(Ok).collect::<Vec<_>>(),
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PATCH: &str = "diff --git a/shop/tier0.py b/shop/tier0.py\nindex 1..2 100644\n--- a/shop/tier0.py\n+++ b/shop/tier0.py\n@@ -1,3 +1,3 @@\n def tier0(units):\n-    return 1\n+    return 2\n     pass\n";

    #[test]
    fn a_patch_is_read_then_edited_by_its_own_hunks() {
        let demo = Demonstration::from_patch(PATCH).unwrap();
        let calls: Vec<_> = demo.calls().collect();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].0, "read_file");
        assert_eq!(calls[0].1["path"], "shop/tier0.py");
        assert_eq!(calls[1].0, "edit_file");
        let diff = calls[1].1["diff"].as_str().unwrap();
        assert!(diff.starts_with("@@ -1,3 +1,3 @@"), "{diff}");
        assert!(!diff.contains("diff --git"), "{diff}");
    }

    #[test]
    fn a_patch_the_edit_tool_cannot_make_is_refused() {
        let created =
            "diff --git a/x b/x\nnew file mode 100644\n--- /dev/null\n+++ b/x\n@@ -0,0 +1 @@\n+x\n";
        assert!(matches!(
            Demonstration::from_patch(created),
            Err(DemonstrationError::Unsupported(_))
        ));
        assert_eq!(
            Demonstration::from_patch("").unwrap_err(),
            DemonstrationError::Empty
        );
    }
}
