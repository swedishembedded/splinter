// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements coding agents that change a repository
// without ever seeing the tests that judge them, for its clients. If your
// team needs expertise in measuring what a code-editing agent really did,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: what a solver's shell is told and given beside the repository.
//!
//! * The shell tells the model where it works and where it may keep scratch
//!   files: `/tmp` is not there, a scratch directory named by `$TMPDIR` is,
//!   and the files it holds are kept between commands and never part of the
//!   patch.
//! * The harness keeps its own state out of the tree the solver reads.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use splinter_agent::solve::SolveOptions;
use splinter_agent::work::{solve_in_workspace, WorkOptions, WorkProtocol};
use splinter_core::experience::Task;
use splinter_sandbox::host::DEFAULT_SEARCH_PATH;
use splinter_sandbox::{
    Limits, ProcessSandbox, ResolvedEnvironment, WorkspaceEnvironment, WorkspaceHost,
};
use sven_sdk::model::{
    CompletionRequest, MessageContent, ModelProvider, ResponseEvent, ResponseStream,
};

/// A model that runs `commands` as shell calls, one per request, then stops;
/// it keeps what it was sent.
struct Shelling {
    commands: Vec<&'static str>,
    seen: Mutex<Vec<CompletionRequest>>,
}

#[async_trait::async_trait]
impl ModelProvider for Shelling {
    fn name(&self) -> &str {
        "scripted"
    }
    fn model_name(&self) -> &str {
        "worker-1"
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let answered = req
            .messages
            .iter()
            .filter(|m| matches!(m.content, MessageContent::ToolResult { .. }))
            .count();
        let events = match self.commands.get(answered) {
            Some(command) => vec![
                ResponseEvent::ToolCall {
                    index: 0,
                    id: format!("c{answered}"),
                    name: "shell".into(),
                    arguments: serde_json::json!({"description": "run", "shell_command": command})
                        .to_string(),
                },
                ResponseEvent::Done,
            ],
            None => vec![ResponseEvent::TextDelta("done".into()), ResponseEvent::Done],
        };
        self.seen.lock().unwrap().push(req);
        Ok(Box::pin(futures::stream::iter(
            events.into_iter().map(Ok).collect::<Vec<_>>(),
        )))
    }
}

impl Shelling {
    /// Everything the tools returned to the model, over all requests.
    fn results(&self) -> String {
        let seen = self.seen.lock().unwrap();
        seen.last()
            .unwrap()
            .messages
            .iter()
            .filter_map(|m| match &m.content {
                MessageContent::ToolResult { content, .. } => Some(format!("{content:?}")),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

fn git(dir: &Path, args: &[&str]) {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["-c", "user.name=t", "-c", "user.email=t@example.com"])
        .args(["-c", "commit.gpgsign=false"])
        .args(args)
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_CONFIG_SYSTEM", "/dev/null")
        .output()
        .unwrap();
    assert!(out.status.success(), "git {args:?}");
}

/// A one-file repository archived as a snapshot, and the workspace of it in
/// the process sandbox.
fn workspace() -> (tempfile::TempDir, PathBuf, WorkspaceEnvironment) {
    let root = tempfile::tempdir().unwrap();
    let repo = root.path().join("origin");
    std::fs::create_dir_all(&repo).unwrap();
    git(&repo, &["init", "-q", "-b", "main"]);
    std::fs::write(repo.join("app.txt"), "broken\n").unwrap();
    git(&repo, &["add", "-A"]);
    git(&repo, &["commit", "-q", "-m", "snapshot"]);
    let archive = root.path().join("snapshot.tar");
    git(
        &repo,
        &[
            "archive",
            "--format=tar",
            "-o",
            archive.to_str().unwrap(),
            "HEAD",
        ],
    );
    std::fs::create_dir_all(root.path().join("calls")).unwrap();
    let host = Arc::new(WorkspaceHost::resolve(DEFAULT_SEARCH_PATH, root.path()).unwrap());
    let sandbox = ProcessSandbox::new(
        root.path().join("calls"),
        BTreeMap::new(),
        &[],
        Limits::default(),
    );
    let snapshot = host.snapshot(&archive).unwrap();
    let environment = WorkspaceEnvironment::new(snapshot, Arc::new(sandbox), host);
    (root, archive, environment)
}

async fn run(
    commands: Vec<&'static str>,
) -> (Arc<Shelling>, splinter_agent::work::WorkspaceSolution) {
    let (_root, archive, environment) = workspace();
    let record = ResolvedEnvironment::Workspace(environment.clone())
        .record()
        .unwrap();
    let task = Task::new("workspace-change", vec![], record, "Fix app.txt.", vec![]).unwrap();
    let model = Arc::new(Shelling {
        commands,
        seen: Mutex::new(Vec::new()),
    });
    let solved = solve_in_workspace(
        &task,
        &environment,
        &archive,
        &[],
        model.clone(),
        WorkOptions::new(SolveOptions::new(Duration::from_secs(120))),
        &WorkProtocol::default(),
    )
    .await
    .unwrap();
    (model, solved)
}

#[tokio::test]
async fn scratch_files_live_between_commands_outside_the_tree_and_never_reach_the_patch() {
    let (model, solved) = run(vec![
        "echo kept > \"$TMPDIR/note.txt\"; echo changed > app.txt",
        "cat \"$TMPDIR/note.txt\"; ls -a",
    ])
    .await;
    let seen = model.results();
    assert!(
        seen.contains("kept"),
        "the next command reads the scratch file: {seen}"
    );
    assert!(
        !seen.contains("note.txt\\n"),
        "the scratch directory is not the tree: {seen}"
    );
    let diff = String::from_utf8(solved.patch.diff.clone()).unwrap();
    assert!(diff.contains("+changed"), "{diff}");
    assert!(
        !diff.contains("note.txt") && !diff.contains("kept"),
        "{diff}"
    );
}

#[tokio::test]
async fn the_shell_says_where_it_works_and_where_scratch_files_go() {
    let (model, _) = run(vec!["true"]).await;
    let seen = model.seen.lock().unwrap();
    let shell = seen[0].tools.iter().find(|t| t.name == "shell").unwrap();
    for part in ["project directory", "$TMPDIR", "/tmp"] {
        assert!(
            shell.description.contains(part),
            "{part}: {}",
            shell.description
        );
    }
}

#[tokio::test]
async fn the_harness_state_is_not_a_directory_the_solver_can_read() {
    // sven's audit log is written beneath the project root's `.sven`; what it
    // holds is the harness's tool log, not the task.
    let (model, solved) = run(vec![
        "ls .sven >/dev/null 2>&1 && echo readable || echo sealed",
    ])
    .await;
    let seen = model.results();
    assert!(seen.contains("sealed"), "{seen}");
    let diff = String::from_utf8(solved.patch.diff).unwrap();
    assert!(!diff.contains(".sven"), "{diff}");
}
