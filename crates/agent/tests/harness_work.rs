// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements coding agents that change a repository
// without ever seeing the tests that judge them, for its clients. If your
// team needs expertise in measuring what a code-editing agent really did,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: a candidate sven harness is run on a task that is a change to a
//! repository exactly as an agent is: a fresh copy of the task's snapshot,
//! the shell confined to it, the change read back from git as the patch that
//! is the solution's final output. What the harness claims it did is not
//! evidence; the tree is.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::Duration;

use splinter_agent::harness_work::{confined, run_harness_in_tree, solve_harness_in_workspace};
use splinter_agent::solve::{SolveError, SolveOptions};
use splinter_agent::work::{WorkOptions, WorkspaceSolution};
use splinter_core::experience::Task;
use splinter_sandbox::host::DEFAULT_SEARCH_PATH;
use splinter_sandbox::{
    Limits, ProcessSandbox, ResolvedEnvironment, Sandbox, WorkspaceEnvironment, WorkspaceHost,
};
use sven_sdk::model::{
    CompletionRequest, MessageContent, ModelProvider, ResponseEvent, ResponseStream,
};
use sven_sdk::{ApprovalPolicy, Engine, RunConclusion, RunOptions, Toolset};

/// A model that answers from a script over the request it is sent.
struct Scripted {
    script: fn(&CompletionRequest) -> Vec<ResponseEvent>,
}

impl Scripted {
    fn new(script: fn(&CompletionRequest) -> Vec<ResponseEvent>) -> Arc<Self> {
        Arc::new(Self { script })
    }
}

#[async_trait::async_trait]
impl ModelProvider for Scripted {
    fn name(&self) -> &str {
        "scripted"
    }
    fn model_name(&self) -> &str {
        "worker-1"
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let events = (self.script)(&req);
        Ok(Box::pin(futures::stream::iter(
            events.into_iter().map(Ok).collect::<Vec<_>>(),
        )))
    }
}

fn text(reply: &str) -> Vec<ResponseEvent> {
    vec![ResponseEvent::TextDelta(reply.into()), ResponseEvent::Done]
}

fn call(id: &str, name: &str, arguments: serde_json::Value) -> Vec<ResponseEvent> {
    vec![
        ResponseEvent::ToolCall {
            index: 0,
            id: id.into(),
            name: name.into(),
            arguments: arguments.to_string(),
        },
        ResponseEvent::Done,
    ]
}

fn tool_results(req: &CompletionRequest) -> usize {
    req.messages
        .iter()
        .filter(|m| matches!(m.content, MessageContent::ToolResult { .. }))
        .count()
}

/// The script every edit test shares: write the fix, look at the tree,
/// then say it is done. After a follow-up it says so again.
fn edits_then_stops(req: &CompletionRequest) -> Vec<ResponseEvent> {
    match tool_results(req) {
        0 => call(
            "c0",
            "write_file",
            serde_json::json!({"path": "app.txt", "text": "fixed\n", "append": false}),
        ),
        1 => call(
            "c1",
            "shell",
            serde_json::json!({"description": "list the tree", "shell_command": "ls -aR"}),
        ),
        _ => text("done"),
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
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// A scratch directory with a repository archived as a snapshot. The
/// repository also holds a test the solver must never see; the archive is
/// made from a commit that leaves it out, as a task generator does.
struct Fixture {
    root: tempfile::TempDir,
    archive: PathBuf,
    host: Arc<WorkspaceHost>,
}

const HIDDEN_NAME: &str = "hidden_acceptance_probe.txt";

impl Fixture {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        let repo = root.path().join("origin");
        std::fs::create_dir_all(&repo).unwrap();
        git(&repo, &["init", "-q", "-b", "main"]);
        std::fs::write(repo.join("app.txt"), "broken\n").unwrap();
        std::fs::write(repo.join("README.txt"), "a tiny project\n").unwrap();
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
        // The withheld material lives beside the snapshot, never in it.
        std::fs::write(root.path().join(HIDDEN_NAME), "the real test\n").unwrap();
        std::fs::create_dir_all(root.path().join("calls")).unwrap();
        let host = WorkspaceHost::resolve(DEFAULT_SEARCH_PATH, root.path()).unwrap();
        Self {
            root,
            archive,
            host: Arc::new(host),
        }
    }

    /// The workspace in the process sandbox.
    fn environment(&self) -> WorkspaceEnvironment {
        let sandbox = ProcessSandbox::new(
            self.root.path().join("calls"),
            BTreeMap::new(),
            &[],
            Limits::default(),
        );
        self.environment_in(Arc::new(sandbox))
    }

    fn environment_in(&self, sandbox: Arc<dyn Sandbox>) -> WorkspaceEnvironment {
        let snapshot = self.host.snapshot(&self.archive).unwrap();
        WorkspaceEnvironment::new(snapshot, sandbox, self.host.clone())
    }

    fn task(&self, environment: &WorkspaceEnvironment) -> Task {
        let record = ResolvedEnvironment::Workspace(environment.clone())
            .record()
            .unwrap();
        Task::new("workspace-change", vec![], record, "Fix app.txt.", vec![]).unwrap()
    }
}

fn options() -> WorkOptions {
    WorkOptions::new(SolveOptions::new(Duration::from_secs(120)))
}

/// Edits `app.txt` with an agent step and returns.
const EDITOR: &str = r#"
api_version: "1"
name: editor
inputs: { task: string }
capabilities: [WriteFile, ReadFile]
limits: { max_steps: 20, max_model_turns: 6, max_tool_calls: 6 }
entry: work
steps:
  work: { agent: { prompt: "{{ inputs.task }}", tools: [write_file, read_file], store_as: note, then: done } }
  done: { return: { value: { finished: true } } }
"#;

/// Claims a fix and changes nothing.
const TALKER: &str = r#"
api_version: "1"
name: talker
inputs: { task: string }
limits: { max_steps: 20, max_model_turns: 2 }
entry: say
steps:
  say: { predict: { prompt: "{{ inputs.task }}", store_as: claim, then: done } }
  done: { return: { value: { claim: "{{ facts.claim }}" } } }
"#;

async fn solve(
    fixture: &Fixture,
    environment: &WorkspaceEnvironment,
    harness: &str,
    model: Arc<dyn ModelProvider>,
) -> Result<WorkspaceSolution, SolveError> {
    solve_harness_in_workspace(
        &fixture.task(environment),
        environment,
        &fixture.archive,
        harness,
        model,
        options(),
    )
    .await
}

#[tokio::test]
async fn the_patch_a_harness_leaves_is_the_solution_and_the_trajectory_names_the_harness() {
    let fixture = Fixture::new();
    let environment = fixture.environment();
    let model = Scripted::new(edits_then_stops);

    let solved = solve(&fixture, &environment, EDITOR, model).await.unwrap();

    assert_eq!(solved.solution.conclusion, RunConclusion::Success);
    let diff = String::from_utf8(solved.patch.diff.clone()).unwrap();
    assert!(
        diff.contains("-broken") && diff.contains("+fixed"),
        "{diff}"
    );
    assert_eq!(solved.patch.files, vec![PathBuf::from("app.txt")]);
    assert_eq!(solved.solution.final_output.as_deref(), Some(diff.as_str()));
    let extra = solved.solution.trajectory.extra.clone().unwrap();
    assert_eq!(extra["harness"]["name"], "editor");
    assert!(extra["harness"]["plan_digest"]
        .as_str()
        .is_some_and(|d| d.len() == 64));
    assert!(
        solved
            .solution
            .trajectory
            .agent
            .extra
            .as_ref()
            .and_then(|e| e.get(splinter_eval::exposure::WORKSPACE_TREE_KEY))
            .is_some(),
        "the exposure audit can tell which tree was this episode's own"
    );
}

#[tokio::test]
async fn a_harness_that_only_claims_a_fix_leaves_an_empty_patch() {
    let fixture = Fixture::new();
    let environment = fixture.environment();
    let model = Scripted::new(|_| text("I fixed app.txt"));

    let solved = solve(&fixture, &environment, TALKER, model).await.unwrap();

    assert!(solved.patch.diff.is_empty(), "no edit, no patch");
    assert_eq!(solved.solution.final_output.as_deref(), Some(""));
}

#[tokio::test]
async fn a_document_that_does_not_compile_is_an_engine_error_not_a_solution() {
    let fixture = Fixture::new();
    let environment = fixture.environment();
    let model = Scripted::new(|_| text("unused"));
    let broken = "api_version: \"1\"\nname: broken\nentry: nowhere\nsteps: {}\n";
    let result = solve(&fixture, &environment, broken, model).await;
    assert!(matches!(result, Err(SolveError::Engine(_))), "{result:?}");
}

/// Reads a file that lies beside the tree, not in it, with the shell.
const SHELLER: &str = r#"
api_version: "1"
name: sheller
capabilities: [ExecuteShell]
limits: { max_steps: 20, max_model_turns: 6, max_tool_calls: 6 }
entry: work
steps:
  work: { agent: { prompt: "Print ../secret.txt.", tools: [shell], store_as: note, then: done } }
  done: { return: { value: { finished: true } } }
"#;

fn reads_beside_the_tree(req: &CompletionRequest) -> Vec<ResponseEvent> {
    match tool_results(req) {
        0 => call(
            "c0",
            "shell",
            serde_json::json!({"description": "print", "shell_command": "cat ../secret.txt"}),
        ),
        _ => text("done"),
    }
}

#[tokio::test]
async fn a_harness_run_in_a_tree_gets_no_answer_from_a_file_beside_the_tree() {
    let fixture = Fixture::new();
    let environment = fixture.environment();
    let attempt = fixture.root.path().join("attempt");
    std::fs::create_dir_all(&attempt).unwrap();
    std::fs::write(attempt.join("secret.txt"), "the hidden diff\n").unwrap();
    let tree = attempt.join("tree");
    environment.materialize(&fixture.archive, &tree).unwrap();
    let episode = fixture.root.path().join("attempt").join("episode");

    let builder = Engine::builder()
        .model_provider(Scripted::new(reads_beside_the_tree) as Arc<dyn ModelProvider>)
        .toolset(Toolset::coding())
        .approvals(ApprovalPolicy::Auto);
    let engine = confined(builder, &environment, &tree, &episode)
        .unwrap()
        .build()
        .unwrap();
    let harness = engine.load_harness_from_str(SHELLER).unwrap();
    let ran = run_harness_in_tree(&harness, &tree, serde_json::json!({}), RunOptions::new())
        .await
        .unwrap();

    assert_eq!(ran.outcome.conclusion, RunConclusion::Success);
    let trajectory = serde_json::to_string(&ran.run.trajectory().unwrap()).unwrap();
    assert!(trajectory.contains("cat ../secret.txt"), "the shell ran");
    assert!(
        !trajectory.contains("the hidden diff"),
        "the shell printed a file beside the tree"
    );
}
