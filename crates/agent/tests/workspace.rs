// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements coding agents that change a repository
// without ever seeing the tests that judge them, for its clients. If your
// team needs expertise in measuring what a code-editing agent really did,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: a task that is a change to a repository is solved in a fresh copy
//! of its snapshot with sven's own tools. What the solver changed is read
//! back from git as the patch, which is the experience's final output; the
//! tree it worked in never holds material that is withheld from it. After
//! each episode the visible checks run, and a red one earns exactly one
//! follow-up user turn carrying that check's name and output tail and
//! nothing else, up to the protocol's bound. Tools that bring in outside
//! state or reach the network or another model are not offered. The tool
//! cap and the repetition bound stop an episode as they do a worker. Where
//! the sandbox is a container the shell runs in it; otherwise sven's own
//! shell runs.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

#[path = "workspace/prompt.rs"]
mod prompt;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use splinter_agent::solve::{SolveError, SolveOptions};
use splinter_agent::work::{
    solve_in_workspace, Observed, WorkOptions, WorkProtocol, WorkspaceSolution, SOLVE_PROFILE_KEY,
    WITHHELD_TOOLS,
};
use splinter_core::digest::Digest;
use splinter_core::experience::Task;
use splinter_sandbox::host::DEFAULT_SEARCH_PATH;
use splinter_sandbox::workspace::DEFAULT_EXCLUDED;
use splinter_sandbox::{
    Backend, CodeCall, CodeResult, CommandCall, Isolation, Limits, ProcessSandbox,
    ResolvedEnvironment, ResolvedRuntime, RuntimeSpec, Sandbox, SandboxError, SandboxSpec,
    WorkspaceEnvironment, WorkspaceHost,
};
use sven_sdk::model::{
    CompletionRequest, MessageContent, ModelProvider, ResponseEvent, ResponseStream, Role,
};
use sven_sdk::{RunConclusion, SessionEvent};

/// A model that answers from a script over the request it is sent, and
/// keeps every request so the test can inspect what it was offered.
struct Scripted {
    script: fn(&CompletionRequest) -> Vec<ResponseEvent>,
    seen: Mutex<Vec<CompletionRequest>>,
}

impl Scripted {
    fn new(script: fn(&CompletionRequest) -> Vec<ResponseEvent>) -> Arc<Self> {
        Arc::new(Self {
            script,
            seen: Mutex::new(Vec::new()),
        })
    }

    /// The text of every user turn of the last request, in order.
    fn user_turns(&self) -> Vec<String> {
        let seen = self.seen.lock().unwrap();
        seen.last()
            .unwrap()
            .messages
            .iter()
            .filter(|m| m.role == Role::User)
            .filter_map(|m| match &m.content {
                MessageContent::Text(t) => Some(t.clone()),
                _ => None,
            })
            .collect()
    }

    /// Everything a tool returned to the model, over all requests.
    fn tool_results(&self) -> String {
        let seen = self.seen.lock().unwrap();
        let mut out = String::new();
        for request in seen.iter() {
            for message in &request.messages {
                if let MessageContent::ToolResult { content, .. } = &message.content {
                    out.push_str(&format!("{content:?}\n"));
                }
            }
        }
        out
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
        self.seen.lock().unwrap().push(req);
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

fn protocol(follow_ups: u32) -> WorkProtocol {
    WorkProtocol {
        follow_ups,
        ..WorkProtocol::default()
    }
}

fn sh(script: &str) -> CommandCall {
    CommandCall {
        argv: vec!["sh".into(), "-c".into(), script.into()],
        stdin: None,
    }
}

async fn solve(
    fixture: &Fixture,
    environment: &WorkspaceEnvironment,
    visible: &[CommandCall],
    model: Arc<dyn ModelProvider>,
    protocol: &WorkProtocol,
) -> Result<WorkspaceSolution, SolveError> {
    solve_in_workspace(
        &fixture.task(environment),
        environment,
        &fixture.archive,
        visible,
        model,
        options(),
        protocol,
    )
    .await
}

#[tokio::test]
async fn the_patch_read_back_is_the_edit_made_and_no_withheld_file_is_in_the_tree() {
    let fixture = Fixture::new();
    let environment = fixture.environment();
    let model = Scripted::new(edits_then_stops);

    let solved = solve(&fixture, &environment, &[], model.clone(), &protocol(0))
        .await
        .unwrap();

    assert_eq!(solved.solution.conclusion, RunConclusion::Success);
    let diff = String::from_utf8(solved.patch.diff.clone()).unwrap();
    assert!(
        diff.contains("-broken") && diff.contains("+fixed"),
        "{diff}"
    );
    assert_eq!(solved.patch.files, vec![PathBuf::from("app.txt")]);
    assert_eq!(
        solved.solution.final_output.as_deref(),
        Some(diff.as_str()),
        "the experience's output is the patch"
    );
    assert_eq!(solved.feedback_rounds, 0);
    let listing = model.tool_results();
    assert!(
        listing.contains("README.txt") && !listing.contains(HIDDEN_NAME),
        "the tree holds the snapshot and nothing withheld: {listing}"
    );
}

#[tokio::test]
async fn a_run_that_stops_announcing_a_call_it_never_made_is_not_a_success_and_earns_no_follow_up()
{
    let fixture = Fixture::new();
    let environment = fixture.environment();
    // A turn that ends on "let me ...:" with no call is not the worker ending.
    let model = Scripted::new(|_| text("Let me look at the tests next:"));

    let solved = solve(&fixture, &environment, &[], model.clone(), &protocol(2))
        .await
        .unwrap();

    let stopped = solved.solution.conclusion == RunConclusion::StoppedWithoutConclusion;
    assert!(stopped && solved.feedback_rounds == 0);
    assert_eq!(model.seen.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn an_observer_made_for_the_tree_reads_the_edits_as_the_calls_finish() {
    let fixture = Fixture::new();
    let environment = fixture.environment();
    let changed_after_each_call: Arc<Mutex<Vec<Vec<PathBuf>>>> = Arc::default();
    let mut bounds = options();
    {
        let (environment, log) = (environment.clone(), changed_after_each_call.clone());
        bounds.tree_observer = Some(Arc::new(move |tree, baseline| {
            let (environment, log) = (environment.clone(), log.clone());
            let (tree, baseline) = (tree.to_path_buf(), baseline.clone());
            Arc::new(move |seen: Observed<'_>| {
                if let Observed::Event(SessionEvent::ToolCallFinished { .. }) = seen {
                    let patch = environment
                        .diff_since(&tree, &baseline, &DEFAULT_EXCLUDED)
                        .unwrap();
                    log.lock().unwrap().push(patch.files);
                }
            })
        }));
    }

    let solved = solve_in_workspace(
        &fixture.task(&environment),
        &environment,
        &fixture.archive,
        &[],
        Scripted::new(edits_then_stops),
        bounds,
        &protocol(0),
    )
    .await
    .unwrap();

    let log = changed_after_each_call.lock().unwrap();
    assert_eq!(
        *log,
        vec![vec![PathBuf::from("app.txt")]; 2],
        "the tree the observer was given is the one the worker edits"
    );
    assert_eq!(solved.patch.files, vec![PathBuf::from("app.txt")]);
}

#[tokio::test]
async fn a_task_recorded_in_another_environment_is_refused() {
    let fixture = Fixture::new();
    let environment = fixture.environment();
    let mut task = fixture.task(&environment);
    task.environment.snapshot = None;

    let refused = solve_in_workspace(
        &task,
        &environment,
        &fixture.archive,
        &[],
        Scripted::new(|_| text("never asked")),
        options(),
        &protocol(0),
    )
    .await;

    assert!(matches!(
        refused,
        Err(SolveError::EnvironmentMismatch { .. })
    ));
}

/// The failing check prints a hundred rows, then names what is wrong.
const RED: &str = "seq 1 100; echo 'expected fixed' >&2; exit 1";

#[tokio::test]
async fn a_red_visible_check_earns_exactly_one_follow_up_with_its_output_tail() {
    let fixture = Fixture::new();
    let environment = fixture.environment();
    let model = Scripted::new(edits_then_stops);

    let solved = solve(
        &fixture,
        &environment,
        &[sh(RED)],
        model.clone(),
        &protocol(1),
    )
    .await
    .unwrap();

    assert_eq!(solved.feedback_rounds, 1, "one follow-up, though still red");
    let turns = model.user_turns();
    assert_eq!(turns.len(), 2, "the task, then one follow-up: {turns:?}");
    assert_eq!(turns[0], "Fix app.txt.");
    let follow_up = &turns[1];
    assert!(
        follow_up.contains(RED) && follow_up.contains("expected fixed"),
        "{follow_up}"
    );
    assert!(
        follow_up.contains("\n100\n") && !follow_up.contains("\n1\n"),
        "only the tail of the output is shown: {follow_up}"
    );
    assert!(
        !follow_up.contains(HIDDEN_NAME) && !follow_up.contains("diff --git"),
        "{follow_up}"
    );
}

#[tokio::test]
async fn a_green_visible_check_earns_no_follow_up() {
    let fixture = Fixture::new();
    let environment = fixture.environment();
    let model = Scripted::new(edits_then_stops);

    let solved = solve(
        &fixture,
        &environment,
        &[sh("grep -q fixed app.txt")],
        model.clone(),
        &protocol(3),
    )
    .await
    .unwrap();

    assert_eq!(solved.feedback_rounds, 0);
    assert_eq!(model.user_turns().len(), 1);
}

#[tokio::test]
async fn a_reply_that_leaves_the_tree_unchanged_earns_a_follow_up_even_when_no_check_is_red() {
    let fixture = Fixture::new();
    let environment = fixture.environment();
    // A model that announces the fix and ends its turn without making it.
    let model = Scripted::new(|req| {
        let users = req.messages.iter().filter(|m| m.role == Role::User).count();
        match (users, tool_results(req)) {
            (1, _) => text("I will now update app.txt."),
            (_, 0) => call(
                "c0",
                "write_file",
                serde_json::json!({"path": "app.txt", "text": "fixed\n", "append": false}),
            ),
            _ => text("done"),
        }
    });

    let solved = solve(&fixture, &environment, &[], model.clone(), &protocol(2))
        .await
        .unwrap();

    assert_eq!(solved.feedback_rounds, 1);
    assert_eq!(solved.patch.files, vec![PathBuf::from("app.txt")]);
    let turns = model.user_turns();
    assert_eq!(turns.len(), 2, "{turns:?}");
    assert!(turns[1].contains("changed no file"), "{}", turns[1]);
}

#[tokio::test]
async fn an_unchanged_tree_earns_no_follow_up_when_none_are_allowed() {
    let fixture = Fixture::new();
    let environment = fixture.environment();
    let model = Scripted::new(|_| text("I will now update app.txt."));

    let solved = solve(&fixture, &environment, &[], model.clone(), &protocol(0))
        .await
        .unwrap();

    assert_eq!(solved.feedback_rounds, 0);
    assert_eq!(model.user_turns().len(), 1);
}

#[tokio::test]
async fn follow_ups_stop_when_the_checks_turn_green() {
    let fixture = Fixture::new();
    let environment = fixture.environment();
    // Red until the second round: the follow-up makes the model fix it.
    let model = Scripted::new(|req| {
        let users = req.messages.iter().filter(|m| m.role == Role::User).count();
        match (users, tool_results(req)) {
            (1, _) => text("I think it is fine"),
            (_, 0) => call(
                "c0",
                "write_file",
                serde_json::json!({"path": "app.txt", "text": "fixed\n", "append": false}),
            ),
            _ => text("now it is"),
        }
    });

    let solved = solve(
        &fixture,
        &environment,
        &[sh("grep -q fixed app.txt")],
        model,
        &protocol(5),
    )
    .await
    .unwrap();

    assert_eq!(solved.feedback_rounds, 1);
    assert_eq!(solved.patch.files, vec![PathBuf::from("app.txt")]);
}

#[tokio::test]
async fn memory_the_web_and_other_models_are_not_offered() {
    let fixture = Fixture::new();
    let environment = fixture.environment();
    let model = Scripted::new(|_| text("nothing to do"));

    solve(&fixture, &environment, &[], model.clone(), &protocol(0))
        .await
        .unwrap();

    let seen = model.seen.lock().unwrap();
    let offered: Vec<&str> = seen[0].tools.iter().map(|t| t.name.as_str()).collect();
    for withheld in [
        "semantic_memory",
        "memory",
        "web_fetch",
        "web_search",
        "task",
    ] {
        assert!(WITHHELD_TOOLS.contains(&withheld), "{withheld} is withheld");
        assert!(
            !offered.contains(&withheld),
            "{withheld} is offered: {offered:?}"
        );
    }
    for needed in ["read_file", "write_file", "edit_file", "shell", "grep"] {
        assert!(offered.contains(&needed), "{needed}: {offered:?}");
    }
}

#[tokio::test]
async fn the_tool_cap_stops_an_episode_and_no_follow_up_follows() {
    let fixture = Fixture::new();
    let environment = fixture.environment();
    let model = Scripted::new(|_| {
        call(
            "c",
            "find_file",
            serde_json::json!({"pattern": "*.nothing"}),
        )
    });
    let capped = WorkProtocol {
        follow_ups: 2,
        max_tool_calls: Some(2),
        max_repeated_results: None,
        ..WorkProtocol::default()
    };

    let solved = solve(&fixture, &environment, &[sh(RED)], model, &capped)
        .await
        .unwrap();

    assert!(solved.report.stopped_by_tool_cap);
    assert_eq!(solved.solution.conclusion, RunConclusion::Cancelled);
    assert!(
        (3..=4).contains(&solved.report.tool_calls),
        "{}",
        solved.report.tool_calls
    );
    assert_eq!(
        solved.feedback_rounds, 0,
        "a stopped episode is not resumed"
    );
}

#[tokio::test]
async fn a_call_that_keeps_returning_the_same_answer_stops_the_episode() {
    let fixture = Fixture::new();
    let environment = fixture.environment();
    let model = Scripted::new(|req| {
        call(
            &format!("c{}", tool_results(req)),
            "read_file",
            serde_json::json!({"path": "app.txt"}),
        )
    });
    let bounded = WorkProtocol {
        follow_ups: 2,
        max_tool_calls: None,
        max_repeated_results: Some(3),
        ..WorkProtocol::default()
    };

    let solved = solve(&fixture, &environment, &[sh(RED)], model, &bounded)
        .await
        .unwrap();

    assert!(solved.report.stopped_by_repetition);
    assert!(solved
        .report
        .repeated_call
        .as_deref()
        .is_some_and(|c| c.starts_with("read_file")));
    assert_eq!(solved.feedback_rounds, 0);
}

/// A sandbox that claims to be a container and records the commands it is
/// asked to run, in the directory it is asked to run them in.
#[derive(Debug, Default)]
struct RecordingContainer {
    ran: Mutex<Vec<(PathBuf, Vec<String>)>>,
}

impl Sandbox for RecordingContainer {
    fn isolation(&self) -> Isolation {
        Isolation::Container
    }
    fn spec(&self) -> SandboxSpec {
        SandboxSpec {
            backend: Backend::Process,
            isolation: Isolation::Container,
            limits: Limits::default(),
            env: BTreeMap::new(),
            unconfined_commands: false,
            network_allowed: false,
            host_proc_visible: false,
        }
    }
    fn resolve(&self, _: &RuntimeSpec) -> Result<ResolvedRuntime, SandboxError> {
        unreachable!("a workspace resolves no runtime")
    }
    fn run(&self, _: &ResolvedRuntime, _: &CodeCall) -> Result<CodeResult, SandboxError> {
        unreachable!("a workspace runs no code calls")
    }
    fn run_command(&self, dir: &Path, call: &CommandCall) -> Result<CodeResult, SandboxError> {
        self.ran
            .lock()
            .unwrap()
            .push((dir.to_path_buf(), call.argv.clone()));
        Ok(CodeResult {
            stdout: "from the container\n".into(),
            stderr: String::new(),
            exit_code: Some(0),
            signal: None,
            timed_out: false,
            stdout_truncated: false,
            stderr_truncated: false,
        })
    }
}

#[tokio::test]
async fn where_the_sandbox_is_a_container_the_shell_runs_in_it() {
    let fixture = Fixture::new();
    let container = Arc::new(RecordingContainer::default());
    let environment = fixture.environment_in(container.clone());
    let model = Scripted::new(|req| match tool_results(req) {
        0 => call(
            "c0",
            "shell",
            serde_json::json!({"description": "say hi", "shell_command": "echo hi; ls"}),
        ),
        _ => text("done"),
    });

    solve(&fixture, &environment, &[], model.clone(), &protocol(0))
        .await
        .unwrap();

    let ran = container.ran.lock().unwrap();
    assert_eq!(ran.len(), 1, "{ran:?}");
    assert_eq!(
        ran[0].1,
        ["bash", "-c", "echo hi; ls"],
        "the model's command runs through bash in the sandbox"
    );
    assert!(
        model.tool_results().contains("from the container"),
        "the sandbox's answer reaches the model: {}",
        model.tool_results()
    );
    let seen = model.seen.lock().unwrap();
    let shells = seen[0].tools.iter().filter(|t| t.name == "shell").count();
    assert_eq!(shells, 1, "the sandbox shell replaces sven's, not joins it");
}

/// A file outside every tree, named the same on every call of this process.
fn outside_file() -> PathBuf {
    std::env::temp_dir().join(format!("splinter-outside-{}.py", std::process::id()))
}

#[tokio::test]
async fn where_the_sandbox_is_the_process_sandbox_the_worker_reaches_nothing_outside_its_tree() {
    let fixture = Fixture::new();
    let environment = fixture.environment();
    let secret = outside_file();
    std::fs::write(&secret, "ORIGINAL_SOURCE = 1\n").unwrap();
    let model = Scripted::new(|req| match tool_results(req) {
        0 => call(
            "c0",
            "read_file",
            serde_json::json!({"path": outside_file(), "start_line": 1, "end_line": 5}),
        ),
        1 => call(
            "c1",
            "shell",
            serde_json::json!({"description": "read it", "shell_command": format!("cat {}; echo from-shell", outside_file().display())}),
        ),
        _ => text("done"),
    });

    let solved = solve(&fixture, &environment, &[], model.clone(), &protocol(0))
        .await
        .unwrap();

    let seen = model.tool_results();
    assert!(seen.contains("from-shell"), "the shell ran: {seen}");
    assert!(
        !seen.contains("ORIGINAL_SOURCE"),
        "neither the file tool nor the shell read outside the tree: {seen}"
    );
    std::fs::remove_file(&secret).unwrap();
    let extra = solved.solution.trajectory.agent.extra.unwrap();
    assert!(extra["workspace_tree"].as_str().unwrap().ends_with("/tree"));
}
