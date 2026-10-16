// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements coding agents that work inside a confined
// directory and leave a complete, replayable record of what they did, for
// its clients. If your team needs expertise in agent runtimes or in
// measuring what an agent really did, you can procure our services by
// sending an email to info@swedishembedded.com.

//! A coding agent working in a directory it is confined to.
//!
//! [`solve`](crate::solve::solve) gives a solver the one tool its
//! environment names. A task that is a change to a repository needs sven's
//! own file, search and shell tools instead, and the bounds a person puts on
//! such a run: [`Worker`] is that agent. It is bounded by sven's run options
//! (deadline, output tokens, cancel) and by a cap on tool calls, which sven
//! has no run option for.
//!
//! * The file tools resolve every path against the workspace and refuse a
//!   path outside it. The shell starts there too, but it is not a sandbox:
//!   a caller that must contain what a command can do runs the worker in a
//!   disposable checkout, as the loop sample does.
//! * Tools that reach the network, hand work to another model or bring in
//!   state from outside the task are withheld ([`WITHHELD_TOOLS`]): the
//!   model the worker runs on is the only one that sees the repository.
//! * Every event sven emits is handed to an [`Observer`] in order, and a
//!   gap in the stream (a slow observer) is reported as [`Observed::Dropped`]
//!   rather than silently skipped.
//! * A worker suspends into sven's serialisable [`AgentState`], and resumes
//!   from one, so a run can outlive its process.
//! * [`solve_in_workspace`] is a whole episode on a task that changes a
//!   repository: a fresh copy of the task's snapshot, a worker in it, the
//!   visible checks run after each reply with their failures put back to the
//!   worker as a follow-up turn, and the change read back from git as the
//!   patch.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures::FutureExt;
use splinter_core::experience::Task;
use splinter_core::outcome::AttemptReport;
use splinter_sandbox::workspace::DEFAULT_EXCLUDED;
use splinter_sandbox::{
    Baseline, CodeResult, CommandCall, Isolation, Patch, ResolvedEnvironment, SandboxError,
    WorkspaceEnvironment,
};
use sven_sdk::config::Config;
use sven_sdk::model::ModelProvider;
use sven_sdk::tool::Tool;
use sven_sdk::{
    Agent, AgentState, ApprovalPolicy, CallError, CancelToken, Engine, EventStream, RunConclusion,
    RunOptions, SessionEvent, Toolset, Usage,
};
use tokio::sync::oneshot;

use crate::solve::{is_empty_reply, Solution, SolveError, SolveOptions, SOLVER_MODE};
use crate::system_prompt::UnderSystemPrompt;
use crate::workspace_shell::SandboxShell;

/// The built-in tools a worker is never given: the two that reach the
/// network, delegation to another agent, asking a person, the tool that
/// switches the agent's mode or model, and the three that bring in state or
/// instructions from outside the task: the user's memory file and the
/// semantic memory store, which outlive a run (and would carry one task's
/// material into the next), and the user's skills, whose listing is part
/// of the tool's description and so of every prompt.
pub const WITHHELD_TOOLS: &[&str] = &[
    "web_fetch",
    "web_search",
    "task",
    "ask_question",
    "system",
    "memory",
    "semantic_memory",
    "skill",
];

/// How often a running worker looks at its caller's cancel token: the
/// longest a cancel waits to be seen between two events.
const CANCEL_POLL: Duration = Duration::from_millis(100);

/// What an [`Observer`] is told about the stream of a worker's events.
#[derive(Debug)]
pub enum Observed<'a> {
    /// The next event, in the order sven emitted it.
    Event(&'a SessionEvent),
    /// This many events were lost because the observer fell behind sven's
    /// bounded event buffer. The history is incomplete at this point.
    Dropped(u64),
}

/// Receives every event of a worker's runs, in order.
pub type Observer = Arc<dyn Fn(Observed<'_>) + Send + Sync>;

/// Makes the observer of an episode on a workspace once its tree is laid
/// out: it is told where the tree is and the baseline the change is read
/// back against, so what the worker has changed can be read between events.
pub type TreeObserver = Arc<dyn Fn(&Path, &Baseline) -> Observer + Send + Sync>;

/// The bounds and the observer of one worker.
#[derive(Clone)]
pub struct WorkOptions {
    /// The bounds sven enforces: deadline, output tokens, cancel, stream
    /// idle limit, and the system prompt the model runs under.
    pub solve: SolveOptions,
    /// Tool calls the model may request in one [`Worker::send`]; the run is
    /// stopped as soon as it asks for one more. `None` sets no cap beyond
    /// sven's own limit on tool rounds.
    pub max_tool_calls: Option<u32>,
    /// How many times one call (same tool, same arguments) may return the
    /// same answer in one [`Worker::send`] before the run is stopped as
    /// going nowhere. A repeated call whose answer changes (a test run after
    /// an edit) is progress and is not counted. `None` sets no bound.
    pub max_repeated_results: Option<u32>,
    /// Told of every event, if set.
    pub observer: Option<Observer>,
    /// Makes a second observer for [`solve_in_workspace`], which lays its
    /// tree out itself; told of every event after `observer`. A worker
    /// started in a directory of the caller's has no use for it.
    pub tree_observer: Option<TreeObserver>,
}

impl WorkOptions {
    /// A worker bounded by `solve` alone.
    #[must_use]
    pub fn new(solve: SolveOptions) -> Self {
        Self {
            solve,
            max_tool_calls: None,
            max_repeated_results: None,
            observer: None,
            tree_observer: None,
        }
    }
}

/// What one [`Worker::send`] produced.
#[derive(Clone, Debug)]
pub struct WorkReport {
    /// The trajectory, final output, conclusion and usage, as for a solve.
    pub solution: Solution,
    /// The tool calls the model requested during this send.
    pub tool_calls: u32,
    /// Whether the run was stopped because it asked for more tool calls than
    /// the cap allows (its conclusion is then [`RunConclusion::Cancelled`]).
    pub stopped_by_tool_cap: bool,
    /// Whether the run was stopped because one call kept returning the same
    /// answer ([`WorkOptions::max_repeated_results`]).
    pub stopped_by_repetition: bool,
    /// The call that kept returning the same answer (tool name and the
    /// start of its arguments), when the run was stopped for it.
    pub repeated_call: Option<String>,
    /// Events the observer missed, if it fell behind.
    pub dropped_events: u64,
}

impl WorkReport {
    /// The facts of this attempt that an experience, being immutable, does
    /// not carry, for the annotation that keeps them. Only what the report
    /// measured is set; the rest stays absent.
    #[must_use]
    pub fn attempt_report(&self, wall: Duration) -> AttemptReport {
        AttemptReport {
            wall_seconds: Some(wall.as_secs_f64()),
            tool_calls: Some(self.tool_calls),
            stopped_by_tool_cap: self.stopped_by_tool_cap,
            stopped_by_repetition: self.stopped_by_repetition,
            ..AttemptReport::default()
        }
    }
}

/// A coding agent working in one directory.
pub struct Worker {
    agent: Agent,
    options: WorkOptions,
}

/// The engine a worker runs on: sven's coding tools, confined to
/// `workspace`, without the tools in [`WITHHELD_TOOLS`], on `model` under
/// the system prompt in `options`. A tool in `replacing` takes the place of
/// the built-in of its name.
fn engine(
    workspace: &Path,
    model: Arc<dyn ModelProvider>,
    options: &WorkOptions,
    replacing: &[Arc<dyn Tool>],
) -> Result<Engine, CallError> {
    let mut config: Config = options.solve.engine_config();
    config.tools.disabled = WITHHELD_TOOLS.iter().map(|t| (*t).to_string()).collect();
    let mut builder = Engine::builder()
        .config(config)
        .model_provider(UnderSystemPrompt::wrap(
            options.solve.provider(model),
            options.solve.system_prompt(),
        ))
        .toolset(Toolset::coding())
        .project_root(workspace)
        .approvals(ApprovalPolicy::Auto);
    for tool in replacing {
        builder = builder.tool(Arc::clone(tool));
    }
    builder.build()
}

impl Worker {
    /// A fresh worker in `workspace` (which must exist) on `model`.
    pub fn start(
        workspace: &Path,
        model: Arc<dyn ModelProvider>,
        options: WorkOptions,
    ) -> Result<Self, SolveError> {
        Self::start_replacing(workspace, model, options, &[])
    }

    /// A fresh worker as [`Worker::start`] makes one, with each tool in
    /// `replacing` taking the place of the built-in tool of its name.
    pub fn start_replacing(
        workspace: &Path,
        model: Arc<dyn ModelProvider>,
        options: WorkOptions,
        replacing: &[Arc<dyn Tool>],
    ) -> Result<Self, SolveError> {
        let agent = engine(workspace, model, &options, replacing)?.agent(SOLVER_MODE);
        Ok(Self { agent, options })
    }

    /// A worker continuing the conversation `state` was suspended from.
    pub fn resume(
        workspace: &Path,
        model: Arc<dyn ModelProvider>,
        options: WorkOptions,
        state: AgentState,
    ) -> Result<Self, SolveError> {
        let agent = engine(workspace, model, &options, &[])?.resume(state)?;
        Ok(Self { agent, options })
    }

    /// The conversation so far, in sven's serialisable form.
    #[must_use]
    pub fn suspend(self) -> AgentState {
        self.agent.suspend()
    }

    /// The bounds the next [`Worker::send`] runs under, which a caller
    /// narrows between sends (the time or tool calls an attempt has left).
    pub fn options_mut(&mut self) -> &mut WorkOptions {
        &mut self.options
    }

    /// What the worker has done, as sven exports it.
    #[must_use]
    pub fn trajectory(&self) -> sven_sdk::atif::Trajectory {
        self.agent.trajectory()
    }

    /// Sends `prompt` and works until the run concludes or a bound stops it.
    /// A run stopped by a bound is a report with that conclusion, not an
    /// error.
    pub async fn send(&mut self, prompt: &str) -> Result<WorkReport, SolveError> {
        let stop = CancelToken::new();
        let mut run = RunOptions::new()
            .deadline(self.options.solve.deadline)
            .cancel(stop.clone());
        if let Some(tokens) = self.options.solve.max_output_tokens {
            run = run.max_output_tokens(tokens);
        }

        let watch = Watch {
            events: self.agent.events(),
            stop,
            external: self.options.solve.cancel.clone(),
            cap: self.options.max_tool_calls,
            repeat_cap: self.options.max_repeated_results,
            repeats: Mutex::new(Repeats::default()),
            repeated: Arc::new(AtomicBool::new(false)),
            repeated_call: Arc::new(Mutex::new(None)),
            observer: self.options.observer.clone(),
            tool_calls: Arc::new(AtomicU32::new(0)),
            capped: Arc::new(AtomicBool::new(false)),
        };
        let (tool_calls, capped, repeated, repeated_call) = (
            watch.tool_calls.clone(),
            watch.capped.clone(),
            watch.repeated.clone(),
            watch.repeated_call.clone(),
        );
        let (done, finished) = oneshot::channel();
        let watcher = tokio::spawn(watch.run(finished));

        let outcome = self.agent.send_with(prompt, run).await;
        // The watcher drains what is buffered before it returns, so the
        // observer has seen every event of this run by the time we do.
        let _ = done.send(());
        let dropped_events = watcher.await.unwrap_or(0);

        let (final_output, conclusion, usage) = match outcome {
            Ok(outcome) => (
                (outcome.conclusion == RunConclusion::Success).then_some(outcome.reply),
                outcome.conclusion,
                outcome.usage,
            ),
            Err(e) if is_empty_reply(&e) => (None, RunConclusion::AgentError, Usage::default()),
            Err(e) => return Err(e.into()),
        };
        Ok(WorkReport {
            solution: Solution {
                trajectory: self.agent.trajectory(),
                final_output,
                conclusion,
                usage,
            },
            tool_calls: tool_calls.load(Ordering::SeqCst),
            stopped_by_tool_cap: capped.load(Ordering::SeqCst),
            stopped_by_repetition: repeated.load(Ordering::SeqCst),
            repeated_call: repeated_call.lock().ok().and_then(|c| c.clone()),
            dropped_events,
        })
    }
}

/// The task that watches one send's events: it counts tool calls, stops the
/// run at the cap or when the caller cancels, and hands every event to the
/// observer.
struct Watch {
    events: EventStream,
    stop: CancelToken,
    external: Option<CancelToken>,
    cap: Option<u32>,
    repeat_cap: Option<u32>,
    repeats: Mutex<Repeats>,
    repeated: Arc<AtomicBool>,
    repeated_call: Arc<Mutex<Option<String>>>,
    observer: Option<Observer>,
    tool_calls: Arc<AtomicU32>,
    capped: Arc<AtomicBool>,
}

/// A call as `name\u{1}arguments`, shortened for a message.
fn describe(asked: &str) -> String {
    let (name, args) = asked.split_once('\u{1}').unwrap_or((asked, ""));
    let args: String = args.chars().take(160).collect();
    format!("{name} {args}")
}

/// What a send has asked and been answered: each call by its id, and how
/// often each (call, answer) pair has come back.
#[derive(Default)]
struct Repeats {
    asked: HashMap<String, String>,
    answered: HashMap<String, u32>,
}

impl Watch {
    /// Runs until the send is `finished`, then drains what is buffered;
    /// returns how many events were lost to a slow observer.
    async fn run(mut self, mut finished: oneshot::Receiver<()>) -> u64 {
        let mut dropped = 0;
        let mut tick = tokio::time::interval(CANCEL_POLL);
        loop {
            tokio::select! {
                received = self.events.recv() => match received {
                    Some(event) => dropped += self.handle(&event.event),
                    None => return dropped,
                },
                _ = tick.tick() => {
                    if self.external.as_ref().is_some_and(CancelToken::is_cancelled) {
                        self.stop.cancel();
                    }
                }
                _ = &mut finished => break,
            }
        }
        // What the finished send left buffered: `now_or_never` reads an
        // event that is ready and stops at the first that is not.
        while let Some(Some(event)) = self.events.recv().now_or_never() {
            dropped += self.handle(&event.event);
        }
        dropped
    }

    /// Takes one event of the stream; returns how many events it says were
    /// lost (a gap marker stands in for them, it is not an event of the run).
    fn handle(&self, event: &SessionEvent) -> u64 {
        if let SessionEvent::EventsDropped { count } = event {
            return self.lost(*count);
        }
        self.see(event);
        0
    }

    fn see(&self, event: &SessionEvent) {
        if matches!(event, SessionEvent::ToolCallStarted(_)) {
            let calls = self.tool_calls.fetch_add(1, Ordering::SeqCst) + 1;
            if self.cap.is_some_and(|cap| calls > cap) {
                self.capped.store(true, Ordering::SeqCst);
                self.stop.cancel();
            }
        }
        self.watch_repetition(event);
        if let Some(observer) = &self.observer {
            observer(Observed::Event(event));
        }
    }

    /// Counts how often a call has returned the same answer and stops the
    /// run when that reaches the bound.
    fn watch_repetition(&self, event: &SessionEvent) {
        let Some(bound) = self.repeat_cap else {
            return;
        };
        let Ok(mut repeats) = self.repeats.lock() else {
            return;
        };
        match event {
            SessionEvent::ToolCallStarted(call) => {
                repeats
                    .asked
                    .insert(call.id.clone(), format!("{}\u{1}{}", call.name, call.args));
            }
            SessionEvent::ToolCallFinished {
                call_id, output, ..
            } => {
                let Some(asked) = repeats.asked.get(call_id).cloned() else {
                    return;
                };
                let seen = repeats
                    .answered
                    .entry(format!("{asked}\u{1}{output}"))
                    .or_insert(0);
                *seen += 1;
                if *seen >= bound {
                    if let Ok(mut call) = self.repeated_call.lock() {
                        call.get_or_insert_with(|| describe(&asked));
                    }
                    self.repeated.store(true, Ordering::SeqCst);
                    self.stop.cancel();
                }
            }
            _ => {}
        }
    }

    fn lost(&self, count: u64) -> u64 {
        if let Some(observer) = &self.observer {
            observer(Observed::Dropped(count));
        }
        count
    }
}

/// The lines of a failing check's output a follow-up shows.
const FEEDBACK_LINES: usize = 40;

/// The most characters of a failing check's output a follow-up shows: a
/// few very long lines must not fill the conversation.
const FEEDBACK_CHARS: usize = 4_000;

/// How one episode on a workspace is run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorkProtocol {
    /// How many follow-up user turns the worker may be sent after the
    /// visible checks stay red; `0` sends none and runs no check.
    pub follow_ups: u32,
    /// Tool calls the model may request over the whole episode, follow-ups
    /// included (each send is held to what is left). `None` sets no cap
    /// beyond sven's own limit on tool rounds.
    pub max_tool_calls: Option<u32>,
    /// How many times one call may return the same answer in the episode
    /// before it is stopped as going nowhere; see
    /// [`WorkOptions::max_repeated_results`].
    pub max_repeated_results: Option<u32>,
}

/// What an episode on a workspace produced.
#[derive(Clone, Debug)]
pub struct WorkspaceSolution {
    /// The trajectory of the whole episode, follow-ups included, and how
    /// its last send concluded. The final output is the patch text: it is
    /// recorded whether or not the run was stopped, because the work on
    /// disk is what is judged. The agent's last reply is in the trajectory.
    pub solution: Solution,
    /// What the worker changed, read back from git (never from what the
    /// worker said), without sven's own working directory.
    pub patch: Patch,
    /// The follow-up turns that were sent.
    pub feedback_rounds: u32,
    /// The sends added up: tool calls and dropped events summed, a stop by
    /// the tool cap or by repetition if any send had one, and the first
    /// repeated call. Its `solution` is the last send's.
    pub report: WorkReport,
}

/// Solves `task`, a change to the repository `env` is a snapshot of, on
/// `model`.
///
/// `archive` is the tar the snapshot digests; it is laid out in a fresh
/// directory that goes when the call returns, and a worker with sven's
/// file, search and shell tools works there on `task`'s instruction. Where
/// `env`'s sandbox is a container the shell runs in it (see
/// [`SandboxShell`]); otherwise sven's own shell runs on the host, and the
/// environment's record says the sandbox does not isolate.
///
/// After each reply that ends the worker's turn, the `visible` checks run in
/// the tree. While one is red and `protocol.follow_ups` and the time and tool
/// calls allow, the worker is sent one more user turn naming each failing
/// check and the tail of its output; a reply that left the tree unchanged is
/// sent one more user turn too, whatever the checks say, asking for the edit. That is all the worker learns from a
/// check: the checks given here are the ones the task shows its solver, and
/// nothing the tree does not hold can reach it. A run stopped by a bound
/// (cancel, deadline, tool cap, repetition) is not resumed.
///
/// `options.solve.deadline` bounds the whole episode, not each send.
///
/// The tree the worker sees is `archive` and nothing else: material a task
/// withholds must not be in it (the task's generator leaves it out), and
/// nothing is given back to the tree after the episode.
pub async fn solve_in_workspace(
    task: &Task,
    env: &WorkspaceEnvironment,
    archive: &Path,
    visible: &[CommandCall],
    model: Arc<dyn ModelProvider>,
    mut options: WorkOptions,
    protocol: &WorkProtocol,
) -> Result<WorkspaceSolution, SolveError> {
    let offered = ResolvedEnvironment::Workspace(env.clone()).record()?;
    if offered != task.environment {
        return Err(SolveError::EnvironmentMismatch {
            task_kind: task.environment.kind.clone(),
            task_snapshot: task.environment.snapshot.clone(),
            offered_kind: offered.kind,
            offered_snapshot: offered.snapshot,
        });
    }
    let scratch = tempfile::tempdir().map_err(|source| SandboxError::Io {
        path: "temporary directory".into(),
        source,
    })?;
    let tree = scratch.path().join("tree");
    let baseline = env.materialize(archive, &tree)?;
    if let Some(make) = options.tree_observer.take() {
        options.observer = Some(both(options.observer.take(), make(&tree, &baseline)));
    }

    let confined: Vec<Arc<dyn Tool>> = match env.sandbox().isolation() {
        Isolation::Container => vec![Arc::new(SandboxShell::new(env.clone(), tree.clone()))],
        Isolation::None => Vec::new(),
    };
    let episode = options.solve.deadline;
    options.max_repeated_results = protocol.max_repeated_results;
    let mut worker = Worker::start_replacing(&tree, model, options, &confined)?;

    let started = Instant::now();
    let mut message = task.instruction.clone();
    let mut feedback_rounds = 0;
    let mut sum = Sum::default();
    let last = loop {
        let bounds = worker.options_mut();
        bounds.solve.deadline = episode.saturating_sub(started.elapsed());
        bounds.max_tool_calls = protocol
            .max_tool_calls
            .map(|cap| cap.saturating_sub(sum.tool_calls));
        let report = worker.send(&message).await?;
        sum.add(&report);

        let ended_itself = report.solution.conclusion == RunConclusion::Success
            && !report.stopped_by_tool_cap
            && !report.stopped_by_repetition;
        if !ended_itself || feedback_rounds >= protocol.follow_ups {
            break report;
        }
        let failures = failing(env, &tree, visible).await?;
        let unchanged = tree_unchanged(env, &tree, &baseline).await?;
        let out_of_time = started.elapsed() >= episode;
        let out_of_calls = protocol
            .max_tool_calls
            .is_some_and(|cap| sum.tool_calls >= cap);
        if (failures.is_empty() && !unchanged) || out_of_time || out_of_calls {
            break report;
        }
        message = if unchanged {
            NO_CHANGE_FOLLOW_UP.to_string()
        } else {
            follow_up(&failures)
        };
        feedback_rounds += 1;
    };

    let patch = {
        let (env, tree) = (env.clone(), tree.clone());
        blocking("reading the change back", move || {
            env.diff_since(&tree, &baseline, &DEFAULT_EXCLUDED)
        })
        .await?
    };
    let mut solution = last.solution;
    solution.final_output = Some(String::from_utf8_lossy(&patch.diff).into_owned());
    Ok(WorkspaceSolution {
        report: sum.into_report(solution.clone()),
        solution,
        patch,
        feedback_rounds,
    })
}

/// An observer that tells `first`, if there is one, and then `second`.
fn both(first: Option<Observer>, second: Observer) -> Observer {
    match first {
        None => second,
        Some(first) => Arc::new(move |seen: Observed<'_>| {
            first(match &seen {
                Observed::Event(event) => Observed::Event(event),
                Observed::Dropped(count) => Observed::Dropped(*count),
            });
            second(seen);
        }),
    }
}

/// The sends of one episode, added up.
#[derive(Default)]
struct Sum {
    tool_calls: u32,
    stopped_by_tool_cap: bool,
    stopped_by_repetition: bool,
    repeated_call: Option<String>,
    dropped_events: u64,
}

impl Sum {
    fn add(&mut self, report: &WorkReport) {
        self.tool_calls += report.tool_calls;
        self.stopped_by_tool_cap |= report.stopped_by_tool_cap;
        self.stopped_by_repetition |= report.stopped_by_repetition;
        if self.repeated_call.is_none() {
            self.repeated_call.clone_from(&report.repeated_call);
        }
        self.dropped_events += report.dropped_events;
    }

    fn into_report(self, solution: Solution) -> WorkReport {
        WorkReport {
            solution,
            tool_calls: self.tool_calls,
            stopped_by_tool_cap: self.stopped_by_tool_cap,
            stopped_by_repetition: self.stopped_by_repetition,
            repeated_call: self.repeated_call,
            dropped_events: self.dropped_events,
        }
    }
}

/// A visible check that did not pass, as the worker is told of it.
struct Failure {
    name: String,
    result: CodeResult,
}

/// Runs `visible` in `tree` and returns the checks that did not exit with
/// zero in time.
async fn failing(
    env: &WorkspaceEnvironment,
    tree: &Path,
    visible: &[CommandCall],
) -> Result<Vec<Failure>, SandboxError> {
    let mut failures = Vec::new();
    for check in visible {
        let (env, tree, check) = (env.clone(), tree.to_path_buf(), check.clone());
        let name = check.argv.join(" ");
        let result = blocking("running a visible check", move || env.run(&tree, &check)).await?;
        if result.timed_out || result.exit_code != Some(0) {
            failures.push(Failure { name, result });
        }
    }
    Ok(failures)
}

/// The user turn sent when a worker ended its turn with the tree as it was
/// given: a model may announce a change and stop without calling an edit
/// tool, and a task whose checks are hidden shows no red check to say so.
const NO_CHANGE_FOLLOW_UP: &str = "You have changed no file yet. Make the change now by calling \
the edit tools; do not only describe it.";

/// Whether the worker has left `tree` as `baseline` laid it out.
async fn tree_unchanged(
    env: &WorkspaceEnvironment,
    tree: &Path,
    baseline: &Baseline,
) -> Result<bool, SandboxError> {
    let (env, tree, baseline) = (env.clone(), tree.to_path_buf(), baseline.clone());
    let patch = blocking("reading the change back", move || {
        env.diff_since(&tree, &baseline, &DEFAULT_EXCLUDED)
    })
    .await?;
    Ok(patch.diff.is_empty())
}

/// Runs the blocking `work` off the async executor.
async fn blocking<T: Send + 'static>(
    what: &str,
    work: impl FnOnce() -> Result<T, SandboxError> + Send + 'static,
) -> Result<T, SandboxError> {
    tokio::task::spawn_blocking(work)
        .await
        .map_err(|e| SandboxError::Workspace {
            what: what.to_string(),
            detail: e.to_string(),
        })?
}

/// The user turn that tells a worker its visible checks are still red.
fn follow_up(failures: &[Failure]) -> String {
    let mut text = String::from(
        "The checks shown with the task still fail on your change. Find the cause and fix it.\n",
    );
    for Failure { name, result } in failures {
        let how = if result.timed_out {
            "timed out".to_string()
        } else {
            result.exit_code.map_or_else(
                || "was killed".to_string(),
                |c| format!("failed with exit code {c}"),
            )
        };
        let output = format!("{}{}", result.stdout, result.stderr);
        text.push_str(&format!("\nCheck `{name}` {how}:\n{}\n", tail(&output)));
    }
    text
}

/// The end of `output`: its last [`FEEDBACK_LINES`] lines, and of those at
/// most [`FEEDBACK_CHARS`] characters. What a follow-up shows of a failing
/// check, and what a supervisor puts in its own feedback.
#[must_use]
pub fn tail(output: &str) -> String {
    let lines: Vec<&str> = output.lines().collect();
    let kept = lines[lines.len().saturating_sub(FEEDBACK_LINES)..].join("\n");
    let skip = kept.chars().count().saturating_sub(FEEDBACK_CHARS);
    kept.chars().skip(skip).collect()
}
