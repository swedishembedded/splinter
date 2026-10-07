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
//! * Tools that reach the network or hand work to another model are
//!   withheld ([`WITHHELD_TOOLS`]): the model the worker runs on is the
//!   only one that sees the repository.
//! * Every event sven emits is handed to an [`Observer`] in order, and a
//!   gap in the stream (a slow observer) is reported as [`Observed::Dropped`]
//!   rather than silently skipped.
//! * A worker suspends into sven's serialisable [`AgentState`], and resumes
//!   from one, so a run can outlive its process.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use sven_sdk::config::Config;
use sven_sdk::model::ModelProvider;
use sven_sdk::{
    Agent, AgentState, ApprovalPolicy, CallError, CancelToken, Engine, RunConclusion, RunOptions,
    SessionEvent, Toolset, Usage,
};
use tokio::sync::broadcast::error::TryRecvError;
use tokio::sync::{broadcast, oneshot};

use crate::solve::{is_empty_reply, Solution, SolveError, SolveOptions, SOLVER_MODE};
use crate::system_prompt::UnderSystemPrompt;

/// The built-in tools a worker is never given: the two that reach the
/// network, delegation to another agent, asking a person, the tool that
/// switches the agent's mode or model, and the two that bring in state or
/// instructions from outside the task: the user's memory file, which
/// outlives a run, and the user's skills, whose listing is part of the
/// tool's description and so of every prompt.
pub const WITHHELD_TOOLS: &[&str] = &[
    "web_fetch",
    "web_search",
    "task",
    "ask_question",
    "system",
    "memory",
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

/// A coding agent working in one directory.
pub struct Worker {
    agent: Agent,
    options: WorkOptions,
}

/// The engine a worker runs on: sven's coding tools, confined to
/// `workspace`, without the tools in [`WITHHELD_TOOLS`], on `model` under
/// the system prompt in `options`.
fn engine(
    workspace: &Path,
    model: Arc<dyn ModelProvider>,
    options: &WorkOptions,
) -> Result<Engine, CallError> {
    let mut config: Config = options.solve.engine_config();
    config.tools.disabled = WITHHELD_TOOLS.iter().map(|t| (*t).to_string()).collect();
    Engine::builder()
        .config(config)
        .model_provider(UnderSystemPrompt::wrap(
            options.solve.provider(model),
            options.solve.system_prompt(),
        ))
        .toolset(Toolset::coding())
        .project_root(workspace)
        .approvals(ApprovalPolicy::Auto)
        .build()
}

impl Worker {
    /// A fresh worker in `workspace` (which must exist) on `model`.
    pub fn start(
        workspace: &Path,
        model: Arc<dyn ModelProvider>,
        options: WorkOptions,
    ) -> Result<Self, SolveError> {
        let agent = engine(workspace, model, &options)?.agent(SOLVER_MODE);
        Ok(Self { agent, options })
    }

    /// A worker continuing the conversation `state` was suspended from.
    pub fn resume(
        workspace: &Path,
        model: Arc<dyn ModelProvider>,
        options: WorkOptions,
        state: AgentState,
    ) -> Result<Self, SolveError> {
        let agent = engine(workspace, model, &options)?.resume(state)?;
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
    events: broadcast::Receiver<SessionEvent>,
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
                    Ok(event) => self.see(&event),
                    Err(broadcast::error::RecvError::Lagged(n)) => dropped += self.lost(n),
                    Err(broadcast::error::RecvError::Closed) => return dropped,
                },
                _ = tick.tick() => {
                    if self.external.as_ref().is_some_and(CancelToken::is_cancelled) {
                        self.stop.cancel();
                    }
                }
                _ = &mut finished => break,
            }
        }
        loop {
            match self.events.try_recv() {
                Ok(event) => self.see(&event),
                Err(TryRecvError::Lagged(n)) => dropped += self.lost(n),
                Err(TryRecvError::Empty | TryRecvError::Closed) => return dropped,
            }
        }
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
