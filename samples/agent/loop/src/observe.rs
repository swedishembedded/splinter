// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements self-improving coding agents whose every
// step is auditable, for its clients. If your team needs expertise in agent
// evaluation or locally operated coding agents, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Turns what sven reports into the loop's trace.
//!
//! sven emits one stream of session events while an agent works. This
//! observer writes the ones that are evidence (tool requests and results,
//! the model's messages, plan updates, usage) as trace events, names the
//! files a tool call changed, keeps the running usage, and stops the run
//! when a remote model's reported cost passes its cap. Streamed text deltas
//! and reasoning chunks are not recorded: the loop audits what the agent
//! did and said, not hidden reasoning.
//!
//! File changes are attributed by comparing the checkout before and after
//! each finishing tool call. sven runs one call at a time but emits its
//! events asynchronously, so an attribution can lag a call by one; the
//! patch read back from git at the end of the attempt is the authority.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};
use splinter_sdk::agent::sven::SessionEvent;
use splinter_sdk::agent::work::{Observed, Observer};
use splinter_sdk::agent::CancelToken;

use crate::repo::{dirty_snapshot, hash_in_rev};
use crate::trace::Tracer;

/// The built-in tools whose result can have changed a file.
const MUTATING_TOOLS: &[&str] = &["write_file", "edit_file", "shell"];

/// What one attempt's observer gathers.
#[derive(Clone, Debug, Default)]
pub struct Gathered {
    /// Input tokens the provider reported.
    pub input_tokens: u64,
    /// Output tokens the provider reported.
    pub output_tokens: u64,
    /// Dollars the provider reported, if any call did.
    pub cost_usd: Option<f64>,
    /// Model calls whose price was not reported.
    pub calls_without_price: u32,
    /// The last complete message the model wrote.
    pub last_message: Option<String>,
    /// Whether the cost cap was passed.
    pub over_budget: bool,
    /// Events lost to a slow observer.
    pub dropped: u64,
}

struct Inner {
    requests: HashMap<String, u64>,
    snapshot: BTreeMap<String, Option<String>>,
    gathered: Gathered,
}

/// The observer of one attempt.
pub struct Observation {
    tracer: Arc<Tracer>,
    work: PathBuf,
    baseline: String,
    remote: bool,
    cost_cap: Option<f64>,
    stop: CancelToken,
    inner: Mutex<Inner>,
}

impl Observation {
    /// An observer of an attempt in `work`, started from `baseline`. `stop`
    /// is cancelled when `cost_cap` is passed.
    #[must_use]
    pub fn new(
        tracer: Arc<Tracer>,
        work: PathBuf,
        baseline: String,
        remote: bool,
        cost_cap: Option<f64>,
        stop: CancelToken,
    ) -> Arc<Self> {
        Arc::new(Self {
            tracer,
            work,
            baseline,
            remote,
            cost_cap,
            stop,
            inner: Mutex::new(Inner {
                requests: HashMap::new(),
                snapshot: BTreeMap::new(),
                gathered: Gathered::default(),
            }),
        })
    }

    /// What has been gathered so far.
    #[must_use]
    pub fn gathered(&self) -> Gathered {
        self.inner
            .lock()
            .map(|i| i.gathered.clone())
            .unwrap_or_default()
    }

    /// The observer to hand to the worker.
    #[must_use]
    pub fn observer(self: &Arc<Self>) -> Observer {
        let this = Arc::clone(self);
        Arc::new(move |seen: Observed<'_>| this.observe(&seen))
    }

    fn emit(&self, kind: &str, parent: Option<u64>, data: Value) -> Option<u64> {
        // A trace that cannot be written must not take the run down with
        // it silently: the error is kept as the next event if possible.
        match self.tracer.emit(kind, parent, data) {
            Ok(id) => Some(id),
            Err(e) => {
                let _ = self
                    .tracer
                    .emit("trace_error", None, json!({"error": format!("{e:#}")}));
                None
            }
        }
    }

    fn observe(&self, seen: &Observed<'_>) {
        match seen {
            Observed::Dropped(n) => {
                if let Ok(mut inner) = self.inner.lock() {
                    inner.gathered.dropped += n;
                }
                self.emit("events_dropped", None, json!({"count": n}));
            }
            Observed::Event(event) => self.event(event),
        }
    }

    fn event(&self, event: &SessionEvent) {
        match event {
            SessionEvent::ToolCallStarted(call) => {
                let id = self.emit(
                    "tool_request",
                    None,
                    json!({
                        "call_id": call.id,
                        "tool": call.name,
                        "arguments": call.args,
                        "cwd": self.work,
                    }),
                );
                if let (Some(id), Ok(mut inner)) = (id, self.inner.lock()) {
                    inner.requests.insert(call.id.clone(), id);
                }
            }
            SessionEvent::ToolCallFinished {
                call_id,
                tool_name,
                output,
                is_error,
            } => {
                let parent = self
                    .inner
                    .lock()
                    .ok()
                    .and_then(|i| i.requests.get(call_id).copied());
                let result = self.emit(
                    "tool_result",
                    parent,
                    json!({
                        "call_id": call_id,
                        "tool": tool_name,
                        "is_error": is_error,
                        "output": output,
                    }),
                );
                if MUTATING_TOOLS.contains(&tool_name.as_str()) {
                    self.mutations(result);
                }
            }
            SessionEvent::TextComplete(text) => {
                if let Ok(mut inner) = self.inner.lock() {
                    inner.gathered.last_message = Some(text.clone());
                }
                self.emit("assistant_message", None, json!({"text": text}));
            }
            SessionEvent::TodoUpdate(items) => {
                let plan = serde_json::to_value(items).unwrap_or(Value::Null);
                self.emit("plan_update", None, json!({"items": plan}));
            }
            SessionEvent::TokenUsage {
                input,
                output,
                cost_usd,
                ..
            } => self.usage(*input, *output, *cost_usd),
            SessionEvent::Error(message) => {
                self.emit("agent_error", None, json!({"message": message}));
            }
            SessionEvent::Aborted { .. } => {
                self.emit("run_aborted", None, json!({}));
            }
            _ => {}
        }
    }

    fn usage(&self, input: u32, output: u32, cost: Option<f64>) {
        let (over, total) = {
            let Ok(mut inner) = self.inner.lock() else {
                return;
            };
            let g = &mut inner.gathered;
            g.input_tokens += u64::from(input);
            g.output_tokens += u64::from(output);
            match cost {
                Some(c) => g.cost_usd = Some(g.cost_usd.unwrap_or(0.0) + c),
                None if self.remote => g.calls_without_price += 1,
                None => {}
            }
            let over = self
                .cost_cap
                .zip(g.cost_usd)
                .is_some_and(|(cap, spent)| spent > cap);
            g.over_budget |= over;
            (over, g.cost_usd)
        };
        self.emit(
            "model_usage",
            None,
            json!({"input_tokens": input, "output_tokens": output, "cost_usd": cost,
                   "cost_total_usd": total, "priced": cost.is_some() || !self.remote}),
        );
        if over {
            self.emit(
                "limit_reached",
                None,
                json!({"limit": "cost", "cap_usd": self.cost_cap, "spent_usd": total}),
            );
            self.stop.cancel();
        }
    }

    /// Names each file that differs from the last snapshot of the checkout.
    fn mutations(&self, parent: Option<u64>) {
        let Ok(now) = dirty_snapshot(&self.work) else {
            return;
        };
        let before = match self.inner.lock() {
            Ok(mut inner) => std::mem::replace(&mut inner.snapshot, now.clone()),
            Err(_) => return,
        };
        let mut changed: Vec<(String, Option<String>, Option<String>)> = Vec::new();
        for (path, after) in &now {
            let was = before.get(path).cloned();
            if was.as_ref() != Some(after) {
                let prior = was.unwrap_or_else(|| {
                    hash_in_rev(&self.work, &self.baseline, path).ok().flatten()
                });
                changed.push((path.clone(), prior, after.clone()));
            }
        }
        for (path, was) in &before {
            if !now.contains_key(path) {
                // Back to the committed content: the file was restored.
                let after = hash_in_rev(&self.work, &self.baseline, path).ok().flatten();
                changed.push((path.clone(), was.clone(), after));
            }
        }
        for (path, before, after) in changed {
            let op = match (&before, &after) {
                (None, Some(_)) => "added",
                (Some(_), None) => "deleted",
                _ => "modified",
            };
            self.emit(
                "file_mutation",
                parent,
                json!({"path": path, "op": op, "before": before, "after": after,
                       "attribution": "approximate: see the attempt's patch"}),
            );
        }
    }
}
