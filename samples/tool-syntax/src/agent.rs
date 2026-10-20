// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The agent an arm is measured with: sven's own coding agent, configured the
//! way a deployed `sven` is, pointed at a named model.
//!
//! # The tool set is a decision, made before measuring
//!
//! The agent is offered sven's coding tools under sven's default **deferred**
//! disclosure: the core tools (`shell`, `read_file`, `write_file`, `edit_file`,
//! `find_file`, `grep`, `todo`) and the `tools` tool carry their full
//! definitions, every other tool is a one-line index entry in the system
//! prompt. Both arms see this one prompt.
//!
//! Why not the full set: it is not what a deployed agent runs, and it puts
//! `task` and `context` - the most complex tools there are - in front of a
//! model on a task that needs none of them. Why not a hand-picked three: the
//! default is a configuration somebody else chose, which a result cannot be
//! accused of tuning. The hardware argument that once forced a choice (a
//! 2 GiB binding cap on one training tensor) no longer applies; the choice
//! stands on the two reasons above.
//!
//! Swedish Embedded AB implements closed-loop learning systems - agents that
//! improve from their own verified experience rather than from hand-written
//! training data - for its clients. If your team needs expertise in agent
//! training loops, sample design, or promotion gating for small models,
//! you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::time::Duration;

use splinter_sdk::agent::sven::{
    config, CancelToken, Engine, RunOptions, SessionEvent, Toolset, Turn,
};

use splinter_sdk::model::RemoteModel;

use crate::ServedModel;

/// Where an arm's model is served from.
#[derive(Clone, Debug)]
pub struct Endpoint<'a> {
    /// The resident (or adapter) the requests name.
    pub model: &'a ServedModel,
    /// The served endpoint; sven's own configuration when `None`.
    pub base_url: Option<&'a str>,
    /// The key for that endpoint.
    pub api_key: Option<&'a str>,
    /// Seed of this episode's sampling, sent with every request so a run can
    /// be repeated; the sampling temperature itself is sven's default.
    pub seed: u64,
}

/// How long an episode may run, from the family's own contract.
#[derive(Clone, Copy, Debug)]
pub struct Bounds {
    /// Tool calls the episode may make (`limits.tool_calls`). A call beyond
    /// it cancels the run; what the workspace holds then is still scored.
    pub tool_calls: usize,
    /// Wall-clock limit for the whole turn.
    pub deadline: Duration,
}

/// What the agent's tools actually did, as the kernel reported it.
#[derive(Clone, Debug, Default)]
pub struct ToolActivity {
    /// Tool calls started, counted from the event stream.
    pub calls: usize,
    /// How the run ended, as sven names it (`Success`, `Cancelled`, ...).
    pub conclusion: String,
    /// Everything the model was shown and everything it said.
    pub transcript: Vec<Turn>,
    /// One entry per failed call: the tool and the first line of what it
    /// said. Kept because "11 tool calls and nothing changed" is not a
    /// diagnosis, and the difference between a model that cannot form a call
    /// and a harness that cannot serve one is the whole question.
    pub failures: Vec<String>,
}

fn first_line(text: &str) -> String {
    text.lines()
        .next()
        .unwrap_or("")
        .chars()
        .take(120)
        .collect()
}

/// Give the agent `request` and let its tool loop run, in the process's
/// current directory. Calls are counted from the kernel's own event stream
/// rather than from anything the model reports about itself.
pub async fn run_agent(
    request: &str,
    endpoint: &Endpoint<'_>,
    bounds: Bounds,
) -> anyhow::Result<ToolActivity> {
    // `SVEN_MODEL` is an argument of the `sven` binary, not something the
    // SDK's loader reads, and the auto-detected default is a sentinel that
    // only resolves when exactly one chat model is served. A sample has
    // to name its own weights regardless.
    let mut settings = config::load(None)?;
    // The provider, the model name, the endpoint, the key, and no reasoning
    // block: the same configuration Splinter's own served arms are built
    // from, and the mode the adapter is trained for.
    RemoteModel {
        spec: endpoint.model.sven_model(),
        base_url: endpoint.base_url.map(String::from),
        api_key: endpoint.api_key.map(String::from),
        thinking: false,
        temperature: None,
    }
    .apply_to(&mut settings)?;
    let mut options = settings
        .model
        .driver_options
        .as_object()
        .cloned()
        .unwrap_or_default();
    options.insert("seed".into(), serde_json::json!(endpoint.seed));
    settings.model.driver_options = serde_json::Value::Object(options);

    // An embedded engine offers no built-in tools unless asked; the coding
    // preset is the agent a user of `sven` gets.
    let engine = Engine::builder()
        .config(settings)
        .toolset(Toolset::coding())
        .build()
        .map_err(|e| anyhow::anyhow!("building the engine: {e}"))?;

    let mut agent = engine.agent("agent");
    let mut events = agent.events();
    let stop = CancelToken::new();
    let budget = stop.clone();
    let counter = tokio::spawn(async move {
        let mut activity = ToolActivity::default();
        while let Some(run_event) = events.recv().await {
            match run_event.event {
                SessionEvent::ToolCallStarted(_) => {
                    activity.calls += 1;
                    if activity.calls > bounds.tool_calls {
                        budget.cancel();
                    }
                }
                SessionEvent::ToolCallFinished {
                    tool_name,
                    output,
                    is_error,
                    ..
                } => {
                    if is_error {
                        activity
                            .failures
                            .push(format!("{tool_name}: {}", first_line(&output)));
                    }
                }
                SessionEvent::TurnComplete | SessionEvent::Aborted { .. } => break,
                _ => {}
            }
        }
        activity
    });

    let sent = agent
        .send_with(
            request,
            RunOptions::new().cancel(stop).deadline(bounds.deadline),
        )
        .await;
    let mut activity = counter.await.unwrap_or_default();
    // The transcript is the episode's evidence, and it is captured whether
    // the turn succeeded or not: a failed episode is exactly the one whose
    // record explains why.
    activity.transcript = agent.state().transcript();
    activity.conclusion = format!("{:?}", sent.map_err(|e| anyhow::anyhow!("{e}"))?.conclusion);
    Ok(activity)
}
