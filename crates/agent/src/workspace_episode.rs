// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agent episodes that run a real agent
// runtime in a disposable workspace and keep its authoritative trajectory,
// for its clients. If your team needs expertise in collecting verified agent
// experience, you can procure our services by sending an email to
// info@swedishembedded.com.

//! One agent episode in a workspace under sven's compact run profile, for a
//! model too small for sven's full coding toolset: sven's own short system
//! prompt and three tools (read a file, write a file, run a command), no
//! compaction, and a bound on tool calls and time.
//!
//! The caller binds the model (`settings` already names the provider, the
//! endpoint and the sampling); this selects the profile and runs the episode
//! with the workspace as the project root. Expert demonstrations, a model's
//! exploration and probes differ only in the endpoint behind the agent, so
//! they are one function. The profile's id is recorded in the trajectory
//! (`agent.extra.sven.profile`).

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{anyhow, Context as _};
use sven_sdk::config::{compact_profile, Config, RunProfile};
use sven_sdk::{CancelToken, Engine, RunOptions, SessionEvent, Toolset};

/// The tools the profile offers, by sven's names.
pub const TOOLS: [&str; 3] = compact_profile::TOOLS;

/// The profile's id, as the trajectory records it.
pub const PROFILE: &str = compact_profile::ID;

/// Bounds on one episode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// Tool calls before the episode is stopped.
    pub max_tool_calls: u32,
    /// Wall-clock limit, in seconds.
    pub deadline_secs: u64,
    /// The model's context window, in tokens.
    pub context_tokens: u32,
    /// Longest reply, in tokens.
    pub max_output_tokens: u32,
}

/// What an episode did.
#[derive(Clone, Debug)]
pub struct Ran {
    /// How the run ended, as sven says.
    pub conclusion: String,
    /// Tool calls the agent started.
    pub tool_calls: u32,
    /// sven's trajectory of the episode (ATIF), as JSON.
    pub trajectory: serde_json::Value,
}

/// `settings` with the small-context profile applied.
#[must_use]
pub fn compact(mut settings: Config, limits: &Limits) -> Config {
    settings.agent.profile = RunProfile::Compact;
    settings.model.max_tokens = Some(limits.context_tokens);
    settings.model.max_output_tokens = Some(limits.max_output_tokens);
    settings.agent.max_tool_rounds = limits.max_tool_calls;
    // The whole transcript is the record; it is never rewritten.
    settings.agent.compaction_threshold = 1.0;
    settings
}

/// Runs one episode of `instruction` in `workspace` with the model `settings`
/// binds, under the small-context profile, with sven's audit trail written
/// to `audit_log`.
///
/// # Errors
/// The agent cannot be built, or its run fails outright (a refused request,
/// an unreachable endpoint). An agent that runs out of tool calls or time
/// is not an error: it is an episode that did not finish, and the verifier
/// judges what it left.
pub async fn run(
    workspace: &Path,
    instruction: &str,
    settings: Config,
    limits: &Limits,
    audit_log: PathBuf,
) -> anyhow::Result<Ran> {
    let engine = Engine::builder()
        .config(compact(settings, limits))
        .toolset(Toolset::coding())
        .project_root(workspace)
        .audit_log(audit_log)
        .build()
        .map_err(|e| anyhow!("building the agent: {e}"))?;
    let mut agent = engine.agent("agent");
    let mut events = agent.events();
    let stop = CancelToken::new();
    let budget = stop.clone();
    let cap = limits.max_tool_calls;
    let counter = tokio::spawn(async move {
        let mut calls = 0u32;
        while let Some(event) = events.recv().await {
            match event.event {
                SessionEvent::ToolCallStarted(_) => {
                    calls += 1;
                    if calls > cap {
                        budget.cancel_with("tool-call budget spent");
                    }
                }
                SessionEvent::TurnComplete | SessionEvent::Aborted { .. } => break,
                _ => {}
            }
        }
        calls
    });
    let sent = agent
        .send_with(
            instruction,
            RunOptions::new()
                .cancel(stop)
                .deadline(Duration::from_secs(limits.deadline_secs)),
        )
        .await;
    let tool_calls = counter.await.unwrap_or(0);
    let outcome = sent.map_err(|e| anyhow!("the agent run failed: {e}"))?;
    let trajectory =
        serde_json::to_value(agent.trajectory()).context("serialising the trajectory")?;
    Ok(Ran {
        conclusion: format!("{:?}", outcome.conclusion),
        tool_calls,
        trajectory,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy_endpoint::Endpoint;
    use serde_json::{json, Value};

    /// A policy that writes `done` to out.txt, then says so.
    fn writer(request: &Value) -> anyhow::Result<Value> {
        let tools = request["tools"].as_array().cloned().unwrap_or_default();
        let offered: Vec<&str> = tools
            .iter()
            .filter_map(|t| t["function"]["name"].as_str())
            .collect();
        assert_eq!(
            offered.len(),
            TOOLS.len(),
            "the profile offers exactly its tools: {offered:?}"
        );
        let results = request["messages"]
            .as_array()
            .map_or(0, |m| m.iter().filter(|m| m["role"] == "tool").count());
        Ok(if results == 0 {
            json!({ "role": "assistant", "content": null, "tool_calls": [{ "id": "c0", "type": "function",
                "function": { "name": "write_file", "arguments": "{\"path\":\"out.txt\",\"text\":\"done\\n\"}" } }] })
        } else {
            json!({ "role": "assistant", "content": "Wrote out.txt." })
        })
    }

    #[test]
    fn an_episode_runs_real_sven_tools_in_the_workspace() {
        let dir = tempfile::tempdir().unwrap();
        let endpoint = Endpoint::start("writer", Box::new(writer)).unwrap();
        let mut settings = Config::default();
        settings.model.provider = "openai".into();
        settings.model.name = "writer".into();
        settings.model.base_url = Some(endpoint.url().to_string());
        settings.model.api_key = Some("test".into());
        let limits = Limits {
            max_tool_calls: 4,
            deadline_secs: 60,
            context_tokens: 32768,
            max_output_tokens: 64,
        };
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let ran = runtime
            .block_on(run(
                dir.path(),
                "write done to out.txt",
                settings,
                &limits,
                dir.path().join("audit.jsonl"),
            ))
            .unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.path().join("out.txt")).unwrap(),
            "done\n"
        );
        assert_eq!(ran.tool_calls, 1);
        assert!(ran.trajectory["steps"]
            .as_array()
            .is_some_and(|s| !s.is_empty()));
        assert_eq!(ran.trajectory["agent"]["extra"]["sven"]["profile"], PROFILE);
    }

    /// The profile's point: the whole request fits a small model's context.
    #[test]
    fn the_request_fits_a_small_context() {
        let dir = tempfile::tempdir().unwrap();
        let endpoint = Endpoint::start("writer", Box::new(writer)).unwrap();
        let mut settings = Config::default();
        settings.model.provider = "openai".into();
        settings.model.name = "writer".into();
        settings.model.base_url = Some(endpoint.url().to_string());
        settings.model.api_key = Some("test".into());
        let limits = Limits {
            max_tool_calls: 4,
            deadline_secs: 60,
            context_tokens: 2048,
            max_output_tokens: 64,
        };
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime
            .block_on(run(
                dir.path(),
                "write done to out.txt",
                settings,
                &limits,
                dir.path().join("audit.jsonl"),
            ))
            .unwrap();
        let request = &endpoint.exchanges()[0].request;
        assert_eq!(
            request["messages"][0]["content"],
            compact_profile::SYSTEM_PROMPT
        );
        assert!(
            request.to_string().len() < 4 * 1024,
            "the first request is {} bytes",
            request.to_string().len()
        );
    }
}
