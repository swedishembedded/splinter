// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The replay driven by a sven harness document (`harness/replay.yaml`).
//!
//! Every prompt, the plan's retry on refusal, the agent's tool allowance and
//! repetition stop and the "you stopped before finishing" follow-up are
//! declared in the document; this module only supplies its inputs and reads
//! the results. It needs sven with the v2 workflow language (the
//! `run.trajectory_file` and `cwd_input` gate inputs, `in_workspace`, model
//! roles): sven at or after 95f668d3.
//!
//! What the document cannot express yet stays on the Rust path, chosen
//! per step by [`HarnessConfig::drives_plan`] and
//! [`HarnessConfig::drives_work`]: the number of plan retries and follow-ups
//! are literals in the document, the thinking budget cannot be left unset,
//! and the supervisor corrections wait for a configurable gate output cap.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context as _, Result};
use serde_json::{json, Value};
use splinter_sdk::agent::sven::atif::Trajectory;
use splinter_sdk::agent::sven::model::ModelProvider;
use splinter_sdk::agent::sven::{Engine, RunConclusion, RunOptions, Toolset};

use crate::control::Control;
use crate::mask::MaskedDiff;
use crate::plan::{PlanCheck, PlanFailure};
use crate::plan_prompt::PlanStyle;

/// The documents the sample ships: the plan stage and the grounded work, and
/// the work of the plan-only control (a sven agent task cannot carry its own
/// system prompt, so the control is a document of its own).
pub const DEFAULT_DOCUMENT: &str = include_str!("../harness/replay.yaml");
/// The plan-only control's document.
pub const DEFAULT_PLAN_ONLY_DOCUMENT: &str = include_str!("../harness/replay-plan-only.yaml");

/// Plan retries and follow-ups the shipped document declares.
const DOCUMENT_PLAN_TRIES: usize = 2;
const DOCUMENT_FOLLOW_UPS: usize = 2;
/// Seconds to wait for a first response when the caller names no limit
/// (the document needs a number; a local model is silent through its prefill).
const DEFAULT_FIRST_RESPONSE_SECS: u64 = 600;

/// How the sampling of every request is stated to the document.
#[derive(Clone, Debug)]
pub struct Decoding {
    /// Sampling temperature.
    pub temperature: f64,
    /// Nucleus sampling.
    pub top_p: f64,
    /// Candidates sampled among.
    pub top_k: u64,
    /// The seed every request carries.
    pub seed: u64,
    /// Whether the model reasons before it answers.
    pub thinking: bool,
}

/// The harness document and what its gates need.
#[derive(Clone, Debug)]
pub struct HarnessConfig {
    /// The text of the document with the plan stage and the grounded work.
    pub document: String,
    /// The text of the document with the work of the plan-only control.
    pub plan_only_document: String,
    /// The `repo-replay` executable the document's gates run.
    pub check_bin: PathBuf,
    /// Sampling stated to the document.
    pub decoding: Decoding,
    /// A thinking budget the caller asked for; the document cannot carry it.
    pub thinking_budget: Option<u32>,
}

impl HarnessConfig {
    /// Whether the document writes the plan under `tries` tries.
    #[must_use]
    pub fn drives_plan(&self, tries: usize) -> bool {
        tries == DOCUMENT_PLAN_TRIES && self.thinking_budget.is_none()
    }

    /// Whether the document runs the work under these settings.
    #[must_use]
    pub fn drives_work(&self, follow_ups: usize, corrections: usize) -> bool {
        follow_ups == DOCUMENT_FOLLOW_UPS && corrections == 0 && self.thinking_budget.is_none()
    }

    fn engine(
        &self,
        planner: Arc<dyn ModelProvider>,
        agent: Arc<dyn ModelProvider>,
    ) -> Result<Engine> {
        Engine::builder()
            .model_provider(Arc::clone(&agent))
            .model_role("planner", planner)
            .model_role("agent", agent)
            .toolset(Toolset::coding())
            .build()
            .context("building the harness engine")
    }

    /// The inputs every stage takes.
    fn inputs(&self, first_response: Option<Duration>) -> serde_json::Map<String, Value> {
        let d = &self.decoding;
        let mut inputs = serde_json::Map::new();
        for (key, value) in [
            ("temperature", json!(d.temperature)),
            ("top_p", json!(d.top_p)),
            ("top_k", json!(d.top_k)),
            ("seed", json!(d.seed)),
            ("thinking", json!(d.thinking)),
            (
                "first_response_secs",
                json!(first_response.map_or(DEFAULT_FIRST_RESPONSE_SECS, |t| t.as_secs().max(1))),
            ),
        ] {
            inputs.insert(key.into(), value);
        }
        inputs
    }
}

/// The last words the agent side of conversation `name` said.
fn last_said_in(trajectory: &Trajectory, name: &str) -> Option<String> {
    let value = serde_json::to_value(trajectory).ok()?;
    let found = value
        .get("subagent_trajectories")
        .or_else(|| value.pointer("/extra/harness/trajectories"))
        .and_then(Value::as_array)?
        .iter()
        .find(|t| {
            t.pointer("/extra/harness/conversation")
                .and_then(Value::as_str)
                == Some(name)
        })?
        .get("steps")?
        .as_array()?
        .iter()
        .rev()
        .find(|s| s.get("source").and_then(Value::as_str) == Some("agent"))?
        .get("message")?
        .as_str()
        .map(str::to_string);
    found
}

/// Writes the plan of a change by running the document's plan stage.
pub async fn write_plan(
    config: &HarnessConfig,
    planner: Arc<dyn ModelProvider>,
    style: PlanStyle,
    (message, diff): (&str, &str),
    check: &PlanCheck<'_>,
    first_response: Option<Duration>,
    deadline: Duration,
) -> Result<String, PlanFailure> {
    let model = |e: String| PlanFailure::Model(e);
    let workspace = tempfile::tempdir().map_err(|e| model(e.to_string()))?;
    let diff_dir = tempfile::tempdir().map_err(|e| model(e.to_string()))?;
    let diff_file = diff_dir.path().join("change.diff");
    std::fs::write(&diff_file, diff).map_err(|e| model(e.to_string()))?;

    let mut inputs = config.inputs(first_response);
    for (key, value) in [
        ("stage", json!("plan")),
        ("style", json!(style.name())),
        ("message", json!(message)),
        ("diff", json!(diff)),
        ("commit", json!(check.commit)),
        ("diff_file", json!(diff_file)),
        ("min_line_words", json!(check.min_line_tokens)),
        ("check_bin", json!(config.check_bin)),
        // The plan stage runs no tools.
        ("plan", json!("")),
        ("masked_diff", json!("")),
        ("calls", json!(1)),
    ] {
        inputs.insert(key.into(), value);
    }
    let engine = config
        .engine(Arc::clone(&planner), planner)
        .map_err(|e| model(format!("{e:#}")))?;
    let harness = engine
        .load_harness_from_str(&config.document)
        .map_err(|e| model(format!("loading the harness document: {e}")))?;
    let mut run = harness
        .start(Value::Object(inputs))
        .and_then(|r| r.in_workspace(workspace.path()))
        .map_err(|e| model(e.to_string()))?;
    let outcome = run
        .execute(RunOptions::new().deadline(deadline))
        .await
        .map_err(|e| model(e.to_string()))?;
    match outcome.result {
        Some(Ok(value)) => value
            .get(format!("plan_{}", style.name()))
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| model("the harness returned no plan".into())),
        Some(Err(failure)) if failure.code == "gate_failed" => {
            // The refused plan is the planner's last words; judging it again
            // names the rejection the gate gave.
            let plan = run
                .trajectory()
                .and_then(|t| last_said_in(&t, &format!("plan_{}", style.name())))
                .ok_or_else(|| model(format!("{}: {}", failure.code, failure.message)))?;
            match crate::plan::validate(&plan, style, check) {
                Err(rejection) => Err(PlanFailure::Rejected(rejection)),
                Ok(()) => Err(model(format!(
                    "the harness refused a plan the gate accepts: {}",
                    failure.message
                ))),
            }
        }
        Some(Err(failure)) => Err(model(format!("{}: {}", failure.code, failure.message))),
        None => Err(model(format!(
            "the plan run ended as {:?}",
            outcome.conclusion
        ))),
    }
}

/// What the work stage left.
pub struct Worked {
    /// The agent's conversation.
    pub trajectory: Trajectory,
    /// Why the run ended early, if it did not conclude by itself.
    pub stopped: Option<String>,
}

/// Runs the document's work stage in `tree`.
pub async fn work(
    config: &HarnessConfig,
    agent: Arc<dyn ModelProvider>,
    tree: &Path,
    (control, plan, masked): (Control, &str, &MaskedDiff),
    calls: usize,
    (first_response, deadline): (Option<Duration>, Duration),
) -> Result<Worked> {
    let mut inputs = config.inputs(first_response);
    let (document, stage_inputs) = match control {
        Control::Grounded => (
            &config.document,
            vec![
                ("stage", json!("work")),
                ("style", json!("")),
                ("message", json!("")),
                ("diff", json!("")),
                ("commit", json!("")),
                ("diff_file", json!("")),
                ("check_bin", json!(config.check_bin)),
                ("min_line_words", json!(0)),
                ("masked_diff", json!(masked.as_str())),
            ],
        ),
        Control::PlanOnly => (&config.plan_only_document, Vec::new()),
    };
    for (key, value) in stage_inputs
        .into_iter()
        .chain([("plan", json!(plan)), ("calls", json!(calls))])
    {
        inputs.insert(key.into(), value);
    }
    let engine = config.engine(Arc::clone(&agent), agent)?;
    let harness = engine
        .load_harness_from_str(document)
        .context("loading the harness document")?;
    let mut run = harness.start(Value::Object(inputs))?.in_workspace(tree)?;
    let outcome = run
        .execute(RunOptions::new().deadline(deadline))
        .await
        .map_err(|e| anyhow::anyhow!("the agent run failed: {e}"))?;
    let trajectory = run
        .trajectory()
        .context("the harness run left no trajectory")?;
    let stop = match &outcome.result {
        Some(Ok(value)) => value
            .get("stop")
            .and_then(Value::as_str)
            .map(str::to_string),
        _ => None,
    };
    let stopped = match (stop.as_deref(), &outcome.result) {
        (Some("tool_cap"), _) => Some("the tool-call cap stopped the run".to_string()),
        (Some("repetition"), _) => Some("a call kept returning the same answer".to_string()),
        (_, Some(Err(failure))) => Some(format!("{}: {}", failure.code, failure.message)),
        _ if outcome.conclusion != RunConclusion::Success => {
            Some(format!("the run ended as {:?}", outcome.conclusion))
        }
        (Some("unfinished"), _) => Some("the run ended as StoppedWithoutConclusion".to_string()),
        _ => None,
    };
    Ok(Worked {
        trajectory,
        stopped,
    })
}
