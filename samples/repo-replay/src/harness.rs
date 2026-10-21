// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The replay driven by sven harness documents (`harness/replay.yaml`,
//! `harness/replay-plan-only.yaml`).
//!
//! Every prompt, the plan's retry on refusal, the agent's tool allowance and
//! repetition stop, the follow-up for a turn that ends announcing a call it
//! did not make, and the supervisor's corrections with the gate that judges
//! the tree (`check`) and the gate that vets the supervisor's message
//! (`message_ok`) are declared in the documents; this module supplies their
//! inputs and reads the results. The gates run this executable's `check`,
//! `check-message` and `check-plan` commands over the hidden side of the
//! attempt ([`crate::case`]). Needs sven at or after 5e2aea0a.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context as _, Result};
use serde_json::{json, Value};
use splinter_sdk::agent::sven::atif::{MessageBody, StepOrigin, Trajectory};
use splinter_sdk::agent::sven::model::ModelProvider;
use splinter_sdk::agent::sven::{Engine, RunConclusion, RunOptions, Toolset};

use crate::control::Control;
use crate::mask::MaskedDiff;
use crate::plan::{PlanCheck, PlanFailure};
use crate::plan_prompt::PlanStyle;
use crate::supervise::{Correction, Refused};

/// The documents the sample ships: the plan stage and the grounded work, and
/// the work of the plan-only control (a sven agent task cannot carry its own
/// system prompt, so the control is a document of its own).
pub const DEFAULT_DOCUMENT: &str = include_str!("../harness/replay.yaml");
/// The plan-only control's document.
pub const DEFAULT_PLAN_ONLY_DOCUMENT: &str = include_str!("../harness/replay-plan-only.yaml");

/// Seconds to wait for a first response when the caller names no limit
/// (the document needs a number; a local model is silent through its prefill).
const DEFAULT_FIRST_RESPONSE_SECS: u64 = 600;
/// Tokens one reply may take when the caller names no limit: sven's own cap
/// (4096) is spent entirely on reasoning by a thinking model.
const DEFAULT_MAX_OUTPUT_TOKENS: u64 = 16_384;

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
    /// The most thinking tokens per reply; `None` is unbounded.
    pub thinking_budget: Option<u32>,
}

/// The harness documents and what their gates need.
#[derive(Clone, Debug)]
pub struct HarnessConfig {
    /// The text of the document with the plan stage and the grounded work.
    pub document: String,
    /// The text of the document with the work of the plan-only control.
    pub plan_only_document: String,
    /// The `repo-replay` executable the documents' gates run.
    pub check_bin: std::path::PathBuf,
    /// Sampling stated to the documents.
    pub decoding: Decoding,
}

/// What one work stage is run with.
pub struct WorkSetup<'a> {
    /// The agent's model.
    pub agent: Arc<dyn ModelProvider>,
    /// The supervisor's model.
    pub supervisor: Arc<dyn ModelProvider>,
    /// The attempt tree.
    pub tree: &'a Path,
    /// The hidden side of the attempt.
    pub case_dir: &'a Path,
    /// What the agent is shown besides the plan.
    pub control: Control,
    /// The task.
    pub plan: &'a str,
    /// The change as the agent sees it (grounded only).
    pub masked: &'a MaskedDiff,
    /// The diff, for the supervisor's eyes.
    pub diff: &'a str,
    /// Tool calls of the first round.
    pub calls: usize,
    /// Tool calls each correction round adds.
    pub per_correction: usize,
    /// Times a turn that ends announcing an unmade call is told to go on.
    pub follow_ups: usize,
    /// Most corrections.
    pub corrections: usize,
    /// Longest wait for a first response.
    pub first_response: Option<Duration>,
    /// Longest reply.
    pub max_output_tokens: Option<u64>,
    /// Wall-clock bound of the run.
    pub deadline: Duration,
}

/// What the work stage left.
pub struct Worked {
    /// The agent's conversation (the supervisor's is not part of it).
    pub trajectory: Trajectory,
    /// Why the run ended early, if it did not conclude by itself.
    pub stopped: Option<String>,
    /// The corrections the agent was sent.
    pub corrections: Vec<Correction>,
    /// Supervisor messages the accept gate refused.
    pub refused: Vec<Refused>,
    /// Why the supervisor could not be reached, if it could not.
    pub supervisor_error: Option<String>,
}

impl HarnessConfig {
    fn engine(
        &self,
        planner: Arc<dyn ModelProvider>,
        agent: Arc<dyn ModelProvider>,
        supervisor: Arc<dyn ModelProvider>,
    ) -> Result<Engine> {
        Engine::builder()
            .model_provider(Arc::clone(&agent))
            .model_role("planner", planner)
            .model_role("agent", agent)
            .model_role("supervisor", supervisor)
            .toolset(Toolset::coding())
            .build()
            .context("building the harness engine")
    }

    /// The inputs both documents take; `first_response` and `max_output_tokens` default as documented.
    fn inputs(
        &self,
        first_response: Option<Duration>,
        max_output_tokens: Option<u64>,
    ) -> serde_json::Map<String, Value> {
        let d = &self.decoding;
        let mut inputs = serde_json::Map::new();
        for (key, value) in [
            ("temperature", json!(d.temperature)),
            ("top_p", json!(d.top_p)),
            ("top_k", json!(d.top_k)),
            ("seed", json!(d.seed)),
            ("thinking", json!(d.thinking)),
            ("thinking_budget", json!(d.thinking_budget)),
            (
                "first_response_secs",
                json!(first_response.map_or(DEFAULT_FIRST_RESPONSE_SECS, |t| t.as_secs().max(1))),
            ),
            (
                "max_output_tokens",
                json!(max_output_tokens.unwrap_or(DEFAULT_MAX_OUTPUT_TOKENS)),
            ),
            ("check_bin", json!(self.check_bin)),
            // Inputs of the other stage still need a value.
            ("plan", json!("")),
            ("diff", json!("")),
            ("calls", json!(1)),
            ("calls_limit", json!(1)),
            ("per_correction", json!(1)),
            ("corrections", json!(0)),
            ("follow_ups", json!(0)),
            ("workspace", json!("")),
            ("case_dir", json!("")),
        ] {
            inputs.insert(key.into(), value);
        }
        inputs
    }
}

/// The last words the agent side of conversation `name` said.
fn last_said_in(trajectory: &Trajectory, name: &str) -> Option<String> {
    trajectory
        .subagent_trajectories
        .iter()
        .flatten()
        .find(|t| {
            t.extra
                .as_ref()
                .and_then(|e| e.pointer("/harness/conversation"))
                .and_then(Value::as_str)
                == Some(name)
        })?
        .steps
        .iter()
        .rev()
        .filter(|s| s.source == StepOrigin::Agent)
        .find_map(|s| match &s.message {
            MessageBody::Text(t) if !t.trim().is_empty() => Some(t.clone()),
            _ => None,
        })
}

/// Writes the plan of a change by running the document's plan stage.
pub async fn write_plan(
    config: &HarnessConfig,
    planner: Arc<dyn ModelProvider>,
    style: PlanStyle,
    (message, diff): (&str, &str),
    (check, tries): (&PlanCheck<'_>, usize),
    (first_response, deadline): (Option<Duration>, Duration),
) -> Result<String, PlanFailure> {
    let model = |e: String| PlanFailure::Model(e);
    let workspace = tempfile::tempdir().map_err(|e| model(e.to_string()))?;
    let diff_dir = tempfile::tempdir().map_err(|e| model(e.to_string()))?;
    let diff_file = diff_dir.path().join("change.diff");
    std::fs::write(&diff_file, diff).map_err(|e| model(e.to_string()))?;

    let mut inputs = config.inputs(first_response, None);
    for (key, value) in [
        ("stage", json!("plan")),
        ("style", json!(style.name())),
        ("message", json!(message)),
        ("diff", json!(diff)),
        ("commit", json!(check.commit)),
        ("diff_file", json!(diff_file)),
        ("min_line_words", json!(check.min_line_tokens)),
        ("plan_retries", json!(tries.saturating_sub(1))),
        ("masked_diff", json!("")),
    ] {
        inputs.insert(key.into(), value);
    }
    let engine = config
        .engine(Arc::clone(&planner), Arc::clone(&planner), planner)
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

/// Runs the document's work stage in the attempt tree.
pub async fn work(config: &HarnessConfig, setup: WorkSetup<'_>) -> Result<Worked> {
    let mut inputs = config.inputs(setup.first_response, setup.max_output_tokens);
    let tree = setup
        .tree
        .canonicalize()
        .unwrap_or(setup.tree.to_path_buf());
    let (document, grounded) = match setup.control {
        Control::Grounded => (&config.document, true),
        Control::PlanOnly => (&config.plan_only_document, false),
    };
    let mut stage_inputs = vec![
        ("plan", json!(setup.plan)),
        ("diff", json!(setup.diff)),
        ("calls", json!(setup.calls)),
        (
            "calls_limit",
            json!(setup.calls + setup.corrections * setup.per_correction),
        ),
        ("per_correction", json!(setup.per_correction)),
        ("corrections", json!(setup.corrections)),
        ("follow_ups", json!(setup.follow_ups)),
        ("workspace", json!(tree)),
        ("case_dir", json!(setup.case_dir)),
    ];
    if grounded {
        stage_inputs.extend([
            ("stage", json!("work")),
            ("style", json!("")),
            ("message", json!("")),
            ("commit", json!("")),
            ("diff_file", json!("")),
            ("min_line_words", json!(0)),
            ("plan_retries", json!(0)),
            ("masked_diff", json!(setup.masked.as_str())),
        ]);
    }
    for (key, value) in stage_inputs {
        inputs.insert(key.into(), value);
    }
    let engine = config.engine(
        Arc::clone(&setup.agent),
        Arc::clone(&setup.agent),
        setup.supervisor,
    )?;
    let harness = engine
        .load_harness_from_str(document)
        .context("loading the harness document")?;
    let mut run = harness.start(Value::Object(inputs))?.in_workspace(&tree)?;
    let outcome = run
        .execute(RunOptions::new().deadline(setup.deadline))
        .await
        .map_err(|e| anyhow::anyhow!("the agent run failed: {e}"))?;
    let mut trajectory = run
        .trajectory()
        .context("the harness run left no trajectory")?;
    // The supervisor's conversation and the run's trace are not the agent's
    // record: the supervisor's prompt never reaches a training record.
    trajectory.subagent_trajectories = None;
    trajectory.extra = None;
    let corrections: Vec<Correction> = run
        .corrections()
        .unwrap_or_default()
        .into_iter()
        .map(|c| Correction {
            message: c.message,
            calls_before: c.tool_calls_before,
        })
        .collect();
    let refused: Vec<Refused> = run
        .refused_messages()
        .unwrap_or_default()
        .into_iter()
        .map(|r| {
            // The class a refusal is counted under is the one the rules give the message.
            let class = crate::case::refusal_class(setup.case_dir, &trajectory, &r.message)
                .unwrap_or_else(|| "refused".into());
            Refused {
                class,
                detail: r.reason,
            }
        })
        .collect();
    let stop = match &outcome.result {
        Some(Ok(value)) => value
            .get("stop")
            .and_then(Value::as_str)
            .map(str::to_string),
        _ => None,
    };
    let supervisor_error = match &outcome.result {
        Some(Err(failure)) if failure.code == "model_unavailable" => Some(failure.message.clone()),
        _ => None,
    };
    let stopped = match (stop.as_deref(), &outcome.result) {
        (Some("tool_cap"), _) => Some("the tool-call cap stopped the run".to_string()),
        (Some("repetition"), _) => Some("a call kept returning the same answer".to_string()),
        (Some("output_truncated"), _) => {
            Some("a reply was cut off at the output limit".to_string())
        }
        (_, Some(Err(failure))) => Some(format!("{}: {}", failure.code, failure.message)),
        _ if outcome.conclusion != RunConclusion::Success => {
            Some(format!("the run ended as {:?}", outcome.conclusion))
        }
        (Some("stopped_without_conclusion"), _) => {
            Some("the run ended as StoppedWithoutConclusion".to_string())
        }
        _ => None,
    };
    Ok(Worked {
        trajectory,
        stopped,
        corrections,
        refused,
        supervisor_error,
    })
}
