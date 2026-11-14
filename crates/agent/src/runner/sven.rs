// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements interchangeable agent runtimes behind one
// contract, so an agent's loop can be measured apart from its model, for its
// clients. If your team needs expertise in comparing coding agents fairly,
// you can procure our services by sending an email to info@swedishembedded.com.

//! Sven behind the [`AgentRunner`] contract: a [`Worker`] on sven's engine in
//! the episode's tree, its trajectory (sven's ATIF export) the native record.

use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use sven_sdk::model::ModelProvider;
use sven_sdk::CancelToken;

use super::{
    AgentRunner, Episode, RunCancel, RunLimits, RunOutcome, RunnerError, RuntimeIdentity, Step,
    StepCall,
};
use crate::work::{WorkOptions, Worker};

/// The sven runtime: its worker, on `model`, bounded by `options` (whose
/// deadline and cancel the run's own replace).
pub struct SvenRunner {
    model: Arc<dyn ModelProvider>,
    options: WorkOptions,
    version: String,
}

impl SvenRunner {
    /// A runner of sven's worker on `model` with `options`; `version` is the
    /// sven build the caller links, as the episode's record names it.
    #[must_use]
    pub fn new(
        model: Arc<dyn ModelProvider>,
        options: WorkOptions,
        version: impl Into<String>,
    ) -> Self {
        Self {
            model,
            options,
            version: version.into(),
        }
    }
}

#[async_trait]
impl AgentRunner for SvenRunner {
    fn identity(&self) -> RuntimeIdentity {
        RuntimeIdentity {
            runtime: "sven".into(),
            version: self.version.clone(),
            binary_sha256: None,
            mode: None,
        }
    }

    async fn run(
        &self,
        episode: &Episode,
        instruction: &str,
        limits: RunLimits,
        cancel: RunCancel,
    ) -> Result<RunOutcome, RunnerError> {
        let mut options = self.options.clone();
        options.solve.deadline = limits.deadline;
        let stop = CancelToken::new();
        options.solve.cancel = Some(stop.clone());
        let relay = {
            let stop = stop.clone();
            tokio::spawn(async move {
                cancel.cancelled().await;
                stop.cancel();
            })
        };
        let started = Instant::now();
        let runtime = |reason: String| RunnerError::Runtime {
            runtime: "sven",
            reason,
        };
        let mut worker = Worker::start(&episode.tree, Arc::clone(&self.model), options)
            .map_err(|e| runtime(e.to_string()))?;
        let report = worker.send(instruction).await;
        relay.abort();
        let report = report.map_err(|e| runtime(e.to_string()))?;
        let document = serde_json::to_value(&report.solution.trajectory)
            .map_err(|e| runtime(e.to_string()))?;
        let native_record = episode.run_dir.join("trajectory.json");
        std::fs::write(
            &native_record,
            serde_json::to_vec_pretty(&document).unwrap_or_default(),
        )
        .map_err(|source| RunnerError::Io {
            path: native_record.clone(),
            source,
        })?;
        Ok(RunOutcome {
            conclusion: report.solution.conclusion,
            native_record,
            steps: normalize(&document),
            wall: started.elapsed(),
        })
    }
}

/// Sven's ATIF trajectory as normalized steps: each agent step's message,
/// reasoning and tool calls with the results its observation holds.
#[must_use]
pub fn normalize(document: &serde_json::Value) -> Vec<Step> {
    let empty = Vec::new();
    let steps = document
        .get("steps")
        .and_then(|s| s.as_array())
        .unwrap_or(&empty);
    steps
        .iter()
        .filter(|s| s.get("source").and_then(|v| v.as_str()) == Some("agent"))
        .map(|step| {
            let results = step
                .pointer("/observation/results")
                .and_then(|r| r.as_array())
                .cloned()
                .unwrap_or_default();
            let tool_calls = step
                .get("tool_calls")
                .and_then(|c| c.as_array())
                .into_iter()
                .flatten()
                .map(|call| {
                    let id = call.get("tool_call_id").and_then(|v| v.as_str());
                    let result = results
                        .iter()
                        .find(|r| r.get("source_call_id").and_then(|v| v.as_str()) == id);
                    let text = result.and_then(|r| r.get("content")).map(|c| match c {
                        serde_json::Value::String(s) => s.clone(),
                        other => other.to_string(),
                    });
                    StepCall {
                        name: call
                            .get("function_name")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default()
                            .into(),
                        input: call
                            .get("arguments")
                            .cloned()
                            .unwrap_or(serde_json::Value::Null),
                        error: result
                            .and_then(|r| r.get("is_error"))
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false),
                        result: text,
                    }
                })
                .collect();
            Step {
                text: step
                    .get("message")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .into(),
                reasoning: step
                    .get("reasoning_content")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .into(),
                tool_calls,
            }
        })
        .collect()
}
