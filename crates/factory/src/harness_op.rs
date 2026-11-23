// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements programmable agent workflows that turn
// documents into structured data with every step on record, for its
// clients. If your team needs expertise in agent harnesses, you can procure
// our services by sending an email to info@swedishembedded.com.

//! `harness`: a sven harness run over the stage's files, for its returned
//! value.
//!
//! Where `agent` is one call, a harness is a workflow of calls (plan, draft,
//! check, repair) the harness document states. Its agent holds no tools:
//! the stage's files reach it as text, a list of `{name, text}` under the
//! harness input `files_input` (`files` by default), beside the stage's
//! literal `inputs`; it runs nothing ([`splinter_agent::harness_call`]). The
//! product holds `output.json` (the returned value) and `trajectory.json`
//! (what the run did). A harness that ends in its own `fail` rejects the
//! input. A harness that must run commands works a task instead, as an
//! attempt stage's `harness:` does.

use std::collections::BTreeMap;
use std::time::Duration;

use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use splinter_agent::harness_call::{check, HarnessCall, HarnessReply};
use splinter_core::factory::execution::Isolation;
use splinter_core::factory::operation::{
    ArtifactKind, OperationDescriptor, Port, Reproducibility, OPERATION_SCHEMA,
};

use crate::agent_op::input_of;
use crate::execute::{Call, Execute, Outcome};
use crate::registry::OperationSpec;

/// The operation's name.
pub const HARNESS: &str = "harness";

fn files_input() -> String {
    "files".into()
}
fn default_deadline() -> u64 {
    600
}
fn no_inputs() -> Value {
    Value::Object(serde_json::Map::new())
}

/// `harness`' parameters.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct HarnessParams {
    /// The harness file, relative to the task repository.
    pub harness: String,
    /// The model role its agent runs on.
    pub model: String,
    /// The harness's other inputs, as literal values.
    #[serde(default = "no_inputs")]
    pub inputs: Value,
    /// The harness input the stage's files are given as.
    #[serde(default = "files_input")]
    pub files_input: String,
    /// How long one run may take, in seconds.
    #[serde(default = "default_deadline")]
    pub deadline_seconds: u64,
}

/// The `harness` operation.
pub struct HarnessOp;

fn params(value: &Value) -> Result<HarnessParams, String> {
    serde_json::from_value(value.clone()).map_err(|e| e.to_string())
}

impl OperationSpec for HarnessOp {
    fn descriptor(&self) -> OperationDescriptor {
        OperationDescriptor {
            schema: OPERATION_SCHEMA.into(),
            name: HARNESS.into(),
            revision: 1,
            summary: "a sven harness run over the stage's files, for its returned value".into(),
            inputs: vec![Port {
                name: "input".into(),
                kind: ArtifactKind::Files,
                many: true,
                optional: true,
            }],
            outputs: vec![Port {
                name: "output".into(),
                kind: ArtifactKind::Files,
                many: false,
                optional: false,
            }],
            parameters: serde_json::to_value(schema_for!(HarnessParams)).unwrap_or(Value::Null),
            reproducibility: Reproducibility::Nondeterministic,
            isolation: Isolation::Any,
        }
    }

    fn check(&self, value: &Value) -> Result<(), String> {
        let p = params(value)?;
        if !p.inputs.is_object() {
            return Err("inputs is an object of the harness's inputs".into());
        }
        if p.inputs.get(&p.files_input).is_some() {
            return Err(format!(
                "inputs names {}, which the stage's files are given as",
                p.files_input
            ));
        }
        if p.deadline_seconds == 0 {
            return Err("deadline_seconds is at least 1".into());
        }
        Ok(())
    }

    fn model_roles(&self, value: &Value) -> Vec<String> {
        params(value).map(|p| vec![p.model]).unwrap_or_default()
    }

    fn files(&self, value: &Value) -> Vec<String> {
        params(value).map(|p| vec![p.harness]).unwrap_or_default()
    }

    fn check_files(&self, value: &Value, files: &BTreeMap<String, String>) -> Vec<String> {
        let Ok(p) = params(value) else {
            return Vec::new();
        };
        match files.get(&p.harness).map(|yaml| check(yaml)) {
            Some(Err(why)) => vec![format!("{}: {why}", p.harness)],
            _ => Vec::new(),
        }
    }

    fn executor(&self) -> Option<&dyn Execute> {
        Some(self)
    }
}

impl Execute for HarnessOp {
    fn implementation(&self) -> String {
        format!("{HARNESS} 1")
    }

    fn execute(&self, call: &Call<'_>) -> Result<Outcome, String> {
        let p = params(call.params)?;
        let model = call
            .models
            .get(&p.model)
            .ok_or_else(|| format!("model role {} is not bound", p.model))?;
        let yaml = call
            .files
            .get(&p.harness)
            .ok_or_else(|| format!("{} is not pinned", p.harness))?
            .clone();
        let mut inputs = p.inputs.clone();
        if !call.inputs.is_empty() {
            let files: Vec<Value> = input_of(&call.inputs)?
                .as_object()
                .into_iter()
                .flat_map(|ports| ports.values())
                .filter_map(Value::as_array)
                .flatten()
                .cloned()
                .collect();
            inputs[&p.files_input] = Value::Array(files);
        }
        let tree = tempfile::tempdir().map_err(|e| e.to_string())?;
        let reply = HarnessCall {
            yaml,
            inputs,
            deadline: Duration::from_secs(p.deadline_seconds),
        }
        .run(model, tree.path())?;
        let (output, trajectory) = match reply {
            HarnessReply::Returned { output, trajectory } => (output, trajectory),
            HarnessReply::Failed(why) => return Ok(Outcome::Rejected(why)),
        };
        write(&call.out.join("output.json"), &output)?;
        if let Some(trajectory) = trajectory {
            write(&call.out.join("trajectory.json"), &trajectory)?;
        }
        Ok(Outcome::Produced)
    }
}

fn write(path: &std::path::Path, value: &Value) -> Result<(), String> {
    let text = serde_json::to_string_pretty(value).map_err(|e| e.to_string())?;
    std::fs::write(path, text + "\n").map_err(|e| format!("{}: {e}", path.display()))
}
