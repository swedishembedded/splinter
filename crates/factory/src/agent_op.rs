// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements structured extraction from documents with
// language models whose every answer is checked against a schema, for its
// clients. If your team needs expertise in reliable model-driven data
// pipelines, you can procure our services by sending an email to
// info@swedishembedded.com.

//! `agent`: one bounded model call for a value of the stage's JSON Schema.
//!
//! The model is given the instruction file, the schema, and the stage's
//! inputs as text (each file by its name, a directory's files by their
//! paths in it). A reply that breaks the schema goes back for correction
//! ([`splinter_agent::schema_call`]); one never corrected rejects the input.
//! The product holds `output.json` (the value) and `call.json` (the model
//! that answered and the tokens it spent). Run once per item (`foreach`),
//! it is the factory's structured extraction step.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;

use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use splinter_agent::schema_call::{check_schema, SchemaCall, SchemaReply};
use splinter_core::factory::execution::Isolation;
use splinter_core::factory::operation::{
    ArtifactKind, OperationDescriptor, Port, Reproducibility, OPERATION_SCHEMA,
};

use crate::execute::{Call, Execute, Outcome};
use crate::registry::OperationSpec;

/// The operation's name.
pub const AGENT: &str = "agent";

/// The files the product store adds at a product's root.
const ENVELOPE_MANIFEST: &str = splinter_store::factory::products::VIEW_MANIFEST;
const ENVELOPE_STAGE: &str = "stage.json";

/// The most input text one call is given, in bytes.
const MAX_INPUT_BYTES: usize = 1 << 20;

fn default_role() -> String {
    "You do exactly the task you are given, from the input you are given.".into()
}
fn default_deadline() -> u64 {
    300
}
fn default_repairs() -> u32 {
    2
}

/// `agent`'s parameters.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AgentParams {
    /// The instruction file (Markdown), relative to the task repository.
    pub instruction: String,
    /// The JSON Schema file the reply must satisfy.
    pub output_schema: String,
    /// The model role.
    pub model: String,
    /// Who the model is told it is.
    #[serde(default = "default_role")]
    pub role: String,
    /// How long one call may take, every correction included, in seconds.
    #[serde(default = "default_deadline")]
    pub deadline_seconds: u64,
    /// Output tokens one call may generate.
    #[serde(default)]
    pub max_output_tokens: Option<u64>,
    /// Replies outside the schema sent back for correction.
    #[serde(default = "default_repairs")]
    pub repairs: u32,
}

/// The `agent` operation.
pub struct Agent;

fn params(value: &Value) -> Result<AgentParams, String> {
    serde_json::from_value(value.clone()).map_err(|e| e.to_string())
}

impl OperationSpec for Agent {
    fn descriptor(&self) -> OperationDescriptor {
        OperationDescriptor {
            schema: OPERATION_SCHEMA.into(),
            name: AGENT.into(),
            revision: 1,
            summary: "one bounded model call for a value of the stage's JSON Schema".into(),
            inputs: vec![Port {
                name: "input".into(),
                kind: ArtifactKind::Files,
                many: true,
                optional: false,
            }],
            outputs: vec![Port {
                name: "output".into(),
                kind: ArtifactKind::Files,
                many: false,
                optional: false,
            }],
            parameters: serde_json::to_value(schema_for!(AgentParams)).unwrap_or(Value::Null),
            reproducibility: Reproducibility::Nondeterministic,
            isolation: Isolation::Any,
        }
    }

    fn check(&self, value: &Value) -> Result<(), String> {
        let p = params(value)?;
        if p.deadline_seconds == 0 {
            return Err("deadline_seconds is at least 1".into());
        }
        Ok(())
    }

    fn model_roles(&self, value: &Value) -> Vec<String> {
        params(value).map(|p| vec![p.model]).unwrap_or_default()
    }

    fn files(&self, value: &Value) -> Vec<String> {
        params(value)
            .map(|p| vec![p.instruction, p.output_schema])
            .unwrap_or_default()
    }

    fn check_files(&self, value: &Value, files: &BTreeMap<String, String>) -> Vec<String> {
        let Ok(p) = params(value) else {
            return Vec::new();
        };
        let Some(text) = files.get(&p.output_schema) else {
            return Vec::new();
        };
        let checked = serde_json::from_str::<Value>(text)
            .map_err(|e| format!("not JSON: {e}"))
            .and_then(|schema| check_schema(&schema));
        match checked {
            Ok(()) => Vec::new(),
            Err(why) => vec![format!("{}: {why}", p.output_schema)],
        }
    }

    fn executor(&self) -> Option<&dyn Execute> {
        Some(self)
    }
}

impl Execute for Agent {
    fn implementation(&self) -> String {
        format!("{AGENT} 1")
    }

    fn execute(&self, call: &Call<'_>) -> Result<Outcome, String> {
        let p = params(call.params)?;
        let model = call
            .models
            .get(&p.model)
            .ok_or_else(|| format!("model role {} is not bound", p.model))?;
        let file = |path: &str| {
            call.files
                .get(path)
                .ok_or_else(|| format!("{path} is not pinned"))
        };
        let schema: Value = serde_json::from_str(file(&p.output_schema)?)
            .map_err(|e| format!("{}: {e}", p.output_schema))?;
        let request = SchemaCall {
            instruction: file(&p.instruction)?.clone(),
            role: p.role,
            schema,
            deadline: Duration::from_secs(p.deadline_seconds),
            max_output_tokens: p.max_output_tokens,
            repairs: p.repairs,
        };
        let input = input_of(&call.inputs)?;
        let (reply, spent) = request.run(model, &input)?;
        let value = match reply {
            SchemaReply::Answered(value) => value,
            SchemaReply::Unanswered(why) => return Ok(Outcome::Rejected(why)),
        };
        write(&call.out.join("output.json"), &value)?;
        write(
            &call.out.join("call.json"),
            &json!({
                "model": model.identity,
                "input_tokens": spent.input_tokens,
                "output_tokens": spent.output_tokens,
            }),
        )?;
        Ok(Outcome::Produced)
    }
}

fn write(path: &Path, value: &Value) -> Result<(), String> {
    let text = serde_json::to_string_pretty(value).map_err(|e| e.to_string())?;
    std::fs::write(path, text + "\n").map_err(|e| format!("{}: {e}", path.display()))
}

/// The inputs as the model reads them: each port's files, by name.
pub(crate) fn input_of(
    inputs: &BTreeMap<String, Vec<std::path::PathBuf>>,
) -> Result<Value, String> {
    let mut budget = MAX_INPUT_BYTES;
    let mut ports = serde_json::Map::new();
    for (port, paths) in inputs {
        let mut files = Vec::new();
        for path in paths {
            texts(path, path, &mut files, &mut budget)?;
        }
        ports.insert(port.clone(), Value::Array(files));
    }
    Ok(Value::Object(ports))
}

/// Every file at `path` (one, or a directory's, in path order) as
/// `{name, text}`, its name relative to `root`'s parent.
fn texts(root: &Path, path: &Path, out: &mut Vec<Value>, budget: &mut usize) -> Result<(), String> {
    let fail = |e: std::io::Error| format!("{}: {e}", path.display());
    if path.is_dir() {
        let mut entries: Vec<_> = std::fs::read_dir(path)
            .map_err(fail)?
            .map(|e| e.map(|e| e.path()))
            .collect::<Result<_, _>>()
            .map_err(fail)?;
        entries.sort();
        for entry in entries {
            texts(root, &entry, out, budget)?;
        }
        return Ok(());
    }
    let name = path
        .strip_prefix(root.parent().unwrap_or(root))
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned();
    // A product's own bookkeeping, at its root, is not input.
    let at_root = root.is_dir() && path.parent() == Some(root);
    if at_root
        && matches!(
            path.file_name().and_then(|n| n.to_str()),
            Some(ENVELOPE_MANIFEST | ENVELOPE_STAGE)
        )
    {
        return Ok(());
    }
    let bytes = std::fs::read(path).map_err(fail)?;
    *budget = budget.checked_sub(bytes.len()).ok_or_else(|| {
        format!("the input is over {MAX_INPUT_BYTES} bytes; split it with foreach")
    })?;
    let text = String::from_utf8(bytes).map_err(|_| format!("{name} is not UTF-8 text"))?;
    out.push(json!({ "name": name, "text": text }));
    Ok(())
}
