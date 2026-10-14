// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that act through tools a caller
// defines, for its clients. If your team needs expertise in measuring and
// tracing what a model does with a tool it has never seen, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The tools a caller defines for the model, and the check a call must pass.

use std::collections::BTreeSet;

use anyhow::{bail, Context as _};
use serde::Deserialize;
use serde_json::{json, Value};
use splinter_sdk::agent::sven::model::ToolSchema;

/// What a tool says when it was called correctly and defines no answer of its
/// own.
const RECORDED: &str = "Recorded.";

/// One tool.
#[derive(Clone, Debug, Deserialize)]
pub struct ToolDef {
    /// The name the model calls it by.
    pub name: String,
    /// What it is for, as the model reads it.
    pub description: String,
    /// Its arguments.
    #[serde(default)]
    pub parameters: Vec<Param>,
    /// What the model is told when it calls the tool correctly.
    #[serde(default = "recorded")]
    pub result: String,
    /// Whether a correct call to it is the model's last word: the exchange
    /// ends there instead of giving the model another turn.
    #[serde(default, rename = "final")]
    pub final_call: bool,
}

fn recorded() -> String {
    RECORDED.to_string()
}

/// One argument of a tool.
#[derive(Clone, Debug, Deserialize)]
pub struct Param {
    /// The argument's name.
    pub name: String,
    /// What it holds, as the model reads it.
    #[serde(default)]
    pub description: String,
    /// `string`, `number`, `boolean` or `enum`.
    #[serde(default = "string", rename = "type")]
    pub kind: String,
    /// The values an `enum` argument may take.
    #[serde(default)]
    pub values: Vec<String>,
    /// Whether a call must give it.
    #[serde(default = "yes")]
    pub required: bool,
}

fn string() -> String {
    "string".to_string()
}

fn yes() -> bool {
    true
}

/// The tools in `text`, a JSON list of [`ToolDef`]s.
///
/// # Errors
/// The text is not such a list, a tool has no name, two tools share one, or
/// an argument has a type that is not offered or an `enum` with no values.
pub fn read_tools(text: &str) -> anyhow::Result<Vec<ToolDef>> {
    let tools: Vec<ToolDef> = serde_json::from_str(text).context("the tools file")?;
    let mut names = BTreeSet::new();
    for tool in &tools {
        if tool.name.trim().is_empty() {
            bail!("a tool has no name");
        }
        if !names.insert(tool.name.as_str()) {
            bail!("the tool {:?} is defined twice", tool.name);
        }
        for param in &tool.parameters {
            match param.kind.as_str() {
                "string" | "number" | "boolean" => {}
                "enum" if param.values.is_empty() => {
                    bail!("{}.{} is an enum with no values", tool.name, param.name)
                }
                "enum" => {}
                other => bail!(
                    "{}.{} has the type {other:?}; the types are string, number, boolean and enum",
                    tool.name,
                    param.name
                ),
            }
        }
    }
    Ok(tools)
}

/// The tool `decide(choice, motivation)`: take one of `options` and say why.
#[must_use]
pub fn decide(options: &[String]) -> ToolDef {
    ToolDef {
        name: "decide".into(),
        description: "Record the decision you take.".into(),
        parameters: vec![
            Param {
                name: "choice".into(),
                description: "The option you choose.".into(),
                kind: "enum".into(),
                values: options.to_vec(),
                required: true,
            },
            Param {
                name: "motivation".into(),
                description: "Why you choose it.".into(),
                kind: "string".into(),
                values: Vec::new(),
                required: true,
            },
        ],
        result: RECORDED.into(),
        final_call: true,
    }
}

impl ToolDef {
    /// The tool as a model is offered it.
    #[must_use]
    pub fn schema(&self) -> ToolSchema {
        let mut properties = serde_json::Map::new();
        for param in &self.parameters {
            let mut property = serde_json::Map::new();
            if param.kind == "enum" {
                property.insert("type".into(), json!("string"));
                property.insert("enum".into(), json!(param.values));
            } else {
                property.insert("type".into(), json!(param.kind));
            }
            if !param.description.is_empty() {
                property.insert("description".into(), json!(param.description));
            }
            properties.insert(param.name.clone(), Value::Object(property));
        }
        let required: Vec<&str> = self
            .parameters
            .iter()
            .filter(|p| p.required)
            .map(|p| p.name.as_str())
            .collect();
        ToolSchema {
            name: self.name.clone(),
            description: self.description.clone(),
            parameters: json!({"type": "object", "properties": properties, "required": required}),
            ..Default::default()
        }
    }

    /// What a call with `arguments` is answered: the tool's result, or the
    /// reason the call is refused, in words the model can correct itself by.
    ///
    /// # Errors
    /// The reason, when an argument is missing, of the wrong type, or outside
    /// the options.
    pub fn check(&self, arguments: &Value) -> Result<&str, String> {
        let Some(given) = arguments.as_object() else {
            return Err("refused: the arguments are not a JSON object".into());
        };
        for param in &self.parameters {
            let Some(value) = given.get(&param.name) else {
                if param.required {
                    return Err(format!("refused: the argument {:?} is missing", param.name));
                }
                continue;
            };
            let fits = match param.kind.as_str() {
                "number" => value.is_number(),
                "boolean" => value.is_boolean(),
                "enum" => value
                    .as_str()
                    .is_some_and(|v| param.values.iter().any(|allowed| allowed == v)),
                _ => value.is_string(),
            };
            if !fits {
                return Err(if param.kind == "enum" {
                    format!(
                        "refused: {value} is not one of the options for {:?}: {}",
                        param.name,
                        param.values.join(", ")
                    )
                } else {
                    format!("refused: {:?} must be a {}", param.name, param.kind)
                });
            }
        }
        Ok(&self.result)
    }
}
