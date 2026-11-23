// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements structured extraction from documents with
// language models whose every answer is checked against a schema, for its
// clients. If your team needs expertise in reliable model-driven data
// pipelines, you can procure our services by sending an email to
// info@swedishembedded.com.

//! A typed call whose shape is a JSON Schema known only at run time: a
//! recipe's `agent` stage names the schema as a file.
//!
//! The call is a [`TypedCall`] for any JSON value; the schema is in the
//! brief, and a postcondition checks every reply against it, so a reply
//! that breaks it goes back to the model with the schema's own complaint.
//! The provider is not asked to constrain decoding to the schema: sven
//! derives a method's response format from its Rust type, and this one has
//! none. A model that cannot answer within its repairs leaves the call
//! unanswered, which is the caller's to record; a failure of the call
//! itself (the endpoint, the deadline) is an error, which may pass.

use std::time::Duration;

use serde_json::Value;
use sven_sdk::CallError;

use crate::solve::Model;
use crate::typed::{Spent, TypedCall};

/// A call for one value of `schema`.
#[derive(Clone, Debug)]
pub struct SchemaCall {
    /// What the model is asked to do.
    pub instruction: String,
    /// Who the model is told it is.
    pub role: String,
    /// The JSON Schema a reply must satisfy.
    pub schema: Value,
    /// How long the call may take, every correction included.
    pub deadline: Duration,
    /// Output tokens the call may generate, every correction included.
    pub max_output_tokens: Option<u64>,
    /// How many replies outside the schema are sent back for correction.
    pub repairs: u32,
}

/// What a call came to.
#[derive(Clone, Debug, PartialEq)]
pub enum SchemaReply {
    /// A value of the schema.
    Answered(Value),
    /// No reply within the repairs satisfied the schema; why the last did
    /// not.
    Unanswered(String),
}

/// Why a schema is refused before any call.
///
/// # Errors
/// It is not a JSON Schema the validator can compile.
pub fn check_schema(schema: &Value) -> Result<(), String> {
    jsonschema::validator_for(schema)
        .map(|_| ())
        .map_err(|e| format!("not a JSON Schema: {e}"))
}

impl SchemaCall {
    /// The call run on `model` over `input`, on a runtime of its own (the
    /// calling thread blocks): the reply and what it cost.
    ///
    /// # Errors
    /// The schema does not compile, or the call failed (the endpoint, the
    /// deadline): nothing was decided about the input.
    pub fn run(self, model: &Model, input: &Value) -> Result<(SchemaReply, Spent), String> {
        let validator = jsonschema::validator_for(&self.schema)
            .map_err(|e| format!("not a JSON Schema: {e}"))?;
        let schema = serde_json::to_string_pretty(&self.schema).map_err(|e| e.to_string())?;
        let task = format!(
            "{}\n\nReply with one JSON value that satisfies this JSON Schema, and nothing else:\n{schema}",
            self.instruction.trim_end()
        );
        let mut call = TypedCall::<Value>::new("answer", task, self.role, self.deadline)
            .repairs(self.repairs)
            .postcondition(move |value| {
                let problems: Vec<String> = validator
                    .iter_errors(value)
                    .map(|e| format!("{}: {e}", e.instance_path()))
                    .collect();
                if problems.is_empty() {
                    Ok(())
                } else {
                    Err(format!(
                        "the reply breaks the schema; correct it: {}",
                        problems.join("; ")
                    ))
                }
            });
        if let Some(tokens) = self.max_output_tokens {
            call = call.max_output_tokens(tokens);
        }
        let session = call.start(model).map_err(|e| e.to_string())?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| e.to_string())?;
        let measured = runtime.block_on(session.call_measured(input));
        let reply = match measured.result {
            Ok(value) => SchemaReply::Answered(value),
            Err(e @ (CallError::Invalid { .. } | CallError::Postcondition { .. })) => {
                SchemaReply::Unanswered(e.to_string())
            }
            Err(e) => return Err(e.to_string()),
        };
        Ok((reply, measured.spent))
    }
}
