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
        let runtime = shared_runtime()?;
        // A connection that drops while the reply streams is transient: the reply is only used
        // whole, so the call is asked again from the start, under the retries of every model call.
        let mut pause = crate::retry::FIRST_PAUSE;
        let mut attempt = 0;
        let measured = loop {
            let measured = runtime.block_on(session.call_measured(input));
            let transient = measured
                .result
                .as_ref()
                .err()
                .is_some_and(|e| crate::retry::is_transient_text(&e.to_string()));
            if !transient || attempt >= crate::retry::RETRIES {
                break measured;
            }
            attempt += 1;
            eprintln!(
                "splinter: the model call failed transiently; asking again in {pause:?} ({attempt}/{})",
                crate::retry::RETRIES
            );
            runtime.block_on(async { tokio::time::sleep(pause).await });
            pause = pause.saturating_mul(2);
        };
        let reply = match measured.result {
            Ok(value) => SchemaReply::Answered(value),
            Err(e) if unanswerable(&e) => SchemaReply::Unanswered(e.to_string()),
            Err(e) => return Err(e.to_string()),
        };
        Ok((reply, measured.spent))
    }
}

/// Whether the call's failure is this input's alone: the model's answer was malformed or broke the schema, or it spent
/// the call's tokens or time without one. The item is rejected with the reason and the others go on. A cancellation or
/// an outage is the run's, not the item's.
fn unanswerable(error: &CallError) -> bool {
    match error {
        CallError::Invalid { .. } | CallError::Postcondition { .. } => true,
        CallError::Stopped { conclusion } => matches!(
            conclusion,
            sven_sdk::RunConclusion::BudgetExhausted | sven_sdk::RunConclusion::Timeout
        ),
        _ => false,
    }
}

/// The runtime every schema call runs on. The model a stage is bound to is one client with one connection pool for all
/// of its calls; a call with a runtime of its own leaves the pooled connections' tasks behind it when it ends, and the
/// next call that picks one fails with "dispatch task is gone". Calls on different threads share this runtime's
/// workers, so a pooled connection outlives the call that opened it.
fn shared_runtime() -> Result<&'static tokio::runtime::Runtime, String> {
    static RUNTIME: std::sync::OnceLock<Result<tokio::runtime::Runtime, String>> =
        std::sync::OnceLock::new();
    RUNTIME
        .get_or_init(|| {
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .thread_name("schema-call")
                .enable_all()
                .build()
                .map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(Clone::clone)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    use serde_json::json;
    use sven_sdk::model::{CompletionRequest, ModelProvider, ResponseEvent, ResponseStream};

    use super::*;

    /// Starts a reply and loses the connection in the middle of it the first
    /// time, then answers with the JSON object asked for.
    struct DropsMidReply(AtomicU32);

    #[async_trait::async_trait]
    impl ModelProvider for DropsMidReply {
        fn name(&self) -> &str {
            "scripted"
        }

        fn model_name(&self) -> &str {
            "scripted"
        }

        async fn complete(&self, _: CompletionRequest) -> anyhow::Result<ResponseStream> {
            let events: Vec<anyhow::Result<ResponseEvent>> =
                if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
                    vec![
                        Ok(ResponseEvent::TextDelta("{\"keep".into())),
                        Err(anyhow::anyhow!(
                        "error decoding response body: connection closed before message completed"
                    )),
                    ]
                } else {
                    vec![
                        Ok(ResponseEvent::TextDelta("{\"keep_through\": 2}".into())),
                        Ok(ResponseEvent::Done),
                    ]
                };
            Ok(Box::pin(futures::stream::iter(events)))
        }
    }

    #[test]
    fn a_schema_call_whose_connection_drops_mid_reply_is_asked_again() {
        let provider = Arc::new(DropsMidReply(AtomicU32::new(0)));
        let model = Model::new(provider.clone(), "scripted");
        let call = SchemaCall {
            instruction: "Locate.".into(),
            role: "You locate.".into(),
            schema: json!({"type": "object", "required": ["keep_through"],
                "properties": {"keep_through": {"type": "integer"}}}),
            deadline: Duration::from_secs(30),
            max_output_tokens: None,
            repairs: 1,
        };
        let (reply, _) = call.run(&model, &json!({"input": "x"})).unwrap();
        assert_eq!(reply, SchemaReply::Answered(json!({"keep_through": 2})));
        assert_eq!(provider.0.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn an_item_that_spends_its_budget_is_rejected_and_a_cancellation_stops_the_run() {
        use sven_sdk::RunConclusion::{BudgetExhausted, Cancelled, Timeout};
        let stopped = |conclusion| CallError::Stopped { conclusion };
        assert!(unanswerable(&stopped(BudgetExhausted)));
        assert!(unanswerable(&stopped(Timeout)));
        assert!(!unanswerable(&stopped(Cancelled)));
        assert!(!unanswerable(&CallError::Infrastructure(anyhow::anyhow!(
            "outage"
        ))));
    }
}
