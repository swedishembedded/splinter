// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements coding agents that work inside a confined
// directory and leave a complete, replayable record of what they did, for
// its clients. If your team needs expertise in agent runtimes or in
// measuring what an agent really did, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The repetition stop, decided where the run decides: before each model
//! request.
//!
//! A watcher that reads the event stream cannot stop a run at a call: it is
//! a task beside the run, so under load the run goes on asking while the
//! watcher waits to be scheduled, and how many calls a stagnating worker got
//! to make depended on the machine. The model request is the one point every
//! further call passes through, and by then the transcript it carries holds
//! every earlier call and its answer, so the same count taken there is the
//! same on every run.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use sven_sdk::model::{
    CompletionRequest, Message, MessageContent, ModelProvider, ResponseStream, Role,
};
use sven_sdk::CancelToken;

use super::watch::describe;

/// What a send is held to; set when it starts, cleared when it ends.
pub(super) struct Armed {
    pub(super) bound: u32,
    pub(super) stop: CancelToken,
    pub(super) repeated: Arc<AtomicBool>,
    pub(super) repeated_call: Arc<Mutex<Option<String>>>,
}

/// The slot a worker's sends arm and its model provider reads.
#[derive(Clone, Default)]
pub(super) struct GuardSlot(Arc<Mutex<Option<Armed>>>);

impl GuardSlot {
    pub(super) fn arm(&self, armed: Armed) {
        if let Ok(mut slot) = self.0.lock() {
            *slot = Some(armed);
        }
    }

    pub(super) fn disarm(&self) {
        if let Ok(mut slot) = self.0.lock() {
            *slot = None;
        }
    }
}

/// The call that has come back with the same answer `bound` times in this
/// send (the turns after the last user turn), as `name\u{1}arguments`.
fn repeated_call(messages: &[Message], bound: u32) -> Option<String> {
    let start = messages
        .iter()
        .rposition(|m| m.role == Role::User)
        .map_or(0, |at| at + 1);
    let mut asked: HashMap<&str, String> = HashMap::new();
    let mut answered: HashMap<String, u32> = HashMap::new();
    for message in &messages[start..] {
        match &message.content {
            MessageContent::ToolCall {
                tool_call_id,
                function,
            } => {
                asked.insert(
                    tool_call_id,
                    format!("{}\u{1}{}", function.name, function.arguments),
                );
            }
            MessageContent::ToolResult {
                tool_call_id,
                content,
            } => {
                let Some(call) = asked.get(tool_call_id.as_str()) else {
                    continue;
                };
                let seen = answered.entry(format!("{call}\u{1}{content}")).or_insert(0);
                *seen += 1;
                if *seen >= bound {
                    return Some(call.clone());
                }
            }
            _ => {}
        }
    }
    None
}

/// `inner`, which stops the armed send instead of answering a request that
/// follows a call's `bound`th identical answer.
pub(crate) struct RepetitionGuard {
    inner: Arc<dyn ModelProvider>,
    slot: GuardSlot,
}

impl RepetitionGuard {
    pub(super) fn wrap(inner: Arc<dyn ModelProvider>, slot: GuardSlot) -> Arc<dyn ModelProvider> {
        Arc::new(Self { inner, slot })
    }

    /// Whether the armed send has just hit its bound; if so it is cancelled
    /// and the flags a report reads are set.
    fn stagnated(&self, request: &CompletionRequest) -> bool {
        let Ok(slot) = self.slot.0.lock() else {
            return false;
        };
        let Some(armed) = slot.as_ref() else {
            return false;
        };
        let Some(call) = repeated_call(&request.messages, armed.bound) else {
            return false;
        };
        if let Ok(mut first) = armed.repeated_call.lock() {
            first.get_or_insert_with(|| describe(&call));
        }
        armed.repeated.store(true, Ordering::SeqCst);
        armed.stop.cancel();
        true
    }
}

#[async_trait::async_trait]
impl ModelProvider for RepetitionGuard {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    async fn complete(&self, request: CompletionRequest) -> anyhow::Result<ResponseStream> {
        if self.stagnated(&request) {
            // The run was just cancelled; it ends without an answer.
            return std::future::pending().await;
        }
        self.inner.complete(request).await
    }

    fn catalog_max_output_tokens(&self) -> Option<u32> {
        self.inner.catalog_max_output_tokens()
    }

    fn catalog_context_window(&self) -> Option<u32> {
        self.inner.catalog_context_window()
    }

    fn config_context_window(&self) -> Option<u32> {
        self.inner.config_context_window()
    }

    fn config_max_output_tokens(&self) -> Option<u32> {
        self.inner.config_max_output_tokens()
    }

    async fn probe_context_window(&self) -> Option<u32> {
        self.inner.probe_context_window().await
    }

    fn supports_images(&self) -> bool {
        self.inner.supports_images()
    }

    fn supports_audio(&self) -> bool {
        self.inner.supports_audio()
    }
}
