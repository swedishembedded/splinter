// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements voice agents that act through tools and
// answer aloud, for its clients. If your team needs expertise in agent
// runtimes behind a spoken interface, you can procure our services by
// sending an email to info@swedishembedded.com.

//! A sven agent that answers as a person and hands on the words it writes.
//!
//! A spoken front end sends the recognised question and speaks what comes
//! back sentence by sentence, so it needs the text as it is written, not at
//! the end of the turn. [`VoiceAgent::ask`] runs one turn of a sven agent and
//! gives the front end every text delta of the model; tool calls, tool output
//! and reasoning are other events and are not handed on, so what is spoken is
//! what the person said. The conversation, tools and session are sven's.

use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};

use sven_sdk::model::ModelProvider;
use sven_sdk::{Agent, ApprovalPolicy, CallError, Engine, SessionEvent, Toolset};

use crate::solve::SOLVER_MODE;
use crate::system_prompt::{with_live_addendum, UnderSystemPrompt};
use crate::teach::Asker;

/// One conversation with a sven agent, asked in turn.
pub struct VoiceAgent {
    engine: Engine,
    agent: Mutex<Agent>,
    taught: Arc<RwLock<String>>,
    runtime: tokio::runtime::Runtime,
}

impl VoiceAgent {
    /// An agent on `model` that answers under the system turn `system` and
    /// nothing else sven would add. With a `workspace` it has sven's coding
    /// tools confined to that directory, with their calls approved; without
    /// one it has no tools.
    ///
    /// # Errors
    /// The engine cannot be built.
    pub fn start(
        model: Arc<dyn ModelProvider>,
        system: &str,
        workspace: Option<&Path>,
    ) -> Result<Self, CallError> {
        let taught = Arc::new(RwLock::new(String::new()));
        let model = compose(model, system, taught.clone());
        let builder = Engine::builder().model_provider(model);
        let engine = match workspace {
            Some(root) => builder
                .toolset(Toolset::coding())
                .project_root(root)
                .approvals(ApprovalPolicy::Auto)
                .build()?,
            None => builder.build()?,
        };
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| CallError::Infrastructure(e.into()))?;
        Ok(Self {
            agent: Mutex::new(engine.agent(SOLVER_MODE)),
            engine,
            taught,
            runtime,
        })
    }

    /// Run one turn on `question`, handing `on_text` each piece of the
    /// model's text as it is written; the whole reply is returned.
    ///
    /// # Errors
    /// The turn failed (the provider errored, or the model said nothing).
    pub fn ask(&self, question: &str, on_text: &mut dyn FnMut(&str)) -> Result<String, CallError> {
        let mut agent = self
            .agent
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.runtime.block_on(async {
            let mut events = agent.events();
            let send = agent.send(question);
            tokio::pin!(send);
            let outcome = loop {
                tokio::select! {
                    outcome = &mut send => break outcome,
                    event = events.recv() => forward(event, on_text),
                }
            };
            // Events still queued when the turn ended.
            while let Ok(event) = events.try_recv() {
                forward(Ok(event), on_text);
            }
            outcome.map(|o| o.reply)
        })
    }
}

/// `model` answering under `system` and, after it, whatever `taught` holds.
fn compose(
    model: Arc<dyn ModelProvider>,
    system: &str,
    taught: Arc<RwLock<String>>,
) -> Arc<dyn ModelProvider> {
    // The taught text is appended to the system turn, so it goes where it is
    // read after that turn has been set.
    UnderSystemPrompt::wrap(with_live_addendum(model, taught), system)
}

/// Hand a text delta on; every other event, and a lagging or closed channel,
/// is not words to speak.
fn forward(
    event: Result<SessionEvent, tokio::sync::broadcast::error::RecvError>,
    on_text: &mut dyn FnMut(&str),
) {
    if let Ok(SessionEvent::TextDelta(piece)) = event {
        on_text(&piece);
    }
}

impl Asker for VoiceAgent {
    fn ask(&self, question: &str, on_text: &mut dyn FnMut(&str)) -> Result<String, CallError> {
        VoiceAgent::ask(self, question, on_text)
    }

    fn taught(&self, text: &str) {
        *self
            .taught
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = text.to_string();
    }

    fn start_over(&self) {
        *self
            .agent
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = self.engine.agent(SOLVER_MODE);
    }
}

#[cfg(test)]
mod specs {
    use sven_sdk::model::{CompletionRequest, Message, ResponseStream, Role};

    use super::*;

    /// A provider that records the system turn of every request it gets.
    struct Capture(Arc<Mutex<Vec<String>>>);

    #[async_trait::async_trait]
    impl ModelProvider for Capture {
        fn name(&self) -> &str {
            "capture"
        }

        fn model_name(&self) -> &str {
            "capture"
        }

        async fn complete(&self, request: CompletionRequest) -> anyhow::Result<ResponseStream> {
            let system = request
                .messages
                .iter()
                .find(|m| m.role == Role::System)
                .and_then(|m| m.as_text())
                .unwrap_or_default()
                .to_string();
            self.0.lock().unwrap().push(system);
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    #[test]
    fn what_was_taught_reaches_the_model_after_the_agents_own_system_turn() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let taught = Arc::new(RwLock::new(String::new()));
        let model = compose(
            Arc::new(Capture(seen.clone())),
            "You are an assistant.",
            taught.clone(),
        );
        let ask = |model: &Arc<dyn ModelProvider>| {
            let request = CompletionRequest {
                messages: vec![Message::system("sven's own"), Message::user("hi")],
                ..Default::default()
            };
            let _stream = tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap()
                .block_on(model.complete(request))
                .unwrap();
        };
        ask(&model);
        *taught.write().unwrap() = "- Fact: the build server is atlas".to_string();
        ask(&model);
        let seen = seen.lock().unwrap();
        assert_eq!(seen[0], "You are an assistant.");
        assert_eq!(
            seen[1],
            "You are an assistant.\n\n- Fact: the build server is atlas"
        );
    }
}
