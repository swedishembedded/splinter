// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A scripted base for specs with no weights: a [`BaseLoader`] whose
//! engines reply with a function of the rendered prompt and the adapter
//! attached when they generate, and which logs every load, adapter switch,
//! generation and free.
//!
//! It tests residency - which base is resident, which adapter a generation
//! ran under - not brain. Built for this crate's own specs and, behind the
//! `scripted` feature, for the specs of the crates above it.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use brain::chat::{ChatDelta, ChatUsage, FinishReason};
use brain::{CancelToken, ChatRequest, ChatResponse};

use super::{BaseLoader, Engine};
use crate::error::PolicyError;

/// What a scripted base did, in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScriptedEvent {
    /// The base at this checkpoint was loaded.
    Loaded(PathBuf),
    /// This adapter was attached.
    Attached(PathBuf),
    /// The attached adapter was detached.
    Detached,
    /// A generation ran with this adapter attached (`None`: the base).
    Generated(Option<PathBuf>),
    /// The base at this checkpoint was freed.
    Freed(PathBuf),
}

type Reply = dyn Fn(&str, Option<&Path>) -> String + Send + Sync;
type Observer = dyn Fn(&ScriptedEvent) + Send + Sync;

/// See the module documentation. Clones share the log.
#[derive(Clone)]
pub struct ScriptedLoader {
    reply: Arc<Reply>,
    events: Arc<Mutex<Vec<ScriptedEvent>>>,
    observer: Option<Arc<Observer>>,
}

impl ScriptedLoader {
    /// A loader whose engines reply `reply(prompt, attached adapter)`.
    pub fn new(reply: impl Fn(&str, Option<&Path>) -> String + Send + Sync + 'static) -> Self {
        Self {
            reply: Arc::new(reply),
            events: Arc::new(Mutex::new(Vec::new())),
            observer: None,
        }
    }

    /// The same loader also telling `observer` each event as it happens.
    #[must_use]
    pub fn observed(mut self, observer: impl Fn(&ScriptedEvent) + Send + Sync + 'static) -> Self {
        self.observer = Some(Arc::new(observer));
        self
    }

    /// Every event so far, in order.
    #[must_use]
    pub fn events(&self) -> Vec<ScriptedEvent> {
        self.events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    fn record(&self, event: ScriptedEvent) {
        if let Some(observer) = &self.observer {
            observer(&event);
        }
        self.events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(event);
    }
}

impl std::fmt::Debug for ScriptedLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScriptedLoader")
            .field("events", &self.events())
            .finish_non_exhaustive()
    }
}

impl BaseLoader for ScriptedLoader {
    fn load(
        &self,
        checkpoint: &Path,
        _context_tokens: u32,
    ) -> Result<Box<dyn Engine>, PolicyError> {
        self.record(ScriptedEvent::Loaded(checkpoint.to_path_buf()));
        Ok(Box::new(ScriptedEngine {
            loader: self.clone(),
            checkpoint: checkpoint.to_path_buf(),
            attached: None,
        }))
    }
}

/// One scripted base, loaded.
struct ScriptedEngine {
    loader: ScriptedLoader,
    checkpoint: PathBuf,
    attached: Option<PathBuf>,
}

impl Engine for ScriptedEngine {
    fn attach_adapter(&mut self, adapter: &Path) -> Result<(), String> {
        self.loader
            .record(ScriptedEvent::Attached(adapter.to_path_buf()));
        self.attached = Some(adapter.to_path_buf());
        Ok(())
    }

    fn detach_adapter(&mut self) -> Result<(), String> {
        self.loader.record(ScriptedEvent::Detached);
        self.attached = None;
        Ok(())
    }

    fn generate_stream(
        &self,
        request: &ChatRequest,
        _cancel: &CancelToken,
        on_delta: &mut dyn FnMut(ChatDelta),
    ) -> Result<ChatResponse, String> {
        let prompt = request.render_prompt().map_err(|e| e.to_string())?;
        self.loader
            .record(ScriptedEvent::Generated(self.attached.clone()));
        let text = (self.loader.reply)(&prompt, self.attached.as_deref());
        on_delta(ChatDelta::Text(text.clone()));
        Ok(ChatResponse {
            text,
            reasoning: String::new(),
            tool_calls: Vec::new(),
            finish_reason: FinishReason::Stop,
            usage: ChatUsage::default(),
        })
    }
}

impl Drop for ScriptedEngine {
    fn drop(&mut self) {
        self.loader
            .record(ScriptedEvent::Freed(self.checkpoint.clone()));
    }
}
