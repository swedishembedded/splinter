// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Fixtures the controller specs share: a scripted policy model that
//! answers from the prompt it is sent, and a context over a scratch state
//! root that serves it as `policy:default`; the release fixtures
//! ([`gate`]), the manual `learn` learns from ([`manual`]) and the small
//! repository `practice` works on with its scripted agents ([`repo`]).

// Each spec binary uses a subset of these helpers.
#![allow(dead_code)]

pub mod fake_brain;
pub mod gate;
pub mod manual;
pub mod repo;
pub mod timeline;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use splinter_agent::solve::Model;
use splinter_core::model_ref::ModelRef;
use splinter_orchestrator::{Config, Context};
use splinter_store::StateRoot;
use sven_sdk::model::{CompletionRequest, ModelProvider, ResponseEvent, ResponseStream, Role};

/// The identity the scripted policy is recorded under.
pub const POLICY: &str = "scripted/policy-1";

type Script = dyn Fn(&str) -> String + Send + Sync;

/// A model whose every reply is a function of the whole prompt it is sent,
/// and which keeps the prompts and the system turns each was sent under.
#[derive(Clone)]
pub struct Scripted {
    reply: Arc<Script>,
    pub prompts: Arc<Mutex<Vec<String>>>,
    pub systems: Arc<Mutex<Vec<Vec<String>>>>,
}

impl Scripted {
    /// A model that replies `reply(prompt)`.
    pub fn new(reply: impl Fn(&str) -> String + Send + Sync + 'static) -> Self {
        Self {
            reply: Arc::new(reply),
            prompts: Arc::new(Mutex::new(Vec::new())),
            systems: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

#[async_trait::async_trait]
impl ModelProvider for Scripted {
    fn name(&self) -> &str {
        "scripted"
    }
    fn model_name(&self) -> &str {
        "policy-1"
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let prompt = format!("{:?}", req.messages);
        let reply = (self.reply)(&prompt);
        self.prompts.lock().unwrap().push(prompt);
        self.systems.lock().unwrap().push(
            req.messages
                .iter()
                .filter(|m| m.role == Role::System)
                .map(|m| m.as_text().unwrap_or_default().to_string())
                .collect(),
        );
        Ok(Box::pin(futures::stream::iter(vec![
            Ok(ResponseEvent::TextDelta(reply)),
            Ok(ResponseEvent::Done),
        ])))
    }
}

/// A scratch directory removed when dropped.
pub struct Scratch(pub PathBuf);

impl Scratch {
    /// A fresh scratch directory named for `test`.
    pub fn new(test: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("splinter-pipelines-{test}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    /// The state root inside it.
    pub fn state(&self) -> StateRoot {
        StateRoot::new(self.0.join("state"))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A configuration over `scratch`'s state root, with no keys and no
/// network opt-in.
pub fn config(scratch: &Scratch) -> Config {
    Config {
        state_root: scratch.state(),
        jobs: Default::default(),
        model_store: scratch.0.join("models"),
        policy_base: scratch.0.join("models/Qwen/Qwen3-0.6B"),
        policy_context_tokens: None,
        allow_remote: false,
        command_env: BTreeMap::new(),
        working_dir: scratch.0.clone(),
        brain_binary: None,
        front_door_model: None,
        assistant_model: None,
        judge_model: None,
        bf16_base: false,
        default_budget: None,
        remote_concurrency: 4,
        solve_deadline: splinter_orchestrator::config::DEFAULT_SOLVE_DEADLINE,
        coding_prompt: splinter_core::prompt::CodingPrompt::default(),
        min_calibration_controls: 2,
        workspace: Default::default(),
        thinking: false,
        served_temperature: None,
        served_top_p: None,
        served_top_k: None,
        served_seed: None,
        served_thinking_budget: None,
        build: Default::default(),
    }
}

/// A context over a fresh scratch state root whose `policy:default` is
/// `policy`, with `allow_remote` as the command line's opt-in.
pub fn scratch_context(test: &str, policy: Scripted, allow_remote: bool) -> (Scratch, Context) {
    let scratch = Scratch::new(test);
    let ctx = Context::new(config(&scratch), allow_remote)
        .unwrap()
        .with_model(
            ModelRef::policy_default(),
            Model::new(Arc::new(policy), POLICY),
        );
    (scratch, ctx)
}

/// A model that answers as `inner` does after a pause of its own for each
/// prompt, and counts the most requests waiting on it at once: what a
/// batching server's slots are measured by. A pause that is shorter for
/// later prompts makes the answers come back in the reverse of the order
/// they were asked.
pub struct Counted {
    inner: Scripted,
    pause: Box<dyn Fn(&str) -> Duration + Send + Sync>,
    waiting: AtomicUsize,
    most: AtomicUsize,
}

impl Counted {
    /// `inner`, answering after `pause(prompt)`.
    pub fn new(
        inner: Scripted,
        pause: impl Fn(&str) -> Duration + Send + Sync + 'static,
    ) -> Arc<Self> {
        Arc::new(Self {
            inner,
            pause: Box::new(pause),
            waiting: AtomicUsize::new(0),
            most: AtomicUsize::new(0),
        })
    }

    /// The most requests that waited on it at once so far.
    pub fn most(&self) -> usize {
        self.most.load(Ordering::SeqCst)
    }
}

#[async_trait::async_trait]
impl ModelProvider for Counted {
    fn name(&self) -> &str {
        "counted"
    }
    fn model_name(&self) -> &str {
        "counted-1"
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let now = self.waiting.fetch_add(1, Ordering::SeqCst) + 1;
        self.most.fetch_max(now, Ordering::SeqCst);
        let pause = (self.pause)(&format!("{:?}", req.messages));
        tokio::time::sleep(pause).await;
        self.waiting.fetch_sub(1, Ordering::SeqCst);
        self.inner.complete(req).await
    }
}
