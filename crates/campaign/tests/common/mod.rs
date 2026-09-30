// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Fixtures the controller specs share: a scripted policy model that
//! answers from the prompt it is sent, and a context over a scratch state
//! root that serves it as `policy:default`.

// Each spec binary uses a subset of these helpers.
#![allow(dead_code)]

pub mod gate;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use splinter_agent::solve::Model;
use splinter_campaign::model_ref::ModelRef;
use splinter_campaign::{Config, Context};
use splinter_store::StateRoot;
use sven_sdk::model::{CompletionRequest, ModelProvider, ResponseEvent, ResponseStream};

/// The identity the scripted policy is recorded under.
pub const POLICY: &str = "scripted/policy-1";

type Script = dyn Fn(&str) -> String + Send + Sync;

/// A model whose every reply is a function of the whole prompt it is sent,
/// and which keeps the prompts.
#[derive(Clone)]
pub struct Scripted {
    reply: Arc<Script>,
    pub prompts: Arc<Mutex<Vec<String>>>,
}

impl Scripted {
    /// A model that replies `reply(prompt)`.
    pub fn new(reply: impl Fn(&str) -> String + Send + Sync + 'static) -> Self {
        Self {
            reply: Arc::new(reply),
            prompts: Arc::new(Mutex::new(Vec::new())),
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
            std::env::temp_dir().join(format!("splinter-campaign-{test}-{}", std::process::id()));
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
        model_store: scratch.0.join("models"),
        policy_base: scratch.0.join("models/Qwen/Qwen3-0.6B"),
        openrouter_api_key: None,
        brain_api_key: None,
        allow_remote: false,
        command_env: BTreeMap::new(),
        working_dir: scratch.0.clone(),
        brain_binary: None,
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
