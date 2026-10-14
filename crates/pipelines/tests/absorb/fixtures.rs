// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Sessions with invented content, a scripted extractor model, and a
//! context over a scratch state root.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use atif::{
    AgentProfile, ObservationEntry, StepObservation, StepOrigin, ToolInvocation, TraceStep,
    Trajectory,
};
use serde_json::json;
use splinter_agent::solve::Model;
use splinter_core::model_ref::ModelRef;
use splinter_orchestrator::{Config, Context};
use splinter_store::StateRoot;
use sven_sdk::model::{CompletionRequest, ModelProvider, ResponseEvent, ResponseStream};

pub type Outcome = anyhow::Result<()>;

pub const WHERE: &str = "Which port does the Tessera dashboard listen on?";
pub const API_KEY: &str = "sk-proj-4f9a8c7b2e1d3f6a5b4c3d2e1f0a9b8c";

/// A model whose reply is a function of the prompt, and which keeps the
/// prompts it was sent.
struct Scripted {
    reply: Box<dyn Fn(&str) -> String + Send + Sync>,
    prompts: Arc<Mutex<Vec<String>>>,
}

#[async_trait::async_trait]
impl ModelProvider for Scripted {
    fn name(&self) -> &str {
        "scripted"
    }
    fn model_name(&self) -> &str {
        "extractor-1"
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let prompt = format!("{:?}", req.messages);
        let reply = (self.reply)(&prompt);
        self.prompts
            .lock()
            .map_err(|_| anyhow::anyhow!("poisoned"))?
            .push(prompt);
        Ok(Box::pin(futures::stream::iter(vec![
            Ok(ResponseEvent::TextDelta(reply)),
            Ok(ResponseEvent::Done),
        ])))
    }
}

/// A context over a scratch directory, whose `policy:default` is the
/// scripted extractor.
pub struct World {
    pub dir: tempfile::TempDir,
    pub ctx: Context,
    pub prompts: Arc<Mutex<Vec<String>>>,
}

impl World {
    /// The prompts the extractor was sent so far, joined.
    pub fn shown(&self) -> String {
        self.prompts
            .lock()
            .map(|p| p.join("\n"))
            .unwrap_or_default()
    }

    /// `trajectory` written to `name` in the scratch directory.
    pub fn write(&self, name: &str, trajectory: &Trajectory) -> anyhow::Result<PathBuf> {
        let path = self.dir.path().join("sessions").join(name);
        write_json(&path, trajectory)?;
        Ok(path)
    }

    pub fn sessions_dir(&self) -> PathBuf {
        self.dir.path().join("sessions")
    }
}

fn write_json(path: &Path, trajectory: &Trajectory) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_vec(trajectory)?)?;
    Ok(())
}

pub fn world(reply: impl Fn(&str) -> String + Send + Sync + 'static) -> anyhow::Result<World> {
    let dir = tempfile::tempdir()?;
    let config = Config {
        state_root: StateRoot::new(dir.path().join("state")),
        model_store: dir.path().join("models"),
        policy_base: dir.path().join("models/base"),
        policy_context_tokens: None,
        openrouter_api_key: None,
        brain_api_key: None,
        allow_remote: false,
        command_env: BTreeMap::new(),
        working_dir: dir.path().to_path_buf(),
        brain_binary: None,
        front_door_model: None,
        assistant_model: None,
        judge_model: None,
        bf16_base: false,
        default_budget: None,
        remote_concurrency: 4,
        min_calibration_controls: 2,
        thinking: false,
        served_temperature: None,
    };
    let prompts = Arc::new(Mutex::new(Vec::new()));
    let model = Scripted {
        reply: Box::new(reply),
        prompts: prompts.clone(),
    };
    let ctx = Context::new(config, false)?.with_model(
        ModelRef::policy_default(),
        Model::new(Arc::new(model), "scripted/extractor-1"),
    );
    Ok(World { dir, ctx, prompts })
}

fn trajectory(steps: Vec<TraceStep>) -> Trajectory {
    let mut t = Trajectory::new("ATIF-v1.7", AgentProfile::new("scripted", "1"));
    t.steps = steps;
    t
}

fn say(id: u64, who: StepOrigin, text: &str) -> TraceStep {
    TraceStep::new(id, who, text)
}

/// The user asks, the agent answers wrongly, the user corrects, the agent
/// acknowledges.
pub fn port_correction() -> Trajectory {
    let mut t = trajectory(vec![
        say(1, StepOrigin::User, WHERE),
        say(
            2,
            StepOrigin::Agent,
            "The Tessera dashboard listens on port 8080.",
        ),
        say(
            3,
            StepOrigin::User,
            "No, that is wrong. It listens on port 9090 since the March move.",
        ),
        say(
            4,
            StepOrigin::Agent,
            "Thanks for the correction: port 9090.",
        ),
    ]);
    t.session_id = Some("monday".into());
    t
}

/// The next day the port changed again.
pub fn port_changed() -> Trajectory {
    let mut t = trajectory(vec![
        say(1, StepOrigin::User, "Quick check on the dashboard."),
        say(2, StepOrigin::Agent, "Ready."),
        say(
            3,
            StepOrigin::User,
            "The Tessera dashboard listens on port 9443 after the TLS change.",
        ),
        say(4, StepOrigin::Agent, "Noted."),
    ]);
    t.session_id = Some("tuesday".into());
    t
}

/// The user asks for a deploy; the agent runs it with a tool.
pub fn deploy() -> Trajectory {
    let mut call = say(2, StepOrigin::Agent, "");
    call.tool_calls = Some(vec![ToolInvocation::new("c1", "shell")
        .with_arguments(json!({"cmd": "brindle deploy --env staging"}))]);
    call.observation = Some(StepObservation::single(ObservationEntry::for_call(
        "c1",
        "deployed brindle 1.4.2 to staging",
    )));
    let mut t = trajectory(vec![
        say(
            1,
            StepOrigin::User,
            "Deploy the Brindle service to staging.",
        ),
        call,
        say(3, StepOrigin::Agent, "Deployed."),
    ]);
    t.session_id = Some("deploy".into());
    t
}

/// A session where the user pastes a key.
pub fn with_secret() -> Trajectory {
    let mut t = trajectory(vec![
        say(
            1,
            StepOrigin::User,
            &format!("Log in with {API_KEY} and tell me the registry port."),
        ),
        say(2, StepOrigin::Agent, "Logged in."),
    ]);
    t.session_id = Some("secret".into());
    t
}

/// A session the training projection cannot render: a second user step
/// beside a tool call.
pub fn mixed() -> Trajectory {
    let mut t = deploy();
    t.steps
        .push(say(4, StepOrigin::User, "Thanks. Now check the status."));
    t.steps.push(say(5, StepOrigin::Agent, "Healthy."));
    t.session_id = Some("mixed".into());
    t
}
