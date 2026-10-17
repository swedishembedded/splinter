// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements self-improving coding agents whose every
// step is auditable, for its clients. If your team needs expertise in agent
// evaluation or locally operated coding agents, you can procure our
// services by sending an email to info@swedishembedded.com.

//! What the loop's specs share: a scripted model standing in for a real one
//! and the disposable world of one spec (a repository with a seeded bug and
//! an acceptance script that lives outside it).

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]
// Each spec file uses a part of this.
#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{json, Value};
use splinter_agent_loop::contract::{Check, Contract, Limits, CONTRACT_SCHEMA};
use splinter_agent_loop::outcome::Outcome;
use splinter_agent_loop::store::LoopHome;
use splinter_agent_loop::trace::read_events;
use splinter_sdk::agent::solve::Model;
use splinter_sdk::config::Config;
use splinter_sdk::store::StateRoot;
use splinter_sdk::vocabulary::model_ref::ModelRef;
use splinter_sdk::Splinter;
use sven_sdk::model::{
    CompletionRequest, MessageContent, ModelProvider, ResponseEvent, ResponseStream,
};

// The loop's own dependency on sven is through the SDK; the tests reach the
// model trait the same way.
use splinter_sdk::agent::sven as sven_sdk;

pub const MODEL: &str = "local:/scripted";

/// Runs `git` in `dir` with a fixed identity and no host configuration.
pub fn git(dir: &std::path::Path, args: &[&str]) -> String {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["-c", "user.name=t", "-c", "user.email=t@example.com"])
        .args(["-c", "commit.gpgsign=false"])
        .args(args)
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_CONFIG_SYSTEM", "/dev/null")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

pub fn head_rev(dir: &std::path::Path) -> String {
    git(dir, &["rev-parse", "HEAD"]).trim().to_string()
}

pub fn is_clean(dir: &std::path::Path) -> bool {
    git(dir, &["status", "--porcelain", "-uall"])
        .trim()
        .is_empty()
}

/// What a scripted model answers a request with.
pub type Script =
    Box<dyn Fn(&CompletionRequest) -> anyhow::Result<Vec<ResponseEvent>> + Send + Sync>;

/// A model that answers from a script over the request it is sent.
pub struct Scripted {
    pub script: Script,
    pub pause: Duration,
    pub requests: Mutex<Vec<CompletionRequest>>,
}

#[async_trait::async_trait]
impl ModelProvider for Scripted {
    fn name(&self) -> &str {
        "scripted"
    }
    fn model_name(&self) -> &str {
        "fixture-1"
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        tokio::time::sleep(self.pause).await;
        let events = (self.script)(&req);
        self.requests.lock().unwrap().push(req);
        Ok(Box::pin(futures::stream::iter(
            events?.into_iter().map(Ok).collect::<Vec<_>>(),
        )))
    }
}

pub fn text(reply: &str) -> Vec<ResponseEvent> {
    vec![ResponseEvent::TextDelta(reply.into()), ResponseEvent::Done]
}

pub fn call(id: &str, name: &str, arguments: Value) -> Vec<ResponseEvent> {
    vec![
        ResponseEvent::ToolCall {
            index: 0,
            id: id.into(),
            name: name.into(),
            arguments: arguments.to_string(),
        },
        ResponseEvent::Done,
    ]
}

pub fn tool_results(req: &CompletionRequest) -> usize {
    req.messages
        .iter()
        .filter(|m| matches!(m.content, MessageContent::ToolResult { .. }))
        .count()
}

pub fn write(path: &str, content: &str) -> Vec<ResponseEvent> {
    call(
        "w",
        "write_file",
        json!({"path": path, "text": content, "append": false}),
    )
}

pub const FIXED: &str = "def add(a, b):\n    return a + b\n";

/// The disposable world of one spec.
pub struct Fixture {
    pub home: tempfile::TempDir,
    pub repo: PathBuf,
    _outside: tempfile::TempDir,
    pub check: PathBuf,
}

impl Fixture {
    pub fn new() -> Self {
        let home = tempfile::tempdir().unwrap();
        let repo = home.path().join("target-repo");
        std::fs::create_dir_all(repo.join("tests")).unwrap();
        // Like most repositories, this one ignores sven's directory and bytecode.
        std::fs::write(repo.join(".gitignore"), ".sven/\n__pycache__/\n").unwrap();
        std::fs::write(repo.join("calc.py"), "def add(a, b):\n    return a - b\n").unwrap();
        std::fs::write(
            repo.join("tests/test_calc.py"),
            "from calc import add\nassert add(2, 3) == 5\n",
        )
        .unwrap();
        git(&repo, &["init", "-q", "-b", "main"]);
        git(&repo, &["add", "-A"]);
        git(&repo, &["commit", "-q", "-m", "seeded bug"]);
        let outside = tempfile::tempdir().unwrap();
        let check = outside.path().join("check.sh");
        std::fs::write(&check, "PYTHONPATH=. python3 tests/test_calc.py\n").unwrap();
        Self {
            home,
            repo,
            _outside: outside,
            check,
        }
    }

    pub fn contract(&self, max_attempts: u32) -> Contract {
        Contract {
            schema: CONTRACT_SCHEMA,
            task: "add() returns the difference; it must return the sum".into(),
            baseline_rev: head_rev(&self.repo),
            repository: self.repo.clone(),
            acceptance: vec![Check {
                name: "unit".into(),
                command: format!("sh {}", self.check.display()),
                visible: false,
            }],
            protected: vec!["tests/".into()],
            limits: Limits {
                attempt_secs: 60,
                total_secs: 120,
                max_output_tokens: 10_000,
                max_tool_calls: 10,
                max_repeated_results: 0,
                max_attempts,
                provider_retries: 0,
                follow_ups: 0,
                max_cost_usd: None,
            },
            model: MODEL.into(),
            allow_api_models: false,
            system_prompt_digest: splinter_sdk::vocabulary::digest::Digest::sha256_of(
                splinter_agent_loop::run::system_prompt(&self.loop_home())
                    .unwrap()
                    .0
                    .as_bytes(),
            )
            .to_string(),
            system_prompt_source: "built-in".into(),
            definitions_digest: String::new(),
            hints: Vec::new(),
        }
    }

    /// The contract with its one check shown to the worker, which is the
    /// only kind of check whose output the worker is ever told.
    pub fn shown(&self, max_attempts: u32) -> Contract {
        let mut contract = self.contract(max_attempts);
        contract.acceptance[0].visible = true;
        contract
    }

    pub fn splinter(&self, model: Arc<Scripted>) -> Splinter {
        let state = StateRoot::under_home(self.home.path());
        let config = Config {
            state_root: state,
            model_store: self.home.path().join("models"),
            policy_base: self.home.path().join("models/policy"),
            policy_context_tokens: None,
            openrouter_api_key: None,
            brain_api_key: None,
            allow_remote: false,
            command_env: Default::default(),
            working_dir: self.home.path().to_path_buf(),
            brain_binary: None,
            front_door_model: None,
            assistant_model: None,
            judge_model: None,
            bf16_base: false,
            default_budget: None,
            remote_concurrency: 1,
            solve_deadline: splinter_sdk::config::DEFAULT_SOLVE_DEADLINE,
            min_calibration_controls: 1,
            workspace: Default::default(),
            thinking: false,
            served_temperature: None,
            build: Default::default(),
        };
        let splinter = Splinter::builder(config).build().unwrap();
        splinter.runtime().add_model(
            MODEL.parse::<ModelRef>().unwrap(),
            Model::new(model, "scripted/fixture-1"),
        );
        splinter
    }

    pub fn loop_home(&self) -> LoopHome {
        LoopHome::under(self.home.path())
    }
}

pub fn model(
    pause: Duration,
    script: impl Fn(&CompletionRequest) -> anyhow::Result<Vec<ResponseEvent>> + Send + Sync + 'static,
) -> Arc<Scripted> {
    Arc::new(Scripted {
        script: Box::new(script),
        pause,
        requests: Mutex::new(Vec::new()),
    })
}

pub fn kinds(events: &[Value]) -> Vec<&str> {
    events.iter().map(|e| e["type"].as_str().unwrap()).collect()
}

pub fn events_of(outcome: &Outcome) -> Vec<Value> {
    read_events(outcome.artifacts.events.parent().unwrap()).unwrap()
}
