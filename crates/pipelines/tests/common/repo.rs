// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that learn a codebase from its own
// tests, for its clients. If your team needs expertise in training
// code-editing agents on executable tasks drawn from a repository, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The small Python project the `practice` and workspace-release specs work
//! on, as a git repository, and the scripted agents that edit it through
//! sven's own tools: one that fixes half its attempts and reaches for the
//! hidden tests in some ([`Agent`]), and a mender that restores a module or
//! does nothing ([`Mender`]).

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;
use std::sync::{Arc, Mutex};

use sven_sdk::model::{
    CompletionRequest, MessageContent, ModelProvider, ResponseEvent, ResponseStream, Role,
};

/// How many modules the project has; each is a family of tasks.
pub const MODULES: usize = 6;

/// The modules whose solver reaches for the hidden tests before it fixes
/// the module.
pub const PEEKING: [usize; 3] = [0, 1, 2];

pub fn module(n: usize) -> String {
    format!(
        "def tier{n}(units):\n    \"\"\"The price tier of an order.\"\"\"\n    if units > 10:\n        \
         return 20 + {n}\n    return 5\n"
    )
}

/// The tests of module `n`. A request lists the symptoms the tests that
/// catch a fault name, so another wording of the names is another request
/// for the same file; one that shares most of its words with an earlier one
/// would be refused as a repeat of it. A name of eight words or more would
/// quote the tests in the request, which is refused too.
pub fn test_of(n: usize, wording: Wording) -> String {
    let (large, small) = match wording {
        Wording::Short => (
            "large_orders_pay_the_large_tier",
            "small_orders_pay_the_small_tier",
        ),
        Wording::Long => (
            "bulk_buyers_pay_the_wholesale_rate",
            "casual_buyers_pay_the_retail_rate",
        ),
    };
    format!(
        "import unittest\n\nfrom shop.tier{n} import tier{n}\n\n\nclass Tier{n}Test(unittest.TestCase):\n    \
         def test_tier{n}_{large}(self):\n        self.assertEqual(tier{n}(11), 20 + {n})\n\n    \
         def test_tier{n}_{small}(self):\n        self.assertEqual(tier{n}(10), 5)\n"
    )
}

/// How the tests of a project name what they check.
#[derive(Clone, Copy)]
pub enum Wording {
    Short,
    Long,
}

pub fn git(dir: &Path, args: &[&str]) {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["-c", "user.name=t", "-c", "user.email=t@example.com"])
        .args(["-c", "commit.gpgsign=false", "-c", "core.autocrlf=false"])
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
}

pub fn write(path: &Path, text: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

/// The project in a git repository under `dir`, with `extra` files beside.
pub fn project(dir: &Path, extra: &[(&str, &str)]) {
    project_worded(dir, extra, Wording::Short);
}

/// [`project`] whose tests are worded as `wording` says.
pub fn project_worded(dir: &Path, extra: &[(&str, &str)], wording: Wording) {
    project_of(dir, extra, wording, MODULES);
}

/// [`project_worded`] with `modules` modules, each a family of tasks.
pub fn project_of(dir: &Path, extra: &[(&str, &str)], wording: Wording, modules: usize) {
    std::fs::create_dir_all(dir).unwrap();
    git(dir, &["init", "-q", "-b", "main"]);
    write(
        &dir.join("pyproject.toml"),
        "[project]\nname = \"shop\"\nversion = \"1\"\n",
    );
    write(&dir.join("shop/__init__.py"), "");
    write(&dir.join("tests/__init__.py"), "");
    for n in 0..modules {
        write(&dir.join(format!("shop/tier{n}.py")), &module(n));
        write(
            &dir.join(format!("tests/test_tier{n}.py")),
            &test_of(n, wording),
        );
    }
    for (path, text) in extra {
        write(&dir.join(path), text);
    }
    git(dir, &["add", "-A", "-f"]);
    git(dir, &["commit", "-q", "-m", "shop"]);
}

/// A scripted agent. Asked to change a module it cannot see, it restores the
/// module it infers from the request (a request lists the symptoms of the
/// failing tests, and each is named for its module). Of the two attempts at a task
/// it makes, the first fixes the module and the second looks and does
/// nothing, so each task is solved about half the time. For the
/// [`PEEKING`] modules the fixing attempt first reads the test file the
/// request is about, which is withheld from the tree.
pub struct Agent {
    /// Attempts begun, by request.
    begun: Mutex<BTreeMap<String, u32>>,
}

impl Agent {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            begun: Mutex::new(BTreeMap::new()),
        })
    }
}

pub fn call(id: &str, name: &str, arguments: serde_json::Value) -> Vec<ResponseEvent> {
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

pub fn shell(command: &str) -> Vec<ResponseEvent> {
    call(
        "c-shell",
        "shell",
        serde_json::json!({"description": "look", "shell_command": command}),
    )
}

/// The first user turn of a conversation: the request.
pub fn request_of(req: &CompletionRequest) -> String {
    req.messages
        .iter()
        .find(|m| m.role == Role::User)
        .and_then(|m| m.as_text().map(str::to_string))
        .unwrap_or_default()
}

/// The module a request is about: the number in the symptoms it lists, which
/// are named for the tests that catch them.
pub fn module_of(request: &str) -> Option<usize> {
    let at = request.find("- tier")? + "- tier".len();
    let digits: String = request[at..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    digits.parse().ok()
}

pub fn tool_calls(req: &CompletionRequest) -> Vec<String> {
    req.messages
        .iter()
        .filter_map(|m| match &m.content {
            MessageContent::ToolCall { function, .. } => Some(function.arguments.clone()),
            _ => None,
        })
        .collect()
}

#[async_trait::async_trait]
impl ModelProvider for Agent {
    fn name(&self) -> &str {
        "scripted"
    }
    fn model_name(&self) -> &str {
        "agent-1"
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let request = request_of(&req);
        let calls = tool_calls(&req);
        let asked_about = module_of(&request);
        let events = match (calls.len(), asked_about) {
            (0, Some(n)) => {
                let attempt = {
                    let mut begun = self.begun.lock().unwrap();
                    let seen = begun.entry(request.clone()).or_insert(0);
                    *seen += 1;
                    *seen
                };
                if attempt % 2 == 1 && PEEKING.contains(&n) {
                    shell(&format!("cat tests/test_tier{n}.py # fix"))
                } else if attempt % 2 == 1 {
                    shell("true # fix")
                } else {
                    shell("true # look")
                }
            }
            (1, Some(n)) if calls[0].contains("# fix") => call(
                "c-fix",
                "write_file",
                serde_json::json!({"path": format!("shop/tier{n}.py"), "text": module(n), "append": false}),
            ),
            _ => vec![ResponseEvent::TextDelta("done".into()), ResponseEvent::Done],
        };
        Ok(Box::pin(futures::stream::iter(
            events.into_iter().map(Ok).collect::<Vec<_>>(),
        )))
    }
}

/// An agent that restores the module a request is about, or looks and does
/// nothing. One that peeks first reads the test file the request is about,
/// which is withheld from its tree: the exposure audit fails every attempt
/// of it, whatever the patch does.
pub struct Mender {
    mends: bool,
    peeks: Vec<usize>,
}

impl Mender {
    /// An agent that restores the module when `mends`, else does nothing.
    pub fn new(mends: bool) -> Arc<Self> {
        Arc::new(Self {
            mends,
            peeks: Vec::new(),
        })
    }

    /// An agent that restores every module, reading the hidden test first
    /// of the `modules` it is asked about.
    pub fn peeking_at(modules: &[usize]) -> Arc<Self> {
        Arc::new(Self {
            mends: true,
            peeks: modules.to_vec(),
        })
    }
}

fn restore(n: usize) -> Vec<ResponseEvent> {
    call(
        "c-fix",
        "write_file",
        serde_json::json!({"path": format!("shop/tier{n}.py"), "text": module(n), "append": false}),
    )
}

#[async_trait::async_trait]
impl ModelProvider for Mender {
    fn name(&self) -> &str {
        "scripted"
    }
    fn model_name(&self) -> &str {
        "mender-1"
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let calls = tool_calls(&req);
        let events = match (calls.len(), module_of(&request_of(&req))) {
            (0, Some(n)) if self.peeks.contains(&n) => {
                shell(&format!("cat tests/test_tier{n}.py # peek"))
            }
            (0, Some(n)) if self.mends => restore(n),
            (1, Some(n)) if self.peeks.contains(&n) && calls[0].contains("# peek") => restore(n),
            _ => vec![ResponseEvent::TextDelta("done".into()), ResponseEvent::Done],
        };
        Ok(Box::pin(futures::stream::iter(
            events.into_iter().map(Ok).collect::<Vec<_>>(),
        )))
    }
}
