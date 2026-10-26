// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agent-driven training-task generation that
// is judged by what the tasks do when run, for its clients. If your team
// needs expertise in synthetic task generation then you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: a harness-driven proposer runs `sven harness run` as a subprocess on
//! a tree it lays out (sections, brief, request, inputs), and reads the
//! `tasks.json` the run leaves back as the one reply shape admission parses.
//! A fake `sven` stands in for the binary, so the subprocess contract is
//! tested without a model.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::json;
use splinter_agent::harness_proposer::{HarnessLaunch, HarnessProposer};
use splinter_knowledge::tasks::propose::{ProposalRequest, ProposeError, Proposed, TaskProposer};
use splinter_knowledge::tasks::Rejection;
use splinter_sandbox::{Limits, ProcessSandbox, RuntimeEnvironment, RuntimeRegistry};

/// What a harness run does in the fake: the tasks file it leaves (if any)
/// and the status it reports.
const FAKE_SVEN: &str = r#"#!/bin/sh
here=$(dirname "$0")
while [ $# -gt 0 ]; do
  case "$1" in
    --workspace) ws=$2 ;;
    --trajectory) trajectory=$2 ;;
    --conversations) conversations=$2 ;;
    --input-file) inputs=$2 ;;
  esac
  shift
done
[ -f "$ws/sections/00.md" ] && [ -f "$ws/sections/01.md" ] && [ -f "$ws/brief.md" ] && [ -f "$ws/request.json" ] || { echo "tree not laid out" >&2; exit 2; }
cp "$inputs" "$here/inputs.json"
cp "$ws/brief.md" "$here/brief.md"
printf '%s' "$PATH" > "$here/path"
[ -f "$here/tasks.json" ] && cp "$here/tasks.json" "$ws/tasks.json"
mkdir -p "$conversations"
echo '{}' > "$trajectory"
echo '{}' > "$conversations/0000-survey.json"
echo '{"status":"completed","conclusion":"success"}'
"#;

struct Fixture {
    root: tempfile::TempDir,
}

impl Fixture {
    fn new(tasks_file: Option<&str>) -> Self {
        let root = tempfile::tempdir().unwrap();
        let sven = root.path().join("sven");
        std::fs::write(&sven, FAKE_SVEN).unwrap();
        std::fs::set_permissions(&sven, std::fs::Permissions::from_mode(0o755)).unwrap();
        if let Some(text) = tasks_file {
            std::fs::write(root.path().join("tasks.json"), text).unwrap();
        }
        Self { root }
    }

    fn path(&self, name: &str) -> PathBuf {
        self.root.path().join(name)
    }

    fn proposer(&self) -> HarnessProposer {
        self.proposer_running(self.path("sven"))
    }

    fn proposer_running(&self, sven: PathBuf) -> HarnessProposer {
        let sandbox = ProcessSandbox::new(
            self.path("sandbox"),
            BTreeMap::new(),
            &[],
            Limits::default(),
        );
        let runtime =
            RuntimeEnvironment::new(&RuntimeRegistry::builtin(), "sh", Arc::new(sandbox)).unwrap();
        HarnessProposer::new(
            HarnessLaunch {
                sven,
                harness: PathBuf::from("harness-dir"),
                model: "served-model".into(),
                endpoint: "http://127.0.0.1:1/v1".into(),
                runs: self.path("runs"),
                first_byte_timeout: None,
            },
            vec![runtime],
            Arc::new(Refuses),
        )
    }
}

/// The typed-call proposer a harness proposer hands other requests to.
struct Refuses;

#[async_trait]
impl TaskProposer for Refuses {
    fn identity(&self) -> &str {
        "refuses"
    }
    async fn propose(&self, _: &ProposalRequest<'_>) -> Result<Proposed, ProposeError> {
        Err(ProposeError("fallback".into()))
    }
}

fn request(method: &str) -> ProposalRequest<'_> {
    ProposalRequest {
        method,
        brief: "Write tasks.",
        role: "You write tasks.",
        input: json!({"source": {"name": "m"}, "sections": [
            {"position": 0, "text": "first section"},
            {"position": 1, "text": "second section"}]}),
        kind: Some("construct-lua"),
        runtime: Some("sh"),
        count: 3,
        deadline: Duration::from_secs(30),
        max_output_tokens: None,
        repairs: 0,
    }
}

const TASKS: &str = r#"{"tasks": [{"instruction": "What is first?", "subject": "m", "reference": "first", "evidence": [{"section": 0}]}]}"#;

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap()
}

#[tokio::test]
async fn the_tasks_a_run_leaves_are_the_reply_and_the_run_is_laid_out_as_the_contract_says() {
    let fixture = Fixture::new(Some(TASKS));
    let proposer = fixture.proposer();

    let proposed = proposer.propose(&request("generate_tasks")).await.unwrap();

    let Proposed::Replied(reply) = proposed else {
        panic!("{proposed:?}");
    };
    assert!(format!("{reply:?}").contains("What is first?"));
    let inputs: serde_json::Value =
        serde_json::from_str(&read(&fixture.path("inputs.json"))).unwrap();
    assert_eq!(
        inputs,
        json!({"count": 3, "kind": "construct-lua", "runtime": "sh"})
    );
    assert_eq!(
        read(&fixture.path("brief.md")),
        "You write tasks.\n\nWrite tasks."
    );
    // The runtime's directory comes first on the child's PATH.
    let first = read(&fixture.path("path"));
    assert!(
        first.starts_with("/usr/bin:") || first.starts_with("/bin:"),
        "{first}"
    );
    // The trajectories are kept, and recorded with how the run ended.
    let runs = proposer.runs();
    assert_eq!(runs.len(), 1);
    assert!(runs[0].trajectory.is_file());
    assert!(runs[0].conversations.join("0000-survey.json").is_file());
    assert_eq!(runs[0].conclusion.as_deref(), Some("success"));
}

#[tokio::test]
async fn a_tasks_file_that_is_not_the_reply_shape_is_declined_as_malformed() {
    let fixture = Fixture::new(Some(r#"{"tasks": [{"instruction": "no reference"}]}"#));
    let proposed = fixture
        .proposer()
        .propose(&request("generate_tasks"))
        .await
        .unwrap();
    assert!(
        matches!(
            proposed,
            Proposed::Declined {
                rejection: Rejection::Malformed,
                ..
            }
        ),
        "{proposed:?}"
    );
}

#[tokio::test]
async fn a_run_that_leaves_no_tasks_file_has_no_reply() {
    let fixture = Fixture::new(None);
    let proposed = fixture
        .proposer()
        .propose(&request("generate_tasks"))
        .await
        .unwrap();
    assert!(
        matches!(
            proposed,
            Proposed::Declined {
                rejection: Rejection::NoReply,
                ..
            }
        ),
        "{proposed:?}"
    );
}

#[tokio::test]
async fn a_binary_that_cannot_run_is_an_error_not_a_declined_request() {
    let fixture = Fixture::new(Some(TASKS));
    let proposer = fixture.proposer_running(fixture.path("missing-sven"));
    assert!(proposer.propose(&request("generate_tasks")).await.is_err());
}

#[tokio::test]
async fn a_request_that_is_not_for_a_kinds_tasks_goes_to_the_typed_call() {
    let fixture = Fixture::new(Some(TASKS));
    let error = fixture
        .proposer()
        .propose(&request("vary_tasks"))
        .await
        .unwrap_err();
    assert_eq!(error.0, "fallback");
}
