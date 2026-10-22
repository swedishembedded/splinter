// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible, auditable workspaces for
// agents that change code, for its clients. If your team needs expertise in
// sandboxed code-editing agents or reproducible agent runs, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec: a task recorded in a workspace is solved in the same workspace
//! later. The environment a record names is resolved from the
//! configuration alone: commands run in a container when the configuration
//! names a docker client and an image pinned by digest, else directly on the
//! host (and the record says it is not isolated); a bad image reference is
//! refused by name; the host tools that lay trees out are resolved once per
//! runtime.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use splinter_orchestrator::config::WorkspaceSettings;
use splinter_orchestrator::{Config, OrchestratorError, Runtime};
use splinter_sandbox::{ResolvedEnvironment, WorkspaceEnvironment};
use splinter_store::StateRoot;

/// The digest an image reference is pinned by.
const IMAGE: &str =
    "registry.example/build@sha256:0000000000000000000000000000000000000000000000000000000000000001";

struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("splinter-orch-ws-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn runtime(scratch: &Scratch, workspace: WorkspaceSettings) -> Runtime {
    let config = Config {
        state_root: StateRoot::new(scratch.0.join("state")),
        model_store: scratch.0.join("models"),
        policy_base: scratch.0.join("models/base"),
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
        thinking: false,
        served_temperature: None,
        served_top_p: None,
        served_top_k: None,
        served_seed: None,
        served_thinking_budget: None,
        build: Default::default(),
        workspace,
    };
    Runtime::new(config, false).unwrap()
}

/// A one-file tree archived as `tar` writes it.
fn archive(scratch: &Scratch) -> PathBuf {
    let tree = scratch.0.join("tree");
    std::fs::create_dir_all(&tree).unwrap();
    std::fs::write(tree.join("app.txt"), "broken\n").unwrap();
    let tar = scratch.0.join("tree.tar");
    let status = Command::new("tar")
        .arg("--create")
        .arg("--file")
        .arg(&tar)
        .arg("--directory")
        .arg(&tree)
        .arg(".")
        .status()
        .unwrap();
    assert!(status.success());
    tar
}

/// The record of `archive` laid out as `runtime`'s workspace backend would.
fn recorded(runtime: &Runtime, archive: &Path) -> splinter_core::experience::Environment {
    let backend = runtime.environments().workspace().unwrap();
    let snapshot = backend.host.snapshot(archive).unwrap();
    let env = WorkspaceEnvironment::new(snapshot, backend.sandbox, backend.host);
    ResolvedEnvironment::Workspace(env).record().unwrap()
}

#[test]
fn a_recorded_workspace_resolves_to_the_environment_it_was_recorded_in() {
    let scratch = Scratch::new("record");
    let runtime = runtime(&scratch, WorkspaceSettings::default());
    let record = recorded(&runtime, &archive(&scratch));
    assert_eq!(record.kind, "workspace");

    let ResolvedEnvironment::Workspace(env) = runtime.environments().for_record(&record).unwrap()
    else {
        panic!("a workspace record resolves to a workspace");
    };
    assert_eq!(
        ResolvedEnvironment::Workspace(env).record().unwrap(),
        record,
        "the same record, so the solve's check against the snapshot passes"
    );
}

#[test]
fn commands_run_on_the_host_and_the_record_says_so_unless_a_container_is_configured() {
    let scratch = Scratch::new("backend");
    let tar = archive(&scratch);

    let host = recorded(&runtime(&scratch, WorkspaceSettings::default()), &tar);
    assert_eq!(host.spec["sandbox"]["backend"], "process", "{host:?}");
    assert_eq!(host.spec["sandbox"]["isolation"], "none", "{host:?}");

    let container = recorded(
        &runtime(
            &scratch,
            WorkspaceSettings {
                docker: Some("/usr/bin/docker".into()),
                image: Some(IMAGE.into()),
                ..WorkspaceSettings::default()
            },
        ),
        &tar,
    );
    assert_eq!(
        container.spec["sandbox"]["backend"], "container",
        "{container:?}"
    );
    assert_eq!(container.spec["sandbox"]["isolation"], "container");
    assert_eq!(container.spec["sandbox"]["image"], IMAGE);
    assert_ne!(
        host.snapshot, container.snapshot,
        "another sandbox, another environment"
    );

    // A docker client alone names no image to run in: commands stay on the host.
    let alone = recorded(
        &runtime(
            &scratch,
            WorkspaceSettings {
                docker: Some("/usr/bin/docker".into()),
                ..WorkspaceSettings::default()
            },
        ),
        &tar,
    );
    assert_eq!(alone.spec["sandbox"]["isolation"], "none");
}

#[test]
fn a_cargo_home_in_the_configuration_reaches_the_commands_and_the_record() {
    let scratch = Scratch::new("cargo");
    let tar = archive(&scratch);
    let plain = recorded(&runtime(&scratch, WorkspaceSettings::default()), &tar);
    let with = recorded(
        &runtime(
            &scratch,
            WorkspaceSettings {
                cargo_home: Some("/srv/cargo-home".into()),
                ..WorkspaceSettings::default()
            },
        ),
        &tar,
    );
    assert_eq!(
        with.spec["sandbox"]["env"]["CARGO_HOME"], "/srv/cargo-home",
        "{with:?}"
    );
    assert!(plain.spec["sandbox"]["env"].get("CARGO_HOME").is_none());
    assert_ne!(plain.snapshot, with.snapshot);
}

#[test]
fn an_image_that_is_not_pinned_by_digest_is_refused_by_name() {
    let scratch = Scratch::new("image");
    let runtime = runtime(
        &scratch,
        WorkspaceSettings {
            docker: Some("/usr/bin/docker".into()),
            image: Some("python:latest".into()),
            ..WorkspaceSettings::default()
        },
    );
    match runtime.environments().workspace() {
        Err(OrchestratorError::Refused(why)) => {
            assert!(why.contains("python:latest"), "{why}");
        }
        other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
    }
}

#[test]
fn a_record_that_is_not_a_workspace_of_a_known_tree_is_refused() {
    let scratch = Scratch::new("bad-record");
    let runtime = runtime(&scratch, WorkspaceSettings::default());
    let mut record = recorded(&runtime, &archive(&scratch));
    record.spec = serde_json::json!({"snapshot": "not a digest"});
    assert!(runtime.environments().for_record(&record).is_err());
    // The tools are resolved once: the same backend is handed out again.
    let (first, second) = (
        runtime.environments().workspace().unwrap(),
        runtime.environments().workspace().unwrap(),
    );
    assert!(Arc::ptr_eq(&first.host, &second.host));
}

#[test]
fn variables_the_configuration_sets_for_project_commands_are_part_of_the_record_and_nothing_else_is(
) {
    let scratch = Scratch::new("env");
    let plain = runtime(&scratch, WorkspaceSettings::default());
    let archive = archive(&scratch);
    let before = recorded(&plain, &archive);
    assert!(
        before.spec["sandbox"]["env"]
            .get("PYTHONDONTWRITEBYTECODE")
            .is_none(),
        "no variable unless asked: an existing task keeps its address"
    );

    let settings = WorkspaceSettings {
        env: BTreeMap::from([("PYTHONDONTWRITEBYTECODE".to_string(), "1".to_string())]),
        ..WorkspaceSettings::default()
    };
    let with = runtime(&scratch, settings);
    let after = recorded(&with, &archive);
    assert_eq!(after.spec["sandbox"]["env"]["PYTHONDONTWRITEBYTECODE"], "1");
    assert_ne!(
        before.snapshot, after.snapshot,
        "a task generated with it names another environment"
    );
    assert!(
        matches!(
            plain.environments().for_record(&after),
            Ok(ResolvedEnvironment::Workspace(_))
        ),
        "resolving is from the configuration; the solve compares the records"
    );
}
