// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for running third-party coding agents
// against local models without exposing their operators' accounts for its
// clients. If your team needs expertise in confining agent runtimes and
// measuring them fairly, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: Claude Code starts only under its launch rules. A launch toward any
//! endpoint but a loopback brain origin, with a key shaped like a hosted
//! provider's, of a binary other than the pinned one, or with a home that would
//! expose a forbidden path is refused; a launch that starts is confined: a
//! canary in a stand-in for the account's login stays unreadable by absolute
//! path, another TCP port is refused while the endpoint's is reached, the
//! environment carries the key and no other credential, and the record has the
//! one shape every runtime's has. A stand-in binary, pinned by its own digest,
//! reports from inside; nothing here starts Claude Code.

#![allow(clippy::unwrap_used)]

use std::io::Read as _;
use std::net::TcpListener;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::time::Duration;

use sha2::{Digest as _, Sha256};
use splinter_agent::runner::claude_code::{
    check_endpoint, check_key, normalize, ClaudeCodeLaunch, ClaudeCodeRunner,
};
use splinter_agent::runner::{collect, prepare, AgentRunner, Episode, RunCancel, RunLimits};
use splinter_sandbox::host::DEFAULT_SEARCH_PATH;
use splinter_sandbox::{Limits, ProcessSandbox, WorkspaceEnvironment, WorkspaceHost};
use std::collections::BTreeMap;
use std::sync::Arc;
use sven_sdk::RunConclusion;

const KEY: &str = "brain-key-0123456789abcdef";

fn stand_in(dir: &Path, script: &str) -> (std::path::PathBuf, String) {
    let install = dir.join("install");
    std::fs::create_dir_all(install.join("bin")).unwrap();
    let entry = install.join("bin/claude");
    std::fs::write(&entry, script).unwrap();
    std::fs::set_permissions(&entry, std::fs::Permissions::from_mode(0o755)).unwrap();
    (
        install,
        format!("{:x}", Sha256::digest(std::fs::read(&entry).unwrap())),
    )
}

fn launch(
    install: &Path,
    sha256: &str,
    endpoint: &str,
    forbid: Vec<std::path::PathBuf>,
) -> ClaudeCodeLaunch {
    ClaudeCodeLaunch {
        install: install.to_path_buf(),
        entry: "bin/claude".into(),
        sha256: sha256.into(),
        version: "test".into(),
        endpoint: endpoint.into(),
        key: KEY.into(),
        model: "main".into(),
        context_window: Some(65536),
        forbid,
    }
}

fn git(dir: &Path, args: &[&str]) {
    let out = std::process::Command::new("git")
        .current_dir(dir)
        .args([
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.com",
            "-c",
            "commit.gpgsign=false",
        ])
        .args(args)
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_CONFIG_SYSTEM", "/dev/null")
        .output()
        .unwrap();
    assert!(out.status.success(), "git {args:?}");
}

/// A one-file repository as a snapshot, its workspace in the process sandbox,
/// and an episode of it prepared under `dir/episode`.
fn episode(dir: &Path) -> (WorkspaceEnvironment, Episode) {
    let repo = dir.join("origin");
    std::fs::create_dir_all(&repo).unwrap();
    git(&repo, &["init", "-q", "-b", "main"]);
    std::fs::write(repo.join("app.txt"), "broken\n").unwrap();
    git(&repo, &["add", "-A"]);
    git(&repo, &["commit", "-q", "-m", "snapshot"]);
    let archive = dir.join("snapshot.tar");
    git(
        &repo,
        &[
            "archive",
            "--format=tar",
            "-o",
            archive.to_str().unwrap(),
            "HEAD",
        ],
    );
    std::fs::create_dir_all(dir.join("calls")).unwrap();
    let host = Arc::new(WorkspaceHost::resolve(DEFAULT_SEARCH_PATH, dir).unwrap());
    let sandbox = ProcessSandbox::new(dir.join("calls"), BTreeMap::new(), &[], Limits::default());
    let env = WorkspaceEnvironment::new(host.snapshot(&archive).unwrap(), Arc::new(sandbox), host);
    let episode = prepare(&env, &archive, &dir.join("episode")).unwrap();
    (env, episode)
}

#[test]
fn an_endpoint_other_than_a_loopback_origin_and_a_hosted_key_are_refused() {
    for endpoint in [
        "https://api.anthropic.com",
        "http://api.anthropic.com:80",
        "https://openrouter.ai/api/v1",
        "http://localhost:8795",
        "http://10.0.0.1:8795",
        "https://127.0.0.1:8795",
        "http://127.0.0.1",
    ] {
        assert!(check_endpoint(endpoint).is_err(), "{endpoint}");
    }
    assert_eq!(check_endpoint("http://127.0.0.1:8795").unwrap(), 8795);
    for key in [
        "sk-ant-api03-aaaaaaaaaaaaaaaaaaaa",
        "sk-or-v1-bbbbbbbbbbbbbbbbbbbb",
        "short",
        "has spaces in it ok",
    ] {
        assert!(check_key(key).is_err(), "{key}");
    }
    check_key(KEY).unwrap();
}

#[tokio::test]
async fn the_confined_run_cannot_read_the_login_nor_reach_another_port() {
    let dir = tempfile::tempdir().unwrap();
    let account = dir.path().join("account");
    std::fs::create_dir_all(account.join(".claude")).unwrap();
    let canary = account.join(".claude/.credentials.json");
    std::fs::write(&canary, "{\"canary\": true}").unwrap();
    let endpoint = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = endpoint.local_addr().unwrap().port();
    let other = TcpListener::bind("127.0.0.1:0").unwrap();
    let other_port = other.local_addr().unwrap().port();
    let script = format!(
        "#!/bin/bash\nprompt=$(cat)\n\
         canary=$(cat {canary:?} 2>&1 && echo READ || echo denied)\n\
         gw=$( (exec 3<>/dev/tcp/127.0.0.1/{port}) 2>&1 && echo reached || echo refused)\n\
         other=$( (exec 3<>/dev/tcp/127.0.0.1/{other_port}) 2>&1 && echo reached || echo refused)\n\
         creds=$(env | cut -d= -f1 | grep -E '_(AUTH|OAUTH)_TOKEN$|OPENROUTER' | tr '\\n' ,)\n\
         echo touched > edited.txt\n\
         printf '%s\\n' '{{\"type\":\"assistant\",\"message\":{{\"id\":\"m1\",\"content\":[{{\"type\":\"text\",\"text\":\"done\"}}]}}}}'\n\
         printf '{{\"type\":\"result\",\"subtype\":\"success\",\"probe\":\"%s|%s|%s|%s|%s|%s|%s\"}}\\n' \"$prompt\" \"${{canary##*$'\\n'}}\" \"${{gw##*$'\\n'}}\" \"${{other##*$'\\n'}}\" \"$creds\" \"$ANTHROPIC_API_KEY\" \"$*\"\n"
    );
    let (install, sha) = stand_in(dir.path(), &script);
    let runner = ClaudeCodeRunner::new(launch(
        &install,
        &sha,
        &format!("http://127.0.0.1:{port}"),
        vec![account.clone()],
    ))
    .unwrap();
    let (env, episode) = episode(dir.path());
    let outcome = runner
        .run(
            &episode,
            "do the task",
            RunLimits {
                deadline: Duration::from_secs(60),
            },
            RunCancel::new(),
        )
        .await
        .unwrap();
    assert_eq!(outcome.conclusion, RunConclusion::Success);
    let mut stream = String::new();
    std::fs::File::open(&outcome.native_record)
        .unwrap()
        .read_to_string(&mut stream)
        .unwrap();
    let (steps, result) = normalize(&stream);
    assert_eq!(steps[0].text, "done");
    let probe = result.unwrap()["probe"].as_str().unwrap().to_string();
    let fields: Vec<&str> = probe.split('|').collect();
    assert_eq!(fields[0], "do the task");
    assert_eq!(
        fields[1], "denied",
        "the canary in the stand-in login stays unreadable"
    );
    assert_eq!(fields[2], "reached", "the endpoint's port is reached");
    assert_eq!(fields[3], "refused", "another port is refused");
    assert_eq!(fields[4], "", "no other credential variable");
    assert_eq!(fields[5], KEY);
    assert!(
        fields[6].starts_with("--bare -p --output-format stream-json"),
        "{}",
        fields[6]
    );
    assert!(
        episode.tree.join("edited.txt").is_file(),
        "the tree is writable"
    );
    assert_eq!(
        std::fs::metadata(&outcome.native_record)
            .unwrap()
            .permissions()
            .mode()
            & 0o222,
        0,
        "the raw stream is read-only"
    );
    let record = collect(&env, &episode, runner.identity(), outcome).unwrap();
    assert_eq!(
        record.patch.files,
        vec![std::path::PathBuf::from("edited.txt")],
        "the change is read back from git"
    );
    assert_eq!(record.runtime.mode.as_deref(), Some("bare"));
    assert_eq!(record.native_sha256.len(), 64);
}

#[tokio::test]
async fn a_binary_other_than_the_pinned_build_or_a_grant_over_the_login_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let (install, _) = stand_in(dir.path(), "#!/bin/sh\nexit 0\n");
    let (_env, episode) = episode(dir.path());
    let limits = RunLimits {
        deadline: Duration::from_secs(30),
    };
    let runner = ClaudeCodeRunner::new(launch(
        &install,
        &"0".repeat(64),
        "http://127.0.0.1:9",
        vec![],
    ))
    .unwrap();
    let err = runner
        .run(&episode, "x", limits, RunCancel::new())
        .await
        .unwrap_err();
    assert!(err.to_string().contains("not the pinned build"), "{err}");

    let (install, sha) = stand_in(dir.path(), "#!/bin/sh\nexit 0\n");
    let login = episode.tree.join(".claude");
    std::fs::create_dir_all(&login).unwrap();
    let runner =
        ClaudeCodeRunner::new(launch(&install, &sha, "http://127.0.0.1:9", vec![login])).unwrap();
    let err = runner
        .run(&episode, "x", limits, RunCancel::new())
        .await
        .unwrap_err();
    assert!(err.to_string().contains("would expose"), "{err}");
}
