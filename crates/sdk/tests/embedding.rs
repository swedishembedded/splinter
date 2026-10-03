// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that turn agent experience
// into better local models, for its clients. If your team needs expertise in
// continual learning, agent evaluation or embedding such a system in a
// product, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: a program that depends on the SDK alone can run a command as a
//! recorded run, read the record back, and see its progress reported.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use serde_json::json;
use splinter_sdk::store::runs::read_run;
use splinter_sdk::store::StateRoot;
use splinter_sdk::{Config, Splinter};

fn config(dir: &std::path::Path) -> Config {
    Config {
        state_root: StateRoot::new(dir.join("state")),
        model_store: dir.join("models"),
        policy_base: dir.join("models/base"),
        openrouter_api_key: None,
        brain_api_key: None,
        allow_remote: false,
        command_env: BTreeMap::new(),
        working_dir: dir.to_path_buf(),
        brain_binary: None,
        front_door_model: None,
        assistant_model: None,
        bf16_base: false,
        default_budget: None,
        remote_concurrency: 4,
    }
}

#[test]
fn a_command_runs_as_a_recorded_run_that_can_be_read_back() {
    let dir = std::env::temp_dir().join(format!("splinter-sdk-embedding-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&seen);
    let splinter = Splinter::builder(config(&dir))
        .progress(Arc::new(move |stage, summary| {
            sink.lock()
                .unwrap()
                .push((stage.to_string(), summary.clone()));
        }))
        .build()
        .unwrap();

    let recorded = splinter
        .run("embedded", &json!({ "why": "a spec" }), |_, run| {
            run.stage("step", &json!({ "did": "something" }))?;
            Ok(json!({ "answer": 42 }))
        })
        .unwrap();

    assert_eq!(recorded.report["answer"], 42);
    let run = read_run(splinter.context().workspace(), &recorded.run).unwrap();
    assert_eq!(run.command, "embedded");
    assert_eq!(run.arguments["why"], "a spec");
    assert_eq!(run.stages.len(), 1);
    assert_eq!(run.outputs["answer"], 42);
    assert_eq!(
        seen.lock().unwrap().as_slice(),
        [("step".to_string(), json!({ "did": "something" }))],
        "the stage was reported as it finished"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn each_command_gets_a_context_of_its_own_on_the_shared_runtime() {
    let dir = std::env::temp_dir().join(format!("splinter-sdk-contexts-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let splinter = Splinter::builder(config(&dir)).build().unwrap();
    let (first, second) = (splinter.context(), splinter.context());
    assert!(
        Arc::ptr_eq(first.runtime(), second.runtime()),
        "one runtime serves every command"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
