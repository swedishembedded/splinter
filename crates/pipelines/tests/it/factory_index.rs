// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! Spec: the experience database indexes a factory run's committed rollout
//! trials as experiences of task-repository tasks with the factory's grade
//! as their verdict; a trial the infrastructure decided gets no verdict, a
//! trial with no trajectory is skipped with why, and indexing again files
//! nothing new.

use std::path::Path;

use common::{config, Scratch};
use serde_json::json;
use splinter_core::clock::{FixedClock, SystemClock};
use splinter_core::digest::Digest;
use splinter_orchestrator::Context;
use splinter_pipelines::factory_index::index_run;
use splinter_store::factory::products::ProductStore;

use crate::common;

fn trial(
    slot: u32,
    status: &str,
    counted: bool,
    passed: Option<bool>,
    trace: bool,
) -> serde_json::Value {
    json!({
        "schema": "splinter.trial/1", "slot": {"pass": "local", "task": "train/006", "slot": slot}, "attempt": 0,
        "task": "train/006", "task_digest": Digest::of(b"task").to_string(),
        "agent": {"name": "sven", "version": "5.9.2", "invocation": Digest::of(b"agent").to_string()},
        "model": "hosted:deepseek", "native_dir": format!("batches/0/t{slot}"), "native": [],
        "grade": {"status": status, "cause": null, "counted": counted, "score": if counted { json!(1.0) } else { json!(null) },
                  "passed": passed, "metrics": {}, "vetoes": []},
        "trace": if trace { json!({"path": format!("batches/0/t{slot}/agent/trajectory.json"), "digest": Digest::of(b"t").to_string()}) } else { json!(null) },
        "started_at": format!("2026-11-22T10:00:0{slot}Z"), "usage": {}, "guided": false,
    })
}

#[test]
fn a_factory_runs_trials_are_indexed_as_graded_experiences_once() {
    let scratch = Scratch::new("factory-index");
    let ctx = Context::new(config(&scratch), false)
        .unwrap()
        .with_clock(Box::new(FixedClock::new("2026-11-23T09:00:00.000Z")));
    let run = scratch.0.join("run");
    let products = ProductStore::new(&run, Box::new(SystemClock));
    let staging = products.begin_staging().unwrap();
    let trajectory = std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../factory/tests/it/fixtures/sven-trial/agent/trajectory.json"),
    )
    .unwrap();
    for slot in 0..3 {
        let path = staging
            .path()
            .join(format!("batches/0/t{slot}/agent/trajectory.json"));
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, &trajectory).unwrap();
    }
    let trials = [
        trial(0, "graded", true, Some(true), true),
        trial(1, "model_error", false, None, true),
        trial(2, "graded", true, Some(false), false),
    ];
    let lines: String = trials.iter().map(|t| format!("{t}\n")).collect();
    std::fs::write(staging.path().join("trials.jsonl"), lines).unwrap();
    products
        .publish(
            &Digest::of(b"stage"),
            "rollouts/local",
            staging,
            &run.join("rollouts/local"),
        )
        .unwrap();

    let first = index_run(&ctx, &run).unwrap();
    assert_eq!(first.len(), 1);
    let pass = &first[0];
    assert_eq!(pass.pass, "rollouts/local");
    assert_eq!(pass.verdicts.get("pass"), Some(&1));
    assert_eq!(
        pass.verdicts.get("abstain"),
        Some(&1),
        "an infrastructure failure is no verdict"
    );
    assert_eq!(
        pass.skipped.len(),
        1,
        "the trial with no trajectory: {:?}",
        pass.skipped
    );
    let set = pass.experience_set.clone().expect("a set");
    assert_eq!(ctx.experiences().get_set(&set).unwrap().members.len(), 2);

    let again = index_run(&ctx, &run).unwrap();
    assert_eq!(
        again[0].experience_set,
        Some(set),
        "indexing again files nothing new"
    );
}
