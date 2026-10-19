// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements interchange of agent benchmark tasks between
// task formats for its clients. If your team needs expertise in importing
// containerised benchmark tasks into reproducible, verifiable task sets, you
// can procure our services by sending an email to info@swedishembedded.com.

//! Spec: Harbor tasks become a task set that records where they came from,
//! Harbor trials become experiences carrying Harbor's own verdict, and a
//! task set can be written back out as Harbor tasks.
//!
//! A trial's verdict is the verifier's: a `reward` of exactly 1 passes, any
//! other reward fails (0.5 is partial credit, not success), and a trial with
//! an exception or no reward has no verdict. A trial without an ATIF
//! trajectory, or of a task that was not imported, is refused. An ingested
//! set is not re-graded by `verify`: its attempts carry no patch.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::fs;
use std::path::{Path, PathBuf};

use common::{config, Scratch};
use splinter_core::clock::FixedClock;
use splinter_knowledge::tasks::workspace::Oracle;
use splinter_orchestrator::Context;
use splinter_pipelines::harbor::{
    discover_harbor, export_harbor, import_harbor, ingest_harbor_trials,
};
use splinter_pipelines::verify::{verify_set, Grading};
use splinter_store::decision::decide;

fn write(path: &Path, text: &str) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, text).unwrap();
}

fn context(test: &str) -> (Scratch, Context) {
    let scratch = Scratch::new(test);
    let ctx = Context::new(config(&scratch), false)
        .unwrap()
        .with_clock(Box::new(FixedClock::new("2026-10-19T09:00:00.000Z")));
    (scratch, ctx)
}

/// A repository task in Harbor's layout whose reward is 1 only for the fix.
fn calc_task(root: &Path, name: &str) -> PathBuf {
    let dir = root.join(name);
    write(
        &dir.join("environment/Dockerfile"),
        "FROM ubuntu:24.04\nWORKDIR /app\nCOPY . /app/\n",
    );
    write(&dir.join("environment/calc.sh"), "echo 1\n");
    write(
        &dir.join("instruction.md"),
        "Make calc.sh print the sum of one and two.\n",
    );
    write(
        &dir.join("task.toml"),
        "schema_version = \"1.4\"\n[task]\nname = \"org/calc\"\n[metadata]\ncategory = \"arithmetic\"\n",
    );
    write(
        &dir.join("tests/test.sh"),
        "#!/bin/bash\ncd /app\nif [ \"$(sh calc.sh)\" = 3 ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi\n",
    );
    write(
        &dir.join("solution/solve.sh"),
        "#!/bin/bash\ncd /app\necho 'echo 3' > calc.sh\n",
    );
    dir
}

const TRAJECTORY: &str = r#"{
  "schema_version": "ATIF-v1.5",
  "agent": {"name": "terminus", "version": "2.0", "model_name": "qwen"},
  "steps": [
    {"step_id": 1, "timestamp": "2026-10-19T08:00:00Z", "source": "user", "message": "Make calc.sh print the sum of one and two."},
    {"step_id": 2, "timestamp": "2026-10-19T08:00:05Z", "source": "agent", "message": "done", "model_name": "qwen"}
  ]
}"#;

fn trial(root: &Path, name: &str, task: &str, verifier: &str, trajectory: bool) -> PathBuf {
    let dir = root.join(name);
    write(
        &dir.join("result.json"),
        &format!(
            r#"{{"task_name": "{task}", "trial_name": "{name}", "task_checksum": "abc",
                "agent_info": {{"name": "terminus", "version": "2.0", "model_info": {{"name": "qwen"}}}}
                {verifier}}}"#
        ),
    );
    if trajectory {
        write(&dir.join("agent/trajectory.json"), TRAJECTORY);
    }
    dir
}

#[test]
fn harbor_tasks_become_a_task_set_that_records_their_source_and_an_unimportable_one_is_refused() {
    let (scratch, ctx) = context("harbor-import");
    let root = scratch.0.join("tasks");
    let good = calc_task(&root, "calc");
    let builds = calc_task(&root, "builds");
    write(
        &builds.join("environment/Dockerfile"),
        "FROM ubuntu\nWORKDIR /app\nRUN make\nCOPY . /app/\n",
    );

    let found = discover_harbor(&root).unwrap();
    assert_eq!(
        found.tasks.iter().filter(|d| d.refusal.is_none()).count(),
        1
    );

    let imported = import_harbor(&ctx, &[good.clone(), builds.clone()], Oracle::Require).unwrap();
    assert_eq!(imported.imported.len(), 1, "{imported:#?}");
    assert_eq!(imported.imported[0].name, "org/calc");
    assert_eq!(imported.imported[0].source.directory, good);
    assert!(
        imported.refused[0].reason.contains("RUN"),
        "{:?}",
        imported.refused
    );
    let set = imported.task_set.expect("a set of one task");
    assert_eq!(ctx.tasks().get_set(&set).unwrap().members.len(), 1);

    // The same task again is already held, and nothing new is made.
    let again = import_harbor(&ctx, &[good], Oracle::Require).unwrap();
    assert!(again.imported.is_empty());
    assert!(
        again.refused[0].reason.contains("AlreadyHeld"),
        "{:?}",
        again.refused
    );

    // And it can be written back out as a Harbor task that records how to
    // rebuild it exactly.
    let out = scratch.0.join("exported");
    let written = export_harbor(&ctx, &set, &out, "ubuntu:24.04").unwrap();
    assert_eq!(written.written.len(), 1);
    let task_toml = fs::read_to_string(written.written[0].directory.join("task.toml")).unwrap();
    assert!(task_toml.contains("splinter.harbor.v1"), "{task_toml}");
}

#[test]
fn trials_become_experiences_carrying_harbors_verdict_and_are_not_regraded() {
    let (scratch, ctx) = context("harbor-trials");
    let root = scratch.0.join("tasks");
    let imported = import_harbor(&ctx, &[calc_task(&root, "calc")], Oracle::Skip).unwrap();
    let set = imported.task_set.unwrap();
    let trials = scratch.0.join("trials");

    let passed = trial(
        &trials,
        "passed",
        "org/calc",
        r#", "verifier_result": {"rewards": {"reward": 1.0}}"#,
        true,
    );
    let partial = trial(
        &trials,
        "partial",
        "org/calc",
        r#", "verifier_result": {"rewards": {"reward": 0.5}}"#,
        true,
    );
    let broken = trial(
        &trials,
        "broken",
        "org/calc",
        r#", "verifier_result": null, "exception_info": {"exception_type": "Timeout"}"#,
        true,
    );
    let no_trajectory = trial(
        &trials,
        "bare",
        "org/calc",
        r#", "verifier_result": {"rewards": {"reward": 1}}"#,
        false,
    );
    let foreign = trial(
        &trials,
        "foreign",
        "org/other",
        r#", "verifier_result": {"rewards": {"reward": 1}}"#,
        true,
    );

    let ingested = ingest_harbor_trials(
        &ctx,
        &set,
        &[passed, partial, broken, no_trajectory, foreign],
    )
    .unwrap();

    assert_eq!(ingested.verdicts.get("pass"), Some(&1));
    assert_eq!(
        ingested.verdicts.get("fail"),
        Some(&1),
        "0.5 is partial credit, a fail"
    );
    assert_eq!(
        ingested.verdicts.get("abstain"),
        Some(&1),
        "an exception is no verdict"
    );
    assert_eq!(ingested.refused.len(), 2, "{:#?}", ingested.refused);
    assert!(ingested
        .refused
        .iter()
        .any(|r| r.reason.contains("trajectory")));
    assert!(ingested
        .refused
        .iter()
        .any(|r| r.reason.contains("not an imported task")));

    let experiences = ingested.experience_set.unwrap();
    let store = ctx.experiences();
    let members = store.get_set(&experiences).unwrap().members;
    let decided: Vec<Option<bool>> = members
        .iter()
        .map(|id| decide(&store.annotations(id).unwrap().annotations).map(|d| d.passed))
        .collect();
    assert_eq!(decided.iter().filter(|d| **d == Some(true)).count(), 1);
    assert_eq!(decided.iter().filter(|d| **d == Some(false)).count(), 1);
    assert_eq!(decided.iter().filter(|d| d.is_none()).count(), 1);

    // `verify` leaves Harbor's verdicts alone: nothing is graded again.
    let verified = verify_set(
        &ctx,
        &experiences,
        Grading::CodeOnly,
        &splinter_agent::CancelToken::new(),
    )
    .unwrap();
    assert_eq!(verified.experiences, 0);
    assert_eq!(verified.unverified.len(), 3);
    let after: Vec<Option<bool>> = members
        .iter()
        .map(|id| decide(&store.annotations(id).unwrap().annotations).map(|d| d.passed))
        .collect();
    assert_eq!(after, decided);
}
