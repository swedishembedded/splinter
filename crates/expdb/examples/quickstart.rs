// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Record one attempt, branch it, and turn the experience into training data.
//!
//! Run with `cargo run --release -p splinter-expdb --example quickstart`.

use std::collections::BTreeMap;

use splinter_expdb::model::{
    Action, Outcome, PolicyRef, ReproLevel, State, TaskDefinition, TaskInstance,
};
use splinter_expdb::train::Recipe;
use splinter_expdb::{Config, ContentId, Database, WriterIdentity};

fn state(label: &str) -> State {
    State::new(
        BTreeMap::from([("files".to_owned(), ContentId::of(label.as_bytes()))]),
        ReproLevel::Exact,
    )
}

fn main() -> splinter_expdb::Result<()> {
    let root =
        tempfile::TempDir::new().map_err(splinter_expdb::Error::io("a scratch directory"))?;
    let db = Database::open(root.path(), Config::default())?;

    // One writer per process; it needs no coordination with any other.
    let mut collector = db.collector(&WriterIdentity::new("demo", "job-1", "host-a", 0))?;
    let definition = TaskDefinition {
        name: "fix the bug".into(),
        description: "make the failing test pass".into(),
        domain: "coding".into(),
    };
    let instance = TaskInstance {
        definition: definition.id()?,
        params: serde_json::json!({ "issue": 742 }),
        environment: None,
    };

    let policy = PolicyRef::new("my-policy", "checkpoint-1");
    let mut run =
        collector.start_attempt(&definition, &instance, &state("repo@a"), &policy, Some(7))?;
    let decision = run.decision().commit(Action::new(
        "grep",
        serde_json::json!({ "pattern": "overflow" }),
    ))?;
    run.transition(&decision, &state("repo@b"), Some(0.2), None)?;
    run.finish(Outcome::Fail)?;

    // Try something else at the same decision. Everything before it is shared.
    let mut branch = collector.fork(&decision, &PolicyRef::new("my-policy", "checkpoint-1"))?;
    branch
        .decision()
        .commit(Action::new("read", serde_json::json!({ "file": "uart.c" })))?;
    branch.finish(Outcome::Pass)?;
    collector.flush()?;

    // A snapshot is a fixed view; pin it so a training run can come back to it.
    let snapshot = db.snapshot()?;
    snapshot.pin("training-run-1")?;

    // A recipe compiled against a snapshot is the dataset; JSON lines are only an export.
    let plan = snapshot.compile(&Recipe::dpo().min_gap(0.5))?;
    println!(
        "snapshot {} -> {} preference pair(s), plan {}",
        snapshot.id(),
        plan.samples.len(),
        plan.id()?
    );
    plan.export_jsonl(&snapshot, &mut std::io::stdout())?;
    Ok(())
}
