// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Shared helpers for the specs: a scratch database root per test.
#![allow(dead_code, clippy::unwrap_used)]

use std::collections::BTreeMap;

use splinter_expdb::model::{
    family_key, Action, Body, Decision, Edge, Observation, Outcome, PolicyRef, Record, Rel,
    ReproLevel, State, TaskDefinition,
};
use splinter_expdb::{Config, ContentId, Database, RecordId, WriterId};
use tempfile::TempDir;

/// A record id of writer `writer`.
pub fn rid(writer: u64, seq: u64) -> RecordId {
    RecordId::new(WriterId::from_raw(writer), seq)
}

/// A state with one named part.
pub fn state(part: &str) -> State {
    State::new(
        BTreeMap::from([("fs".to_owned(), ContentId::of(part.as_bytes()))]),
        ReproLevel::Exact,
    )
}

/// A decision record with a distinctive action.
pub fn decision(writer: u64, seq: u64, family: ContentId, action: &str) -> Record {
    let st = state("s0").id().unwrap();
    Record::new(
        rid(writer, seq),
        seq,
        Body::Decision(Decision {
            state: st,
            observation: None,
            policy: PolicyRef::new("policy", "1"),
            context: None,
            action: Action::new(action, serde_json::json!({ "n": seq })),
            old_logprob: Some(-0.5),
            value_estimate: None,
        }),
    )
    .in_attempt(rid(writer, 0), family, ContentId::of(b"instance"))
}

/// A family key for test `n`.
pub fn family(n: u64) -> ContentId {
    family_key(&ContentId::of(&n.to_le_bytes()), &state("s0").id().unwrap())
}

/// A mixed set of records and edges covering several kinds.
pub fn mixed(writer: u64, count: u64) -> (Vec<Record>, Vec<Edge>) {
    let fam = family(1);
    let mut records = Vec::new();
    let mut edges = Vec::new();
    for seq in 0..count {
        let record = match seq % 4 {
            0 => Record::new(
                rid(writer, seq),
                seq,
                Body::TaskDefinition(TaskDefinition {
                    name: format!("task {seq}"),
                    description: "fix the bug".into(),
                    domain: "coding".into(),
                }),
            ),
            1 => Record::new(
                rid(writer, seq),
                seq,
                Body::Observation(Observation {
                    state: state("s0").id().unwrap(),
                    content: splinter_expdb::model::Content::text(format!("output {seq}")),
                }),
            ),
            2 => decision(writer, seq, fam, "grep"),
            _ => Record::new(
                rid(writer, seq),
                seq,
                Body::AttemptEnd {
                    attempt: rid(writer, 0),
                    outcome: Outcome::Pass,
                },
            )
            .with_parent(rid(writer, seq - 1)),
        };
        if seq > 0 {
            edges.push(Edge::new(
                rid(writer, seq),
                Rel::DerivedFrom,
                rid(writer, seq - 1),
            ));
        }
        records.push(record);
    }
    (records, edges)
}

/// A database in its own temporary directory, removed on drop.
pub struct Scratch {
    pub dir: TempDir,
}

impl Scratch {
    pub fn new() -> Self {
        Self {
            dir: TempDir::new().unwrap(),
        }
    }

    pub fn open(&self) -> Database {
        Database::open(self.dir.path(), Config::default()).unwrap()
    }

    pub fn open_with(&self, config: Config) -> Database {
        Database::open(self.dir.path(), config).unwrap()
    }
}

/// A task definition, instance and initial state for a coding task.
pub fn coding_task(n: u64) -> (TaskDefinition, splinter_expdb::model::TaskInstance, State) {
    let definition = TaskDefinition {
        name: "fix the bug".into(),
        description: "make the failing test pass".into(),
        domain: "coding".into(),
    };
    let instance = splinter_expdb::model::TaskInstance {
        definition: definition.id().unwrap(),
        params: serde_json::json!({ "issue": n }),
        environment: None,
    };
    (definition, instance, state("repo@abc123"))
}

/// Records one attempt of `chain` decisions, each followed by a transition,
/// ending as `outcome`. Returns the decisions made and the attempt id.
pub fn attempt(
    collector: &mut splinter_expdb::ingest::Collector,
    task: u64,
    policy: &str,
    chain: usize,
    outcome: Outcome,
) -> (Vec<splinter_expdb::ingest::DecisionRef>, RecordId) {
    let (definition, instance, initial) = coding_task(task);
    let mut run = collector
        .start_attempt(
            &definition,
            &instance,
            &initial,
            &PolicyRef::new(policy, "1"),
            None,
        )
        .unwrap();
    let attempt = run.attempt();
    let mut decisions = Vec::new();
    for step in 0..chain {
        let d = run
            .decision()
            .commit(Action::new("act", serde_json::json!({ "step": step })))
            .unwrap();
        run.transition(&d, &state(&format!("{policy}-{task}-{step}")), None, None)
            .unwrap();
        decisions.push(d);
    }
    run.finish(outcome).unwrap();
    (decisions, attempt)
}

/// A skill body with a distinctive name.
pub fn skill(name: &str) -> splinter_expdb::model::Skill {
    splinter_expdb::model::Skill {
        name: name.into(),
        description: format!("{name}: do it before anything else"),
        trigger: "an asynchronous operation fails".into(),
        action_pattern: "inspect the event source".into(),
        expected_effect: "the failing layer is found first".into(),
        parents: vec![],
        prerequisites: vec![],
    }
}
