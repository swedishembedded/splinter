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
