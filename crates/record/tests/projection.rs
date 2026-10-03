// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements durable, content-addressed experience
// stores for learning agents, for its clients. If your team needs expertise
// in training-data lineage or crash-safe storage, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: an experience is also the graph the experience database reads. Each
//! stored experience is an attempt at its task with a decision per agent
//! step; a verdict is evidence about that attempt, so the database's recipes
//! select and pair Splinter's own experience; a relation is an edge between
//! attempts and a step label is evidence about a decision.
#![allow(clippy::unwrap_used)]

use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

use serde_json::json;
use splinter_core::annotation::{
    Annotation, AnnotationBody, Label, Outcome, Producer, RelationKind, Strength,
};
use splinter_core::clock::FixedClock;
use splinter_core::experience::{
    Environment, Experience, ExperienceId, Privileged, PrivilegedKind, Provenance, Task,
};
use splinter_expdb::model::{RecordKind, Rel};
use splinter_expdb::query::Query;
use splinter_expdb::train::{Recipe, SampleBody};
use splinter_expdb::{Config, Database};
use splinter_record::experiences::ExperienceStore;
use splinter_record::workspace::Workspace;
use splinter_record::StateRoot;
use sven_sdk::atif::{AgentProfile, StepOrigin, TraceStep, Trajectory};

struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        static N: AtomicU32 = AtomicU32::new(0);
        let path = std::env::temp_dir().join(format!(
            "splinter-proj-{name}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&path);
        Self(path)
    }

    fn store(&self) -> ExperienceStore {
        ExperienceStore::new(&Workspace::at(&StateRoot::new(&self.0)))
    }

    fn snapshot(&self) -> splinter_expdb::manifest::Snapshot {
        Database::open(StateRoot::new(&self.0).expdb(), Config::default())
            .unwrap()
            .snapshot()
            .unwrap()
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn task(instruction: &str) -> Task {
    Task::new(
        "recall",
        vec![],
        Environment {
            kind: "closed-book".into(),
            spec: json!({}),
            snapshot: None,
        },
        instruction,
        vec![Privileged {
            kind: PrivilegedKind::Reference,
            content: "forty-two".into(),
            span: None,
        }],
    )
    .unwrap()
}

/// An experience of `instruction` whose agent says each of `says` in turn.
fn experience(instruction: &str, says: &[&str]) -> Experience {
    let mut trajectory = Trajectory::new(
        "ATIF-v1.7",
        AgentProfile {
            name: "test".into(),
            version: "1".into(),
            model_name: Some("scripted".into()),
            tool_definitions: None,
            extra: None,
        },
    );
    trajectory
        .steps
        .push(TraceStep::new(1, StepOrigin::User, instruction));
    for (n, said) in says.iter().enumerate() {
        trajectory
            .steps
            .push(TraceStep::new(2 + n as u64, StepOrigin::Agent, *said));
    }
    Experience::new(
        task(instruction),
        trajectory,
        says.last().map(|s| (*s).to_string()),
        Provenance::new(
            "scripted/solver",
            &FixedClock::new("2026-09-30T12:00:00.000Z"),
        ),
    )
    .unwrap()
}

fn grade(id: &ExperienceId, outcome: Outcome, strength: Strength) -> Annotation {
    Annotation {
        experience: id.clone(),
        producer: Producer {
            name: "grader".into(),
            version: "1".into(),
        },
        body: AnnotationBody::Verdict {
            outcome,
            strength,
            evidence: json!({}),
        },
    }
}

fn count(snapshot: &splinter_expdb::manifest::Snapshot, kind: RecordKind) -> usize {
    snapshot
        .query(&Query::all().kind(kind))
        .unwrap()
        .records
        .len()
}

#[test]
fn a_stored_experience_is_an_attempt_with_a_decision_per_agent_step() {
    let scratch = Scratch::new("shape");
    let store = scratch.store();
    store
        .put(&experience(
            "What is six times seven?",
            &["Let me think.", "42"],
        ))
        .unwrap();

    let snapshot = scratch.snapshot();
    assert_eq!(count(&snapshot, RecordKind::Attempt), 1);
    assert_eq!(count(&snapshot, RecordKind::Decision), 2);
    assert_eq!(
        count(&snapshot, RecordKind::Observation),
        1,
        "the user's turn is what was shown"
    );
    assert_eq!(snapshot.families().unwrap().len(), 1);
}

#[test]
fn storing_the_same_experience_again_adds_no_attempt() {
    let scratch = Scratch::new("once");
    let exp = experience("What is six times seven?", &["42"]);
    scratch.store().put(&exp).unwrap();
    scratch.store().put(&exp).unwrap();
    assert_eq!(count(&scratch.snapshot(), RecordKind::Attempt), 1);
}

#[test]
fn attempts_at_one_task_form_one_family() {
    let scratch = Scratch::new("family");
    let store = scratch.store();
    let ask = "What is six times seven?";
    store.put(&experience(ask, &["41"])).unwrap();
    store.put(&experience(ask, &["42"])).unwrap();
    store
        .put(&experience("What is two plus two?", &["4"]))
        .unwrap();

    let families = scratch.snapshot().families().unwrap();
    let mut sizes: Vec<usize> = families.iter().map(|f| f.attempts.len()).collect();
    sizes.sort();
    assert_eq!(sizes, vec![1, 2]);
}

#[test]
fn a_verdict_is_evidence_about_the_attempt_and_recipes_read_it_by_rank() {
    let scratch = Scratch::new("verdict");
    let store = scratch.store();
    let formal = store.put(&experience("six times seven?", &["42"])).unwrap();
    let judged = store
        .put(&experience("six times seven?", &["forty-two"]))
        .unwrap();
    let failed = store.put(&experience("six times seven?", &["41"])).unwrap();
    store
        .annotate(&grade(&formal, Outcome::Pass, Strength::Formal))
        .unwrap();
    store
        .annotate(&grade(&judged, Outcome::Pass, Strength::Judged))
        .unwrap();
    store
        .annotate(&grade(&failed, Outcome::Fail, Strength::Formal))
        .unwrap();

    let snapshot = scratch.snapshot();
    let rewards: Vec<Option<f64>> = {
        let mut all: Vec<_> = snapshot
            .families()
            .unwrap()
            .into_iter()
            .flat_map(|f| f.attempts)
            .map(|a| (a.attempt, a.reward))
            .collect();
        all.sort_by_key(|(attempt, _)| *attempt);
        all.into_iter().map(|(_, r)| r).collect()
    };
    assert_eq!(rewards, vec![Some(1.0), Some(1.0), Some(0.0)]);

    let strong = snapshot
        .compile(&Recipe::sft().episodes().min_rank(Strength::Formal.rank()))
        .unwrap();
    assert_eq!(strong.samples.len(), 1, "only the formally verified pass");
    let any = snapshot.compile(&Recipe::sft().episodes()).unwrap();
    assert_eq!(any.samples.len(), 2, "both passes, never the fail");

    let pairs = snapshot.compile(&Recipe::dpo().by_task()).unwrap();
    assert_eq!(
        pairs.samples.len(),
        1,
        "the formal pass against the formal fail"
    );
}

#[test]
fn a_relation_is_an_edge_between_attempts() {
    let scratch = Scratch::new("relation");
    let store = scratch.store();
    let failed = store.put(&experience("six times seven?", &["41"])).unwrap();
    let retry = store.put(&experience("six times seven?", &["42"])).unwrap();
    store
        .annotate(&Annotation {
            experience: retry.clone(),
            producer: Producer {
                name: "agent".into(),
                version: "1".into(),
            },
            body: AnnotationBody::Relation {
                kind: RelationKind::RetryOf,
                other: failed.clone(),
            },
        })
        .unwrap();

    let plan = scratch
        .snapshot()
        .compile(&Recipe::relations(Rel::RetryOf))
        .unwrap();
    assert_eq!(plan.samples.len(), 1);
    assert!(matches!(
        plan.samples[0].body,
        SampleBody::Related {
            rel: Rel::RetryOf,
            ..
        }
    ));
    assert_eq!(store.annotations(&retry).unwrap().annotations.len(), 1);
    assert!(store.annotations(&failed).unwrap().annotations.is_empty());
}

#[test]
fn a_step_label_is_evidence_about_that_steps_decision() {
    let scratch = Scratch::new("label");
    let store = scratch.store();
    let id = store
        .put(&experience("six times seven?", &["Hmm.", "42"]))
        .unwrap();
    store
        .annotate(&Annotation {
            experience: id.clone(),
            producer: Producer {
                name: "labeler".into(),
                version: "1".into(),
            },
            body: AnnotationBody::StepLabel {
                step: 3,
                label: Label::Good,
                evidence: json!({}),
            },
        })
        .unwrap();

    let plan = scratch
        .snapshot()
        .compile(&Recipe::prm().criterion("step_label"))
        .unwrap();
    assert_eq!(plan.samples.len(), 1);
    assert_eq!(store.annotations(&id).unwrap().annotations.len(), 1);
}

#[test]
fn an_experience_stored_without_its_graph_is_healed_by_storing_it_again() {
    let scratch = Scratch::new("heal");
    let exp = experience("six times seven?", &["42"]);
    let id = exp.id().unwrap();
    // A write that died between the experience and its graph.
    let db = Database::open(StateRoot::new(&scratch.0).expdb(), Config::default()).unwrap();
    let mut session = splinter_expdb::Session::open(
        &db,
        &splinter_expdb::WriterIdentity::new("test", "partial", "node", 0),
    )
    .unwrap();
    session
        .put_entity(&splinter_expdb::model::Entity::keyed(
            "experience",
            splinter_record::address::content_id(&id.0).unwrap(),
            serde_json::to_value(&exp).unwrap(),
        ))
        .unwrap();
    session.flush().unwrap();

    let store = scratch.store();
    assert!(store.contains(&id).unwrap());
    assert!(
        store
            .annotate(&grade(&id, Outcome::Pass, Strength::Formal))
            .is_err(),
        "without its graph there is nothing to record evidence about"
    );
    assert_eq!(store.put(&exp).unwrap(), id);
    store
        .annotate(&grade(&id, Outcome::Pass, Strength::Formal))
        .unwrap();
    assert_eq!(store.annotations(&id).unwrap().annotations.len(), 1);
    assert_eq!(count(&scratch.snapshot(), RecordKind::Attempt), 1);
}
