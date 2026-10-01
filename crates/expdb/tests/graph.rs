// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: the experience is a graph, not a list of trajectories. Branches
//! share their common prefix, ancestry is cheap however long the path, and
//! typed links answer lineage questions forwards and backwards.
#![allow(clippy::unwrap_used)]

mod common;

use common::{attempt, coding_task, Scratch};
use splinter_expdb::model::{Body, Outcome, PolicyRef, Rel};
use splinter_expdb::{RecordId, WriterIdentity};

fn identity() -> WriterIdentity {
    WriterIdentity::new("exp", "job", "node", 0)
}

#[test]
fn a_branch_shares_its_prefix_instead_of_copying_it() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = db.collector(&identity()).unwrap();
    let (decisions, _) = attempt(&mut c, 1, "p", 5, Outcome::Fail);
    let before = {
        c.flush().unwrap();
        db.snapshot().unwrap().records().unwrap().len()
    };

    // Branch at the fourth decision: only the alternative's own records are new.
    let mut branch = c.fork(&decisions[3], &PolicyRef::new("p", "1")).unwrap();
    let alt = branch
        .decision()
        .commit(splinter_expdb::model::Action::new(
            "other",
            serde_json::Value::Null,
        ))
        .unwrap();
    branch.finish(Outcome::Pass).unwrap();
    c.flush().unwrap();

    let snapshot = db.snapshot().unwrap();
    // attempt + counterfactual set + decision + end; nothing before the fork is repeated.
    assert_eq!(snapshot.records().unwrap().len() - before, 4);
    let index = snapshot.index().unwrap();
    assert_eq!(index.parent(alt.id), index.parent(decisions[3].id));
    assert!(index
        .children(index.parent(alt.id).unwrap())
        .contains(&decisions[3].id));
}

#[test]
fn parents_and_children_follow_the_path_through_the_graph() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = db.collector(&identity()).unwrap();
    let (decisions, _) = attempt(&mut c, 2, "p", 3, Outcome::Pass);
    c.flush().unwrap();
    let index = db.snapshot().unwrap().index().unwrap();

    let transition_of_first = index.children(decisions[0].id)[0];
    assert_eq!(index.parent(transition_of_first), Some(decisions[0].id));
    assert_eq!(index.children(transition_of_first), [decisions[1].id]);
}

#[test]
fn a_distant_ancestor_is_found_without_walking_every_step() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = db.collector(&identity()).unwrap();
    let (decisions, _) = attempt(&mut c, 3, "p", 150, Outcome::Pass);
    c.flush().unwrap();
    let index = db.snapshot().unwrap().index().unwrap();

    let last = decisions[149].id;
    // Walk by hand, one parent at a time, to check the shortcut against.
    let mut walked = vec![last];
    while let Some(parent) = index.parent(*walked.last().unwrap()) {
        walked.push(parent);
    }
    for k in [0, 1, 2, 7, 64, 100, walked.len() - 1] {
        assert_eq!(index.nth_ancestor(last, k), Some(walked[k]), "{k} steps up");
    }
    assert_eq!(index.nth_ancestor(last, walked.len()), None);
    assert_eq!(index.depth(last), Some(walked.len() - 1));
}

#[test]
fn the_common_prefix_of_two_branches_is_where_they_diverge() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = db.collector(&identity()).unwrap();
    let (decisions, _) = attempt(&mut c, 4, "p", 6, Outcome::Fail);
    let mut branch = c.fork(&decisions[2], &PolicyRef::new("p", "1")).unwrap();
    let alt = branch
        .decision()
        .commit(splinter_expdb::model::Action::new(
            "other",
            serde_json::Value::Null,
        ))
        .unwrap();
    branch.finish(Outcome::Pass).unwrap();
    c.flush().unwrap();
    let index = db.snapshot().unwrap().index().unwrap();

    let shared = index.common_prefix(decisions[5].id, alt.id).unwrap();
    assert_eq!(Some(shared), index.parent(decisions[2].id));
    assert_eq!(
        index.common_prefix(decisions[5].id, decisions[5].id),
        Some(decisions[5].id)
    );
}

#[test]
fn typed_links_answer_lineage_forwards_and_backwards() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = db.collector(&identity()).unwrap();
    let (decisions, _) = attempt(&mut c, 5, "p", 1, Outcome::Pass);
    let experience = decisions[0].id;
    let skill = c.record(Body::Skill(skill())).unwrap();
    let sample = c.record(Body::Derivation(derivation())).unwrap();
    let run = c.record(Body::Derivation(derivation())).unwrap();
    let model = c.record(Body::Derivation(derivation())).unwrap();
    c.link(skill, Rel::ExtractedFrom, experience).unwrap();
    c.link(sample, Rel::DerivedFrom, skill).unwrap();
    c.link(sample, Rel::UsedBy, run).unwrap();
    c.link(run, Rel::ProducedBy, model).unwrap();
    c.flush().unwrap();
    let index = db.snapshot().unwrap().index().unwrap();

    // Which experiences does this model rest on? Follow the links away from it...
    let upstream: Vec<RecordId> = index.reachable_from(run, None);
    assert!(upstream.contains(&model));
    // ...and what was made from an experience? Follow them towards it.
    let downstream: Vec<RecordId> = index.reaching(experience, None);
    for derived in [skill, sample] {
        assert!(
            downstream.contains(&derived),
            "{derived} derives from the experience"
        );
    }
    assert_eq!(index.edges_from(sample, Some(Rel::UsedBy)), [run]);
    assert_eq!(index.edges_to(skill, Some(Rel::DerivedFrom)), [sample]);
}

#[test]
fn alternatives_at_one_decision_are_grouped_whichever_fork_made_them() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = db.collector(&identity()).unwrap();
    let (decisions, _) = attempt(&mut c, 6, "p", 3, Outcome::Fail);
    let origin = decisions[1];
    let mut alternatives = vec![origin.id];
    for n in 0..3 {
        let mut branch = c.fork(&origin, &PolicyRef::new("p", "1")).unwrap();
        let d = branch
            .decision()
            .commit(splinter_expdb::model::Action::new(
                "try",
                serde_json::json!(n),
            ))
            .unwrap();
        branch.finish(Outcome::Pass).unwrap();
        alternatives.push(d.id);
    }
    c.flush().unwrap();

    let sets = db.snapshot().unwrap().counterfactual_sets().unwrap();
    assert_eq!(sets.len(), 1, "forks of one decision are one set");
    assert_eq!(sets[0].origin_decision, origin.id);
    let mut found = sets[0].alternatives.clone();
    found.sort();
    alternatives.sort();
    assert_eq!(found, alternatives);
}

fn skill() -> splinter_expdb::model::Skill {
    splinter_expdb::model::Skill {
        name: "inspect first".into(),
        description: String::new(),
        trigger: String::new(),
        action_pattern: String::new(),
        expected_effect: String::new(),
        parents: vec![],
        prerequisites: vec![],
    }
}

fn derivation() -> splinter_expdb::model::Derivation {
    splinter_expdb::model::Derivation {
        algorithm: "x".into(),
        version: "1".into(),
        inputs: vec![],
        code_ref: None,
        model: None,
        params: serde_json::Value::Null,
        seed: None,
    }
}

#[test]
fn a_record_is_read_back_by_id_from_whichever_segment_holds_it() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = db.collector(&identity()).unwrap();
    let (decisions, _) = attempt(&mut c, 7, "p", 2, Outcome::Pass);
    c.flush().unwrap();
    let (_, instance, _) = coding_task(7);
    let snapshot = db.snapshot().unwrap();
    let record = snapshot.get(decisions[1].id).unwrap().unwrap();
    assert_eq!(record.task_instance, Some(instance.id().unwrap()));
    assert!(snapshot
        .get(RecordId::new(splinter_expdb::WriterId::from_raw(1), 1))
        .unwrap()
        .is_none());
}
