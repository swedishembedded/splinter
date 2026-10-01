// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: everything derived says how it came to be, and the links run the
//! whole way from a task to a model and back.
#![allow(clippy::unwrap_used)]

mod common;

use common::{attempt, skill, Scratch};
use splinter_expdb::model::{
    Body, CreditAssignment, DatasetNode, Derivation, ModelNode, Outcome, RecordKind, TrainingRun,
};
use splinter_expdb::{ContentId, WriterIdentity};

fn extractor() -> Derivation {
    Derivation {
        algorithm: "skill-extractor".into(),
        version: "3".into(),
        inputs: vec![],
        code_ref: Some("git:f172bd8".into()),
        model: Some("qwen3-32b".into()),
        params: serde_json::json!({ "temperature": 0.2 }),
        seed: Some(7),
    }
}

#[test]
fn a_derived_record_names_its_derivation_and_its_inputs() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = db
        .collector(&WriterIdentity::new("exp", "job", "node", 0))
        .unwrap();
    let (decisions, _) = attempt(&mut c, 1, "p", 2, Outcome::Pass);
    let derivation = Derivation {
        inputs: vec![decisions[0].id, decisions[1].id],
        ..extractor()
    };
    let (derivation_id, outputs) = c
        .derived(
            derivation.clone(),
            vec![Body::Skill(skill("inspect first"))],
        )
        .unwrap();
    c.flush().unwrap();
    let snapshot = db.snapshot().unwrap();

    assert_eq!(
        snapshot.derivation_of(outputs[0]).unwrap(),
        Some((derivation_id, derivation))
    );
    let sources = snapshot.lineage_back(outputs[0]).unwrap();
    assert!(sources.contains(&decisions[0].id) && sources.contains(&decisions[1].id));
    assert_eq!(
        snapshot.derivation_of(decisions[0].id).unwrap(),
        None,
        "raw experience is not derived"
    );
}

#[test]
fn a_model_traces_back_to_the_experiences_it_rests_on_and_an_experience_forward_to_the_models() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = db
        .collector(&WriterIdentity::new("exp", "job", "node", 0))
        .unwrap();
    let (decisions, _) = attempt(&mut c, 1, "p", 2, Outcome::Pass);
    let (other, _) = attempt(&mut c, 2, "p", 2, Outcome::Pass);

    let (_, skills) = c
        .derived(
            Derivation {
                inputs: vec![decisions[0].id],
                ..extractor()
            },
            vec![Body::Skill(skill("inspect first"))],
        )
        .unwrap();
    let dataset = c
        .record_dataset(
            DatasetNode {
                recipe: serde_json::json!({ "objective": "sft" }),
                snapshot: ContentId::of(b"m1"),
                samples: 10,
            },
            &skills,
        )
        .unwrap();
    let run = c
        .record_training_run(TrainingRun {
            dataset,
            objective: "sft".into(),
            base_model: None,
        })
        .unwrap();
    let model = c
        .record_model(ModelNode {
            name: "model-47".into(),
            run: Some(run),
            parent: None,
        })
        .unwrap();
    c.flush().unwrap();
    let snapshot = db.snapshot().unwrap();

    let behind = snapshot.lineage_back(model).unwrap();
    for stage in [run, dataset, skills[0], decisions[0].id] {
        assert!(behind.contains(&stage), "{stage} is behind the model");
    }
    let index = snapshot.index().unwrap();
    let experiences: Vec<_> = behind
        .iter()
        .filter(|id| index.kind_of(**id) == Some(RecordKind::Decision))
        .collect();
    assert_eq!(
        experiences,
        [&decisions[0].id],
        "only the experience the skill was extracted from"
    );
    assert!(!behind.contains(&other[0].id));

    let ahead = snapshot.lineage_forward(decisions[0].id).unwrap();
    assert!(ahead.contains(&model));
    assert!(!snapshot
        .lineage_forward(other[0].id)
        .unwrap()
        .contains(&model));
}

#[test]
fn everything_a_given_algorithm_version_made_can_be_found_to_remove_it_from_a_dataset() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = db
        .collector(&WriterIdentity::new("exp", "job", "node", 0))
        .unwrap();
    let (decisions, _) = attempt(&mut c, 1, "p", 1, Outcome::Pass);
    let v3 = Derivation {
        inputs: vec![decisions[0].id],
        ..extractor()
    };
    let v4 = Derivation {
        version: "4".into(),
        ..v3.clone()
    };
    let (_, from_v3) = c
        .derived(v3, vec![Body::Skill(skill("a")), Body::Skill(skill("b"))])
        .unwrap();
    let (_, from_v4) = c.derived(v4, vec![Body::Skill(skill("c"))]).unwrap();
    c.flush().unwrap();
    let snapshot = db.snapshot().unwrap();

    assert_eq!(
        snapshot.derived_by("skill-extractor", "3").unwrap(),
        from_v3
    );
    assert_eq!(
        snapshot.derived_by("skill-extractor", "4").unwrap(),
        from_v4
    );
    assert!(snapshot
        .derived_by("skill-extractor", "9")
        .unwrap()
        .is_empty());
}

#[test]
fn two_credit_algorithms_can_disagree_about_one_decision_and_both_opinions_stand() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = db
        .collector(&WriterIdentity::new("exp", "job", "node", 0))
        .unwrap();
    let (decisions, _) = attempt(&mut c, 1, "p", 3, Outcome::Pass);
    let target = decisions[1].id;
    let credit = |algorithm: &str, value: f64| CreditAssignment {
        target,
        algorithm: algorithm.into(),
        value,
        confidence: 0.8,
        supporting_branches: vec![],
        derivation: None,
    };
    c.flush().unwrap();
    let original = db.snapshot().unwrap().get(target).unwrap().unwrap();

    c.assign_credit(credit("counterfactual-credit-v4", 0.91))
        .unwrap();
    c.assign_credit(credit("uniform-v1", 0.33)).unwrap();
    c.flush().unwrap();
    let snapshot = db.snapshot().unwrap();

    let mut found: Vec<_> = snapshot
        .credits_for(target)
        .unwrap()
        .into_iter()
        .map(|c| (c.algorithm, c.value))
        .collect();
    found.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(
        found,
        [
            ("counterfactual-credit-v4".to_owned(), 0.91),
            ("uniform-v1".to_owned(), 0.33)
        ]
    );
    assert_eq!(
        snapshot.get(target).unwrap().unwrap(),
        original,
        "the decision itself is never mutated"
    );
}
