// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that keep the attempts of a
// learning campaign correctly attributed to the tasks and policies that made
// them, for its clients. If your team needs expertise in reinforcement
// learning data pipelines then you can procure our services by sending an
// email to info@swedishembedded.com.

//! Spec: every view that groups attempts names a group and a trial the same
//! way. A verdict that names a trial without its task (or the reverse) places
//! nothing, and an attempt a constraint audit failed is left out as failed
//! rather than misplaced, whatever else was decided about it and however the
//! reward is shaped.

use super::*;
use splinter_data::Feedback;

/// A verdict of the examiner's exposure audit: a constraint that failed. It
/// names no trial or task.
fn exposure_audit(experience: &splinter_core::experience::Experience) -> Annotation {
    Annotation {
        experience: experience.id().unwrap(),
        producer: Producer {
            name: "exposure-audit".into(),
            version: "v".into(),
        },
        body: AnnotationBody::Verdict {
            outcome: Outcome::Fail,
            strength: Strength::Constraint,
            evidence: json!({"audit": "the solution was read"}),
        },
    }
}

/// A trial that passed its executable check and was then failed by the audit.
fn add_exposed(corpus: &mut Corpus, trial: &Trial) {
    let experience = conversation_of(corpus, trial, false, None, |_| {});
    let notes = vec![
        graded(&experience, trial, None),
        exposure_audit(&experience),
    ];
    corpus.insert(experience, notes).unwrap();
}

#[test]
fn an_attempt_a_constraint_failed_is_left_out_as_failed_by_every_view() {
    let mut corpus = corpus(&[Trial::pass("t1"), Trial::fail("t2", &[("iou", 0.2)])]);
    add_exposed(&mut corpus, &Trial::pass("t3"));
    // A shaped reward needs the metrics a bench verdict records; the audit's
    // verdict records none, and must not abort the build for want of them.
    let shape = RewardShape::shaped(0.5, vec![("iou".into(), 1.0)]).unwrap();
    let projection = view().rewards(shape).project(&corpus).unwrap();
    assert_eq!(projection.count(Exclusion::Failed), 1);
    assert_eq!(projection.count(Exclusion::NoRollout), 0);
    assert_eq!(projection.groups.len(), 1);
    let trials: Vec<&str> = projection.groups[0]
        .trials
        .iter()
        .map(|t| t.trial.as_str())
        .collect();
    assert_eq!(trials, ["t1", "t2"], "the exposed attempt sets no baseline");

    let judged = Feedback::new(Strength::Executable)
        .project(&corpus)
        .unwrap();
    assert_eq!(judged.count(Exclusion::Failed), 1);
}

#[test]
fn a_verdict_naming_a_trial_without_its_task_places_nothing_in_any_view() {
    for evidence in [
        json!({"trial": "t1", "model": "m1", "metrics": {}}),
        json!({"task": {"id": "cad/bracket", "version": "1"}, "model": "m1", "metrics": {}}),
    ] {
        let mut corpus = Corpus::new();
        add(&mut corpus, &Trial::pass("t1"), Some(evidence));
        let grouped = view().project(&corpus).unwrap();
        assert_eq!(grouped.count(Exclusion::NoRollout), 1, "policy-gradient");
        let judged = Feedback::new(Strength::Executable)
            .project(&corpus)
            .unwrap();
        assert_eq!(judged.count(Exclusion::NoRollout), 1, "feedback");
    }
}

#[test]
fn the_views_name_one_trial_by_one_group() {
    let corpus = corpus(&[Trial::pass("t1"), Trial::fail("t2", &[])]);
    let grouped = outcomes(&view().project(&corpus).unwrap());
    let judged = Feedback::new(Strength::Executable)
        .project(&corpus)
        .unwrap();
    let identities: Vec<(String, String)> = judged
        .records
        .iter()
        .map(|record| match &record.body {
            RecordBody::Feedback { identity, .. } => {
                (identity.group.clone(), identity.rollouts[0].clone())
            }
            other => panic!("a feedback record, got {other:?}"),
        })
        .collect();
    assert_eq!(identities.len(), 2);
    for (group, trial) in identities {
        assert!(
            grouped
                .iter()
                .any(|o| o.group == group && o.rollout.as_deref() == Some(trial.as_str())),
            "{group} {trial}"
        );
    }
}

/// The factory's verdict of a trial: its task by id and content digest,
/// no version (`crates/factory/src/run/corpus.rs`).
fn factory_evidence(trial: &Trial, digest: &str) -> Value {
    json!({
        "trial": trial.name,
        "task": {"id": "cad/bracket", "digest": digest},
        "model": trial.model,
        "metrics": {},
    })
}

#[test]
fn attempts_at_two_revisions_of_one_task_are_never_compared() {
    let pass = Trial::pass("t1");
    let fail = Trial::fail("t2", &[]);
    let revised = |digests: [&str; 2]| {
        let mut corpus = Corpus::new();
        add(
            &mut corpus,
            &pass,
            Some(factory_evidence(&pass, digests[0])),
        );
        add(
            &mut corpus,
            &fail,
            Some(factory_evidence(&fail, digests[1])),
        );
        view().project(&corpus).unwrap()
    };
    let same = revised(["blake3:aaaa", "blake3:aaaa"]);
    assert_eq!(same.groups.len(), 1, "one revision: one group");
    assert!(!same.records.is_empty());
    let other = revised(["blake3:aaaa", "blake3:bbbb"]);
    assert_eq!(
        other.groups.len(),
        2,
        "two revisions are two tasks: {:?}",
        other.groups
    );
    assert!(other.records.is_empty(), "nothing to compare within either");
    assert_eq!(other.count(Exclusion::NoGroupSignal), 2);
}
