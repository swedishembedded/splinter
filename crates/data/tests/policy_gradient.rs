// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reinforcement learning from the verified
// rewards of agent rollouts for its clients. If your team needs expertise in
// training agents against verifiers that do not trust them, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec: the policy-gradient view turns several attempts (trials) of one task
//! by one policy into training records that carry a group-relative advantage.
//! Every validly graded trial is admitted whatever its verdict; each of its
//! model actions becomes a record with the trial's advantage, so a failed
//! attempt pushes its actions down and a better one pulls its actions up.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use atif::{StepOrigin, TraceStep};
use common::*;
use serde_json::{json, Value};
use splinter_core::annotation::{Annotation, AnnotationBody, Outcome, Producer, Strength};
use splinter_core::experience::{Environment, Task};
use splinter_data::{
    write_dataset, AdvantageKind, Corpus, Exclusion, Objective, PolicyGradient, Projection,
    RecordBody, RewardShape, Unchecked, View, ViewError,
};

const PERFECT: &[(&str, f64)] = &[("iou", 1.0), ("feature_match", 1.0)];

fn bench_task(version: &str) -> Task {
    Task::new(
        "bench/cad",
        Vec::new(),
        Environment::new(
            Environment::BENCH,
            json!({"task": {"id": "cad/bracket", "version": version}}),
        ),
        "Design a bracket.",
        Vec::new(),
    )
    .unwrap()
}

/// One trial of a task: what the verifier said, the metrics it measured, the
/// model that ran it and the role of the conversation.
struct Trial {
    name: &'static str,
    version: &'static str,
    model: &'static str,
    role: &'static str,
    verdict: Outcome,
    metrics: &'static [(&'static str, f64)],
}

impl Trial {
    fn pass(name: &'static str) -> Self {
        Self {
            name,
            version: "1",
            model: "m1",
            role: "designer",
            verdict: Outcome::Pass,
            metrics: PERFECT,
        }
    }

    fn fail(name: &'static str, metrics: &'static [(&'static str, f64)]) -> Self {
        Self {
            verdict: Outcome::Fail,
            metrics,
            ..Self::pass(name)
        }
    }

    fn version(self, version: &'static str) -> Self {
        Self { version, ..self }
    }

    fn model(self, model: &'static str) -> Self {
        Self { model, ..self }
    }

    fn role(self, role: &'static str) -> Self {
        Self { role, ..self }
    }
}

/// The verdict annotation the bench ingestion writes for a trial.
fn graded(
    experience: &splinter_core::experience::Experience,
    trial: &Trial,
    evidence: Option<Value>,
) -> Annotation {
    let metrics: serde_json::Map<String, Value> = trial
        .metrics
        .iter()
        .map(|(k, v)| ((*k).to_string(), json!(v)))
        .collect();
    Annotation {
        experience: experience.id().unwrap(),
        producer: Producer {
            name: "bench".into(),
            version: "v".into(),
        },
        body: AnnotationBody::Verdict {
            outcome: trial.verdict,
            strength: Strength::Executable,
            evidence: evidence.unwrap_or_else(|| {
                json!({
                    "trial": trial.name,
                    "task": {"id": "cad/bracket", "version": trial.version},
                    "model": trial.model,
                    "metrics": Value::Object(metrics),
                })
            }),
        },
    }
}

/// A conversation of the trial: one action that calls a tool, one final
/// answer.
fn add(corpus: &mut Corpus, trial: &Trial, evidence: Option<Value>) {
    let mut call_step = call(3, "c0", "shell", json!({"command": "ls"}), "part.py");
    call_step.reasoning_content = Some("look".into());
    let mut trajectory = trajectory(vec![
        TraceStep::new(1, StepOrigin::System, "You are a careful engineer."),
        user(2, "Design a bracket."),
        call_step,
        say(4, "Done."),
    ]);
    trajectory.agent.tool_definitions = Some(vec![json!({"type": "function", "function": {
        "name": "shell", "description": "Run a command.", "parameters": {"type": "object"}}})]);
    trajectory.extra = Some(json!({"splinter_harness": {
        "conversation": trial.role, "role": trial.role, "role_source": "export",
        "shown_images": false}}));
    let at = format!("2026-11-01T00:00:00.{:03}Z", corpus.entries().len());
    let experience = experience(&bench_task(trial.version), trajectory, Some("Done."), &at);
    let note = graded(&experience, trial, evidence);
    corpus.insert(experience, vec![note]).unwrap();
}

fn corpus(trials: &[Trial]) -> Corpus {
    let mut corpus = Corpus::new();
    for trial in trials {
        add(&mut corpus, trial, None);
    }
    corpus
}

fn view() -> PolicyGradient {
    PolicyGradient::new(Strength::Executable)
}

fn advantages(projection: &Projection) -> Vec<f64> {
    projection
        .records
        .iter()
        .map(|r| match &r.body {
            RecordBody::Chat { advantage, .. } => advantage.unwrap(),
            other => panic!("a chat record, got {other:?}"),
        })
        .collect()
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-6
}

#[test]
fn every_action_of_every_graded_trial_carries_its_trials_advantage() {
    // Two passes and two fails: rewards 1, 1, 0, 0; mean 0.5, std 0.5.
    let corpus = corpus(&[
        Trial::pass("t1"),
        Trial::pass("t2"),
        Trial::fail("t3", &[]),
        Trial::fail("t4", &[]),
    ]);
    let projection = view().project(&corpus).unwrap();
    assert_eq!(projection.objective, Objective::PolicyGradient);
    // Two actions per trial (the tool call and the final answer), failed
    // trials included.
    let got = advantages(&projection);
    assert_eq!(got.len(), 8);
    let expected = [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0];
    for (g, e) in got.iter().zip(expected) {
        assert!(close(*g, e * 0.5 / (0.5 + 1e-6)), "{got:?}");
    }
    // Only the action is supervised in each record.
    for record in &projection.records {
        let RecordBody::Chat { messages, .. } = &record.body else {
            panic!("chat");
        };
        assert_eq!(messages.iter().filter(|m| m.train).count(), 1);
    }
}

#[test]
fn the_conversations_of_one_trial_share_its_advantage() {
    // One trial answered by two roles of one model: its two conversations are
    // one rollout, so the trial counts once in its group.
    let corpus = corpus(&[
        Trial::pass("t1").role("designer"),
        Trial::pass("t1").role("critic"),
        Trial::fail("t2", &[]),
    ]);
    let projection = view().project(&corpus).unwrap();
    let group = &projection.groups[0];
    assert_eq!(group.trials.len(), 2);
    let got = advantages(&projection);
    // 4 actions of t1 (+), 2 of t2 (-).
    assert_eq!(got.len(), 6);
    assert!(got[..4].iter().all(|a| *a > 0.0));
    assert!(got[4..].iter().all(|a| *a < 0.0));
    assert!(close(got[0], got[3]));
}

#[test]
fn a_failed_trial_is_rewarded_by_the_shaped_metrics_below_the_ceiling() {
    let shape = RewardShape::shaped(
        0.5,
        vec![
            ("iou".into(), 0.5),
            ("feature_match".into(), 0.25),
            ("deviation_score".into(), 0.25),
        ],
    )
    .unwrap();
    let near: &[(&str, f64)] = &[
        ("iou", 1.0),
        ("feature_match", 1.0),
        ("deviation_score", 1.0),
    ];
    let half: &[(&str, f64)] = &[
        ("iou", 0.5),
        ("feature_match", 0.0),
        ("deviation_score", 1.0),
    ];
    let corpus = corpus(&[
        Trial::pass("t1"),
        Trial::fail("t2", near),
        Trial::fail("t3", half),
    ]);
    let projection = view().rewards(shape).project(&corpus).unwrap();
    let rewards: Vec<f64> = projection.groups[0]
        .trials
        .iter()
        .map(|t| t.reward)
        .collect();
    // A pass is 1; a failure is at most the ceiling, so a pass always
    // outranks any failure.
    assert!(close(rewards[0], 1.0));
    assert!(close(rewards[1], 0.5));
    assert!(close(rewards[2], 0.5 * (0.25 + 0.0 + 0.25)));
    assert!(rewards[0] > rewards[1] && rewards[1] > rewards[2]);
}

#[test]
fn without_shaping_a_failure_earns_nothing() {
    let corpus = corpus(&[Trial::pass("t1"), Trial::fail("t2", PERFECT)]);
    let projection = view().project(&corpus).unwrap();
    let rewards: Vec<f64> = projection.groups[0]
        .trials
        .iter()
        .map(|t| t.reward)
        .collect();
    assert_eq!(rewards, [1.0, 0.0]);
}

#[test]
fn a_configured_metric_a_failed_trial_lacks_is_an_error_naming_it() {
    let shape =
        RewardShape::shaped(0.5, vec![("iou".into(), 0.5), ("chamfer".into(), 0.5)]).unwrap();
    let corpus = corpus(&[Trial::pass("t1"), Trial::fail("t2", &[("iou", 0.3)])]);
    let error = view().rewards(shape).project(&corpus).unwrap_err();
    assert!(matches!(error, ViewError::Parameter { .. }));
    let text = error.to_string();
    assert!(text.contains("chamfer") && text.contains("t2"), "{text}");
}

#[test]
fn a_metric_outside_zero_to_one_is_an_error() {
    let shape = RewardShape::shaped(0.5, vec![("iou".into(), 1.0)]).unwrap();
    let corpus = corpus(&[Trial::pass("t1"), Trial::fail("t2", &[("iou", 1.5)])]);
    assert!(view().rewards(shape).project(&corpus).is_err());
}

#[test]
fn reward_shapes_that_could_not_rank_a_pass_first_are_refused() {
    let w = |pairs: &[(&str, f64)]| pairs.iter().map(|(k, v)| ((*k).to_string(), *v)).collect();
    for ceiling in [-0.1, 1.0, 1.5, f64::NAN] {
        assert!(
            RewardShape::shaped(ceiling, w(&[("a", 1.0)])).is_err(),
            "{ceiling}"
        );
    }
    for weights in [
        w(&[("a", 0.5)]),
        w(&[("a", 1.5), ("b", -0.5)]),
        w(&[("a", f64::INFINITY)]),
        w(&[("a", 0.5), ("a", 0.5)]),
        w(&[]),
    ] {
        assert!(RewardShape::shaped(0.5, weights).is_err());
    }
    assert!(RewardShape::shaped(0.5, w(&[("a", 0.5), ("b", 0.5)])).is_ok());
}

#[test]
fn mean_centred_advantages_are_not_divided_by_the_spread() {
    let corpus = corpus(&[
        Trial::pass("t1"),
        Trial::fail("t2", &[]),
        Trial::fail("t3", &[]),
    ]);
    let projection = view()
        .advantage(AdvantageKind::MeanCentred)
        .project(&corpus)
        .unwrap();
    let trials = &projection.groups[0].trials;
    assert!(close(trials[0].advantage, 2.0 / 3.0));
    assert!(close(trials[1].advantage, -1.0 / 3.0));
    // The default divides by the spread (population standard deviation).
    let standardised = view().project(&corpus).unwrap();
    let std = (2.0f64 / 9.0).sqrt();
    assert!(close(
        standardised.groups[0].trials[0].advantage,
        (2.0 / 3.0) / (std + 1e-6)
    ));
}

#[test]
fn a_group_without_a_spread_of_rewards_teaches_nothing() {
    // Three passes: nobody did better than anybody. One lone trial: nothing
    // to compare it with.
    let corpus = corpus(&[
        Trial::pass("a1"),
        Trial::pass("a2"),
        Trial::pass("a3"),
        Trial::pass("b1").version("2"),
    ]);
    let error = write_dataset_of(&view().project(&corpus).unwrap());
    assert!(matches!(error, Err(ViewError::Empty { .. })));
    let projection = view().project(&corpus).unwrap();
    assert!(projection.records.is_empty());
    // Counted per experience, and the groups are still reported.
    assert_eq!(projection.count(Exclusion::NoGroupSignal), 4);
    assert_eq!(projection.groups.len(), 2);
    assert!(projection.groups.iter().all(|g| !g.signal));
}

fn write_dataset_of(projection: &Projection) -> Result<(), ViewError> {
    let dir = std::env::temp_dir().join(format!("pg-spec-{}", std::process::id()));
    let result = write_dataset(&dir.join("d.jsonl"), projection, &Unchecked).map(|_| ());
    let _ = std::fs::remove_dir_all(&dir);
    result
}

#[test]
fn trials_of_another_task_version_or_another_model_are_other_groups() {
    let corpus = corpus(&[
        Trial::pass("a1"),
        Trial::fail("a2", &[]),
        Trial::pass("b1").version("2"),
        Trial::fail("b2", &[]).version("2"),
        Trial::pass("c1").model("m2"),
        Trial::fail("c2", &[]).model("m2"),
    ]);
    let projection = view().project(&corpus).unwrap();
    assert_eq!(projection.groups.len(), 3);
    assert!(projection.groups.iter().all(|g| g.trials.len() == 2));
    // Each group is standardised on its own.
    let got = advantages(&projection);
    assert_eq!(got.len(), 12);
    assert!(got.iter().all(|a| close(a.abs(), 0.5 / (0.5 + 1e-6))));
}

#[test]
fn the_report_names_the_task_the_policy_the_rewards_and_the_advantages() {
    let corpus = corpus(&[Trial::pass("t1"), Trial::fail("t2", &[])]);
    let group = &view().project(&corpus).unwrap().groups[0];
    assert_eq!(group.task, "cad/bracket");
    assert_eq!(group.version, "1");
    assert_eq!(group.policy, "m1");
    let names: Vec<_> = group.trials.iter().map(|t| t.trial.as_str()).collect();
    assert_eq!(names, ["t1", "t2"]);
    assert!(close(group.trials[0].advantage, -group.trials[1].advantage));
}

#[test]
fn only_the_wanted_roles_are_trained_and_a_trial_without_one_does_not_count() {
    let corpus = corpus(&[
        Trial::pass("t1").role("designer"),
        Trial::fail("t2", &[]).role("designer"),
        Trial::pass("t3").role("critic"),
    ]);
    let projection = view()
        .only_roles(vec!["designer".into()])
        .project(&corpus)
        .unwrap();
    assert_eq!(projection.count(Exclusion::RoleNotWanted), 1);
    assert_eq!(projection.groups[0].trials.len(), 2);
    assert_eq!(projection.records.len(), 4);
}

#[test]
fn an_experience_with_no_decision_or_too_weak_a_one_is_left_out() {
    let mut corpus = corpus(&[Trial::pass("t1"), Trial::fail("t2", &[])]);
    // Abstained: no pass/fail decision.
    let abstained = Trial {
        verdict: Outcome::Abstain,
        ..Trial::pass("t3")
    };
    add(&mut corpus, &abstained, None);
    let projection = PolicyGradient::new(Strength::Executable)
        .project(&corpus)
        .unwrap();
    assert_eq!(projection.count(Exclusion::Undecided), 1);
    assert_eq!(projection.groups[0].trials.len(), 2);
    // Everything here is decided at executable strength; ask for more.
    let strict = PolicyGradient::new(Strength::Constraint)
        .project(&corpus)
        .unwrap();
    assert_eq!(strict.count(Exclusion::TooWeak), 2);
}

#[test]
fn an_experience_that_names_no_trial_cannot_be_grouped() {
    let mut corpus = Corpus::new();
    add(&mut corpus, &Trial::pass("t1"), Some(json!({})));
    let projection = view().project(&corpus).unwrap();
    assert_eq!(projection.count(Exclusion::NoRollout), 1);
}

#[test]
fn the_same_corpus_projects_to_the_same_bytes() {
    let corpus = corpus(&[
        Trial::pass("t1"),
        Trial::fail("t2", &[]),
        Trial::fail("t3", &[]).version("2"),
        Trial::pass("t4").version("2"),
    ]);
    let once = serde_json::to_string(&view().project(&corpus).unwrap()).unwrap();
    let again = serde_json::to_string(&view().project(&corpus).unwrap()).unwrap();
    assert_eq!(once, again);
}

#[test]
fn the_dataset_line_carries_the_advantage_for_brain() {
    let corpus = corpus(&[Trial::pass("t1"), Trial::fail("t2", &[])]);
    let projection = view().project(&corpus).unwrap();
    let dir = std::env::temp_dir().join(format!("pg-line-{}", std::process::id()));
    let path = dir.join("d.jsonl");
    let written = write_dataset(&path, &projection, &Unchecked).unwrap();
    assert_eq!(written.format, splinter_data::Format::GenericMessagesV2);
    let text = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(lines.len(), 4);
    assert!(lines.iter().all(|l| l["advantage"].is_f64()));
    assert!(lines[0]["advantage"].as_f64().unwrap() > 0.0);
    assert!(lines[2]["advantage"].as_f64().unwrap() < 0.0);
    // The group report is part of the manifest, so a dataset says how its
    // advantages were made.
    let manifest = std::fs::read_to_string(splinter_data::manifest_path(&path)).unwrap();
    assert!(manifest.contains("\"groups\""), "{manifest}");
    let _ = std::fs::remove_dir_all(&dir);
}
