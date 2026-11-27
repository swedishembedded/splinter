// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reinforcement learning from the verified
// rewards of agent rollouts for its clients. If your team needs expertise in
// training agents against verifiers that do not trust them, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec: the policy-gradient view turns several attempts (trials) of one task
//! by one policy into training records that carry the trial's reward, the
//! group's key and the trial's name (brain's training-record contract v2), so
//! the trainer computes each trial's group-relative advantage. Every validly
//! graded trial is admitted whatever its verdict; each of its model actions
//! becomes a record of the trial, so a failed attempt pushes its actions down
//! and a better one pulls its actions up.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use crate::common;

use atif::{StepOrigin, TraceStep};
use common::*;
use serde_json::{json, Value};
use splinter_core::annotation::{Annotation, AnnotationBody, Outcome, Producer, Strength};
use splinter_core::experience::{Environment, Task};
use splinter_data::{
    write_dataset, AdvantageFloor, AdvantageKind, AdvantageNorm, Corpus, Exclusion, Objective,
    Overlong, PolicyGradient, Projection, RecordBody, RewardShape, RolloutOutcome, Unchecked, View,
    ViewError,
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
    name: String,
    version: String,
    model: String,
    role: String,
    verdict: Outcome,
    metrics: &'static [(&'static str, f64)],
}

impl Trial {
    fn pass(name: &str) -> Self {
        Self {
            name: name.into(),
            version: "1".into(),
            model: "m1".into(),
            role: "designer".into(),
            verdict: Outcome::Pass,
            metrics: PERFECT,
        }
    }

    fn fail(name: &str, metrics: &'static [(&'static str, f64)]) -> Self {
        Self {
            verdict: Outcome::Fail,
            metrics,
            ..Self::pass(name)
        }
    }

    fn version(self, version: &str) -> Self {
        Self {
            version: version.into(),
            ..self
        }
    }

    fn model(self, model: &str) -> Self {
        Self {
            model: model.into(),
            ..self
        }
    }

    fn role(self, role: &str) -> Self {
        Self {
            role: role.into(),
            ..self
        }
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
    add_shown(corpus, trial, evidence, false);
}

/// [`add`], the conversation shown images when `shown_images`.
fn add_shown(corpus: &mut Corpus, trial: &Trial, evidence: Option<Value>, shown_images: bool) {
    add_with(corpus, trial, evidence, shown_images, None, |_| {});
}

/// [`add_shown`], with `tweak` free to change the trajectory's steps (the
/// system step, the user step, the tool-call step and the answer, in order)
/// and `calls` the requests the harness recorded the model was sent.
fn add_with(
    corpus: &mut Corpus,
    trial: &Trial,
    evidence: Option<Value>,
    shown_images: bool,
    calls: Option<&[splinter_core::harness::ModelCall]>,
    tweak: impl FnOnce(&mut [TraceStep]),
) {
    let experience = conversation_of(corpus, trial, shown_images, calls, tweak);
    let note = graded(&experience, trial, evidence);
    corpus.insert(experience, vec![note]).unwrap();
}

/// The experience of one conversation of `trial`, not yet in `corpus`.
fn conversation_of(
    corpus: &Corpus,
    trial: &Trial,
    shown_images: bool,
    calls: Option<&[splinter_core::harness::ModelCall]>,
    tweak: impl FnOnce(&mut [TraceStep]),
) -> splinter_core::experience::Experience {
    let mut call_step = call(3, "c0", "shell", json!({"command": "ls"}), "part.py");
    call_step.reasoning_content = Some("look".into());
    let mut steps = vec![
        TraceStep::new(1, StepOrigin::System, "You are a careful engineer."),
        user(2, "Design a bracket."),
        call_step,
        say(4, "Done."),
    ];
    tweak(&mut steps);
    let mut trajectory = trajectory(steps);
    trajectory.agent.tool_definitions = Some(vec![json!({"type": "function", "function": {
        "name": "shell", "description": "Run a command.", "parameters": {"type": "object"}}})]);
    trajectory.extra = Some(json!({"splinter_harness": {
        "conversation": trial.role, "role": trial.role, "role_source": "export",
        "shown_images": shown_images}}));
    if let (Some(calls), Some(extra)) = (calls, trajectory.extra.as_mut()) {
        extra["splinter_harness_calls"] =
            splinter_core::harness::RecordedCalls::record("abc", calls).to_value();
    }
    let at = format!("2026-11-01T00:00:00.{:03}Z", corpus.entries().len());
    experience(&bench_task(&trial.version), trajectory, Some("Done."), &at)
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

fn outcomes(projection: &Projection) -> Vec<RolloutOutcome> {
    projection
        .records
        .iter()
        .map(|r| match &r.body {
            RecordBody::Chat { outcome, .. } => outcome.clone().unwrap(),
            other => panic!("a chat record, got {other:?}"),
        })
        .collect()
}

fn rewards(projection: &Projection) -> Vec<f64> {
    outcomes(projection).iter().map(|o| o.reward).collect()
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-6
}

#[test]
fn every_action_of_every_graded_trial_carries_its_trials_reward_group_and_name() {
    // Two passes and two fails: rewards 1, 1, 0, 0.
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
    let got = outcomes(&projection);
    assert_eq!(
        rewards(&projection),
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    );
    assert!(
        got.iter().all(|o| o.group == got[0].group),
        "one group: {got:?}"
    );
    let rollouts: Vec<&str> = got.iter().map(|o| o.rollout.as_deref().unwrap()).collect();
    assert_eq!(rollouts, ["t1", "t1", "t2", "t2", "t3", "t3", "t4", "t4"]);
    // The trainer is told to standardise the rewards, as GRPO does.
    assert_eq!(
        projection.advantage.map(AdvantageKind::norm),
        Some(AdvantageNorm::ZScore)
    );
    // Only the action is supervised in each record.
    for record in &projection.records {
        let RecordBody::Chat { messages, .. } = &record.body else {
            panic!("chat");
        };
        assert_eq!(messages.iter().filter(|m| m.train).count(), 1);
    }
}

#[test]
fn packed_each_trial_is_one_record_supervising_every_action_with_its_reward_and_group() {
    let corpus = corpus(&[Trial::pass("t1"), Trial::pass("t2"), Trial::fail("t3", &[])]);
    let projection = view().packed(true).project(&corpus).unwrap();
    assert!(
        projection.packed,
        "the manifest says the records are packed"
    );
    let got = outcomes(&projection);
    let rollouts: Vec<&str> = got.iter().map(|o| o.rollout.as_deref().unwrap()).collect();
    assert_eq!(rollouts, ["t1", "t2", "t3"]);
    assert_eq!(rewards(&projection), [1.0, 1.0, 0.0]);
    assert!(got.iter().all(|o| o.group == got[0].group));
    // Both actions of a trial (the tool call and the final answer) are
    // supervised in its one record, as the per-action records supervise them.
    for record in &projection.records {
        let RecordBody::Chat { messages, .. } = &record.body else {
            panic!("chat");
        };
        assert_eq!(messages.iter().filter(|m| m.train).count(), 2);
    }
}

#[test]
fn the_conversations_of_one_trial_are_one_rollout_of_its_group() {
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
    let got = outcomes(&projection);
    // 4 actions of t1 (rewarded), 2 of t2: two rollouts of one group.
    assert_eq!(got.len(), 6);
    assert!(got[..4]
        .iter()
        .all(|o| o.reward == 1.0 && o.rollout.as_deref() == Some("t1")));
    assert!(got[4..]
        .iter()
        .all(|o| o.reward == 0.0 && o.rollout.as_deref() == Some("t2")));
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
fn the_advantage_kind_names_the_trainers_normalisation_and_writes_every_attempt() {
    let corpus = corpus(&[
        Trial::pass("t1"),
        Trial::fail("t2", &[]),
        Trial::fail("t3", &[]),
    ]);
    for (kind, norm) in [
        (AdvantageKind::Standardised, AdvantageNorm::ZScore),
        (AdvantageKind::MeanCentred, AdvantageNorm::MeanCentred),
        (AdvantageKind::LeaveOneOut, AdvantageNorm::LeaveOneOut),
    ] {
        let projection = view().advantage(kind).project(&corpus).unwrap();
        assert_eq!(projection.advantage.map(AdvantageKind::norm), Some(norm));
        assert_eq!(
            rewards(&projection),
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "{kind:?}"
        );
    }
    assert_eq!(
        ["z-score", "mean-centred", "leave-one-out"],
        [
            AdvantageNorm::ZScore,
            AdvantageNorm::MeanCentred,
            AdvantageNorm::LeaveOneOut
        ]
        .map(AdvantageNorm::name),
        "brain's spellings"
    );
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
    // Each group has its own key.
    let got = outcomes(&projection);
    assert_eq!(got.len(), 12);
    let keys: std::collections::BTreeSet<&str> = got.iter().map(|o| o.group.as_str()).collect();
    assert_eq!(keys.len(), 3);
}

#[test]
fn the_report_names_the_task_the_policy_the_rewards_and_what_was_written() {
    let corpus = corpus(&[Trial::pass("t1"), Trial::fail("t2", &[])]);
    let group = &view().project(&corpus).unwrap().groups[0];
    assert_eq!(group.task, "cad/bracket");
    assert_eq!(group.version, "1");
    assert_eq!(group.policy, "m1");
    let names: Vec<_> = group.trials.iter().map(|t| t.trial.as_str()).collect();
    assert_eq!(names, ["t1", "t2"]);
    let rewards: Vec<f64> = group.trials.iter().map(|t| t.reward).collect();
    assert_eq!(rewards, [1.0, 0.0]);
    assert!(group.trials.iter().all(|t| t.written));
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
fn the_dataset_line_carries_the_reward_group_and_rollout_for_brain() {
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
    assert!(
        lines.iter().all(|l| l.get("advantage").is_none()),
        "contract v2 has no precomputed advantage"
    );
    assert_eq!(lines[0]["reward"], 1.0);
    assert_eq!(lines[2]["reward"], 0.0);
    assert!(lines
        .iter()
        .all(|l| l["group"] == lines[0]["group"] && l["group"].is_string()));
    assert_eq!(
        (lines[0]["rollout"].as_str(), lines[2]["rollout"].as_str()),
        (Some("t1"), Some("t2"))
    );
    // The group report and the normalisation are part of the manifest, so a
    // dataset says how its advantages are to be made.
    let manifest = std::fs::read_to_string(splinter_data::manifest_path(&path)).unwrap();
    assert!(
        manifest.contains("\"groups\"") && manifest.contains("\"advantage\":\"standardised\""),
        "{manifest}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// L5, the producer's side of brain's training-record contract (version 2),
/// held to brain's own fixtures (`tests/fixtures/training-records/v2`): every
/// line the view writes uses only the fields brain's valid rollout records
/// use, never the removed `advantage`, with keys brain accepts, and keeps the
/// rules brain checks between records: the records of one rollout agree on
/// its reward, and every group holds at least two rollouts.
#[test]
fn the_records_the_view_writes_keep_brains_training_record_contract() {
    // doc-reference: the fixtures are brain's contract, copied unchanged.
    let fixtures =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/training-records/v2");
    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(fixtures.join("manifest.json")).unwrap())
            .unwrap();
    assert_eq!(manifest["contract"], "training-records-v2");
    let fixture_lines = |name: &str| -> Vec<Value> {
        std::fs::read_to_string(fixtures.join(name))
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    };
    // The fields a rollout record may carry: those of brain's valid rollout
    // records, and the `tools` and `metadata` every chat record may carry.
    let mut allowed: std::collections::BTreeSet<String> =
        ["tools", "metadata"].map(String::from).into();
    for line in fixture_lines("chat_rollout.jsonl")
        .iter()
        .chain(&fixture_lines("chat.jsonl"))
    {
        allowed.extend(line.as_object().unwrap().keys().cloned());
    }
    assert!(allowed.contains("reward") && allowed.contains("group") && allowed.contains("rollout"));
    // A baseline-only record: brain's valid one (line 2 of its refusal file,
    // which only refuses line 3) names the fields it carries.
    let baselines = fixture_lines("invalid/baseline_trains.jsonl");
    let baseline_fields: std::collections::BTreeSet<String> = baselines[1]
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .chain(["tools", "metadata"].map(String::from))
        .collect();
    assert_eq!(baselines[1]["baseline_only"], true);
    let refused = fixture_lines("invalid/advantage_removed.jsonl");
    assert!(
        refused[0].get("advantage").is_some(),
        "the removed field the contract refuses"
    );

    // Graded groups of several trials, one trial answered by two roles (two
    // conversations of one rollout), a group of one that must not be written,
    // and a trial whose only conversation cannot be rendered (a baseline).
    let mut corpus = corpus(&[
        Trial::pass("t1").role("designer"),
        Trial::pass("t1").role("critic"),
        Trial::fail("t2", &[]),
        Trial::fail("t3", &[]),
        Trial::pass("b1").version("2"),
        Trial::fail("b2", &[]).version("2"),
        Trial::pass("alone").model("m2"),
    ]);
    add_shown(&mut corpus, &Trial::fail("t4", &[]), None, true);
    for kind in [
        AdvantageKind::Standardised,
        AdvantageKind::MeanCentred,
        AdvantageKind::LeaveOneOut,
        AdvantageKind::Positive,
    ] {
        let projection = view().advantage(kind).project(&corpus).unwrap();
        let dir = std::env::temp_dir().join(format!("pg-contract-{}-{kind:?}", std::process::id()));
        let path = dir.join("d.jsonl");
        write_dataset(&path, &projection, &Unchecked).unwrap();
        let lines: Vec<Value> = std::fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        let _ = std::fs::remove_dir_all(&dir);
        let mut rollouts: std::collections::BTreeMap<(String, String), f64> =
            std::collections::BTreeMap::new();
        for line in &lines {
            let keys: Vec<&String> = line.as_object().unwrap().keys().collect();
            if line.get("baseline_only").is_some() {
                assert!(
                    keys.iter().all(|k| baseline_fields.contains(*k)),
                    "{kind:?}: {keys:?} beyond {baseline_fields:?}"
                );
                assert!(
                    line["messages"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .all(|m| m["train"] != true),
                    "{kind:?}: a baseline supervises nothing"
                );
            } else {
                assert!(
                    keys.iter().all(|k| allowed.contains(*k)),
                    "{kind:?}: {keys:?} beyond {allowed:?}"
                );
            }
            let (group, rollout) = (
                line["group"].as_str().unwrap(),
                line["rollout"].as_str().unwrap(),
            );
            for key in [group, rollout] {
                assert!(
                    !key.is_empty() && key.len() <= 512 && !key.chars().any(char::is_control),
                    "{key:?}"
                );
            }
            let reward = line["reward"].as_f64().unwrap();
            assert!(reward.is_finite());
            let seen = rollouts
                .entry((group.to_string(), rollout.to_string()))
                .or_insert(reward);
            assert_eq!(
                *seen, reward,
                "{kind:?}: the records of one rollout agree on its reward"
            );
        }
        let mut per_group: std::collections::BTreeMap<&str, usize> =
            std::collections::BTreeMap::new();
        for (group, _) in rollouts.keys() {
            *per_group.entry(group.as_str()).or_insert(0) += 1;
        }
        assert!(!per_group.is_empty(), "{kind:?}: something is written");
        assert_eq!(
            lines
                .iter()
                .filter(|l| l.get("baseline_only").is_some())
                .count(),
            1,
            "{kind:?}: the unrendered trial is one baseline"
        );
        assert!(
            per_group.values().all(|n| *n >= 2),
            "{kind:?}: no group of one: {per_group:?}"
        );
        assert_eq!(
            serde_json::to_value(projection.advantage.unwrap().norm()).unwrap(),
            Value::from(projection.advantage.unwrap().norm().name()),
            "the manifest names brain's advantage_norm by its own spelling"
        );
    }
}

mod exact_tokens;
mod external;
mod identity;
mod request_log;
mod self_distill;
mod shaping;
