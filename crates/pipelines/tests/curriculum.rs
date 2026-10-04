// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements curricula that spend a learner's solver and
// training budget where it still learns, for its clients. If your team needs
// expertise in curriculum design or continual learning, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: the curriculum spends solver and training budget where learning
//! happens.
//!
//! * Before its attempts become training data, a task's pass@k is
//!   measured: k solves, each graded by the task's own verifiers and kept
//!   as an experience. A task the policy never solves is solved once more
//!   by a teacher, open-book, shown the task's grounding material. A task
//!   the policy always solves is dropped (no signal), one it solves
//!   sometimes is kept, one it never solves is kept when the teacher's
//!   answer is verified (taught) and dropped otherwise (nothing to learn
//!   from); the distribution is reported and the measurement recorded per
//!   task.
//! * `learn` measures the frontier by default.
//! * Concept mastery is tallied per release from the policy's recorded
//!   solves, and `status` lists the weakest concepts.
//! * A retention suite the gate saw drop queues its forgotten concepts for
//!   new tasks, which the next `learn` generates from.
//! * A training set is deduplicated and capped per concept, kind and
//!   strength, deterministically.
//!
//! Models are scripted, so every pass@k outcome is controlled.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use common::gate::{
    anchor_file, candidate, decide, fact, gate_context, knower, released, Brain, ANCHOR,
};
use common::manual::{BAUD_QUESTION, BAUD_QUOTE, MANUAL};
use common::{scratch_context, Scripted, POLICY};
use serde_json::json;
use splinter_agent::solve::Model;
use splinter_agent::CancelToken;
use splinter_core::annotation::Strength;
use splinter_core::digest::Digest;
use splinter_core::experience::{Environment, ExperienceId, Privileged, PrivilegedKind, Task};
use splinter_core::model_ref::ModelRef;
use splinter_core::release::ReleaseId;
use splinter_eval::gate::SuiteSummary;
use splinter_eval::gate::{self, Check, GateConfig, GateReport};
use splinter_eval::paired::PairedOutcome;
use splinter_knowledge::concepts::Concept;
use splinter_orchestrator::Context;
use splinter_pipelines::curriculum::frontier::{measure, MeasureRequest, PassAtK};
use splinter_pipelines::curriculum::mastery::weakest;
use splinter_pipelines::curriculum::queue::{enqueue_retention, pending};
use splinter_pipelines::curriculum::quota::{select, Candidate, Quotas};
use splinter_pipelines::curriculum::BASE_POLICY;
use splinter_pipelines::learn::{learn, LearnRequest, Learned};
use splinter_pipelines::release::anchor;
use splinter_pipelines::solving::solve_set;
use splinter_pipelines::status::status;
use splinter_pipelines::verify::verify_set;
use splinter_store::decision::decide as decision;
use splinter_store::tasks::{TaskEntry, TaskSet, TaskSetId};

/// A closed-book recall task asking `instruction`, referenced by `answer`.
fn recall(instruction: &str, answer: &str) -> Task {
    Task::new(
        "recall",
        Vec::new(),
        Environment::closed_book(),
        instruction,
        vec![Privileged {
            kind: PrivilegedKind::Reference,
            content: answer.into(),
            span: None,
        }],
    )
    .unwrap()
}

/// Stores `tasks` as a task set.
fn task_set(ctx: &Context, tasks: &[Task]) -> TaskSetId {
    let members = tasks
        .iter()
        .map(|task| TaskEntry {
            task: ctx.tasks().put(task).unwrap(),
            generator: None,
            prompt: None,
            variant_of: None,
            subject: None,
        })
        .collect();
    ctx.tasks()
        .put_set(&TaskSet {
            name: "curriculum spec".into(),
            members,
        })
        .unwrap()
}

/// A policy that always knows the capital, never the moon's mass, and
/// knows the boiling point every other time it is asked.
fn uneven_policy() -> Scripted {
    let asked = Arc::new(AtomicUsize::new(0));
    Scripted::new(move |prompt| {
        if prompt.contains("capital of France") {
            "Paris".into()
        } else if prompt.contains("boiling point") {
            let n = asked.fetch_add(1, Ordering::SeqCst);
            if n.is_multiple_of(2) { "100 C" } else { "90 C" }.into()
        } else {
            "I do not know.".into()
        }
    })
}

/// The passage the speed-of-light task is grounded in, which only its
/// teacher is shown.
const LIGHT_PASSAGE: &str = "Light travels through a vacuum at 299792458 m/s.";

/// A teacher that answers the speed of light when it is shown the
/// passage, and nothing else.
fn light_teacher() -> Scripted {
    Scripted::new(|prompt| {
        if prompt.contains(LIGHT_PASSAGE) {
            "299792458 m/s".into()
        } else {
            "I do not know.".into()
        }
    })
}

#[test]
fn pass_at_k_keeps_the_frontier_and_the_taught_and_records_every_attempt() {
    let (_scratch, ctx) = scratch_context("curriculum-frontier", uneven_policy(), false);
    let always = recall("What is the capital of France?", "Paris");
    let never = recall("What is the mass of the moon?", "7.35e22 kg");
    let sometimes = recall("What is the boiling point of water at sea level?", "100 C");
    let taught = Task::new(
        "recall",
        Vec::new(),
        Environment::closed_book(),
        "How fast does light travel in a vacuum?",
        vec![
            Privileged {
                kind: PrivilegedKind::Reference,
                content: "299792458 m/s".into(),
                span: None,
            },
            Privileged {
                kind: PrivilegedKind::Passage,
                content: LIGHT_PASSAGE.into(),
                span: None,
            },
        ],
    )
    .unwrap();
    let set = task_set(
        &ctx,
        &[
            always.clone(),
            never.clone(),
            sometimes.clone(),
            taught.clone(),
        ],
    );
    let policy = ModelRef::policy_default();
    let teacher: ModelRef = "local:./teacher".parse().unwrap();
    ctx.add_model(
        teacher.clone(),
        Model::new(Arc::new(light_teacher()), "scripted/teacher"),
    );

    let measured = measure(
        &ctx,
        &MeasureRequest {
            task_set: &set,
            solver: &policy,
            teacher: Some(&teacher),
            pass_at_k: PassAtK::default(),
            deadline: None,
            cancel: CancelToken::new(),
        },
    )
    .unwrap();

    let frontier = &measured.frontier;
    assert_eq!(frontier.k, 4);
    let d = frontier.distribution;
    assert_eq!(
        (d.always, d.never, d.frontier, d.taught, d.unmeasured),
        (1, 1, 1, 1, 0)
    );
    assert_eq!(
        frontier.sampling, None,
        "a model handed in samples as its provider does"
    );
    let kept = ctx.tasks().get_set(&frontier.frontier_task_set).unwrap();
    let kept: Vec<&Digest> = kept.members.iter().map(|e| &e.task).collect();
    assert_eq!(
        kept,
        [&sometimes.task.id, &taught.task.id],
        "the frontier and the taught are kept"
    );

    // The teacher named solved, open-book, the two tasks never solved:
    // the one with a passage it answered verifiably; the one grounded in
    // nothing it was not asked.
    let teach = &measured.teach;
    assert_eq!(
        (frontier.teacher.as_str(), teach.solve.solver.as_str()),
        ("scripted/teacher", "scripted/teacher")
    );
    assert_eq!(
        (teach.solve.solved, teach.verify.passed),
        (1, 1),
        "{teach:#?}"
    );
    let [skipped] = &teach.solve.skipped[..] else {
        panic!("one task skipped: {teach:#?}")
    };
    assert_eq!(skipped.task, never.task.id);
    let store = ctx.experiences();
    for id in store.get_set(&teach.solve.experience_set).unwrap().members {
        let experience = store.get(&id).unwrap();
        assert!(experience.provenance.teacher);
        assert_eq!(experience.task.id, taught.task.id);
    }

    // Every attempt is an experience with its verdicts, numbered and
    // labelled with the release the policy was: the base.
    let attempts = store.get_set(&measured.solve.experience_set).unwrap();
    assert_eq!(attempts.members.len(), 16, "k solves of each of 4 tasks");
    let mut numbered = Vec::new();
    for id in &attempts.members {
        let experience = store.get(id).unwrap();
        assert_eq!(experience.provenance.policy.as_deref(), Some(BASE_POLICY));
        assert_eq!(experience.provenance.solver, POLICY);
        let notes = store.annotations(id).unwrap().annotations;
        assert!(decision(&notes).is_some(), "each attempt is graded");
        if experience.task.id == sometimes.task.id {
            numbered.push(experience.provenance.attempt.unwrap());
        }
    }
    numbered.sort_unstable();
    assert_eq!(numbered, [0, 1, 2, 3]);
    let frontier_attempts = store.get_set(&frontier.frontier_experience_set).unwrap();
    assert_eq!(
        frontier_attempts.members.len(),
        9,
        "the frontier task's attempts, the taught task's and its verified answer"
    );

    // The measurement is recorded per task: attempts, passes, rate, and
    // the release measured.
    let recorded: serde_json::Value = ctx
        .workspace()
        .get_document("frontier_measurement", &frontier.measurement)
        .unwrap()
        .unwrap();
    assert_eq!(
        recorded["policy"],
        json!({ "alias": "default", "release": null })
    );
    let row = |task: &Task| {
        recorded["tasks"]
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["task"] == task.task.id.to_string())
            .cloned()
            .unwrap()
    };
    let sometimes_row = row(&sometimes);
    assert_eq!(
        (
            &sometimes_row["attempts"],
            &sometimes_row["passes"],
            &sometimes_row["rate"]
        ),
        (&json!(4), &json!(2), &json!(0.5))
    );
    assert_eq!(sometimes_row["class"], "frontier");
    assert_eq!(row(&always)["class"], "always");
    assert_eq!(
        (&row(&never)["rate"], &row(&never)["class"]),
        (&json!(0.0), &json!("never"))
    );
    let taught_row = row(&taught);
    assert_eq!(
        (
            &taught_row["passes"],
            &taught_row["teacher"],
            &taught_row["class"]
        ),
        (
            &json!(0),
            &json!({ "attempts": 1, "graded": 1, "passes": 1 }),
            &json!("taught")
        )
    );
    assert_eq!(recorded["teacher"], "scripted/teacher");
    assert_eq!(
        sometimes_row["concepts"],
        json!([{ "concept": "kind", "kind": "recall" }]),
        "a task with no evidence and no declaration exercises its kind"
    );

    // Sampling that cannot be set is refused, not ignored.
    let refused = measure(
        &ctx,
        &MeasureRequest {
            task_set: &set,
            solver: &policy,
            teacher: None,
            pass_at_k: PassAtK::new(4, Some(1.0), None),
            deadline: None,
            cancel: CancelToken::new(),
        },
    )
    .unwrap_err();
    assert!(refused.is_refusal(), "{refused}");
    assert!(
        measure(
            &ctx,
            &MeasureRequest {
                task_set: &set,
                solver: &policy,
                teacher: None,
                pass_at_k: PassAtK::new(1, None, None),
                deadline: None,
                cancel: CancelToken::new(),
            },
        )
        .unwrap_err()
        .is_refusal(),
        "k = 1 cannot find a task sometimes solved"
    );
}

/// The manual's policy, knowing the baud rate every other time it is asked
/// and never the idle current.
fn unsure_manual_policy() -> Scripted {
    let asked = Arc::new(AtomicUsize::new(0));
    Scripted::new(move |prompt| {
        if prompt.contains("You write training tasks") {
            json!({ "tasks": [
                {
                    "instruction": BAUD_QUESTION,
                    "subject": "Frobnicator",
                    "reference": "115200 baud",
                    "evidence": [{ "section": 1, "quote": BAUD_QUOTE }]
                },
                {
                    "instruction": "How much current does the Frobnicator draw when idle?",
                    "subject": "Frobnicator",
                    "reference": "40 mA",
                    "evidence": [{ "section": 2, "quote": "40 mA when idle" }]
                }
            ]})
            .to_string()
        } else if prompt.contains("You are reviewing an attempt") {
            "The baud rate is wrong.".into()
        } else if prompt.contains("baud rate") {
            let n = asked.fetch_add(1, Ordering::SeqCst);
            if n.is_multiple_of(2) {
                "115200 baud"
            } else {
                "9600 baud"
            }
            .into()
        } else {
            "12 mA".into()
        }
    })
}

#[test]
fn learn_keeps_the_frontier_by_default_and_generates_for_queued_concepts() {
    let (scratch, ctx) = scratch_context("curriculum-learn", unsure_manual_policy(), false);
    let manual = scratch.0.join("manual.md");
    std::fs::write(&manual, MANUAL).unwrap();
    let request = LearnRequest {
        sources: vec![manual.display().to_string()],
        no_release: true,
        ..LearnRequest::default()
    };
    let trainer = common::gate::FakeTrainer::knowing(&[]);

    let Learned::Ran(run) = learn(&ctx, &request, &trainer).unwrap() else {
        panic!("a learn that is not a dry run runs");
    };
    let report = &run.report;
    let solved = report.solve.as_ref().unwrap();
    assert_eq!((solved.attempts, solved.solved), (4, 8), "{solved:#?}");
    let frontier = report.frontier.as_ref().unwrap();
    let d = frontier.distribution;
    assert_eq!((d.always, d.never, d.frontier), (0, 1, 1), "{frontier:#?}");
    let recorded: serde_json::Value = ctx
        .workspace()
        .get_document("frontier_measurement", &frontier.measurement)
        .unwrap()
        .unwrap();
    let baud = recorded["tasks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["class"] == "frontier")
        .unwrap();
    let section = |concept: &serde_json::Value| concept["section"].as_u64();
    assert_eq!(
        baud["concepts"]
            .as_array()
            .unwrap()
            .iter()
            .map(section)
            .collect::<Vec<_>>(),
        [Some(1)],
        "the baud-rate task exercises the manual's console section"
    );
    let select = report.select.as_ref().unwrap();
    assert_eq!(
        select.selection.selected.len(),
        1,
        "the passing attempts of one task are one example: {select:#?}"
    );
    assert!(select.selection.duplicates >= 1, "{select:#?}");
    let stages: Vec<String> = splinter_store::runs::read_run(ctx.workspace(), &run.run)
        .unwrap()
        .stages
        .into_iter()
        .map(|s| s.stage)
        .collect();
    assert!(
        stages.iter().any(|s| s == "frontier") && stages.iter().any(|s| s == "select"),
        "{stages:?}"
    );

    // The gate saw the baud-rate task forgotten: its concept is queued, and
    // the next learn generates from its section and empties the queue.
    let outcomes = vec![PairedOutcome {
        item: baud["task"].as_str().unwrap().to_string(),
        candidate: Some(false),
        baseline: Some(true),
        cluster: None,
    }];
    let suite = SuiteSummary {
        name: "retention".into(),
        tasks: 1,
        excluded: Default::default(),
    };
    let earlier = ReleaseId(Digest::of(b"an earlier release"));
    let report = GateReport::new(
        GateConfig::default(),
        Check::unmeasured("not asked"),
        gate::retention(vec![(earlier, suite, outcomes)], 0.05),
        Check::unmeasured("not asked"),
        Check::unmeasured("not asked"),
    );
    let queued = enqueue_retention(&ctx, &report).unwrap();
    let [Concept::Section(console)] = queued.as_slice() else {
        panic!("one section concept: {queued:?}");
    };
    assert_eq!(console.section, 1);
    let waiting = pending(&ctx).unwrap();
    assert_eq!(waiting.len(), 1);
    assert_eq!(waiting[0].sections, std::slice::from_ref(console));

    let Learned::Ran(again) = learn(&ctx, &request, &trainer).unwrap() else {
        panic!("a learn that is not a dry run runs");
    };
    let tasks = again.report.tasks.as_ref().unwrap();
    assert_eq!(tasks.sections, 1, "{tasks:#?}");
    assert!(pending(&ctx).unwrap().is_empty(), "the queue is drained");

    // --no-frontier solves once and keeps every task.
    let Learned::Ran(plain) = learn(
        &ctx,
        &LearnRequest {
            no_frontier: true,
            ..request.clone()
        },
        &trainer,
    )
    .unwrap() else {
        panic!("a learn that is not a dry run runs");
    };
    assert!(plain.report.frontier.is_none());
    assert_eq!(plain.report.solve.as_ref().unwrap().attempts, 1);
}

/// The fact tasks of `topics`, `n` each, as one set.
fn facts(ctx: &Context, topics: &[&str], n: usize) -> TaskSetId {
    let tasks: Vec<Task> = topics
        .iter()
        .flat_map(|topic| (0..n).map(move |i| fact(topic, i)))
        .collect();
    task_set(ctx, &tasks)
}

#[test]
fn mastery_is_tallied_per_release_and_status_lists_the_weakest() {
    let (scratch, ctx) = gate_context("curriculum-mastery", Brain::Honest);
    anchor::freeze(&ctx, &anchor_file(&scratch.0, 4)).unwrap();
    let policy = ModelRef::policy_default();
    ctx.add_model(policy.clone(), knower(&[ANCHOR, "gamma"]));
    let set = facts(&ctx, &["alpha", "gamma"], 3);
    let solved = solve_set(&ctx, &set, &policy, None, &CancelToken::new()).unwrap();
    verify_set(&ctx, &solved.experience_set, None, &CancelToken::new()).unwrap();

    let base = weakest(&ctx, 5).unwrap();
    assert_eq!(base.policy.release, None);
    assert_eq!((base.concepts, base.measured), (2, 2));
    let names: Vec<String> = base.weakest.iter().map(|c| c.concept.to_string()).collect();
    assert_eq!(names, ["declared:alpha", "declared:gamma"]);
    assert_eq!(base.weakest[0].current.as_ref().unwrap().rate, Some(0.0));

    // A release that learned alpha; the policy now answers as it does.
    let release = released(&ctx, "alpha", &[ANCHOR, "alpha", "gamma"]);
    ctx.add_model(policy.clone(), knower(&[ANCHOR, "alpha"]));
    let solved = solve_set(&ctx, &set, &policy, None, &CancelToken::new()).unwrap();
    verify_set(&ctx, &solved.experience_set, None, &CancelToken::new()).unwrap();

    let now = status(&ctx).unwrap().concepts;
    assert_eq!(now.policy.release, Some(release.clone()));
    let weakest = &now.weakest[0];
    assert_eq!(
        weakest.concept,
        Concept::Declared {
            name: "gamma".into()
        }
    );
    let current = weakest.current.as_ref().unwrap();
    assert_eq!(
        (current.release.as_ref(), current.rate),
        (Some(&release), Some(0.0))
    );
    let earlier: Vec<(Option<&ReleaseId>, Option<f64>)> = weakest
        .releases
        .iter()
        .map(|m| (m.release.as_ref(), m.rate))
        .collect();
    assert_eq!(earlier, [(None, Some(1.0)), (Some(&release), Some(0.0))]);
    let alpha = &now.weakest[1];
    assert_eq!(
        alpha.concept,
        Concept::Declared {
            name: "alpha".into()
        }
    );
    assert_eq!(alpha.current.as_ref().unwrap().rate, Some(1.0));
    assert!(
        weakest.releases.iter().all(|m| m.graded == 3),
        "each release's own attempts, never pooled: {weakest:#?}"
    );
}

#[test]
fn a_retention_drop_queues_the_forgotten_concept() {
    let (scratch, ctx) = gate_context("curriculum-retention", Brain::Honest);
    anchor::freeze(&ctx, &anchor_file(&scratch.0, 4)).unwrap();
    released(&ctx, "alpha", &[ANCHOR, "alpha"]);

    // Beta learned, alpha forgotten.
    let (forgetful, _) = candidate(&ctx, "beta", &[ANCHOR, "beta"]);
    let decided = decide(&ctx, &forgetful);
    assert!(!decided.gate.retention.passed, "{:#?}", decided.gate);
    let alpha = Concept::Declared {
        name: "alpha".into(),
    };
    assert_eq!(decided.requeued, std::slice::from_ref(&alpha));
    let waiting = pending(&ctx).unwrap();
    assert_eq!(waiting.len(), 1);
    assert_eq!(waiting[0].concept, alpha);
    assert!(
        waiting[0].reasons[0].contains("dropped 1.0000"),
        "{:?}",
        waiting[0].reasons
    );
    assert_eq!(status(&ctx).unwrap().concepts.queued, 1);
}

fn candidate_of(
    id: u8,
    instruction: &str,
    concept: &str,
    kind: &str,
    strength: Strength,
) -> Candidate {
    Candidate {
        id: ExperienceId(Digest::of(&[id])),
        instruction: instruction.into(),
        concepts: vec![Concept::Declared {
            name: concept.into(),
        }],
        kind: kind.into(),
        strength,
    }
}

#[test]
fn quotas_cap_each_share_after_removing_near_duplicates_deterministically() {
    use Strength::{Executable, Formal};
    let pool = vec![
        candidate_of(
            1,
            "What baud rate does the console run at?",
            "console",
            "recall",
            Formal,
        ),
        candidate_of(
            2,
            "What baud rate does the console run at, exactly?",
            "console",
            "recall",
            Formal,
        ),
        candidate_of(
            3,
            "How many data bits does the console use?",
            "console",
            "recall",
            Formal,
        ),
        candidate_of(
            4,
            "Which parity does the console use?",
            "console",
            "recall",
            Formal,
        ),
        candidate_of(
            5,
            "How much current is drawn when idle?",
            "power",
            "recall",
            Formal,
        ),
        candidate_of(
            6,
            "Write a function returning the idle current.",
            "power",
            "construct",
            Executable,
        ),
        candidate_of(
            7,
            "Which pin carries the transmit line?",
            "pins",
            "recall",
            Formal,
        ),
        candidate_of(8, "Which pin is ground?", "pins", "recall", Formal),
        candidate_of(
            9,
            "What voltage does the board take?",
            "supply",
            "recall",
            Formal,
        ),
    ];
    let quotas = Quotas {
        max_concept_share: 0.25,
        max_kind_share: 0.6,
        max_strength_share: 0.9,
    };
    let chosen = select(pool.clone(), &quotas);
    assert_eq!(chosen.candidates, 9);
    assert_eq!(chosen.duplicates, 1, "the reworded baud-rate question");
    // 8 left, 4 concepts: at most ceil(max(0.25, 1/4) * 8) = 2 each; 2
    // kinds: ceil(0.6 * 8) = 5 recall; 2 strengths: ceil(0.9 * 8) = 8.
    assert_eq!(chosen.caps["concept"]["declared:console"], 2);
    assert_eq!(chosen.caps["kind"]["recall"], 5);
    let kept = |c: &splinter_pipelines::curriculum::quota::Selection, name: &str| {
        c.selected
            .iter()
            .filter(|id| {
                pool.iter().any(|p| {
                    &p.id == *id && p.concepts[0].to_string() == format!("declared:{name}")
                })
            })
            .count()
    };
    assert!(kept(&chosen, "console") <= 2, "{chosen:#?}");
    let recall = chosen
        .selected
        .iter()
        .filter(|id| pool.iter().any(|p| &p.id == *id && p.kind == "recall"))
        .count();
    assert!(recall <= 5, "{chosen:#?}");
    assert_eq!(chosen.selected.len(), 6, "{chosen:#?}");
    assert_eq!(
        chosen.selected[0],
        ExperienceId(Digest::of(&[6])),
        "the strongest verdict first"
    );

    let mut reversed = pool.clone();
    reversed.reverse();
    assert_eq!(select(reversed, &quotas), chosen, "order does not matter");

    let one_kind: Vec<Candidate> = pool.into_iter().filter(|c| c.kind == "recall").collect();
    let whole = select(one_kind, &Quotas::default());
    assert_eq!(
        whole.capped.get("kind"),
        None,
        "a pool of one kind is never capped by kind"
    );
}
