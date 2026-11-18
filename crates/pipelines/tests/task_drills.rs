// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements leakage-free held-out splits of
// code-editing task sets, for its clients. If your team needs expertise in
// holding out whole task families before any training data is built, you
// can procure our services by sending an email to info@swedishembedded.com.

//! Spec: `tasks drills` makes comprehension drills from a practised
//! repository and keeps the exam sealed.
//!
//! * a drill of a file the exam reserved is an exam drill, a drill of any
//!   other file a training drill, and the two sets partition the drills;
//! * the same seed makes the same sets;
//! * an answer to an exam drill cannot be built into a training set, and
//!   one to a training drill can, which remembers its file as trained on;
//! * with `for_tasks`, only the functions that contain a line a repair task's
//!   reference change changes are drilled, and a reference change that is not
//!   of the revision is refused;
//! * drills export as Harbor tasks like the other tasks of a set.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::collections::BTreeSet;

use common::gate::{put_base, FakeTrainer};
use common::repo::{git, project, write, Agent};
use common::{config, Scratch};
use serde_json::json;
use splinter_agent::solve::Model;
use splinter_core::annotation::{Annotation, AnnotationBody, Outcome, Producer, Strength};
use splinter_core::clock::FixedClock;
use splinter_core::experience::{Experience, PrivilegedKind, Provenance};
use splinter_core::family::touches;
use splinter_core::model_ref::ModelRef;
use splinter_core::rng::Seed;
use splinter_knowledge::tasks::workspace::DrillOrigin;
use splinter_orchestrator::Context;
use splinter_pipelines::curriculum::frontier::PassAtK;
use splinter_pipelines::datasets::{build, BuildRequest, ViewName, VoiceBuild};
use splinter_pipelines::exam_seal;
use splinter_pipelines::harbor::export_harbor;
use splinter_pipelines::practice::{practice, PracticeRequest};
use splinter_pipelines::tasks::drills::{drills, DrillRequest, Drilled};
use splinter_store::experiences::ExperienceSet;
use splinter_store::tasks::TaskSetId;

fn context(test: &str) -> (Scratch, Context) {
    let scratch = Scratch::new(test);
    let ctx = Context::new(config(&scratch), false)
        .unwrap()
        .with_clock(Box::new(FixedClock::new("2026-11-06T08:00:00.000Z")))
        .with_model(
            ModelRef::policy_default(),
            Model::new(Agent::new(), "scripted/agent-1"),
        );
    put_base(&ctx);
    (scratch, ctx)
}

/// A practised repository: its admitted set and the files its exam holds.
fn practised(test: &str) -> (Scratch, Context, TaskSetId, BTreeSet<String>) {
    let (scratch, ctx) = context(test);
    let repo = scratch.0.join("repo");
    project(&repo, &[]);
    let request = PracticeRequest {
        repository: repo,
        revision: "HEAD".into(),
        max_tasks: Some(32),
        mutants_per_file: Some(1),
        exam_families: Some(2),
        pass_at_k: PassAtK::new(2, None, None),
        no_release: true,
        ..PracticeRequest::default()
    };
    let report = practice(&ctx, &request, &FakeTrainer::knowing(&[]))
        .unwrap()
        .report;
    let exam = report.reserve.unwrap().families.into_iter().collect();
    (scratch, ctx, report.tasks.unwrap().task_set.unwrap(), exam)
}

fn request(scratch: &Scratch, seed: u64) -> DrillRequest {
    DrillRequest {
        repository: scratch.0.join("repo"),
        revision: "HEAD".into(),
        allow_dirty: false,
        per_function: 2,
        // seed: the second run must draw the same drills as the first
        seed: Seed::new(seed),
        files: Vec::new(),
        for_tasks: None,
    }
}

fn members(ctx: &Context, set: &Option<TaskSetId>) -> Vec<splinter_core::experience::Task> {
    set.as_ref()
        .map(|s| ctx.tasks().get_set(s).unwrap().members)
        .unwrap_or_default()
        .into_iter()
        .map(|m| ctx.tasks().get(&m.task).unwrap())
        .collect()
}

#[test]
fn a_drill_of_an_exam_file_is_an_exam_drill_and_the_rest_may_be_trained_on() {
    let (scratch, ctx, _, exam) = practised("task-drills-exam");
    assert_eq!(exam.len(), 2);

    let made = drills(&ctx, &request(&scratch, 7)).unwrap();
    assert!(made.admitted > 0, "{made:#?}");
    assert_eq!(made.exam_drills + made.training_drills, made.admitted);
    assert!(
        made.exam_drills > 0 && made.training_drills > 0,
        "{made:#?}"
    );

    let in_exam = members(&ctx, &made.exam_task_set);
    let in_training = members(&ctx, &made.training_task_set);
    for task in &in_exam {
        let origin = DrillOrigin::of_task(task).unwrap();
        assert!(
            touches(&origin.file, &exam).is_some(),
            "{} is not exam",
            origin.file
        );
        assert_eq!(task.concepts[0], origin.file);
    }
    for task in &in_training {
        let origin = DrillOrigin::of_task(task).unwrap();
        assert_eq!(
            touches(&origin.file, &exam),
            None,
            "{} was trained on",
            origin.file
        );
    }
    let all: BTreeSet<_> = members(&ctx, &made.task_set)
        .into_iter()
        .map(|t| t.task.id)
        .collect();
    let split: BTreeSet<_> = in_exam
        .iter()
        .chain(&in_training)
        .map(|t| t.task.id.clone())
        .collect();
    assert_eq!(all, split);
    assert_eq!(all.len(), in_exam.len() + in_training.len());
    assert_eq!(made.families.values().sum::<usize>(), made.admitted);

    // Again with the same seed, the same sets.
    let again = drills(&ctx, &request(&scratch, 7)).unwrap();
    assert_eq!(made.task_set, again.task_set);
    assert_eq!(made.exam_task_set, again.exam_task_set);
    assert_eq!(made.training_task_set, again.training_task_set);
}

fn answered(ctx: &Context, task: &splinter_core::experience::Task) -> Experience {
    let reference = task
        .privileged
        .iter()
        .find(|p| p.kind == PrivilegedKind::Reference)
        .unwrap()
        .content
        .clone();
    let experience = Experience::answered_without_a_run(
        task.clone(),
        &format!("```\n{reference}\n```"),
        Provenance::new(
            "scripted/solver-1",
            &FixedClock::new("2026-11-06T09:00:00.000Z"),
        ),
    )
    .unwrap();
    let id = ctx.experiences().put(&experience).unwrap();
    ctx.experiences()
        .annotate(&Annotation {
            experience: id,
            producer: Producer {
                name: "grader".into(),
                version: "1".into(),
            },
            body: AnnotationBody::Verdict {
                outcome: Outcome::Pass,
                strength: Strength::Formal,
                evidence: json!({}),
            },
        })
        .unwrap();
    experience
}

fn sft_of(ctx: &Context, experience: &Experience) -> Result<(), String> {
    let set = ctx
        .experiences()
        .put_set(&ExperienceSet {
            name: "answers".into(),
            members: vec![experience.id().unwrap()],
        })
        .unwrap();
    build(
        ctx,
        &BuildRequest {
            sets: vec![set],
            view: ViewName::SftFinal,
            strip: None,
            min_strength: Some(Strength::Formal),
            system_prompt: None,
            export_only: false,
            limit: None,
            voice: VoiceBuild::default(),
            max_block: None,
            pack: false,
            split_over_block: false,
            control: Default::default(),
            serialise_calls: false,
            only_actions: None,
            roles: None,
            images: false,
            rl: Default::default(),
            split_seed: None,
            history: Default::default(),
            base: None,
        },
    )
    .map(|_| ())
    .map_err(|e| e.to_string())
}

#[test]
fn an_answer_to_an_exam_drill_is_never_built_into_a_training_set_and_one_to_a_training_drill_is() {
    let (scratch, ctx, _, exam) = practised("task-drills-seal");
    let made = drills(&ctx, &request(&scratch, 7)).unwrap();

    let before = exam_seal::trained(&ctx).unwrap();
    let exam_task = members(&ctx, &made.exam_task_set).remove(0);
    let refused = sft_of(&ctx, &answered(&ctx, &exam_task)).unwrap_err();
    let file = &exam_task.concepts[0];
    assert!(
        refused.contains("leakage") && refused.contains(file.as_str()),
        "{refused}"
    );
    assert_eq!(exam_seal::trained(&ctx).unwrap(), before);

    let training_task = members(&ctx, &made.training_task_set).remove(0);
    sft_of(&ctx, &answered(&ctx, &training_task)).unwrap();
    let mut after = before;
    after.insert(training_task.concepts[0].clone());
    assert_eq!(exam_seal::trained(&ctx).unwrap(), after);
    assert_eq!(touches(&training_task.concepts[0], &exam), None);
}

fn lines_of(drilled: &Drilled) -> BTreeSet<String> {
    drilled.drilled.keys().cloned().collect()
}

#[test]
fn drills_for_tasks_cover_exactly_the_functions_their_reference_changes_touch() {
    let (scratch, ctx, tasks, _) = practised("task-drills-for-tasks");
    let mut asked = request(&scratch, 7);
    asked.for_tasks = Some(tasks.clone());
    let made = drills(&ctx, &asked).unwrap();
    let scope = made.scope.clone().unwrap();
    assert!(scope.tasks > 0 && scope.without_reference == 0, "{scope:?}");
    assert!(!scope.functions.is_empty(), "{scope:?}");
    // Every drill is of a function a change lies in, and of nothing else.
    assert!(
        lines_of(&made).is_subset(&scope.functions.iter().cloned().collect()),
        "{made:#?}"
    );
    assert!(made.admitted > 0, "{made:#?}");

    // The same drills of the whole repository are more than those.
    let whole = drills(&ctx, &request(&scratch, 7)).unwrap();
    assert!(whole.admitted >= made.admitted);
}

#[test]
fn a_reference_change_made_for_another_revision_is_refused_naming_the_revision() {
    let (scratch, ctx, tasks, _) = practised("task-drills-other-revision");
    let repo = scratch.0.join("repo");
    for n in 0..common::repo::MODULES {
        let path = repo.join(format!("shop/tier{n}.py"));
        let text = std::fs::read_to_string(&path).unwrap();
        write(&path, &format!("# moved down a line\n{text}"));
    }
    git(&repo, &["add", "-A"]);
    git(&repo, &["commit", "-q", "-m", "shift"]);
    let mut asked = request(&scratch, 7);
    asked.for_tasks = Some(tasks);
    let refused = drills(&ctx, &asked).unwrap_err().to_string();
    assert!(refused.contains("drill the revision"), "{refused}");
}

#[test]
fn drills_export_as_harbor_tasks_like_the_other_tasks_of_a_set() {
    let (scratch, ctx, _, _) = practised("task-drills-harbor");
    let made = drills(&ctx, &request(&scratch, 7)).unwrap();
    let out = scratch.0.join("harbor");
    let exports = export_harbor(&ctx, &made.task_set.unwrap(), &out, "python:3.12-slim").unwrap();
    assert_eq!(exports.written.len(), made.admitted);
    for exported in &exports.written {
        for file in [
            "instruction.md",
            "task.toml",
            "tests/test.sh",
            "tests/expected_output.txt",
        ] {
            assert!(exported.directory.join(file).is_file(), "{file}");
        }
    }
}
