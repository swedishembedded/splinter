// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements on-policy learning from privileged
// demonstrations, for its clients. If your team needs expertise in turning a
// guided solution into training data the student could have produced itself,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: path-guided resampling. A failed attempt's task has a verified path
//! a person guided (one of its turns is the person's); the student's own
//! model is forked at that path's states with the person's turns taken out,
//! continues there, and the task's verifier decides.
//!
//! * The fork starts after the path's latest action; nothing the person typed
//!   reaches any request of the student's continuation.
//! * An accepted continuation is kept with the path's steps as copied context
//!   (never supervised) and the student's own steps after them; it holds no
//!   human turn, so a dataset may train on its suffix.
//! * A task without a guided path is not given the intervention at all.

#![allow(clippy::unwrap_used)]

use std::fs;
use std::sync::Arc;

use serde_json::json;
use splinter_agent::solve::Model;
use splinter_core::annotation::{Annotation, AnnotationBody, Outcome, Producer, Strength};
use splinter_core::model_ref::ModelRef;
use splinter_pipelines::bench::repair::{repair, InterventionKind, RepairOutcome};
use sven_sdk::model::ModelProvider;

use crate::bench_repair::{
    examiner, practice_run, request, steps, student_trajectory, teacher_ref, Step,
};
use crate::common::{self, Scratch};

const GUIDANCE: &str = "The answer is 42: write that.";

fn student_ref() -> ModelRef {
    "remote:scripted/student".parse().unwrap()
}

/// A verified pass of the failed attempt's task by another source: the
/// student's own reads, then (when `person`) the person's turn, then the
/// right answer, then (when `verify`) a read of it, by `solver`.
fn privileged_path(
    ctx: &splinter_orchestrator::Context,
    student: &splinter_core::experience::ExperienceId,
    person: bool,
    verify: bool,
    solver: &str,
) -> splinter_core::experience::ExperienceId {
    let store = ctx.experiences();
    let mut path = store.get(student).unwrap();
    let mut steps: Vec<atif::TraceStep> = path
        .trajectory
        .steps
        .iter()
        .filter(|s| {
            s.tool_calls
                .as_ref()
                .is_none_or(|c| c.iter().all(|c| c.function_name != "write_file"))
        })
        // Without a person, the path goes on from its reads: no reply ended it there.
        .filter(|s| {
            person
                || s.source != atif::StepOrigin::Agent
                || s.tool_calls.as_ref().is_some_and(|c| !c.is_empty())
        })
        .cloned()
        .collect();
    if person {
        let mut typed = atif::TraceStep::new(0, atif::StepOrigin::User, GUIDANCE);
        typed.extra = Some(json!({"sven": {"origin": "human"}}));
        steps.push(typed);
    }
    let mut write = atif::TraceStep::new(0, atif::StepOrigin::Agent, "");
    write.tool_calls = Some(vec![atif::ToolInvocation::new("guided-1", "write_file")
        .with_arguments(json!({"path": "answer.json", "text": "42"}))]);
    write.observation = Some(atif::StepObservation::single(
        atif::ObservationEntry::for_call("guided-1", "wrote 2 bytes to answer.json"),
    ));
    steps.push(write);
    if verify {
        let mut read = atif::TraceStep::new(0, atif::StepOrigin::Agent, "");
        read.tool_calls = Some(vec![atif::ToolInvocation::new("guided-2", "read_file")
            .with_arguments(json!({"path": "answer.json"}))]);
        read.observation = Some(atif::StepObservation::single(
            atif::ObservationEntry::for_call("guided-2", "L1:42"),
        ));
        steps.push(read);
    }
    steps.push(atif::TraceStep::new(0, atif::StepOrigin::Agent, "Done."));
    for (index, step) in steps.iter_mut().enumerate() {
        step.step_id = index as u64 + 1;
    }
    path.trajectory.steps = steps;
    path.provenance.solver = solver.into();
    let id = store.put(&path).unwrap();
    store
        .annotate(&Annotation {
            experience: id.clone(),
            producer: Producer {
                name: "verifier".into(),
                version: "1".into(),
            },
            body: AnnotationBody::Verdict {
                outcome: Outcome::Pass,
                strength: Strength::Executable,
                evidence: json!({"human_guided": person}),
            },
        })
        .unwrap();
    id
}

fn guided_path(
    ctx: &splinter_orchestrator::Context,
    student: &splinter_core::experience::ExperienceId,
) {
    privileged_path(ctx, student, true, false, "harbor/sven@5/student");
}

#[test]
fn the_student_continues_from_the_guided_paths_state_without_the_persons_turn() {
    let scratch = Scratch::new("path-guided");
    let student = steps(
        "pathstudent",
        vec![Step::Call(
            "write_file",
            json!({"path": "answer.json", "text": "42"}),
        )],
    );
    let ctx = splinter_orchestrator::Context::new(common::config(&scratch), true)
        .unwrap()
        .with_model(
            student_ref(),
            Model::new(
                Arc::clone(&student) as Arc<dyn ModelProvider>,
                "scripted/student",
            ),
        )
        // The run names a teacher; only the student acts here.
        .with_model(
            teacher_ref(),
            Model::new(
                steps("teacher", vec![]) as Arc<dyn ModelProvider>,
                "scripted/teacher",
            ),
        );
    let tree = scratch.0.join("student-tree");
    fs::create_dir_all(&tree).unwrap();
    let trajectory = ctx.block_on(student_trajectory(&tree));
    let runs = vec![practice_run(&ctx, &scratch.0, &trajectory)];
    let checkout = examiner(&scratch.0);
    let teacher = teacher_ref();
    let student_model = student_ref();
    let only = [InterventionKind::PathGuided];

    // No guided path of the task yet: the intervention is not tried.
    let mut asked = request(&runs, &checkout, &teacher);
    asked.student = Some(&student_model);
    asked.interventions = &only;
    let report = repair(&ctx, &asked).unwrap();
    assert!(report.repairs.is_empty(), "{:?}", report.repairs);

    let failed = splinter_pipelines::bench::split::trials(&ctx).unwrap()[0]
        .experience
        .clone();
    guided_path(&ctx, &failed);
    let report = repair(&ctx, &asked).unwrap();
    let [record] = report.repairs.as_slice() else {
        panic!("one repair: {:?}", report.repairs)
    };
    assert_eq!(record.intervention, "path_guided");
    assert_eq!(
        record.outcome,
        RepairOutcome::Accepted,
        "{:?}",
        record.reason
    );
    let path = record.notes.privileged_path.as_ref().unwrap();
    assert_eq!(path["removed_turns"], json!(1));
    // The latest state: after the path's own write of the answer.
    assert_eq!(json!(record.keep_through), path["fork_after"]);

    let shown = student.seen.lock().unwrap().join("\n");
    assert!(!shown.is_empty(), "the student was asked");
    assert!(
        !shown.contains(GUIDANCE),
        "the person's turn reached the student: {shown}"
    );

    let kept = ctx
        .experiences()
        .get(record.experience.as_ref().unwrap())
        .unwrap();
    assert!(!splinter_data::has_human_turn(&kept.trajectory));
    let prefix = record.prefix_steps.unwrap();
    assert!(kept.trajectory.steps[..prefix]
        .iter()
        .all(|s| s.is_copied_context == Some(true)));
    assert!(kept.trajectory.steps[prefix..]
        .iter()
        .all(|s| s.is_copied_context.is_none()));
    assert!(kept
        .trajectory
        .steps
        .iter()
        .all(|s| s.message.as_text() != Some(GUIDANCE)));
}

/// A teacher's verified pass is a privileged path too, when the run names the
/// set it is in: the student is forked at its states and continues; the
/// dataset trains on the student's own steps only, never on the teacher's
/// copied prefix (rule 1, same observation).
#[test]
fn a_teachers_pass_named_by_its_set_is_a_path_and_only_the_students_steps_are_trained() {
    let scratch = Scratch::new("path-guided-teacher");
    let student = steps(
        "pathstudent",
        vec![Step::Call(
            "write_file",
            json!({"path": "answer.json", "text": "42"}),
        )],
    );
    let ctx = splinter_orchestrator::Context::new(common::config(&scratch), true)
        .unwrap()
        .with_model(
            student_ref(),
            Model::new(
                Arc::clone(&student) as Arc<dyn ModelProvider>,
                "scripted/student",
            ),
        )
        .with_model(
            teacher_ref(),
            Model::new(
                steps("teacher", vec![]) as Arc<dyn ModelProvider>,
                "scripted/teacher",
            ),
        );
    let tree = scratch.0.join("student-tree");
    fs::create_dir_all(&tree).unwrap();
    let trajectory = ctx.block_on(student_trajectory(&tree));
    let runs = vec![practice_run(&ctx, &scratch.0, &trajectory)];
    let checkout = examiner(&scratch.0);
    let failed = splinter_pipelines::bench::split::trials(&ctx).unwrap()[0]
        .experience
        .clone();
    let path = privileged_path(&ctx, &failed, false, false, "harbor/sven@5/teacher-27b");
    let store = ctx.experiences();
    let set = store
        .put_set(&splinter_store::experiences::ExperienceSet {
            name: "teacher passes".into(),
            members: vec![path],
        })
        .unwrap();
    let teacher = teacher_ref();
    let student_model = student_ref();
    let only = [InterventionKind::PathGuided];
    let mut asked = request(&runs, &checkout, &teacher);
    asked.student = Some(&student_model);
    asked.interventions = &only;

    // Not named: a pass by another model is no privileged path by itself.
    assert!(repair(&ctx, &asked).unwrap().repairs.is_empty());

    let sets = [set];
    asked.paths = &sets;
    let report = repair(&ctx, &asked).unwrap();
    let [record] = report.repairs.as_slice() else {
        panic!("one repair: {:?}", report.repairs)
    };
    assert_eq!(
        record.outcome,
        RepairOutcome::Accepted,
        "{:?}",
        record.reason
    );
    assert_eq!(
        record.notes.privileged_path.as_ref().unwrap()["removed_turns"],
        json!(0)
    );

    // The loss mask: of the kept repair, only the student's own action is a
    // training target; every step of the teacher's path is context.
    let kept = record.experience.clone().unwrap();
    use splinter_data::View;
    let corpus = splinter_data::Corpus::load(&store, std::slice::from_ref(&kept)).unwrap();
    let projection = splinter_data::SftStep::new(Strength::Executable)
        .project(&corpus)
        .unwrap();
    let trained: Vec<String> = projection
        .records
        .iter()
        .flat_map(|record| match &record.body {
            splinter_data::RecordBody::Chat { messages, .. } => messages
                .iter()
                .filter(|m| m.train)
                .map(|m| serde_json::to_string(m).unwrap())
                .collect(),
            _ => Vec::new(),
        })
        .collect();
    let suffix: usize = store.get(&kept).unwrap().trajectory.steps[record.prefix_steps.unwrap()..]
        .iter()
        .filter(|s| s.source == atif::StepOrigin::Agent)
        .count();
    assert!(suffix > 0, "the student continued");
    assert_eq!(trained.len(), suffix, "{trained:?}");
    assert!(
        trained
            .iter()
            .all(|m| !m.contains("guided-1") && !m.contains("Done.")),
        "a step of the teacher's path is a target: {trained:?}"
    );
}

/// A student that ends the run when it is shown the answer written already,
/// and writes it otherwise.
struct Finisher {
    seen: std::sync::Mutex<usize>,
}

#[async_trait::async_trait]
impl ModelProvider for Finisher {
    fn name(&self) -> &str {
        "scripted"
    }
    fn model_name(&self) -> &str {
        "finisher"
    }
    async fn complete(
        &self,
        req: sven_sdk::model::CompletionRequest,
    ) -> anyhow::Result<sven_sdk::model::ResponseStream> {
        use sven_sdk::model::{MessageContent, ResponseEvent};
        *self.seen.lock().unwrap() += 1;
        let results: Vec<(&str, String)> = req
            .messages
            .iter()
            .filter_map(|m| match &m.content {
                MessageContent::ToolResult {
                    tool_call_id,
                    content,
                } => Some((tool_call_id.as_str(), format!("{content:?}"))),
                _ => None,
            })
            .collect();
        let written = results
            .iter()
            .any(|(id, text)| id.starts_with("finisher") || text.contains("wrote 2 bytes"));
        let events = if written {
            vec![
                Ok(ResponseEvent::TextDelta("Done.".into())),
                Ok(ResponseEvent::Done),
            ]
        } else {
            vec![
                Ok(ResponseEvent::ToolCall {
                    index: 0,
                    id: "finisher-0".into(),
                    name: "write_file".into(),
                    arguments: json!({"path": "answer.json", "text": "42"}).to_string(),
                }),
                Ok(ResponseEvent::Done),
            ]
        };
        Ok(Box::pin(futures::stream::iter(events)))
    }
}

/// A continuation that passes without one action of the student's own is no
/// repair: the path had done the work. The chain moves to an earlier state,
/// where the student acts itself.
#[test]
fn a_continuation_without_an_action_of_the_students_own_moves_the_fork_earlier() {
    let scratch = Scratch::new("path-guided-act");
    let finisher = Arc::new(Finisher {
        seen: std::sync::Mutex::new(0),
    });
    let ctx = splinter_orchestrator::Context::new(common::config(&scratch), true)
        .unwrap()
        .with_model(
            student_ref(),
            Model::new(
                Arc::clone(&finisher) as Arc<dyn ModelProvider>,
                "scripted/student",
            ),
        )
        .with_model(
            teacher_ref(),
            Model::new(
                steps("teacher", vec![]) as Arc<dyn ModelProvider>,
                "scripted/teacher",
            ),
        );
    let tree = scratch.0.join("student-tree");
    fs::create_dir_all(&tree).unwrap();
    let trajectory = ctx.block_on(student_trajectory(&tree));
    let runs = vec![practice_run(&ctx, &scratch.0, &trajectory)];
    let checkout = examiner(&scratch.0);
    let failed = splinter_pipelines::bench::split::trials(&ctx).unwrap()[0]
        .experience
        .clone();
    // Reads, the write, a read of the answer: forks after the write, then
    // after the reads.
    let path = privileged_path(&ctx, &failed, false, true, "harbor/sven@5/teacher-27b");
    let set = ctx
        .experiences()
        .put_set(&splinter_store::experiences::ExperienceSet {
            name: "teacher passes".into(),
            members: vec![path],
        })
        .unwrap();
    let teacher = teacher_ref();
    let student_model = student_ref();
    let only = [InterventionKind::PathGuided];
    let sets = [set];
    let mut asked = request(&runs, &checkout, &teacher);
    asked.student = Some(&student_model);
    asked.interventions = &only;
    asked.paths = &sets;
    asked.samples = 2;
    let report = repair(&ctx, &asked).unwrap();
    let outcomes: Vec<_> = report
        .repairs
        .iter()
        .map(|r| (r.outcome, r.reason.clone()))
        .collect();
    assert_eq!(outcomes.len(), 2, "two samples: {outcomes:?}");
    let by = |o| report.repairs.iter().find(|r| r.outcome == o);
    let (Some(ended), Some(acted)) = (by(RepairOutcome::Refused), by(RepairOutcome::Accepted))
    else {
        panic!("one refused, one accepted: {outcomes:?}")
    };
    assert!(ended
        .reason
        .as_deref()
        .is_some_and(|r| r.contains("no action of its own")));
    assert!(ended.experience.is_none());
    assert_eq!(acted.outcome, RepairOutcome::Accepted, "{outcomes:?}");
    // The next state: after the reads (step 3), not the write (step 4).
    let fork = &acted.notes.privileged_path.as_ref().unwrap();
    assert_eq!(fork["fork_after"], json!(3));
    let kept = ctx
        .experiences()
        .get(acted.experience.as_ref().unwrap())
        .unwrap();
    let own = &kept.trajectory.steps[acted.prefix_steps.unwrap()..];
    assert!(own
        .iter()
        .any(|s| s.tool_calls.as_ref().is_some_and(|c| !c.is_empty())));
}

/// A student whose first two continuations write a wrong answer and every
/// later one the right answer.
struct Flaky {
    started: std::sync::Mutex<usize>,
}

#[async_trait::async_trait]
impl ModelProvider for Flaky {
    fn name(&self) -> &str {
        "scripted"
    }
    fn model_name(&self) -> &str {
        "flaky"
    }
    async fn complete(
        &self,
        req: sven_sdk::model::CompletionRequest,
    ) -> anyhow::Result<sven_sdk::model::ResponseStream> {
        use sven_sdk::model::{MessageContent, ResponseEvent};
        let answered = req.messages.iter().any(|m| {
            matches!(&m.content, MessageContent::ToolResult { tool_call_id, .. }
                if tool_call_id.starts_with("flaky"))
        });
        let events = if answered {
            vec![
                Ok(ResponseEvent::TextDelta("Done.".into())),
                Ok(ResponseEvent::Done),
            ]
        } else {
            let mut started = self.started.lock().unwrap();
            *started += 1;
            let text = if *started <= 2 { "7" } else { "42" };
            vec![
                Ok(ResponseEvent::ToolCall {
                    index: 0,
                    id: "flaky-0".into(),
                    name: "write_file".into(),
                    arguments: json!({"path": "answer.json", "text": text}).to_string(),
                }),
                Ok(ResponseEvent::Done),
            ]
        };
        Ok(Box::pin(futures::stream::iter(events)))
    }
}

/// Samples beyond a path's states fork at its states again, the latest first:
/// a state the student failed from once is tried again under a seed of its
/// own.
#[test]
fn samples_beyond_the_paths_states_try_its_states_again() {
    let scratch = Scratch::new("path-guided-again");
    let flaky = Arc::new(Flaky {
        started: std::sync::Mutex::new(0),
    });
    let ctx = splinter_orchestrator::Context::new(common::config(&scratch), true)
        .unwrap()
        .with_model(
            student_ref(),
            Model::new(
                Arc::clone(&flaky) as Arc<dyn ModelProvider>,
                "scripted/student",
            ),
        )
        .with_model(
            teacher_ref(),
            Model::new(
                steps("teacher", vec![]) as Arc<dyn ModelProvider>,
                "scripted/teacher",
            ),
        );
    let tree = scratch.0.join("student-tree");
    fs::create_dir_all(&tree).unwrap();
    let trajectory = ctx.block_on(student_trajectory(&tree));
    let runs = vec![practice_run(&ctx, &scratch.0, &trajectory)];
    let checkout = examiner(&scratch.0);
    let failed = splinter_pipelines::bench::split::trials(&ctx).unwrap()[0]
        .experience
        .clone();
    // Two states: after the reads (step 3) and after the listing (step 2).
    let path = privileged_path(&ctx, &failed, false, false, "harbor/sven@5/teacher-27b");
    let set = ctx
        .experiences()
        .put_set(&splinter_store::experiences::ExperienceSet {
            name: "teacher passes".into(),
            members: vec![path],
        })
        .unwrap();
    let teacher = teacher_ref();
    let student_model = student_ref();
    let only = [InterventionKind::PathGuided];
    let sets = [set];
    let mut asked = request(&runs, &checkout, &teacher);
    asked.student = Some(&student_model);
    asked.interventions = &only;
    asked.paths = &sets;
    asked.samples = 3;
    asked.budget = 3;
    let report = repair(&ctx, &asked).unwrap();
    let mut forks: Vec<_> = report
        .repairs
        .iter()
        .map(|r| {
            (
                r.sample,
                r.outcome,
                r.notes
                    .privileged_path
                    .as_ref()
                    .map(|p| p["fork_after"].clone()),
            )
        })
        .collect();
    forks.sort_by_key(|f| f.0);
    assert_eq!(
        forks,
        [
            (0, RepairOutcome::Rejected, Some(json!(3))),
            (1, RepairOutcome::Rejected, Some(json!(2))),
            (2, RepairOutcome::Accepted, Some(json!(3))),
        ]
    );
}
