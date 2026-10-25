// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements coding agents whose failures become training
// data by letting a stronger teacher continue the state the student actually
// reached, for its clients. If your team needs expertise in on-policy agent
// data then you can procure our services by sending an email to
// info@swedishembedded.com.

//! A teacher continuing an episode the student could not finish.
//!
//! [`continue_attempt`] starts from a recorded attempt, not from a clean
//! start: the student's own transcript is replayed into a fresh copy of the
//! task's snapshot by sven's fork, which refuses the fork unless every
//! replayed observation matches the recorded one, the workspace it leaves is
//! the one the student's recorded patch says it reached, and the transcript
//! hashes to what was recorded. The teacher then continues that same tool
//! loop, in the same workspace, under the same system prompt and the steps
//! the episode had left.
//!
//! What comes back is still one ATIF trajectory ([`compose`]), not a second
//! format: the student's recorded steps lead it, marked
//! `is_copied_context` so the sft-step view can never supervise them, and
//! the teacher's steps follow, each carrying the model that produced it in
//! `model_name` and in `extra.correction`. The teacher's patch is the
//! solution's final output, so the task's ordinary verifier grades it in a
//! workspace of its own.
//!
//! What the teacher sees is exactly what the student saw: the fork replays
//! the student's transcript and nothing else. Hidden tests, the reference
//! change and the task's privileged material are never in it.

use std::path::Path;
use std::sync::Arc;

use serde_json::{json, Map, Value};
use splinter_core::experience::{Experience, Task};
use splinter_sandbox::workspace::DEFAULT_EXCLUDED;
use splinter_sandbox::{Patch, ResolvedEnvironment, SandboxError, WorkspaceEnvironment};
use sven_sdk::atif::{AgentProfile, StepOrigin, TraceStep, Trajectory};
use sven_sdk::fork::{transcript_digest, workspace_digest, Fork, ForkError, RecordedState};
use sven_sdk::{ToolCallRecord, Turn};

use crate::solve::{Model, Solution, SolveError, SolveOptions};
use crate::work::{engine, WorkOptions};
use crate::workspace_shell::SandboxShell;

/// The extra key on each composed step naming who produced it.
pub const CORRECTION_KEY: &str = "correction";

/// One student attempt, to be continued by `model`.
pub struct CorrectionRequest<'a> {
    /// The task the student attempted.
    pub task: &'a Task,
    /// The environment its snapshot names.
    pub environment: &'a WorkspaceEnvironment,
    /// The snapshot's archive, as the student's episode laid it out.
    pub archive: &'a Path,
    /// The student's recorded attempt: its trajectory and its patch.
    pub student: &'a Experience,
    /// The system prompt the student ran under.
    pub system: &'a str,
    /// Tool-call rounds the continuation may take: what the student's episode
    /// had left. `None` leaves sven's own limit.
    pub remaining_rounds: Option<u32>,
}

/// A teacher's continuation, composed onto the student's transcript.
pub struct Correction {
    /// The composed trajectory and the teacher's patch.
    pub solution: Solution,
    /// The steps of the transcript the student produced, never supervised.
    pub prefix_actions: u32,
    /// The steps the teacher produced, the correction's training material.
    pub teacher_actions: u32,
}

/// Why a correction could not be made.
#[derive(Debug, thiserror::Error)]
pub enum CorrectionError {
    /// The task is not a change to a repository.
    #[error("the task is not a change to a repository, so there is no workspace to fork")]
    NotWorkspace,
    /// The student's patch does not apply to the snapshot it was made from,
    /// so the workspace it reached cannot be reproduced.
    #[error("the student's patch does not apply to its snapshot: {0}")]
    StudentPatch(String),
    /// The student's transcript holds something this build cannot replay.
    #[error("the student's trajectory cannot be forked: {0}")]
    Trajectory(String),
    /// The continuation exported no steps for the recorded prefix, so the
    /// provenance of its actions cannot be told apart.
    #[error(
        "the continuation holds {got} steps, fewer than the {expected} of the student's trajectory"
    )]
    Provenance {
        /// The student's own trajectory's step count.
        expected: usize,
        /// What the continuation exported.
        got: usize,
    },
    /// The fork was refused.
    #[error(transparent)]
    Fork(#[from] ForkError),
    /// The workspace or sandbox failed.
    #[error(transparent)]
    Sandbox(#[from] SandboxError),
    /// The episode could not be run.
    #[error(transparent)]
    Call(#[from] sven_sdk::CallError),
}

impl From<SolveError> for CorrectionError {
    fn from(error: SolveError) -> Self {
        match error {
            SolveError::Environment(error) => Self::Sandbox(error),
            SolveError::Engine(error) => Self::Call(error),
            other => Self::Trajectory(other.to_string()),
        }
    }
}

/// Continues `request`'s recorded attempt with `model`, under `options`.
///
/// # Errors
///
/// [`CorrectionError`] when the task is not a workspace task, the student's
/// patch cannot reproduce its tree, its transcript cannot be replayed, the
/// fork is refused, or the continuation cannot run. A refused fork is the
/// normal outcome for an attempt whose observations do not replay, and it
/// is reported rather than approximated.
pub async fn continue_attempt(
    request: &CorrectionRequest<'_>,
    model: &Model,
    options: SolveOptions,
) -> Result<Correction, CorrectionError> {
    let offered = ResolvedEnvironment::Workspace(request.environment.clone()).record()?;
    if offered != request.task.environment {
        return Err(CorrectionError::Trajectory(format!(
            "the task records {}, not the offered workspace",
            request
                .task
                .environment
                .snapshot
                .as_ref()
                .map_or_else(|| "no snapshot".to_string(), |digest| digest.to_string())
        )));
    }
    let scratch = tempfile::tempdir().map_err(|source| SandboxError::Io {
        path: "temporary directory".into(),
        source,
    })?;
    let tree = scratch.path().join("tree");
    let baseline = request.environment.materialize(request.archive, &tree)?;

    // The workspace the student's own recording says it reached: its base
    // snapshot with its recorded patch applied, hashed independently of the
    // replay. This is what a fork of the attempt is held to.
    let expected_tree = scratch.path().join("expected");
    request
        .environment
        .materialize(request.archive, &expected_tree)?;
    if let Some(patch) = request.student.final_output.as_deref() {
        if !patch.trim().is_empty() {
            request
                .environment
                .apply(
                    &expected_tree,
                    &Patch {
                        diff: patch.as_bytes().to_vec(),
                        files: Vec::new(),
                    },
                )
                .map_err(|e| CorrectionError::StudentPatch(e.to_string()))?;
        }
    }
    let reached = workspace_digest(&expected_tree).map_err(|source| ForkError::Workspace {
        path: expected_tree.clone(),
        source,
    })?;

    let transcript = transcript_of(request.student, request.system)?;
    let recorded = RecordedState::new(reached, transcript_digest(&transcript));

    let work = WorkOptions::new(SolveOptions {
        max_tool_rounds: request.remaining_rounds.or(options.max_tool_rounds),
        // The continuation must not compact: the record it is projected into has
        // to hold the student's whole transcript, and a compaction replaces it
        // with a summary, which no longer supervises the teacher's action on it.
        compaction: false,
        ..options
    });
    let confined: Arc<dyn sven_sdk::tool::Tool> = Arc::new(SandboxShell::for_episode(
        request.environment.clone(),
        tree.clone(),
        scratch.path(),
    )?);
    let fork_engine = engine(
        &tree,
        model.provider.clone(),
        &work,
        std::slice::from_ref(&confined),
        false,
    )?;
    let fork = Fork::new(transcript, &tree, recorded);
    let mut agent = fork_engine.fork(&fork).await?;
    let outcome = agent.continue_with(work.solve.run_options()).await?;
    let continuation = agent.trajectory();
    let patch = request
        .environment
        .diff_since(&tree, &baseline, &DEFAULT_EXCLUDED)?;

    let trajectory = compose(
        &request.student.trajectory,
        &continuation,
        &request.student.provenance.solver,
        &model.identity,
    )?;
    let prefix_actions = request
        .student
        .trajectory
        .steps
        .iter()
        .filter(|step| step.source == StepOrigin::Agent)
        .count() as u32;
    let teacher_actions = trajectory
        .steps
        .iter()
        .skip(request.student.trajectory.steps.len())
        .filter(|step| step.source == StepOrigin::Agent)
        .count() as u32;
    Ok(Correction {
        solution: Solution {
            trajectory,
            final_output: Some(String::from_utf8_lossy(&patch.diff).into_owned()),
            conclusion: outcome.conclusion,
            usage: outcome.usage,
        },
        prefix_actions,
        teacher_actions,
    })
}

/// The student's recorded trajectory as sven's visible transcript, under
/// `system`.
///
/// The trajectory is the recording of what the model was shown: each step's
/// message, each tool call with its arguments, and each observation that
/// answered it. A message this build cannot represent as text, or an
/// observation that answers no call, is refused rather than dropped - a
/// transcript with a silent hole would fork a conversation the student never
/// had.
///
/// # Errors
///
/// [`CorrectionError::Trajectory`] naming the step.
pub fn transcript_of(student: &Experience, system: &str) -> Result<Vec<Turn>, CorrectionError> {
    let mut turns = vec![Turn::System {
        text: system.to_string(),
    }];
    for step in &student.trajectory.steps {
        let text = step
            .message
            .as_text()
            .ok_or_else(|| {
                CorrectionError::Trajectory(format!(
                    "step {} holds content a transcript cannot replay",
                    step.step_id
                ))
            })?
            .to_string();
        match step.source {
            StepOrigin::System => {
                return Err(CorrectionError::Trajectory(format!(
                    "step {} is a system turn inside the conversation",
                    step.step_id
                )))
            }
            StepOrigin::User => turns.push(Turn::User { text }),
            StepOrigin::Agent => {
                let calls: Vec<ToolCallRecord> = step
                    .tool_calls
                    .iter()
                    .flatten()
                    .map(|call| ToolCallRecord {
                        id: call.tool_call_id.clone(),
                        name: call.function_name.clone(),
                        arguments: call.arguments.to_string(),
                    })
                    .collect();
                turns.push(Turn::Assistant {
                    text,
                    tool_calls: calls,
                });
                for result in step.observation.iter().flat_map(|o| &o.results) {
                    let call_id = result.source_call_id.clone().ok_or_else(|| {
                        CorrectionError::Trajectory(format!(
                            "step {} observes a result that answers no call",
                            step.step_id
                        ))
                    })?;
                    let content: &str = result
                        .content
                        .as_ref()
                        .and_then(|body| body.as_text())
                        .ok_or_else(|| {
                            CorrectionError::Trajectory(format!(
                                "step {} observed content a transcript cannot replay",
                                step.step_id
                            ))
                        })?;
                    turns.push(Turn::ToolResult {
                        call_id,
                        text: content.to_string(),
                    });
                }
            }
        }
    }
    Ok(turns)
}

/// The student's steps followed by the teacher's, as one trajectory.
///
/// `student`'s steps keep their ids, model and observations, are renumbered
/// from one and are marked `is_copied_context`, so no view can supervise
/// them. `continuation` begins from the student's own transcript, so its first
/// as many steps as `student` has are that transcript replayed, already
/// present in `student`; its later steps are the teacher's, marked with the
/// teacher's model in `model_name` and in `extra.correction`. Both trajectories
/// are exports of a recorded conversation by the same derivation and the fork
/// holds the replayed transcript to the student's digest, so the counts are
/// comparable where a count of messages is not: one message per turn, but a
/// step pairs an action with the observation that answered it.
///
/// # Errors
///
/// [`CorrectionError::Provenance`] when `continuation` exported fewer steps
/// than `student` has, which would make its actions unplaceable.
pub fn compose(
    student: &Trajectory,
    continuation: &Trajectory,
    student_model: &str,
    teacher_model: &str,
) -> Result<Trajectory, CorrectionError> {
    let prefix_steps = student.steps.len();
    if continuation.steps.len() < prefix_steps {
        return Err(CorrectionError::Provenance {
            expected: prefix_steps,
            got: continuation.steps.len(),
        });
    }
    let agent = AgentProfile {
        model_name: Some(teacher_model.to_string()),
        ..continuation.agent.clone()
    };
    let mut composed = Trajectory::new(continuation.schema_version.clone(), agent);
    composed.session_id = continuation.session_id.clone();
    composed.notes = student.notes.clone();
    composed.final_metrics = continuation.final_metrics.clone();
    // The episode's own profile (the prompt and bounds the student ran under)
    // stays the trajectory's, with the correction's provenance beside it.
    let mut extra = student.extra.clone().unwrap_or_else(|| json!({}));
    if let Value::Object(fields) = &mut extra {
        fields.insert(
            CORRECTION_KEY.to_string(),
            json!({ "teacher": teacher_model, "prefix_steps": prefix_steps }),
        );
    }
    composed.extra = Some(extra);
    let mut id = 1;
    for step in &student.steps {
        let mut step = step.clone();
        step.step_id = id;
        id += 1;
        step.is_copied_context = Some(true);
        step.model_name = Some(student_model.to_string());
        step.extra = marked(step.extra, "student", student_model);
        composed.steps.push(step);
    }
    for step in continuation.steps.iter().skip(prefix_steps) {
        let mut step = step.clone();
        step.step_id = id;
        id += 1;
        step.is_copied_context = None;
        step.model_name = Some(teacher_model.to_string());
        step.extra = marked(step.extra, "teacher", teacher_model);
        composed.steps.push(step);
    }
    Ok(composed)
}

/// `extra` with the correction's own provenance beside whatever it held.
fn marked(extra: Option<Value>, role: &str, model: &str) -> Option<Value> {
    let mut object = match extra {
        Some(Value::Object(object)) => object,
        Some(other) => {
            let mut object = Map::new();
            object.insert("step".to_string(), other);
            object
        }
        None => Map::new(),
    };
    object.insert(
        CORRECTION_KEY.to_string(),
        json!({ "role": role, "model": model }),
    );
    Some(Value::Object(object))
}

/// What one composed step says about who produced it, for a report.
#[must_use]
pub fn provenance_of(step: &TraceStep) -> Option<(&str, &str)> {
    let correction = step.extra.as_ref()?.get(CORRECTION_KEY)?;
    Some((
        correction.get("role")?.as_str()?,
        correction.get("model")?.as_str()?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use splinter_core::clock::FixedClock;
    use splinter_core::experience::{Environment, Experience, Provenance, Task};
    use sven_sdk::atif::{ObservationEntry, StepObservation, ToolInvocation};
    use sven_sdk::fork::message_count;

    fn call_step(step: u64, id: &str, result: &str) -> TraceStep {
        let mut step = TraceStep::new(step, StepOrigin::Agent, "");
        step.tool_calls = Some(vec![
            ToolInvocation::new(id, "write_file").with_arguments(json!({"path": "a"}))
        ]);
        step.observation = Some(StepObservation::single(ObservationEntry::for_call(
            id, result,
        )));
        step
    }

    fn user_step(step: u64) -> TraceStep {
        TraceStep::new(step, StepOrigin::User, "make it pass")
    }

    #[test]
    fn compose_marks_every_student_action_as_context_and_every_teacher_action_as_supervised() {
        let mut student = Trajectory::new("ATIF-v1.7", AgentProfile::new("scripted", "1"));
        student.steps = vec![user_step(1), call_step(2, "s1", "red")];
        let mut continuation = Trajectory::new("ATIF-v1.7", AgentProfile::new("teacher", "1"));
        continuation.steps = vec![
            user_step(1),
            call_step(2, "s1", "red"),
            call_step(3, "t1", "green"),
            TraceStep::new(4, StepOrigin::Agent, "done"),
        ];

        let composed = compose(&student, &continuation, "student/1", "teacher/1").unwrap();
        let roles: Vec<_> = composed
            .steps
            .iter()
            .map(|step| provenance_of(step).unwrap())
            .collect();
        assert_eq!(
            roles,
            [
                ("student", "student/1"),
                ("student", "student/1"),
                ("teacher", "teacher/1"),
                ("teacher", "teacher/1")
            ]
        );
        assert!(composed.steps[0].is_copied_context == Some(true));
        assert!(composed.steps[1].is_copied_context == Some(true));
        assert!(composed.steps[2].is_copied_context.is_none());
        assert!(composed.steps[3].is_copied_context.is_none());
        assert_eq!(composed.steps[2].model_name.as_deref(), Some("teacher/1"));
        assert_eq!(
            composed.sft_steps().count(),
            2,
            "only the teacher's actions are eligible for training"
        );
    }

    #[test]
    fn the_prefix_is_the_students_steps_not_the_messages_they_export_as() {
        // One step with a tool call exports two messages (the call and the
        // result that answered it), so a two-step student is a three-message
        // transcript. The continuation replays that transcript as two steps
        // and adds one: the step count, not the message count, places them.
        let mut student = Trajectory::new("ATIF-v1.7", AgentProfile::new("scripted", "1"));
        student.steps = vec![user_step(1), call_step(2, "s1", "red")];
        let mut continuation = Trajectory::new("ATIF-v1.7", AgentProfile::new("teacher", "1"));
        continuation.steps = vec![
            user_step(1),
            call_step(2, "s1", "red"),
            call_step(3, "t1", "green"),
        ];
        let composed = compose(&student, &continuation, "student/1", "teacher/1").unwrap();
        assert_eq!(composed.steps.len(), 3);
        assert_eq!(composed.sft_steps().count(), 1);
        assert_eq!(
            provenance_of(&composed.steps[2]),
            Some(("teacher", "teacher/1"))
        );
    }

    #[test]
    fn a_continuation_shorter_than_the_prefix_cannot_be_attributed() {
        let mut student = Trajectory::new("ATIF-v1.7", AgentProfile::new("scripted", "1"));
        student.steps = vec![user_step(1), call_step(2, "s1", "red")];
        let continuation = Trajectory::new("ATIF-v1.7", AgentProfile::new("teacher", "1"));
        assert!(matches!(
            compose(&student, &continuation, "student/1", "teacher/1"),
            Err(CorrectionError::Provenance {
                expected: 2,
                got: 0
            })
        ));
    }

    #[test]
    fn the_visible_transcript_is_the_recorded_steps_under_the_recorded_prompt() {
        let task = Task::new(
            "workspace-change",
            vec![],
            Environment::new(Environment::WORKSPACE, json!({"path": "repo"})),
            "make it pass",
            vec![],
        )
        .unwrap();
        let mut trajectory = Trajectory::new("ATIF-v1.7", AgentProfile::new("scripted", "1"));
        trajectory.steps = vec![user_step(1), call_step(2, "s1", "red")];
        let student = Experience::new(
            task,
            trajectory,
            Some("diff".to_string()),
            Provenance::new("student/1", &FixedClock::new("2026-10-25T00:00:00.000Z")),
        )
        .unwrap();

        let turns = transcript_of(&student, "SYS").unwrap();
        assert_eq!(
            turns,
            vec![
                Turn::System { text: "SYS".into() },
                Turn::User {
                    text: "make it pass".into()
                },
                Turn::Assistant {
                    text: String::new(),
                    tool_calls: vec![ToolCallRecord {
                        id: "s1".into(),
                        name: "write_file".into(),
                        arguments: json!({"path": "a"}).to_string(),
                    }],
                },
                Turn::ToolResult {
                    call_id: "s1".into(),
                    text: "red".into()
                },
            ]
        );
        assert_eq!(
            message_count(&turns),
            3,
            "the system prompt is not a conversation message"
        );
        assert_ne!(
            transcript_digest(&turns),
            transcript_digest(&transcript_of(&student, "OTHER").unwrap()),
            "the prompt is part of what the model saw"
        );
    }
}
