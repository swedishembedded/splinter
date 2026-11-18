// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements coding agents whose failures become training
// data by letting a stronger teacher continue the state the student actually
// reached, for its clients. If your team needs expertise in on-policy agent
// data then you can procure our services by sending an email to
// info@swedishembedded.com.

//! A teacher continuing a recorded attempt from one of its steps, in a tree
//! the caller laid out.
//!
//! [`crate::correction`] continues a repository task's attempt from its end,
//! in a workspace this crate materialises. An attempt recorded by an examiner
//! comes with its own workspace (the examiner lays out the pristine tree the
//! agent was given) and its own tool profile, and is best continued from the
//! last good step: the step before the decisive error a diagnosis names.
//! [`continue_from_step`] keeps the student's steps up to that one, has sven's
//! fork replay them into the given tree under the profile the student ran
//! (every replayed observation held to the recorded one, and the tree to the
//! recorded digest when the caller knows it), and lets the teacher continue
//! the same tool loop. The result is the one composed trajectory
//! [`crate::correction::compose`] makes: the student's kept steps as copied
//! context, never supervised, then the teacher's.
//!
//! The model's commands run confined to the tree and a scratch directory of
//! the caller's (sven's Landlock confinement of the `shell` tool, no
//! network); the file tools are held to the tree.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use sven_sdk::atif::{StepOrigin, Trajectory};
use sven_sdk::config::{CommandConfinement, RunProfile};
use sven_sdk::fork::{transcript_digest, Fork, RecordedState};
use sven_sdk::model::time_notice::without_time_notice;
use sven_sdk::{ApprovalPolicy, Engine, Toolset, Turn};

use crate::correction::{compose, trajectory_transcript, Correction, CorrectionError};
use crate::solve::{Model, Solution, SolveOptions};
use crate::work::WITHHELD_TOOLS;

/// The tool profile a recorded attempt ran under, which its fork must offer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Profile {
    /// sven's `compact` profile: `read_file`, `write_file` and `shell`, as
    /// the examiner's agent adapter runs a small model.
    Compact,
    /// sven's default coding tools.
    Default,
}

/// One recorded attempt, to be continued from a step.
pub struct ForkAt<'a> {
    /// The tree to replay into: the pristine workspace the student was
    /// given, laid out by its examiner.
    pub tree: &'a Path,
    /// A directory outside the tree for the commands' `HOME` and `TMPDIR`.
    pub scratch: &'a Path,
    /// The student's recorded trajectory, without its system prompt.
    pub student: &'a Trajectory,
    /// The model that produced it, as its steps are attributed.
    pub student_model: &'a str,
    /// The system prompt the student ran under.
    pub system: &'a str,
    /// The last step of the student kept: the fork continues after it.
    pub keep_through: u64,
    /// The digest of the tree at that step, when the recording has one;
    /// without it the fork is held to its observations alone.
    pub workspace: Option<String>,
    /// The tool profile the student ran under.
    pub profile: Profile,
    /// The observations as the student's model was shown them, by call id,
    /// when its requests were recorded: an agent runner may record a tool's
    /// raw output in the trajectory and show the model the marked and
    /// truncated one (`error: ...`), and the fork replays, and the teacher is
    /// shown, what the model saw.
    pub shown: Option<&'a BTreeMap<String, String>>,
    /// Directories the confined commands may read and run from besides the
    /// system's: the toolchains of their `PATH`.
    pub read_exec: Vec<std::path::PathBuf>,
}

/// The student's steps through `keep_through`, as their own trajectory.
///
/// # Errors
///
/// [`CorrectionError::Trajectory`] when no step is kept, or the last kept
/// step is not one after which a model turn is asked: a fork continues after
/// a user turn or an agent step that observed its calls.
pub fn kept_prefix(student: &Trajectory, keep_through: u64) -> Result<Trajectory, CorrectionError> {
    let mut prefix = student.clone();
    prefix.steps.retain(|step| step.step_id <= keep_through);
    let Some(last) = prefix.steps.last() else {
        return Err(CorrectionError::Trajectory(format!(
            "no step of the attempt is at or before step {keep_through}"
        )));
    };
    let continues = match last.source {
        StepOrigin::User => true,
        StepOrigin::Agent => last.tool_calls.as_ref().is_some_and(|c| !c.is_empty()),
        StepOrigin::System => false,
    };
    if !continues {
        return Err(CorrectionError::Trajectory(format!(
            "step {} is a reply that called no tool: the attempt ended there, so there is no \
             turn to continue after it",
            last.step_id
        )));
    }
    prefix.final_metrics = None;
    Ok(prefix)
}

/// Continues `request`'s attempt after its kept step with `model`, under
/// `options`; see the module documentation.
///
/// # Errors
///
/// [`CorrectionError`] when the prefix cannot be kept or replayed, the fork
/// is refused (an observation or the tree differs from the recording), or
/// the continuation cannot run.
pub async fn continue_from_step(
    request: &ForkAt<'_>,
    model: &Model,
    options: SolveOptions,
) -> Result<Correction, CorrectionError> {
    let prefix = kept_prefix(request.student, request.keep_through)?;
    let mut transcript = trajectory_transcript(&prefix, request.system)?;
    if let Some(shown) = request.shown {
        for turn in &mut transcript {
            if let Turn::ToolResult { call_id, text } = turn {
                // The time-left line a request ends with is the request's,
                // never the history's: the teacher's runtime adds its own.
                if let Some(seen) = shown.get(call_id.as_str()) {
                    *text = without_time_notice(seen).to_string();
                }
            }
        }
    }
    let digest = transcript_digest(&transcript);
    let recorded = match &request.workspace {
        Some(workspace) => RecordedState::new(workspace.clone(), digest),
        None => RecordedState::observed(digest),
    };
    let engine = fork_engine(request, model, &options)?;
    let mut agent = engine
        .fork(&Fork::new(transcript, request.tree, recorded))
        .await?;
    let outcome = agent.continue_with(options.run_options()).await?;
    let continuation = agent.trajectory();
    let trajectory = compose(
        &prefix,
        &continuation,
        request.student_model,
        &model.identity,
    )?;
    let prefix_actions = count_agent(&prefix.steps);
    let teacher_actions = count_agent(&trajectory.steps[prefix.steps.len()..]);
    Ok(Correction {
        solution: Solution {
            trajectory,
            final_output: None,
            conclusion: outcome.conclusion,
            usage: outcome.usage,
        },
        prefix_actions,
        teacher_actions,
    })
}

fn count_agent(steps: &[sven_sdk::atif::TraceStep]) -> u32 {
    u32::try_from(
        steps
            .iter()
            .filter(|s| s.source == StepOrigin::Agent)
            .count(),
    )
    .unwrap_or(u32::MAX)
}

/// The engine the fork runs on: the student's profile, confined to the tree.
fn fork_engine(
    request: &ForkAt<'_>,
    model: &Model,
    options: &SolveOptions,
) -> Result<Engine, CorrectionError> {
    let mut config = options.engine_config();
    config.tools.disabled = WITHHELD_TOOLS.iter().map(|t| (*t).to_string()).collect();
    if request.profile == Profile::Compact {
        config.agent.profile = RunProfile::Compact;
    }
    let tmp = request.scratch.join("tmp");
    std::fs::create_dir_all(&tmp).map_err(|e| {
        CorrectionError::Trajectory(format!("the scratch directory {}: {e}", tmp.display()))
    })?;
    config.tools.confine = Some(CommandConfinement {
        writable: vec![request.tree.to_path_buf(), request.scratch.to_path_buf()],
        read_exec: request.read_exec.clone(),
        tmpdir: Some(tmp),
        connect_ports: Vec::new(),
    });
    let provider = Arc::clone(&model.provider);
    let engine = Engine::builder()
        .config(config)
        .model_provider(crate::system_prompt::UnderSystemPrompt::wrap(
            options.provider(provider),
            request.system,
        ))
        .toolset(Toolset::coding())
        .project_root(request.tree)
        .approvals(ApprovalPolicy::Auto)
        .build()?;
    Ok(engine)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use sven_sdk::atif::{
        AgentProfile, ObservationEntry, StepObservation, ToolInvocation, TraceStep,
    };

    fn student() -> Trajectory {
        let mut t = Trajectory::new("ATIF-v1.7", AgentProfile::new("sven", "1"));
        t.steps
            .push(TraceStep::new(1, StepOrigin::User, "write answer.json"));
        for (id, path) in [(2, "a.csv"), (3, "b.csv")] {
            let mut step = TraceStep::new(id, StepOrigin::Agent, "");
            let call = format!("c{id}");
            step.tool_calls = Some(vec![
                ToolInvocation::new(&call, "read_file").with_arguments(json!({"path": path}))
            ]);
            step.observation = Some(StepObservation::single(ObservationEntry::for_call(
                &call, "L1:x",
            )));
            t.steps.push(step);
        }
        t.steps
            .push(TraceStep::new(4, StepOrigin::Agent, "the answer is 4"));
        t
    }

    #[test]
    fn the_fork_keeps_the_students_steps_through_the_last_good_one() {
        let prefix = kept_prefix(&student(), 2).unwrap();
        assert_eq!(
            prefix.steps.iter().map(|s| s.step_id).collect::<Vec<_>>(),
            [1, 2]
        );
        assert!(kept_prefix(&student(), 1).is_ok(), "after the instruction");
        assert!(
            kept_prefix(&student(), 4).is_err(),
            "the final reply called no tool: nothing follows it"
        );
        assert!(kept_prefix(&student(), 0).is_err());
    }
}
