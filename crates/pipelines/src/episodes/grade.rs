// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that turn practice task folders
// into graded, replayable agent experience for its clients. If your team
// needs expertise in on-policy experience collection for agent training you
// can procure our services by sending an email to info@swedishembedded.com.

//! What an attempt's verdicts and trajectory say: its outcome, its usage and
//! its policy trace.

use atif::{StepOrigin, Trajectory};
use splinter_core::annotation::{AnnotationBody, Outcome};
use splinter_core::experience::Task;
use splinter_core::outcome::AttemptReport;
use splinter_eval::exposure::EXPOSURE_PRODUCER;
use splinter_eval::verifiers::workspace::{WorkspaceCheck, WORKSPACE_PRODUCER};
use splinter_eval::verifiers::Verification;
use splinter_knowledge::tasks::workspace::{is_harbor_runner, NO_REWARD_EXIT};

use super::{EpisodeOutcome, EpisodeUsage, TraceSummary};
use crate::served::{RolloutTrace, TraceSequence};

/// The exit status of the Harbor runner when it could not set itself up.
const RUNNER_SETUP_EXIT: i64 = 2;

/// How the verdicts of `verification` grade an attempt at `task`.
pub(super) fn grade(task: &Task, verification: &Verification) -> EpisodeOutcome {
    let verdict = |producer: &str| {
        verification
            .annotations
            .iter()
            .filter(|note| note.producer.name == producer)
            .find_map(|note| match &note.body {
                AnnotationBody::Verdict {
                    outcome, evidence, ..
                } => Some((*outcome, evidence)),
                _ => None,
            })
    };
    if matches!(verdict(EXPOSURE_PRODUCER), Some((Outcome::Fail, _))) {
        return EpisodeOutcome::Exposed;
    }
    let Some((outcome, evidence)) = verdict(WORKSPACE_PRODUCER) else {
        let errors: Vec<String> = verification
            .errors
            .iter()
            .map(|(producer, error)| format!("{}: {error}", producer.name))
            .collect();
        return EpisodeOutcome::VerifierError {
            detail: if errors.is_empty() {
                "the workspace verifier gave no verdict".into()
            } else {
                errors.join("; ")
            },
        };
    };
    match outcome {
        Outcome::Pass => EpisodeOutcome::Passed,
        Outcome::Abstain => EpisodeOutcome::VerifierError {
            detail: format!(
                "the workspace verifier abstained: {}",
                evidence["abstained"]
            ),
        },
        Outcome::Fail => failed(task, evidence),
    }
}

/// Whether a failed check failed the agent's change or failed to grade it.
fn failed(task: &Task, evidence: &serde_json::Value) -> EpisodeOutcome {
    // A change set aside before any command ran (none, empty, not applying,
    // touching what it may not) is the agent's.
    if evidence.get("refused").is_some() {
        return EpisodeOutcome::Failed;
    }
    let commands = evidence["commands"].as_array().cloned().unwrap_or_default();
    if commands.iter().any(|c| c["timed_out"] == true) {
        return EpisodeOutcome::VerifierTimeout;
    }
    if let Some(signal) = commands.iter().find_map(|c| c["signal"].as_i64()) {
        return EpisodeOutcome::VerifierError {
            detail: format!("a check was killed by signal {signal}"),
        };
    }
    // A task graded by its own Harbor verifier: the runner says when the
    // script left no reward, or when it could not set itself up.
    let harbor = WorkspaceCheck::of_task(task)
        .ok()
        .flatten()
        .is_some_and(|(_, check)| check.fail_to_pass.iter().any(is_harbor_runner));
    if harbor {
        let exit = commands
            .iter()
            .find(|c| c["phase"] == "fail_to_pass")
            .and_then(|c| c["exit_code"].as_i64());
        if exit == Some(i64::from(NO_REWARD_EXIT)) {
            return EpisodeOutcome::VerifierError {
                detail: "the task's verifier left no reward".into(),
            };
        }
        if exit == Some(RUNNER_SETUP_EXIT) {
            return EpisodeOutcome::VerifierError {
                detail: "the task's verifier could not be set up".into(),
            };
        }
    }
    EpisodeOutcome::Failed
}

/// The model calls of `trajectory`, each with the ids and log-probabilities
/// the provider returned, when it returned them all: a call without them,
/// or with ids that are not token ids, leaves the trace unprovable.
pub(super) fn rollout_trace(trajectory: &Trajectory) -> RolloutTrace {
    let mut sequences = Vec::new();
    let mut provable = true;
    for step in agent_steps(trajectory) {
        match traced(step) {
            Some(sequence) => sequences.push(sequence),
            None => provable = false,
        }
    }
    RolloutTrace {
        provable: provable && !sequences.is_empty(),
        sequences,
    }
}

/// The steps of `trajectory` that are model calls.
fn agent_steps(trajectory: &Trajectory) -> impl Iterator<Item = &atif::TraceStep> {
    trajectory
        .steps
        .iter()
        .filter(|step| step.source == StepOrigin::Agent)
}

/// The call `step` as sampled, when the provider returned all of it.
fn traced(step: &atif::TraceStep) -> Option<TraceSequence> {
    let metrics = step.metrics.as_ref()?;
    let ids = |ids: &Option<Vec<i64>>| -> Option<Vec<u32>> {
        ids.as_ref()?
            .iter()
            .map(|&id| u32::try_from(id).ok())
            .collect()
    };
    let prompt = ids(&metrics.prompt_token_ids)?;
    let generated = ids(&metrics.completion_token_ids)?;
    // The trace keeps single precision: what a trainer reads.
    #[allow(clippy::cast_possible_truncation)]
    let logprobs: Vec<f32> = metrics
        .logprobs
        .as_ref()?
        .iter()
        .map(|&l| l as f32)
        .collect();
    let whole = !prompt.is_empty() && !generated.is_empty() && logprobs.len() == generated.len();
    whole.then_some(TraceSequence {
        prompt_token_ids: prompt,
        generation_token_ids: generated,
        generation_log_probs: logprobs,
        truncated: false,
    })
}

/// The policy trace of `trajectory`, summarized.
pub(super) fn trace_summary(trajectory: &Trajectory) -> TraceSummary {
    let calls = agent_steps(trajectory).count();
    let traced = agent_steps(trajectory)
        .filter(|step| traced(step).is_some())
        .count();
    TraceSummary {
        calls,
        traced,
        provable: calls > 0 && traced == calls,
    }
}

/// What an attempt used: its model calls and tokens from the trajectory,
/// its tool calls and time from the attempt report.
pub(super) fn usage(trajectory: &Trajectory, report: &AttemptReport) -> EpisodeUsage {
    let totals = trajectory.final_metrics.as_ref();
    let summed = |pick: fn(&atif::StepMetrics) -> Option<u64>| -> Option<u64> {
        let counts: Option<Vec<u64>> = agent_steps(trajectory)
            .map(|step| step.metrics.as_ref().and_then(pick))
            .collect();
        counts.filter(|c| !c.is_empty()).map(|c| c.iter().sum())
    };
    EpisodeUsage {
        turns: agent_steps(trajectory).count() as u64,
        tool_calls: report.tool_calls,
        prompt_tokens: totals
            .and_then(|t| t.total_prompt_tokens)
            .or_else(|| summed(|m| m.prompt_tokens)),
        completion_tokens: totals
            .and_then(|t| t.total_completion_tokens)
            .or_else(|| summed(|m| m.completion_tokens)),
        wall_seconds: report.wall_seconds,
    }
}
