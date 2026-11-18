// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements causal diagnosis of why an agent failed, from
// the evidence its run left, for its clients. If your team needs expertise in
// telling model failures from infrastructure failures before you train on
// them, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Deterministic diagnosis of a failed attempt from the evidence its run left.
//! Pure: no model, no files.
//!
//! The evidence ([`AttemptEvidence`]) is what an examined attempt carries:
//! the examiner's score status (`bench.score/1`) and its cause, the
//! verifier's own reason, whether the agent changed the workspace, the
//! gateway's HTTP statuses, and the agent's ATIF trajectory. The rules run in
//! order, strongest evidence first, and the first that holds decides
//! ([`diagnose`]):
//!
//! 1. **The run's infrastructure.** A status the examiner gives a sandbox,
//!    staging or agent-runtime failure is [`FailureClass::Environment`]; a
//!    model request that failed is [`FailureClass::Gateway`] when the gateway
//!    could not get an answer (HTTP 502 or 504, or the endpoint was
//!    unreachable) and [`FailureClass::ModelServer`] otherwise; a verifier
//!    that crashed is [`FailureClass::Verification`]. A defective task is
//!    [`FailureClass::UnclearTask`] and a capability the agent lacks
//!    [`FailureClass::Unsupported`]. None of these is ever a model class, so
//!    no trajectory pattern can turn one into a model failure.
//! 2. **The security veto.** Reaching for hidden answers is
//!    [`FailureClass::HistoryProbing`], decided at the first call that reads
//!    the repository's history ([`reads_repository_history`]) when one does.
//! 3. **A graded answer.** A verifier that read the answer and found it wrong
//!    is [`FailureClass::WrongContent`]; one that could not read an answer
//!    the agent did write is [`FailureClass::MalformedArtifact`]; both are
//!    decided at the last step that wrote a file.
//! 4. **No answer.** The agent repeated one call, unchanged, three times
//!    ([`FailureClass::Repetition`], decided at the first repeat); its last
//!    actions failed and it stopped there ([`FailureClass::Tool`], decided at
//!    the first of them); a time, turn or token limit ended it
//!    ([`FailureClass::Resource`]); or it stopped without writing the answer
//!    ([`FailureClass::NoAnswer`], decided at the step it stopped).
//!
//! Anything else is [`FailureClass::Unclassified`]. Every result names its
//! rule and carries the spans it read: the step and the excerpt, or the
//! record field.

use std::collections::BTreeMap;

use atif::{StepOrigin, Trajectory};
use splinter_core::diagnosis::{DiagnosticResult, EvidenceSpan};
use splinter_core::outcome::FailureClass;

pub use splinter_core::repository_history::reads_repository_history;

/// The producer name a diagnosis is recorded under.
pub const PRODUCER: &str = "splinter-eval/diagnosis";

/// The version of the rules; a change to any rule raises it.
pub const RULES_VERSION: &str = "1";

/// How many identical calls make a repetition.
pub const REPEATED_CALLS: usize = 3;

/// The verifier's words for an attempt that submitted no work it could grade:
/// no answer file it could read, or (a task graded on the state the agent
/// left in a service) a veto that nothing was done.
const NO_SUBMISSION: [&str; 2] = ["no readable answer", "VETO something_was_done"];

/// What a failed attempt left behind. Every field is optional evidence: what
/// the run did not record is absent.
#[derive(Clone, Copy, Debug, Default)]
pub struct AttemptEvidence<'a> {
    /// The examiner's status (`graded`, `timeout`, `model_error`, ...).
    pub status: Option<&'a str>,
    /// The machine-readable code of the status's cause.
    pub cause_code: Option<&'a str>,
    /// The cause's message.
    pub cause_message: Option<&'a str>,
    /// The verifier's reason for its verdict.
    pub verification_reason: Option<&'a str>,
    /// Whether the agent's resulting workspace differs from the one it was
    /// given (the examiner exported a worktree change).
    pub workspace_changed: Option<bool>,
    /// The HTTP statuses of the attempt's model requests, in order.
    pub gateway_statuses: &'a [u16],
    /// The agent's trajectory.
    pub trajectory: Option<&'a Trajectory>,
}

/// The class of a failed attempt, by the rules of the module documentation.
#[must_use]
pub fn diagnose(evidence: &AttemptEvidence<'_>) -> DiagnosticResult {
    if let Some(result) = infrastructure(evidence) {
        return result;
    }
    let actions = evidence.trajectory.map(actions_of).unwrap_or_default();
    let mut spans = status_spans(evidence);
    if evidence.status == Some("security_violation") {
        let probe = actions
            .iter()
            .find(|a| reads_repository_history(&a.arguments));
        if let Some(probe) = probe {
            spans.push(probe.call_span());
        }
        return DiagnosticResult::new(
            FailureClass::HistoryProbing,
            "security_violation",
            probe.map(|a| a.step),
            spans,
        );
    }
    let ran_out = matches!(evidence.status, Some("timeout" | "budget_exhausted"));
    if !ran_out {
        if let Some(reason) = evidence.verification_reason {
            spans.push(EvidenceSpan::new(None, "verification.reason", reason));
            let last_write = actions.iter().rev().find(|a| a.writes_file());
            if !NO_SUBMISSION.iter().any(|marker| reason.contains(marker)) {
                if let Some(write) = last_write {
                    spans.push(write.call_span());
                }
                return DiagnosticResult::new(
                    FailureClass::WrongContent,
                    "graded_wrong_content",
                    last_write.or(actions.last()).map(|a| a.step),
                    spans,
                );
            }
            if evidence.workspace_changed == Some(true) {
                if let Some(write) = last_write {
                    spans.push(write.call_span());
                }
                return DiagnosticResult::new(
                    FailureClass::MalformedArtifact,
                    "answer_unreadable",
                    last_write.or(actions.last()).map(|a| a.step),
                    spans,
                );
            }
        }
    }
    no_answer(evidence, &actions, spans, ran_out)
}

/// Rule 1: a status the infrastructure or the task decided.
fn infrastructure(evidence: &AttemptEvidence<'_>) -> Option<DiagnosticResult> {
    let status = evidence.status?;
    let class = match status {
        "environment_error" | "agent_error" => FailureClass::Environment,
        "verifier_error" => FailureClass::Verification,
        "invalid_task" => FailureClass::UnclearTask,
        "unsupported" => FailureClass::Unsupported,
        "model_error" => {
            return Some(model_request_failure(evidence));
        }
        _ => return None,
    };
    Some(DiagnosticResult::new(
        class,
        status,
        None,
        status_spans(evidence),
    ))
}

/// A failed model request: the gateway's when it got no answer, the model
/// server's otherwise.
fn model_request_failure(evidence: &AttemptEvidence<'_>) -> DiagnosticResult {
    let mut spans = status_spans(evidence);
    let failed = evidence
        .gateway_statuses
        .iter()
        .rev()
        .find(|s| **s >= 500)
        .copied()
        .or_else(|| evidence.cause_message.and_then(http_status));
    if let Some(code) = failed {
        spans.push(EvidenceSpan::new(None, "gateway.status", &code.to_string()));
    }
    let gateway =
        evidence.cause_code == Some("endpoint_unreachable") || matches!(failed, Some(502 | 504));
    if gateway {
        DiagnosticResult::new(FailureClass::Gateway, "gateway_no_answer", None, spans)
    } else {
        DiagnosticResult::new(FailureClass::ModelServer, "model_server", None, spans)
    }
}

/// The first `HTTP <code>` a message names.
fn http_status(message: &str) -> Option<u16> {
    let at = message.find("HTTP ")?;
    message[at + 5..].get(..3)?.parse().ok()
}

/// Rule 4: no answer the verifier could read.
fn no_answer(
    evidence: &AttemptEvidence<'_>,
    actions: &[Action],
    mut spans: Vec<EvidenceSpan>,
    ran_out: bool,
) -> DiagnosticResult {
    if let Some((first, repeat)) = repetition(actions) {
        spans.push(first.call_span());
        spans.push(repeat.call_span());
        return DiagnosticResult::new(
            FailureClass::Repetition,
            "repeated_call",
            Some(repeat.step),
            spans,
        );
    }
    let failing: Vec<&Action> = actions.iter().rev().take_while(|a| a.failed).collect();
    if let Some(first) = failing.last() {
        spans.push(first.call_span());
        if let Some(text) = &first.observation {
            spans.push(EvidenceSpan::new(Some(first.step), "observation", text));
        }
        return DiagnosticResult::new(
            FailureClass::Tool,
            "stopped_on_failing_calls",
            Some(first.step),
            spans,
        );
    }
    let limit = evidence
        .trajectory
        .and_then(|t| t.final_metrics.as_ref())
        .and_then(|m| m.extra.as_ref())
        .and_then(|extra| extra.get("outcome"))
        .and_then(serde_json::Value::as_str)
        .filter(|outcome| {
            ["timeout", "limit", "budget", "cap"]
                .iter()
                .any(|word| outcome.contains(word))
        });
    if ran_out || limit.is_some() {
        if let Some(outcome) = limit {
            spans.push(EvidenceSpan::new(None, "final_metrics.outcome", outcome));
        }
        return DiagnosticResult::new(FailureClass::Resource, "limit_reached", None, spans);
    }
    let Some(trajectory) = evidence.trajectory else {
        return DiagnosticResult::new(FailureClass::Unclassified, "no_evidence", None, spans);
    };
    let stop = trajectory
        .steps
        .iter()
        .rev()
        .find(|s| s.source == StepOrigin::Agent);
    match stop {
        Some(step) if evidence.verification_reason.is_some() => {
            spans.push(EvidenceSpan::new(
                Some(step.step_id),
                "message",
                step.message.as_text().unwrap_or_default(),
            ));
            DiagnosticResult::new(
                FailureClass::NoAnswer,
                "stopped_without_answer",
                Some(step.step_id),
                spans,
            )
        }
        _ => DiagnosticResult::new(FailureClass::Unclassified, "no_rule", None, spans),
    }
}

/// The record fields every result shows: the status and its cause.
fn status_spans(evidence: &AttemptEvidence<'_>) -> Vec<EvidenceSpan> {
    let mut spans = Vec::new();
    if let Some(status) = evidence.status {
        spans.push(EvidenceSpan::new(None, "score.status", status));
    }
    if let Some(message) = evidence.cause_message {
        spans.push(EvidenceSpan::new(None, "score.cause", message));
    }
    spans
}

/// One tool call of the trajectory and what answered it.
#[derive(Debug)]
struct Action {
    step: u64,
    tool: String,
    /// The arguments as canonical JSON text.
    arguments: String,
    observation: Option<String>,
    failed: bool,
}

impl Action {
    /// Whether the call wrote a file: `write_file`, or a command that
    /// redirects into one or writes it from a script.
    fn writes_file(&self) -> bool {
        if self.failed {
            return false;
        }
        self.tool == "write_file"
            || (self.tool == "shell"
                && [
                    "> ",
                    ">>",
                    "| tee ",
                    "json.dump(",
                    "'w')",
                    "\"w\")",
                    ".write_text(",
                ]
                .iter()
                .any(|marker| self.arguments.contains(marker)))
    }

    fn call_span(&self) -> EvidenceSpan {
        EvidenceSpan::new(
            Some(self.step),
            "tool_call",
            &format!("{} {}", self.tool, self.arguments),
        )
    }
}

/// Every tool call in the order the model made them.
fn actions_of(trajectory: &Trajectory) -> Vec<Action> {
    let mut actions = Vec::new();
    for step in &trajectory.steps {
        let results: BTreeMap<&str, &str> = step
            .observation
            .iter()
            .flat_map(|o| &o.results)
            .filter_map(|r| Some((r.source_call_id.as_deref()?, r.content.as_ref()?.as_text()?)))
            .collect();
        for call in step.tool_calls.iter().flatten() {
            let observation = results
                .get(call.tool_call_id.as_str())
                .map(|s| (*s).to_string());
            actions.push(Action {
                step: step.step_id,
                tool: call.function_name.clone(),
                arguments: call.arguments.to_string(),
                failed: observation.as_deref().is_some_and(failed_observation),
                observation,
            });
        }
    }
    actions
}

/// Whether an observation says the call failed: the tool's own error, a
/// missing file, or a command that exited non-zero.
#[must_use]
pub fn failed_observation(text: &str) -> bool {
    let lower = text.trim_start().to_ascii_lowercase();
    if lower.starts_with("error") || lower.starts_with("read error") {
        return true;
    }
    text.rsplit_once("Exit: ")
        .and_then(|(_, tail)| tail.split(|c: char| !c.is_ascii_digit()).next())
        .and_then(|code| code.parse::<i64>().ok())
        .is_some_and(|code| code != 0)
}

/// The first call made [`REPEATED_CALLS`] times unchanged: its first
/// occurrence and its first repeat.
fn repetition(actions: &[Action]) -> Option<(&Action, &Action)> {
    let mut seen: BTreeMap<(&str, &str), Vec<&Action>> = BTreeMap::new();
    for action in actions {
        let key = (action.tool.as_str(), action.arguments.as_str());
        let calls = seen.entry(key).or_default();
        calls.push(action);
        if calls.len() >= REPEATED_CALLS {
            return Some((calls[0], calls[1]));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use atif::{AgentProfile, ObservationEntry, StepObservation, ToolInvocation, TraceStep};
    use serde_json::json;

    fn trajectory(calls: &[(&str, serde_json::Value, &str)], last: &str) -> Trajectory {
        let mut t = Trajectory::new("ATIF-v1.7", AgentProfile::new("sven", "1"));
        t.steps.push(TraceStep::new(1, StepOrigin::User, "do it"));
        for (i, (tool, args, result)) in calls.iter().enumerate() {
            let id = format!("c{i}");
            let mut step = TraceStep::new(i as u64 + 2, StepOrigin::Agent, "");
            step.tool_calls = Some(vec![
                ToolInvocation::new(&id, *tool).with_arguments(args.clone())
            ]);
            step.observation = Some(StepObservation::single(ObservationEntry::for_call(
                &id, *result,
            )));
            t.steps.push(step);
        }
        t.steps.push(TraceStep::new(
            calls.len() as u64 + 2,
            StepOrigin::Agent,
            last,
        ));
        t
    }

    #[test]
    fn an_infrastructure_status_is_never_a_model_class_whatever_the_trajectory_shows() {
        let t = trajectory(
            &[(
                "read_file",
                json!({"path": "x"}),
                "read error: No such file or directory (os error 2)",
            )],
            "",
        );
        for (status, cause, class) in [
            ("environment_error", None, FailureClass::Environment),
            ("agent_error", None, FailureClass::Environment),
            ("verifier_error", None, FailureClass::Verification),
            (
                "model_error",
                Some("the trial's last model request got HTTP 504: timeout"),
                FailureClass::Gateway,
            ),
            (
                "model_error",
                Some("the trial's last model request got HTTP 500: boom"),
                FailureClass::ModelServer,
            ),
        ] {
            let result = diagnose(&AttemptEvidence {
                status: Some(status),
                cause_message: cause,
                verification_reason: Some("no readable answer"),
                trajectory: Some(&t),
                ..AttemptEvidence::default()
            });
            assert_eq!(result.class, class, "{status} {cause:?}");
            assert!(!result.class.is_model());
        }
    }

    #[test]
    fn a_failure_to_read_the_answer_is_malformed_only_when_one_was_written() {
        let wrote = trajectory(
            &[(
                "write_file",
                json!({"path": "answer.json", "text": "{x"}),
                "wrote 2 bytes",
            )],
            "done",
        );
        let evidence = |changed| AttemptEvidence {
            status: Some("graded"),
            verification_reason: Some("no readable answer; functional_correctness 0.0000"),
            workspace_changed: Some(changed),
            trajectory: Some(&wrote),
            ..AttemptEvidence::default()
        };
        let malformed = diagnose(&evidence(true));
        assert_eq!(malformed.class, FailureClass::MalformedArtifact);
        assert_eq!(malformed.decisive_step, Some(2));
        let silent = trajectory(
            &[("read_file", json!({"path": "a.csv"}), "L1:x")],
            "the answer is 4",
        );
        let none = diagnose(&AttemptEvidence {
            trajectory: Some(&silent),
            workspace_changed: Some(false),
            ..evidence(false)
        });
        assert_eq!(none.class, FailureClass::NoAnswer);
        assert_eq!(none.decisive_step, Some(3));
        assert_eq!(none.last_good_step(), Some(2));
    }

    #[test]
    fn repetition_and_failing_calls_are_decided_at_the_step_they_began() {
        let read = ("read_file", json!({"path": "a"}), "L1:a");
        let t = trajectory(&[read.clone(), read.clone(), read], "");
        let result = diagnose(&AttemptEvidence {
            status: Some("timeout"),
            trajectory: Some(&t),
            ..AttemptEvidence::default()
        });
        assert_eq!(
            (result.class, result.decisive_step),
            (FailureClass::Repetition, Some(3))
        );

        let t = trajectory(
            &[
                ("read_file", json!({"path": "a"}), "L1:a"),
                (
                    "read_file",
                    json!({"path": "b"}),
                    "read error: No such file or directory (os error 2)",
                ),
                (
                    "shell",
                    json!({"shell_command": "cat c"}),
                    "cat: c: No such file\n\nExit: 1 | Duration: 0.0s",
                ),
            ],
            "I could not find it",
        );
        let result = diagnose(&AttemptEvidence {
            status: Some("graded"),
            verification_reason: Some("no readable answer"),
            workspace_changed: Some(false),
            trajectory: Some(&t),
            ..AttemptEvidence::default()
        });
        assert_eq!(
            (result.class, result.decisive_step),
            (FailureClass::Tool, Some(3))
        );
        assert!(result
            .evidence
            .iter()
            .any(|s| s.step == Some(3) && s.field == "observation"));
    }

    #[test]
    fn the_security_veto_is_decided_at_the_first_history_read() {
        let t = trajectory(
            &[
                (
                    "shell",
                    json!({"shell_command": "ls"}),
                    "a\n\nExit: 0 | Duration: 0.0s",
                ),
                (
                    "shell",
                    json!({"shell_command": "git log --all"}),
                    "commit 1\n\nExit: 0 | Duration: 0.0s",
                ),
            ],
            "",
        );
        let result = diagnose(&AttemptEvidence {
            status: Some("security_violation"),
            trajectory: Some(&t),
            ..AttemptEvidence::default()
        });
        assert_eq!(
            (result.class, result.decisive_step),
            (FailureClass::HistoryProbing, Some(3))
        );
    }

    #[test]
    fn an_exit_status_decides_whether_a_command_failed() {
        assert!(failed_observation("x\n\nExit: 1 | Duration: 0.1s"));
        assert!(!failed_observation("x\n\nExit: 0 | Duration: 0.1s"));
        assert!(failed_observation("error: unknown argument"));
        assert!(!failed_observation("L1:Exit: none"));
    }
}
