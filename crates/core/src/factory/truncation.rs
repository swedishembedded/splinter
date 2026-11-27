// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agent training loops that tell a limit
// reached from a failure of the infrastructure, for its clients. If your team
// needs expertise in reinforcement learning on long agent episodes, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The one classifier of a limit that ended an agent's run early
//! ([`Truncation`]), whatever ran it: a run in this process (its conclusion,
//! or the error it ended with) and a trial on the execution worker (its
//! grade and evidence) are read onto the same causes, so an offline pass and
//! an online run name the same ending the same way.
//!
//! A limit is never an infrastructure failure: running the attempt again
//! meets it again. A run whose conversation outgrew the served context is
//! told by its server's refusal (`context_length_exceeded`, the code brain
//! and OpenAI-compatible servers answer with); a run stopped at its budget,
//! its turn cap or its deadline by how it concluded.

use super::group::Truncation;
use super::trial::{Grade, TrialEvidence, TrialStatus};

/// The error code a server answers a request longer than its context with.
pub const CONTEXT_EXCEEDED: &str = "context_length_exceeded";

/// The limit an error a run ended with names: the served context, when the
/// server refused a request as longer than its context.
#[must_use]
pub fn of_error(error: &str) -> Option<Truncation> {
    error
        .contains(CONTEXT_EXCEEDED)
        .then_some(Truncation::Context)
}

/// The limit a run's conclusion names, as a report names it (`timeout`,
/// `budget_exhausted`, `tool_cap_stopped`); `None` for any other ending.
#[must_use]
pub fn of_conclusion(label: &str) -> Option<Truncation> {
    match label {
        "timeout" => Some(Truncation::Deadline),
        "budget_exhausted" => Some(Truncation::OutputBudget),
        "tool_cap_stopped" => Some(Truncation::Turns),
        _ => None,
    }
}

/// The limit that ended a trial on the execution worker: a refused request
/// for its length, a budget the trial was classified as spending, or a time
/// limit that ended the agent while its model was answering.
#[must_use]
pub fn of_trial(grade: &Grade, evidence: &TrialEvidence) -> Option<Truncation> {
    let refused = evidence
        .requests
        .iter()
        .filter_map(|r| r.error.as_deref())
        .chain(grade.cause.as_ref().map(|c| c.message.as_str()))
        .find_map(of_error);
    if refused.is_some() {
        return refused;
    }
    match grade.status {
        TrialStatus::BudgetExhausted if turn_cap(grade) => Some(Truncation::Turns),
        TrialStatus::BudgetExhausted => Some(Truncation::OutputBudget),
        TrialStatus::Timeout => Some(Truncation::Deadline),
        _ => None,
    }
}

/// Whether a budget cause names the run's turn cap rather than its tokens.
fn turn_cap(grade: &Grade) -> bool {
    grade
        .cause
        .as_ref()
        .is_some_and(|c| c.message.contains("turn"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factory::trial::{grade, GradingPolicy, ModelRequest, RunBudget};
    use serde_json::json;

    fn evidence(result: serde_json::Value) -> TrialEvidence {
        TrialEvidence {
            result,
            ..TrialEvidence::default()
        }
    }

    #[test]
    fn a_run_in_this_process_is_classified_from_its_ending() {
        assert_eq!(
            of_error("server error (context_length_exceeded): 9000 tokens > 8192"),
            Some(Truncation::Context)
        );
        assert_eq!(of_error("connection reset by peer"), None);
        assert_eq!(of_conclusion("timeout"), Some(Truncation::Deadline));
        assert_eq!(
            of_conclusion("budget_exhausted"),
            Some(Truncation::OutputBudget)
        );
        assert_eq!(of_conclusion("tool_cap_stopped"), Some(Truncation::Turns));
        for other in ["success", "agent_error", "repetition_stopped", "cancelled"] {
            assert_eq!(of_conclusion(other), None, "{other}");
        }
    }

    #[test]
    fn a_trial_on_the_worker_is_classified_from_its_grade_and_evidence() {
        let passed = evidence(json!({"verifier_result": {"rewards": {"reward": 1.0}}}));
        let graded = grade(&passed, &GradingPolicy::default());
        assert_eq!(of_trial(&graded, &passed), None);

        let mut refused = passed.clone();
        refused.requests = vec![
            ModelRequest {
                status: Some(200),
                ..ModelRequest::default()
            },
            ModelRequest {
                status: Some(400),
                error: Some("context_length_exceeded: the prompt is 9000 tokens".into()),
                ..ModelRequest::default()
            },
        ];
        let graded = grade(&refused, &GradingPolicy::default());
        assert_eq!(of_trial(&graded, &refused), Some(Truncation::Context));

        let mut spent = passed;
        spent.agent_budget_exhausted = true;
        let policy = GradingPolicy {
            budget: Some(RunBudget {
                completion_tokens: 1000,
                turns: 50,
            }),
            ..GradingPolicy::default()
        };
        let graded = grade(&spent, &policy);
        assert_eq!(graded.status, TrialStatus::BudgetExhausted);
        assert_eq!(of_trial(&graded, &spent), Some(Truncation::OutputBudget));
        spent.agent_budget_exhausted = false;
        spent.turns = Some(51);
        let graded = grade(&spent, &policy);
        assert_eq!(of_trial(&graded, &spent), Some(Truncation::Turns));
    }
}
