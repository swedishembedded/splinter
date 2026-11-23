// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agent evaluations in which an
// infrastructure failure is never mistaken for the agent's failure, for its
// clients. If your team needs expertise in trustworthy benchmark scoring,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! How one executed trial ended: the one classifier every consumer uses.
//!
//! A trial is classified from its evidence ([`TrialEvidence`]): the
//! execution backend's result record, the verifier's verification record,
//! the model requests the gateway logged and what the agent reported about
//! itself. Classification ([`TrialStatus`] and its [`Cause`]) is separate
//! from the score: a trial can score 0 and still have ended on a time limit.
//! [`grade`] combines both under a [`GradingPolicy`] into a [`Grade`], which
//! says whether the trial is counted in pass rates and whether it passed.
//!
//! The rules, in the order they are applied:
//!
//! 1. A leak of hidden material is a security violation, whatever the score.
//! 2. A reasoning loop the agent reported ended the run: a counted fail.
//! 3. Under a run budget, stopping at it, or spending past it, is a counted
//!    fail (`run_budget`).
//! 4. A verifier that reported an environment error is an environment error.
//! 5. Model requests that all failed with a 5xx (or, with no gateway log, an
//!    agent request log of nothing but errors) are a model error: the agent
//!    never completed a model turn.
//! 6. A last model request that failed with a 5xx, or that the endpoint
//!    refused (401, 403, 404: the run named a model or key it does not
//!    serve), is a model error; so are requests the endpoint rejected
//!    (4xx) of which none was answered: the endpoint does not take what the
//!    run's agent sends, and the agent never had a model turn. Once one was
//!    answered, a rejected request is the agent's own doing.
//! 7. An exception the backend recorded is mapped by its type (time limits,
//!    sandbox, verifier, model API, budget, unsupported capability); a
//!    non-zero agent exit or a safety refusal stays graded when the verifier
//!    rewarded the work.
//! 8. A time limit the agent reported is a timeout (an infrastructure
//!    timeout under a run budget, whose own bounds are the measure).
//! 9. A malformed or missing reward is a verifier error; anything else is
//!    graded.
//!
//! This module performs no I/O; the caller reads the files.

use std::collections::BTreeMap;
use std::sync::LazyLock;

use regex::Regex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The reward a pass needs: full credit, as every verifier's own verdict.
pub const FULL_CREDIT: f64 = 1.0;

/// The cause code of a trial that ran out of its run's budget: the one
/// budget exhaustion that is counted.
pub const RUN_BUDGET: &str = "run_budget";

/// How a trial ended.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum TrialStatus {
    /// The verifier judged the agent's work.
    Graded,
    /// The task itself is defective (no tests, a verifier that refuses its
    /// own truth).
    InvalidTask,
    /// Sandbox, staging, gateway, or a simulator the verifier needs failed.
    EnvironmentError,
    /// The model endpoint failed: refused, 5xx, an empty or unusable reply.
    ModelError,
    /// The agent process crashed for a reason other than its model.
    AgentError,
    /// The verifier crashed, timed out, or left no reward.
    VerifierError,
    /// A time limit ended the agent; the task's policy says whether it is
    /// graded as it is.
    Timeout,
    /// A token, context or usage budget ended the agent; the run's own budget
    /// is a counted fail.
    BudgetExhausted,
    /// The model's reasoning looped and ended the run: a counted fail.
    ReasoningLoop,
    /// A wall-clock limit ended the agent of a budgeted run: the
    /// infrastructure's, never graded.
    InfraTimeout,
    /// A sandbox escape or a hidden-answer leak: a hard veto.
    SecurityViolation,
    /// The agent or model cannot take a capability the task needs (coverage,
    /// not a 0).
    Unsupported,
    /// The trial's trace could not be read or failed validation; its raw
    /// evidence is kept and it is quarantined from training. Set by trial
    /// admission, never by [`classify`].
    InvalidTrace,
}

impl TrialStatus {
    /// Whether the run's infrastructure, not the agent's skill, decided the
    /// trial. Such a trial is never a training negative.
    #[must_use]
    pub fn is_infrastructure(self) -> bool {
        matches!(
            self,
            Self::EnvironmentError
                | Self::ModelError
                | Self::AgentError
                | Self::VerifierError
                | Self::InvalidTask
                | Self::InfraTimeout
        )
    }

    /// Whether a fresh physical attempt at the same logical slot may change
    /// the outcome: an infrastructure failure other than a defective task.
    #[must_use]
    pub fn is_retryable(self) -> bool {
        self.is_infrastructure() && self != Self::InvalidTask
    }
}

/// Why a trial is not plainly graded: a short machine-readable code and a
/// message for a person.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Cause {
    /// Machine-readable code (`agent_timeout`, `server_error`, an exception
    /// type, ...).
    pub code: String,
    /// What happened, for a person.
    pub message: String,
    /// The task's timeout policy, on a timeout.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy: Option<TimeoutPolicy>,
}

impl Cause {
    fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            policy: None,
        }
    }
}

/// Whether a trial a time limit ended is graded on what the agent left, or
/// excluded from pass rates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TimeoutPolicy {
    /// Graded as it stands.
    #[default]
    Graded,
    /// Left out of the denominators.
    Excluded,
}

/// The bounds a budgeted run gives each trial.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RunBudget {
    /// Completion tokens a trial may spend.
    pub completion_tokens: u64,
    /// Agent turns a trial may take.
    pub turns: u64,
}

impl RunBudget {
    /// Why a trial that spent `completion_tokens` in `turns` is past the
    /// budget; `None` when it is within it.
    #[must_use]
    pub fn overrun(&self, completion_tokens: u64, turns: Option<u64>) -> Option<String> {
        if completion_tokens > self.completion_tokens {
            return Some(format!(
                "spent {completion_tokens} completion tokens of a budget of {}",
                self.completion_tokens
            ));
        }
        match turns {
            Some(turns) if turns > self.turns => {
                Some(format!("took {turns} turns of a budget of {}", self.turns))
            }
            _ => None,
        }
    }
}

/// One model request the gateway logged for a trial.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ModelRequest {
    /// The HTTP status the request ended with; `None` when none was recorded.
    #[serde(default)]
    pub status: Option<u16>,
    /// The error the request ended with.
    #[serde(default)]
    pub error: Option<String>,
    /// Prompt tokens processed.
    #[serde(default)]
    pub prompt_tokens: Option<u64>,
    /// Completion tokens generated.
    #[serde(default)]
    pub completion_tokens: Option<u64>,
    /// Prompt tokens served from cache.
    #[serde(default)]
    pub cached_tokens: Option<u64>,
}

/// The facts a trial is classified from.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TrialEvidence {
    /// The execution backend's result record (Harbor's trial `result.json`).
    pub result: Value,
    /// The verifier's verification record, when it wrote one.
    #[serde(default)]
    pub verification: Option<Value>,
    /// The agent reported that a time limit ended it.
    #[serde(default)]
    pub agent_timed_out: bool,
    /// The agent reported stopping at the run's budget.
    #[serde(default)]
    pub agent_budget_exhausted: bool,
    /// The agent reported that a reasoning loop ended the run.
    #[serde(default)]
    pub agent_reasoning_loop: bool,
    /// What the leak audit found, one line each.
    #[serde(default)]
    pub leaks: Vec<String>,
    /// The agent's turns, when the trace records them.
    #[serde(default)]
    pub turns: Option<u64>,
    /// The model responses the agent's own request log records.
    #[serde(default)]
    pub model_responses: Option<u64>,
    /// How many of those carry an error instead of an answer.
    #[serde(default)]
    pub model_errors: Option<u64>,
    /// The model requests the gateway logged, in order.
    #[serde(default)]
    pub requests: Vec<ModelRequest>,
}

/// What grading needs beside the evidence.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct GradingPolicy {
    /// The task's timeout policy.
    #[serde(default)]
    pub timeout: TimeoutPolicy,
    /// The run's budget; `None` for an unbudgeted run.
    #[serde(default)]
    pub budget: Option<RunBudget>,
}

/// A classified, scored trial.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Grade {
    /// How the trial ended.
    pub status: TrialStatus,
    /// Why, when it is not plainly graded.
    pub cause: Option<Cause>,
    /// In pass rates: graded, a timeout its task grades, or a counted fail.
    pub counted: bool,
    /// The verifier's reward; `None` when it left none or the status makes it
    /// meaningless.
    pub score: Option<f64>,
    /// `score >= FULL_CREDIT` with no veto; `None` when not counted (`false`
    /// under a veto).
    pub passed: Option<bool>,
    /// The verifier's named partial-credit metrics.
    pub metrics: BTreeMap<String, f64>,
    /// Hard vetoes that force `passed = false` whatever the score.
    pub vetoes: Vec<String>,
}

// Harbor's exception types and Python's transport errors, by the status
// they mean.
const EXCEPTIONS: &[(&str, TrialStatus)] = &[
    ("AgentTimeoutError", TrialStatus::Timeout),
    (
        "EnvironmentStartTimeoutError",
        TrialStatus::EnvironmentError,
    ),
    ("AgentSetupTimeoutError", TrialStatus::EnvironmentError),
    ("SandboxBuildFailedError", TrialStatus::EnvironmentError),
    ("HealthcheckError", TrialStatus::EnvironmentError),
    ("MemoryLimitExceededError", TrialStatus::EnvironmentError),
    ("AgentAuthenticationError", TrialStatus::EnvironmentError),
    ("CancelledError", TrialStatus::EnvironmentError),
    ("VerifierTimeoutError", TrialStatus::VerifierError),
    ("RewardFileNotFoundError", TrialStatus::VerifierError),
    ("RewardFileEmptyError", TrialStatus::VerifierError),
    ("VerifierOutputParseError", TrialStatus::VerifierError),
    ("DownloadVerifierDirError", TrialStatus::VerifierError),
    ("AddTestsDirError", TrialStatus::InvalidTask),
    ("ApiInternalServerError", TrialStatus::ModelError),
    ("ApiOverloadedError", TrialStatus::ModelError),
    ("ApiConnectionClosedError", TrialStatus::ModelError),
    ("ApiResponseStalledError", TrialStatus::ModelError),
    ("ApiRateLimitError", TrialStatus::ModelError),
    ("ApiProviderResourceNotFoundError", TrialStatus::ModelError),
    ("UnknownApiError", TrialStatus::ModelError),
    ("ModelNotFoundError", TrialStatus::ModelError),
    ("NetworkConnectionError", TrialStatus::ModelError),
    ("URLError", TrialStatus::ModelError),
    ("ConnectionError", TrialStatus::ModelError),
    ("ConnectionRefusedError", TrialStatus::ModelError),
    ("ConnectionResetError", TrialStatus::ModelError),
    ("RemoteDisconnected", TrialStatus::ModelError),
    ("IncompleteRead", TrialStatus::ModelError),
    ("ApiUsageLimitError", TrialStatus::BudgetExhausted),
    ("ContextWindowExceededError", TrialStatus::BudgetExhausted),
    ("OutputTokenExceededError", TrialStatus::BudgetExhausted),
    (
        "ServiceOperationsUnsupportedError",
        TrialStatus::Unsupported,
    ),
];

// Exceptions that end the agent but leave its work for the verifier.
const GRADED_WHEN_REWARDED: &[&str] = &["NonZeroAgentExitCodeError", "AgentSafetyRefusalError"];

// Statuses with which an endpoint refuses what the run named rather than what
// the agent sent.
const ENDPOINT_REFUSALS: &[u16] = &[401, 403, 404];

// The patterns are literals a test compiles; a failure is a bug here.
#[allow(clippy::expect_used)]
static UNREACHABLE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"Connection refused|Connection reset|\[Errno (?:111|104)\]|RemoteDisconnected|ECONNREFUSED")
        .expect("a valid pattern")
});
#[allow(clippy::expect_used)]
static UNSUPPORTED: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\bdoes not support\b").expect("a valid pattern"));

/// One reward value: a finite number in `[0, 1]`, given as a number or text.
///
/// # Errors
/// A boolean, text that is not a number, or a number outside `[0, 1]`.
pub fn parse_reward(value: &Value) -> Result<f64, String> {
    let reward = match value {
        Value::Number(number) => number.as_f64(),
        Value::String(text) => text.trim().parse::<f64>().ok(),
        _ => None,
    }
    .ok_or_else(|| format!("reward is not a number: {value}"))?;
    if !reward.is_finite() || !(0.0..=1.0).contains(&reward) {
        return Err(format!(
            "reward must be a finite number in [0, 1], got {value}"
        ));
    }
    Ok(reward)
}

/// The scalar reward of a rewards object: its `reward` key, else its only
/// value. `Ok(None)` when there is no verdict, or several keys and none
/// named `reward` (picking one would be a guess).
///
/// # Errors
/// The chosen value is not a valid reward.
pub fn reward_of(rewards: Option<&serde_json::Map<String, Value>>) -> Result<Option<f64>, String> {
    let Some(rewards) = rewards.filter(|r| !r.is_empty()) else {
        return Ok(None);
    };
    if let Some(value) = rewards.get("reward") {
        return parse_reward(value).map(Some);
    }
    if let (1, Some(value)) = (rewards.len(), rewards.values().next()) {
        return parse_reward(value).map(Some);
    }
    Ok(None)
}

fn rewards(result: &Value) -> Option<&serde_json::Map<String, Value>> {
    result
        .pointer("/verifier_result/rewards")
        .and_then(Value::as_object)
}

/// The reward, and why it is unusable (`None` when usable or absent).
fn usable_reward(result: &Value) -> (Option<f64>, Option<String>) {
    let rewards = rewards(result);
    match reward_of(rewards) {
        Err(problem) => (None, Some(problem)),
        Ok(None) if rewards.is_some_and(|r| !r.is_empty()) => {
            let names: Vec<&String> = rewards.into_iter().flat_map(|r| r.keys()).collect();
            (
                None,
                Some(format!("several rewards and none named reward: {names:?}")),
            )
        }
        Ok(reward) => (reward, None),
    }
}

fn present(value: Option<&Value>) -> bool {
    value.is_some_and(|v| !v.is_null() && v.as_object().is_none_or(|o| !o.is_empty()))
}

fn exception_status(result: &Value, kind: &str, message: &str) -> Option<(TrialStatus, Cause)> {
    if UNSUPPORTED.is_match(message) {
        return Some((
            TrialStatus::Unsupported,
            Cause::new("unsupported_capability", message),
        ));
    }
    if GRADED_WHEN_REWARDED.contains(&kind) && usable_reward(result).0.is_some() {
        return None;
    }
    if kind == "AgentTimeoutError" {
        let message = if message.is_empty() {
            "the agent time limit ended the agent"
        } else {
            message
        };
        return Some((TrialStatus::Timeout, Cause::new("agent_timeout", message)));
    }
    if kind == "CancelledError" {
        return Some((
            TrialStatus::EnvironmentError,
            Cause::new(
                "cancelled",
                "the run was interrupted before the trial ended",
            ),
        ));
    }
    let described = format!("{kind}: {message}");
    if let Some((_, status)) = EXCEPTIONS.iter().find(|(name, _)| *name == kind) {
        if *status == TrialStatus::ModelError && UNREACHABLE.is_match(message) {
            return Some((*status, Cause::new("endpoint_unreachable", described)));
        }
        return Some((*status, Cause::new(kind, described)));
    }
    if UNREACHABLE.is_match(message) {
        return Some((
            TrialStatus::ModelError,
            Cause::new("endpoint_unreachable", described),
        ));
    }
    if !present(result.get("agent_execution")) {
        return Some((
            TrialStatus::EnvironmentError,
            Cause::new("setup_failed", format!("before the agent ran: {described}")),
        ));
    }
    if present(result.get("verifier")) && usable_reward(result).0.is_none() {
        return Some((
            TrialStatus::VerifierError,
            Cause::new(kind, format!("during verification: {described}")),
        ));
    }
    Some((TrialStatus::AgentError, Cause::new(kind, described)))
}

fn last_server_status(requests: &[ModelRequest]) -> Option<(u16, &str)> {
    requests
        .iter()
        .rev()
        .find_map(|r| r.status.map(|s| (s, r.error.as_deref().unwrap_or(""))))
}

fn detail(error: &str) -> String {
    if error.is_empty() {
        String::new()
    } else {
        format!(": {error}")
    }
}

/// How a trial ended, by the rules of the module documentation.
#[must_use]
pub fn classify(
    evidence: &TrialEvidence,
    budget: Option<&RunBudget>,
) -> (TrialStatus, Option<Cause>) {
    let result = &evidence.result;
    if let Some(leak) = evidence.leaks.first() {
        return (
            TrialStatus::SecurityViolation,
            Some(Cause::new("leak", format!("leak audit: {leak}"))),
        );
    }
    if evidence.agent_reasoning_loop {
        return (
            TrialStatus::ReasoningLoop,
            Some(Cause::new(
                "reasoning_loop",
                "the model's reasoning looped again after the agent's one retry, and that ended the run",
            )),
        );
    }
    if let Some(budget) = budget {
        if evidence.agent_budget_exhausted {
            return (
                TrialStatus::BudgetExhausted,
                Some(Cause::new(
                    RUN_BUDGET,
                    "the agent stopped at the run's budget",
                )),
            );
        }
        let spent = evidence
            .requests
            .iter()
            .filter_map(|r| r.completion_tokens)
            .sum();
        if let Some(overrun) = budget.overrun(spent, evidence.turns) {
            return (
                TrialStatus::BudgetExhausted,
                Some(Cause::new(RUN_BUDGET, format!("the agent {overrun}"))),
            );
        }
    }
    if let Some(verification) = &evidence.verification {
        if verification.get("verdict").and_then(Value::as_str) == Some("environment_error") {
            let reason = verification
                .get("reason")
                .and_then(Value::as_str)
                .unwrap_or("the verifier could not judge the trial");
            return (
                TrialStatus::EnvironmentError,
                Some(Cause::new("verifier_environment_error", reason)),
            );
        }
    }
    let statuses: Vec<u16> = evidence.requests.iter().filter_map(|r| r.status).collect();
    if statuses.iter().any(|s| *s >= 500) && !statuses.iter().any(|s| (200..300).contains(s)) {
        let failed: Vec<&ModelRequest> = evidence
            .requests
            .iter()
            .filter(|r| r.status.unwrap_or(0) >= 500)
            .collect();
        let error = failed.iter().find_map(|r| r.error.as_deref()).unwrap_or("");
        return (
            TrialStatus::ModelError,
            Some(Cause::new(
                "no_model_turn",
                format!(
                    "none of the trial's {} model requests was answered and the model server failed {} of them \
                     (HTTP 5xx), so the agent never completed a model turn{}",
                    statuses.len(),
                    failed.len(),
                    detail(error)
                ),
            )),
        );
    }
    if let (true, Some(responses), Some(errors)) = (
        evidence.requests.is_empty(),
        evidence.model_responses,
        evidence.model_errors,
    ) {
        if responses > 0 && errors == responses {
            return (
                TrialStatus::ModelError,
                Some(Cause::new(
                    "no_model_turn",
                    format!(
                        "no gateway request of the trial is recorded, and every one of the {responses} model \
                         responses in the agent's own request log is an error: the agent never completed a model turn"
                    ),
                )),
            );
        }
    }
    if let Some((status, error)) = last_server_status(&evidence.requests) {
        if status >= 500 {
            return (
                TrialStatus::ModelError,
                Some(Cause::new(
                    "server_error",
                    format!(
                        "the trial's last model request got HTTP {status}{}",
                        detail(error)
                    ),
                )),
            );
        }
        if ENDPOINT_REFUSALS.contains(&status) {
            return (
                TrialStatus::ModelError,
                Some(Cause::new(
                    "endpoint_refused",
                    format!(
                        "the endpoint refused the trial's last model request with HTTP {status}{} (the model is not \
                         served or the key is not accepted)",
                        detail(error)
                    ),
                )),
            );
        }
    }
    let rejected: Vec<&ModelRequest> = evidence
        .requests
        .iter()
        .filter(|r| r.status.is_some_and(|s| (400..499).contains(&s)))
        .collect();
    if !rejected.is_empty() && !statuses.iter().any(|s| (200..300).contains(s)) {
        let error = rejected
            .iter()
            .find_map(|r| r.error.as_deref())
            .unwrap_or("");
        return (
            TrialStatus::ModelError,
            Some(Cause::new(
                "endpoint_rejected",
                format!(
                    "none of the trial's {} model requests was answered and the endpoint rejected {} of them \
                     (HTTP 4xx), so the agent never completed a model turn{}",
                    statuses.len(),
                    rejected.len(),
                    detail(error)
                ),
            )),
        );
    }
    if let Some(exception) = result.get("exception_info").filter(|e| present(Some(e))) {
        let text = |key: &str| match exception.get(key) {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Null) | None => String::new(),
            Some(other) => other.to_string(),
        };
        if let Some((status, cause)) =
            exception_status(result, &text("exception_type"), &text("exception_message"))
        {
            if status == TrialStatus::Timeout && budget.is_some() {
                return (
                    TrialStatus::InfraTimeout,
                    Some(Cause::new(
                        "wall_clock",
                        format!(
                            "the wall-clock backstop of a budgeted run ended the agent: {}",
                            cause.message
                        ),
                    )),
                );
            }
            return (status, Some(cause));
        }
    }
    if evidence.agent_timed_out {
        if budget.is_some() {
            return (
                TrialStatus::InfraTimeout,
                Some(Cause::new(
                    "wall_clock",
                    "an agent step or run time limit ended the agent of a budgeted run",
                )),
            );
        }
        return (
            TrialStatus::Timeout,
            Some(Cause::new(
                "agent_timeout",
                "an agent step or run time limit ended the agent",
            )),
        );
    }
    match usable_reward(result) {
        (_, Some(problem)) => (
            TrialStatus::VerifierError,
            Some(Cause::new("malformed_reward", problem)),
        ),
        (None, None) => (
            TrialStatus::VerifierError,
            Some(Cause::new("no_reward", "the verifier recorded no reward")),
        ),
        (Some(_), None) => (TrialStatus::Graded, None),
    }
}

/// The graded aspects beside the reward: every other numeric key of the
/// rewards object; empty when one of them is not a valid reward.
fn metrics(result: &Value) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    for (name, value) in rewards(result).into_iter().flatten() {
        if name == "reward" || !value.is_number() {
            continue;
        }
        match parse_reward(value) {
            Ok(reward) => {
                out.insert(name.clone(), reward);
            }
            Err(_) => return BTreeMap::new(),
        }
    }
    out
}

/// Classify and score one trial under `policy`.
#[must_use]
pub fn grade(evidence: &TrialEvidence, policy: &GradingPolicy) -> Grade {
    let (status, mut cause) = classify(evidence, policy.budget.as_ref());
    let reward = usable_reward(&evidence.result).0;
    let run_budget = status == TrialStatus::BudgetExhausted
        && cause.as_ref().is_some_and(|c| c.code == RUN_BUDGET);
    let looped = status == TrialStatus::ReasoningLoop;
    let keeps_score = matches!(
        status,
        TrialStatus::Graded | TrialStatus::Timeout | TrialStatus::SecurityViolation
    ) || run_budget
        || looped;
    let score = if keeps_score { reward } else { None };
    let mut counted = status == TrialStatus::Graded || run_budget || looped;
    if status == TrialStatus::Timeout {
        let mut timed_out =
            cause.unwrap_or_else(|| Cause::new("agent_timeout", "a time limit ended the agent"));
        timed_out.policy = Some(policy.timeout);
        cause = Some(timed_out);
        counted = policy.timeout == TimeoutPolicy::Graded && score.is_some();
    }
    let vetoes = if status == TrialStatus::SecurityViolation {
        evidence.leaks.clone()
    } else {
        Vec::new()
    };
    let passed = if run_budget || looped {
        Some(false)
    } else if counted {
        Some(score.is_some_and(|s| s >= FULL_CREDIT) && vetoes.is_empty())
    } else if vetoes.is_empty() {
        None
    } else {
        Some(false)
    };
    let metrics = if score.is_some() {
        metrics(&evidence.result)
    } else {
        BTreeMap::new()
    };
    Grade {
        status,
        cause,
        counted,
        score,
        passed,
        metrics,
        vetoes,
    }
}
