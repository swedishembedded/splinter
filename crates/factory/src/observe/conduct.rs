// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements failure analysis that says whether an agent
// failed for want of training or for want of a working interface, for its
// clients. If your team needs expertise in diagnosing agent failures, you can
// procure our services by sending an email to info@swedishembedded.com.

//! How graded attempts went, read from each trial's grade and trajectory:
//! which class each failure belongs to, whether the agent's own claim of
//! completion agreed with the grader, and whether a task is known to be
//! solvable at all.
//!
//! # Failure attribution
//!
//! Every attempt that did not pass goes to exactly one class, by the first of
//! these rules that holds:
//!
//! 1. the attempt was not counted (the infrastructure decided it):
//!    `infrastructure`;
//! 2. the agent or model could not take a capability the task needs
//!    (`unsupported`): `missing_control`;
//! 3. a time, turn or token budget ended the agent (`timeout`,
//!    `budget_exhausted`): `budget`;
//! 4. the model's reasoning looped until it was ended (`reasoning_loop`):
//!    `reasoning`;
//! 5. the task's reference solution ran and failed its own verifier, so no
//!    witness shows the goal reachable: `unreachable_goal`;
//! 6. the trajectory cannot be read: `unattributed`;
//! 7. the agent made no tool call: `action_selection` (it never acted on the
//!    environment);
//! 8. the agent made a call it had just made (same tool, same arguments) and
//!    was shown the same result again, so it learned nothing new and did not
//!    change course: `failure_to_recover`;
//! 9. otherwise `unattributed`: no rule decides it, and none is guessed.
//!
//! `missing_observation`, `ambiguous_predicate`, `wrong_environment_model`
//! and `unused_evidence` have no rule yet: deciding them needs what a run does
//! not record (what each tool reads and writes, the final state against the
//! agent's account of it), so they are listed as undecided, never counted.
//!
//! # Self-verification
//!
//! An agent claims completion when its trajectory ends with an answer of its
//! own: the last step is the agent's and makes no tool call, and no budget or
//! time limit ended it. The claim is read from that structure, not from the
//! answer's words. A claimed attempt that failed is an undetected failure; an
//! unclaimed one that passed succeeded without saying so. Whether a success
//! was verified (the agent read the final state back before claiming) needs
//! to know which calls change state, which the records do not hold, so it is
//! not measured.

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use splinter_core::factory::product::TrialResult;
use splinter_core::factory::trial::TrialStatus;
use splinter_eval::pilot::Share;

/// The class of an attempt that did not pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureClass {
    /// The interface offers no action the goal needs.
    MissingControl,
    /// The interface does not show what the goal depends on.
    MissingObservation,
    /// No witness shows the goal reachable from the start.
    UnreachableGoal,
    /// A time, turn or token budget ended the attempt.
    Budget,
    /// The goal's predicate admits more than one reading.
    AmbiguousPredicate,
    /// The environment does not behave as the task assumes.
    WrongEnvironmentModel,
    /// The agent chose actions that cannot reach the goal.
    ActionSelection,
    /// The model's reasoning failed (looped).
    Reasoning,
    /// What the agent was shown contradicted it and it went on regardless.
    UnusedEvidence,
    /// The agent repeated what had just failed instead of changing course.
    FailureToRecover,
    /// The infrastructure decided the attempt; it is not counted.
    Infrastructure,
    /// No rule decides it.
    Unattributed,
}

impl FailureClass {
    /// What fixes a failure of this class: `interface` (the environment or
    /// the task), `policy` (training), `infrastructure` or `unattributed`.
    #[must_use]
    pub fn side(self) -> &'static str {
        match self {
            Self::MissingControl
            | Self::MissingObservation
            | Self::UnreachableGoal
            | Self::Budget
            | Self::AmbiguousPredicate
            | Self::WrongEnvironmentModel => "interface",
            Self::ActionSelection
            | Self::Reasoning
            | Self::UnusedEvidence
            | Self::FailureToRecover => "policy",
            Self::Infrastructure => "infrastructure",
            Self::Unattributed => "unattributed",
        }
    }
}

/// The classes no rule decides from what a run records.
pub const UNDECIDED: [FailureClass; 4] = [
    FailureClass::MissingObservation,
    FailureClass::AmbiguousPredicate,
    FailureClass::WrongEnvironmentModel,
    FailureClass::UnusedEvidence,
];

/// Whether a task is known to be solvable: its reference solution passed its
/// verifier (a constructive witness that the goal is reachable), ran and
/// failed, or never ran.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Witness {
    /// The reference solution passed.
    Witnessed,
    /// The reference solution ran and failed every counted attempt.
    Refuted,
    /// No counted attempt of the reference solution.
    Unknown,
}

/// What a trajectory shows, for the rules above.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Trace {
    /// The tool calls the agent made.
    pub calls: usize,
    /// The trajectory ends with the agent's own answer and no call.
    pub ended_with_answer: bool,
    /// The agent made a call it had just made and was shown the same result.
    pub repeated_without_change: bool,
}

impl Trace {
    /// What the ATIF trajectory `trajectory` shows.
    #[must_use]
    pub fn of(trajectory: &Value) -> Self {
        let steps = trajectory["steps"]
            .as_array()
            .map_or(&[][..], Vec::as_slice);
        let mut results: HashMap<&str, &Value> = HashMap::new();
        for result in steps
            .iter()
            .flat_map(|s| s["observation"]["results"].as_array().into_iter().flatten())
        {
            if let Some(id) = result["source_call_id"].as_str() {
                results.insert(id, &result["content"]);
            }
        }
        let calls: Vec<(&Value, &Value, Option<&Value>)> = steps
            .iter()
            .filter(|s| s["source"] == "agent")
            .flat_map(|s| s["tool_calls"].as_array().into_iter().flatten())
            .map(|c| {
                let result = c["tool_call_id"]
                    .as_str()
                    .and_then(|id| results.get(id))
                    .copied();
                (&c["function_name"], &c["arguments"], result)
            })
            .collect();
        let repeated_without_change = calls.windows(2).any(|w| {
            let ((name_a, args_a, result_a), (name_b, args_b, result_b)) = (w[0], w[1]);
            name_a == name_b && args_a == args_b && result_a.is_some() && result_a == result_b
        });
        let ended_with_answer = steps.last().is_some_and(|last| {
            last["source"] == "agent"
                && last["tool_calls"].as_array().is_none_or(Vec::is_empty)
                && last["message"]
                    .as_str()
                    .is_some_and(|m| !m.trim().is_empty())
        });
        Self {
            calls: calls.len(),
            ended_with_answer,
            repeated_without_change,
        }
    }
}

/// The class of `trial`, which did not pass, given its trajectory (`None`
/// when it cannot be read) and its task's witness (the module's rules).
#[must_use]
pub fn attribute(trial: &TrialResult, trace: Option<Trace>, witness: Witness) -> FailureClass {
    let status = trial.grade.status;
    if status == TrialStatus::Unsupported {
        return FailureClass::MissingControl;
    }
    if !trial.grade.counted {
        return FailureClass::Infrastructure;
    }
    match status {
        TrialStatus::Timeout | TrialStatus::BudgetExhausted => return FailureClass::Budget,
        TrialStatus::ReasoningLoop => return FailureClass::Reasoning,
        _ => {}
    }
    if witness == Witness::Refuted {
        return FailureClass::UnreachableGoal;
    }
    let Some(trace) = trace else {
        return FailureClass::Unattributed;
    };
    if trace.calls == 0 {
        FailureClass::ActionSelection
    } else if trace.repeated_without_change {
        FailureClass::FailureToRecover
    } else {
        FailureClass::Unattributed
    }
}

/// Whether the agent claimed completion: it ended with its own answer and no
/// limit ended it. `None` when its trajectory cannot be read.
#[must_use]
pub fn claimed(trial: &TrialResult, trace: Option<Trace>) -> Option<bool> {
    let trace = trace?;
    Some(trial.grade.status == TrialStatus::Graded && trace.ended_with_answer)
}

/// The failure classes of a set of attempts and how their claims agreed with
/// the grader.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Conduct {
    /// Every attempt.
    pub trials: usize,
    /// Attempts the grader counted.
    pub counted: usize,
    /// Counted attempts that passed.
    pub passed: usize,
    /// Attempts that did not pass (counted failures and uncounted ones).
    pub failed: usize,
    /// The failed attempts by class; each failed attempt is in one class.
    pub attribution: BTreeMap<FailureClass, usize>,
    /// The failed attempts by what fixes them (`interface`, `policy`,
    /// `infrastructure`, `unattributed`).
    pub by_side: BTreeMap<String, usize>,
    /// The classes no rule decides from these records.
    pub undecided_classes: Vec<FailureClass>,
    /// Claims of completion against the grader.
    pub self_verification: SelfVerification,
}

/// How an agent's claims of completion agreed with the grader, over counted
/// attempts whose trajectory could be read.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SelfVerification {
    /// Counted attempts whose trajectory could be read.
    pub judged: usize,
    /// Counted attempts whose trajectory could not be read.
    pub unreadable: usize,
    /// Claimed completion and passed.
    pub claimed_passed: usize,
    /// Claimed completion and failed: undetected failures.
    pub undetected_failures: usize,
    /// Did not claim completion and failed.
    pub unclaimed_failed: usize,
    /// Did not claim completion and passed.
    pub unclaimed_passed: usize,
    /// Of the claims, the share that failed, with its Wilson interval.
    pub undetected_failure_rate: Option<Share>,
    /// Of the judged attempts, the share whose claim disagreed with the
    /// grader (claimed and failed, or unclaimed and passed).
    pub disagreement: Option<Share>,
    /// Successes not read back before the claim: not measured.
    pub unverified_successes: Option<usize>,
    /// Why `unverified_successes` is not measured.
    pub unverified_successes_unmeasured: Option<String>,
}

/// Why unverified successes are absent.
const UNVERIFIED_UNMEASURED: &str = "telling a read-back from a write needs which calls change \
state, which the run's records do not hold";

impl Conduct {
    /// Adds `trial`, given its trajectory and its task's witness.
    pub fn add(&mut self, trial: &TrialResult, trace: Option<Trace>, witness: Witness) {
        self.trials += 1;
        let passed = trial.grade.counted && trial.grade.passed == Some(true);
        if trial.grade.counted {
            self.counted += 1;
            self.passed += usize::from(passed);
            let sv = &mut self.self_verification;
            match claimed(trial, trace) {
                None => sv.unreadable += 1,
                Some(claim) => {
                    sv.judged += 1;
                    match (claim, passed) {
                        (true, true) => sv.claimed_passed += 1,
                        (true, false) => sv.undetected_failures += 1,
                        (false, true) => sv.unclaimed_passed += 1,
                        (false, false) => sv.unclaimed_failed += 1,
                    }
                }
            }
        }
        if !passed {
            self.failed += 1;
            let class = attribute(trial, trace, witness);
            *self.attribution.entry(class).or_default() += 1;
            *self.by_side.entry(class.side().to_owned()).or_default() += 1;
        }
    }

    /// The conduct with its shares computed.
    #[must_use]
    pub fn finished(mut self) -> Self {
        self.undecided_classes = UNDECIDED.to_vec();
        let sv = &mut self.self_verification;
        let claims = sv.claimed_passed + sv.undetected_failures;
        sv.undetected_failure_rate = Share::wilson(sv.undetected_failures, claims);
        sv.disagreement = Share::wilson(sv.undetected_failures + sv.unclaimed_passed, sv.judged);
        sv.unverified_successes = None;
        sv.unverified_successes_unmeasured = Some(UNVERIFIED_UNMEASURED.to_owned());
        self
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use splinter_core::factory::trial::Grade;

    use super::*;

    fn trial(status: TrialStatus, counted: bool, passed: Option<bool>) -> TrialResult {
        let mut t: TrialResult = serde_json::from_value(json!({
            "schema": "splinter.trial/1",
            "slot": {"pass": "p", "task": "t", "slot": 0},
            "attempt": 0, "task": "t",
            "task_digest": format!("blake3:{}", "0".repeat(64)),
            "agent": {"name": "sven", "version": null, "invocation": format!("blake3:{}", "0".repeat(64))},
            "model": null,
            "grade": {"status": "graded", "cause": null, "counted": true, "score": 0.0, "passed": false, "metrics": {}, "vetoes": []},
            "trace": null, "native_dir": "", "native": []
        }))
        .unwrap();
        t.grade = Grade {
            status,
            counted,
            passed,
            ..t.grade
        };
        t
    }

    fn call(id: &str, args: Value) -> Value {
        json!({"tool_call_id": id, "function_name": "edit", "arguments": args})
    }

    fn step(calls: Vec<Value>, results: Vec<(&str, &str)>) -> Value {
        let results: Vec<Value> = results
            .into_iter()
            .map(|(id, c)| json!({"source_call_id": id, "content": c}))
            .collect();
        json!({"source": "agent", "message": "doing it", "tool_calls": calls, "observation": {"results": results}})
    }

    fn answer(text: &str) -> Value {
        json!({"source": "agent", "message": text})
    }

    #[test]
    fn a_repeat_counts_only_when_the_same_call_got_the_same_result() {
        let same = json!({"steps": [
            step(vec![call("a", json!({"x": 1}))], vec![("a", "no match")]),
            step(vec![call("b", json!({"x": 1}))], vec![("b", "no match")]),
            answer("done"),
        ]});
        let trace = Trace::of(&same);
        assert_eq!(trace.calls, 2);
        assert!(trace.repeated_without_change && trace.ended_with_answer);

        let changed = json!({"steps": [
            step(vec![call("a", json!({"x": 1}))], vec![("a", "no match")]),
            step(vec![call("b", json!({"x": 1}))], vec![("b", "replaced 1")]),
        ]});
        let trace = Trace::of(&changed);
        assert!(!trace.repeated_without_change);
        assert!(
            !trace.ended_with_answer,
            "it ended on a call, not an answer"
        );
    }

    #[test]
    fn each_failure_has_one_class_by_the_first_rule_that_holds() {
        let acted = Trace {
            calls: 3,
            ended_with_answer: true,
            repeated_without_change: true,
        };
        let cases = [
            (
                trial(TrialStatus::ModelError, false, None),
                Witness::Refuted,
                FailureClass::Infrastructure,
            ),
            (
                trial(TrialStatus::Unsupported, false, None),
                Witness::Unknown,
                FailureClass::MissingControl,
            ),
            (
                trial(TrialStatus::Timeout, true, Some(false)),
                Witness::Refuted,
                FailureClass::Budget,
            ),
            (
                trial(TrialStatus::BudgetExhausted, true, Some(false)),
                Witness::Unknown,
                FailureClass::Budget,
            ),
            (
                trial(TrialStatus::ReasoningLoop, true, Some(false)),
                Witness::Unknown,
                FailureClass::Reasoning,
            ),
            (
                trial(TrialStatus::Graded, true, Some(false)),
                Witness::Refuted,
                FailureClass::UnreachableGoal,
            ),
            (
                trial(TrialStatus::Graded, true, Some(false)),
                Witness::Witnessed,
                FailureClass::FailureToRecover,
            ),
        ];
        for (t, witness, expected) in cases {
            assert_eq!(
                attribute(&t, Some(acted), witness),
                expected,
                "{:?}",
                t.grade.status
            );
        }
        let failed = trial(TrialStatus::Graded, true, Some(false));
        assert_eq!(
            attribute(&failed, None, Witness::Witnessed),
            FailureClass::Unattributed
        );
        let idle = Trace {
            calls: 0,
            ..Trace::default()
        };
        assert_eq!(
            attribute(&failed, Some(idle), Witness::Unknown),
            FailureClass::ActionSelection
        );
        let plain = Trace {
            calls: 2,
            ..Trace::default()
        };
        assert_eq!(
            attribute(&failed, Some(plain), Witness::Witnessed),
            FailureClass::Unattributed
        );
    }

    #[test]
    fn a_claim_that_failed_is_an_undetected_failure_and_a_limit_is_no_claim() {
        let answered = Some(Trace {
            calls: 1,
            ended_with_answer: true,
            repeated_without_change: false,
        });
        let mut conduct = Conduct::default();
        conduct.add(
            &trial(TrialStatus::Graded, true, Some(true)),
            answered,
            Witness::Unknown,
        );
        conduct.add(
            &trial(TrialStatus::Graded, true, Some(false)),
            answered,
            Witness::Unknown,
        );
        conduct.add(
            &trial(TrialStatus::Timeout, true, Some(false)),
            answered,
            Witness::Unknown,
        );
        conduct.add(
            &trial(TrialStatus::Graded, true, Some(false)),
            None,
            Witness::Unknown,
        );
        conduct.add(
            &trial(TrialStatus::ModelError, false, None),
            None,
            Witness::Unknown,
        );
        let conduct = conduct.finished();
        let sv = &conduct.self_verification;
        assert_eq!(
            (
                conduct.trials,
                conduct.counted,
                conduct.passed,
                conduct.failed
            ),
            (5, 4, 1, 4)
        );
        assert_eq!(
            (
                sv.claimed_passed,
                sv.undetected_failures,
                sv.unclaimed_failed
            ),
            (1, 1, 1)
        );
        assert_eq!((sv.judged, sv.unreadable), (3, 1));
        assert_eq!(sv.undetected_failure_rate.unwrap().value, 0.5);
        assert_eq!(conduct.attribution.values().sum::<usize>(), conduct.failed);
        assert_eq!(conduct.by_side["infrastructure"], 1);
        assert!(sv.unverified_successes.is_none() && sv.unverified_successes_unmeasured.is_some());
    }
}
