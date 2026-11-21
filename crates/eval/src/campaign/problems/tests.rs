// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for choosing the next training
// experiment from the evidence already paid for, for its clients. If your team
// needs expertise in planning model-improvement campaigns you can procure our
// services by sending an email to info@swedishembedded.com.

//! Specifications of the problem state: known-answer priorities and order,
//! the belief update of each verdict, the verdict of a measured effect and of
//! the gate, the single-variable and firewall checks, and replay.

#![allow(clippy::unwrap_used, clippy::float_cmp)]

use serde_json::json;

use super::*;
use crate::promotion::{Aggregate, Pair, Promotion, Reason, Rejection, Role, Rule};

fn candidate(id: &str, belief: f64) -> Candidate {
    Candidate {
        id: id.into(),
        intervention: "curation".into(),
        statement: format!("{id} does something"),
        variable: "train.learning_rate".into(),
        value: None,
        phase: Phase::Hypothesis,
        belief,
        prior: String::new(),
        status: CandidateStatus::Pending,
        evidence: Vec::new(),
        history: Vec::new(),
        revisions: Vec::new(),
    }
}

fn problem(id: &str, attempts: u32, candidates: Vec<Candidate>) -> Problem {
    Problem {
        id: id.into(),
        statement: format!("{id} limits the model"),
        metric: "pass@1".into(),
        status: ProblemStatus::Open,
        attempts,
        evidence: vec![Evidence {
            reference: "knowledge:202".into(),
            note: String::new(),
        }],
        candidates,
        revisions: Vec::new(),
    }
}

fn seed(problems: Vec<Problem>) -> Event {
    Event::Add {
        document: Document {
            schema: STATE_SCHEMA.into(),
            parameters: None,
            problems,
        },
    }
}

fn verdict(candidate: &str, experiment: &str, verdict: Verdict) -> Event {
    Event::Verdict {
        candidate: candidate.into(),
        experiment: experiment.into(),
        verdict,
        effect: None,
        evidence: Vec::new(),
        reason: "measured".into(),
    }
}

fn state(events: &[Event]) -> ProblemState {
    ProblemState::replay(events).unwrap().unwrap()
}

/// Priorities by hand: a 0.75 on a problem tried twice is 0.25; a 0.3 on an
/// untried one is 0.3; a 0.5 tried once is 0.25 and loses the tie to the
/// 0.75 by belief. Resolved problems, adopted candidates, other phases and a
/// belief of 0 are never proposed.
#[test]
fn candidates_are_ranked_by_belief_over_attempts_plus_one() {
    let mut engineering = candidate("e.jobs", 0.9);
    engineering.phase = Phase::Engineering;
    engineering.variable = "practice.jobs".into();
    let mut adopted = candidate("a.adopted", 0.99);
    adopted.status = CandidateStatus::Adopted;
    let mut resolved = problem("resolved", 0, vec![candidate("r.c", 0.99)]);
    resolved.status = ProblemStatus::Resolved;
    let s = state(&[seed(vec![
        problem(
            "tried-twice",
            2,
            vec![candidate("t.c", 0.75), candidate("t.zero", 0.0)],
        ),
        problem("untried", 0, vec![candidate("u.c", 0.3), engineering]),
        problem("tried-once", 1, vec![candidate("o.c", 0.5), adopted]),
        resolved,
    ])]);

    let ranked = s.ranked(Phase::Hypothesis);
    let order: Vec<(&str, f64)> = ranked
        .iter()
        .map(|p| (p.candidate.as_str(), p.priority))
        .collect();
    assert_eq!(
        order,
        vec![("u.c", 0.3), ("t.c", 0.25), ("o.c", 0.25)],
        "b / (N + 1), ties to the higher belief"
    );
    assert_eq!(priority(0.75, 2), 0.25);
    assert_eq!(priority(0.3, 0), 0.3);
    assert_eq!(s.next(Phase::Hypothesis).unwrap().candidate, "u.c");
    assert_eq!(s.next(Phase::Engineering).unwrap().candidate, "e.jobs");
}

/// Each verdict's update, from a 0.5 prior: supported +0.15 and adopted,
/// falsified -0.20, inconclusive and invalid unchanged; every one counts one
/// attempt against the problem.
#[test]
fn a_verdict_moves_the_belief_by_its_kind_and_always_counts_an_attempt() {
    let cases = [
        (Verdict::Supported, 0.65, CandidateStatus::Adopted),
        (Verdict::Falsified, 0.3, CandidateStatus::Pending),
        (Verdict::Inconclusive, 0.5, CandidateStatus::Pending),
        (Verdict::Invalid, 0.5, CandidateStatus::Pending),
    ];
    for (kind, belief, status) in cases {
        let s = state(&[
            seed(vec![problem("p", 3, vec![candidate("p.c", 0.5)])]),
            verdict("p.c", "x1", kind),
        ]);
        let (p, c) = s.candidate("p.c").unwrap();
        assert!((c.belief - belief).abs() < 1e-12, "{kind:?}: {}", c.belief);
        assert_eq!(c.status, status, "{kind:?}");
        assert_eq!(p.attempts, 4, "{kind:?} counts one attempt");
        assert_eq!(c.history.len(), 1);
        assert_eq!(c.history[0].belief_before, 0.5);
    }
}

/// An inconclusive verdict lowers the priority through `N` only: the
/// candidate keeps its belief, so it is not treated as falsified.
#[test]
fn inconclusive_is_not_falsified() {
    let s = state(&[
        seed(vec![problem("p", 0, vec![candidate("p.c", 0.5)])]),
        verdict("p.c", "x1", Verdict::Inconclusive),
        verdict("p.c", "x2", Verdict::Inconclusive),
    ]);
    let next = s.next(Phase::Hypothesis).unwrap();
    assert_eq!(next.belief, 0.5);
    assert!((next.priority - 0.5 / 3.0).abs() < 1e-12);
}

/// Beliefs stay in [0, 1]: supports stop at 1 and falsifications at 0, and a
/// candidate at 0 is no longer proposed.
#[test]
fn beliefs_are_clamped_and_a_zero_belief_is_never_proposed() {
    let p = Parameters::default();
    assert_eq!(updated_belief(0.95, Verdict::Supported, &p), 1.0);
    assert_eq!(updated_belief(0.1, Verdict::Falsified, &p), 0.0);
    let s = state(&[
        seed(vec![problem("p", 0, vec![candidate("p.c", 0.2)])]),
        verdict("p.c", "x1", Verdict::Falsified),
    ]);
    assert_eq!(s.candidate("p.c").unwrap().1.belief, 0.0);
    assert!(s.next(Phase::Hypothesis).is_none());
}

/// A measured effect against a target of 0.05 at alpha 0.05.
#[test]
fn an_effect_is_judged_against_the_target_not_against_zero() {
    let e = |difference, low, high, p| Effect {
        difference,
        low,
        high,
        p,
    };
    let judge = |effect: &Effect| classify_effect(Some(effect), 0.05, 0.05);
    assert_eq!(judge(&e(0.08, 0.02, 0.14, Some(0.01))), Verdict::Supported);
    assert_eq!(
        judge(&e(0.08, 0.02, 0.14, Some(0.2))),
        Verdict::Inconclusive,
        "the paired test must agree"
    );
    assert_eq!(
        judge(&e(0.08, 0.02, 0.14, None)),
        Verdict::Inconclusive,
        "no test, no support"
    );
    assert_eq!(
        judge(&e(0.016, -0.009, 0.040, Some(0.29))),
        Verdict::Falsified,
        "item 25 E1b: +0.05 is ruled out"
    );
    assert_eq!(
        judge(&e(0.02, 0.005, 0.04, Some(0.01))),
        Verdict::Falsified,
        "a real gain smaller than the target is not the hypothesised gain"
    );
    assert_eq!(
        judge(&e(0.029, -0.050, 0.105, Some(0.88))),
        Verdict::Inconclusive,
        "item 21 S1b (g): the interval holds 0 and the target"
    );
    assert_eq!(
        judge(&e(0.04, 0.01, 0.09, Some(0.01))),
        Verdict::Inconclusive,
        "shown above zero but below the target"
    );
    assert_eq!(classify_effect(None, 0.05, 0.05), Verdict::Invalid);
    assert_eq!(judge(&e(f64::NAN, 0.0, 0.1, Some(0.01))), Verdict::Invalid);
}

fn aggregate(gain: f64, interval: (f64, f64)) -> Aggregate {
    Aggregate {
        schema: "bench.examination/2".into(),
        collection: "dev".into(),
        exams: Vec::new(),
        sealed: false,
        items: 96,
        alpha: 0.05,
        gain: Some(gain),
        interval: Some(interval),
        one_sided_lower: Some(interval.0),
        families: Some(12),
        burns: Vec::new(),
    }
}

fn promotion(
    promoted: bool,
    reasons: Vec<Reason>,
    dev: (f64, f64),
    retention: (f64, f64),
) -> Promotion {
    Promotion {
        promoted,
        rule: Rule::Examined,
        exam_verified: true,
        reasons,
        pair: Pair {
            candidate: Some("sha256:c".into()),
            predecessor: None,
        },
        dev: Some(aggregate((dev.0 + dev.1) / 2.0, dev)),
        retention: Some(aggregate((retention.0 + retention.1) / 2.0, retention)),
        sealed: None,
    }
}

/// The gate's decision as a verdict, against a target of 0.05.
#[test]
fn the_gate_decides_supported_and_its_reasons_decide_the_rest() {
    let no_gain = Reason::NoDevGain {
        gain: Some(0.0),
        low: -0.02,
        high: 0.02,
    };
    let judge = |p: &Promotion| classify_promotion(p, 0.05).0;
    assert_eq!(
        judge(&promotion(true, vec![], (0.01, 0.07), (-0.02, 0.02))),
        Verdict::Supported
    );
    let unaccepted = Reason::Verdict {
        role: Role::Dev,
        rejection: Rejection::BadSignature,
    };
    assert_eq!(
        judge(&promotion(
            false,
            vec![unaccepted],
            (0.01, 0.07),
            (-0.02, 0.02)
        )),
        Verdict::Invalid
    );
    assert_eq!(
        judge(&promotion(
            false,
            vec![no_gain.clone()],
            (-0.02, 0.02),
            (-0.02, 0.02)
        )),
        Verdict::Falsified,
        "the dev interval rules the target out"
    );
    assert_eq!(
        judge(&promotion(
            false,
            vec![no_gain.clone()],
            (-0.02, 0.09),
            (-0.02, 0.02)
        )),
        Verdict::Inconclusive
    );
    let inferior = Reason::Inferior {
        lower: -0.2,
        margin: crate::promotion::RETENTION_MARGIN,
    };
    assert_eq!(
        judge(&promotion(
            false,
            vec![no_gain, inferior],
            (-0.02, 0.09),
            (-0.2, -0.08)
        )),
        Verdict::Falsified,
        "harm past the retention margin is shown"
    );
    let (_, effect) = classify_promotion(&promotion(true, vec![], (0.01, 0.07), (0.0, 0.0)), 0.05);
    assert_eq!(
        effect,
        Some(Effect {
            difference: 0.04,
            low: 0.01,
            high: 0.07,
            p: None
        })
    );
}

/// The arm changes exactly the declared variable, or a leaf under it.
#[test]
fn an_arm_that_changes_one_declared_variable_passes() {
    let champion =
        json!({"train": {"rank": 16, "learning_rate": 1e-5}, "curate": {"view": "sft-step"}});
    let arm = json!({"train": {"rank": 16, "learning_rate": 3e-6}, "curate": {"view": "sft-step"}});
    assert_eq!(
        single_variable(&champion, &arm, "train.learning_rate").unwrap(),
        vec!["train.learning_rate".to_string()]
    );
    let missing_equals_null = json!({"train": {"rank": 16, "learning_rate": 1e-5, "steps": null}, "curate": {"view": "sft-step"}});
    assert_eq!(
        changed(&champion, &missing_equals_null),
        Vec::<String>::new()
    );
}

/// An arm that changes a second variable is refused with every extra path
/// named; so is an arm that changes nothing, and a path that only shares a
/// prefix with the variable.
#[test]
fn a_multi_variable_arm_is_refused() {
    let champion = json!({"train": {"rank": 16, "learning_rate": 1e-5, "ranking": 1}, "curate": {"view": "sft-step"}});
    let arm = json!({"train": {"rank": 32, "learning_rate": 3e-6, "ranking": 2}, "curate": {"view": "sft-step"}});
    assert_eq!(
        single_variable(&champion, &arm, "train.rank"),
        Err(VariableError::Undeclared {
            variable: "train.rank".into(),
            paths: vec!["train.learning_rate".into(), "train.ranking".into()],
        })
    );
    assert_eq!(
        single_variable(&champion, &champion, "train.rank"),
        Err(VariableError::NoChange)
    );
    assert!(under("train.rank", "train"));
    assert!(!under("train.ranking", "train.rank"));
}

/// An engineering change may touch only the engineering paths; a change to
/// the algorithm beside it is a breach, named.
#[test]
fn an_engineering_change_that_reaches_the_algorithm_is_refused() {
    let allowed = vec!["practice.jobs".to_string(), "practice.harbor".to_string()];
    let champion = json!({"practice": {"jobs": 8}, "train": {"learning_rate": 1e-5}});
    let faster = json!({"practice": {"jobs": 32}, "train": {"learning_rate": 1e-5}});
    assert_eq!(
        firewall(&champion, &faster, &allowed).unwrap(),
        vec!["practice.jobs".to_string()]
    );
    let sneaky = json!({"practice": {"jobs": 32}, "train": {"learning_rate": 2e-5}});
    assert_eq!(
        firewall(&champion, &sneaky, &allowed),
        Err(VariableError::Breach {
            allowed,
            paths: vec!["train.learning_rate".into()],
        })
    );
}

/// Replaying the same events gives the same state and digest; an event the
/// state refuses changes nothing; a second verdict from one experiment, a
/// verdict on an adopted candidate and an event before any document are
/// refused.
#[test]
fn replay_is_deterministic_and_refusals_change_nothing() {
    let events = vec![
        seed(vec![problem(
            "p",
            0,
            vec![candidate("p.c", 0.5), candidate("p.d", 0.4)],
        )]),
        verdict("p.c", "x1", Verdict::Falsified),
        Event::Belief {
            candidate: "p.d".into(),
            belief: 0.7,
            reason: "an insight from x1".into(),
        },
        verdict("p.d", "x2", Verdict::Supported),
    ];
    let once = state(&events);
    let twice = state(&events);
    assert_eq!(once, twice);
    assert_eq!(once.digest(), twice.digest());

    let mut s = once.clone();
    assert_eq!(
        s.apply(&verdict("p.c", "x1", Verdict::Supported)),
        Err(ProblemError::DuplicateExperiment {
            candidate: "p.c".into(),
            experiment: "x1".into()
        })
    );
    assert_eq!(
        s.apply(&verdict("p.d", "x3", Verdict::Supported)),
        Err(ProblemError::Adopted("p.d".into()))
    );
    assert_eq!(s, once, "a refused event changes nothing");
    assert_eq!(
        ProblemState::replay(&[verdict("p.c", "x1", Verdict::Falsified)]),
        Err(ProblemError::Unseeded)
    );
}

/// A document is refused when a metric has no target, an engineering
/// candidate's variable is not an engineering path, an id repeats, or a
/// revision is given without a reason.
#[test]
fn documents_and_revisions_out_of_range_are_refused() {
    let mut no_target = problem("p", 0, vec![]);
    no_target.metric = "unmeasured".into();
    assert!(matches!(
        ProblemState::replay(&[seed(vec![no_target])]),
        Err(ProblemError::Invalid(_))
    ));
    let mut engineering = candidate("p.e", 0.5);
    engineering.phase = Phase::Engineering;
    assert!(matches!(
        ProblemState::replay(&[seed(vec![problem("p", 0, vec![engineering])])]),
        Err(ProblemError::Invalid(_))
    ));
    assert!(matches!(
        ProblemState::replay(&[seed(vec![
            problem("p", 0, vec![candidate("x", 0.5)]),
            problem("q", 0, vec![candidate("x", 0.5)]),
        ])]),
        Err(ProblemError::Invalid(_))
    ));
    let mut s = state(&[seed(vec![problem("p", 0, vec![candidate("p.c", 0.5)])])]);
    assert!(matches!(
        s.apply(&Event::Status {
            problem: "p".into(),
            status: ProblemStatus::Resolved,
            reason: " ".into(),
        }),
        Err(ProblemError::Invalid(_))
    ));
}
