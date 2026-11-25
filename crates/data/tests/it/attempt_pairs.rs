// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that turn graded agent attempts
// into preference and feedback training data for its clients. If your team
// needs expertise in learning from verified successes and failures of agents
// then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: the preference view pairs a verified pass with a verified fail of
//! the same task by the same policy, shown the same prompt, from where their
//! runs part; a repair pairs a failed attempt's prefix continued to a pass
//! with that attempt's own failed suffix, the prefix never trained on. The
//! feedback view judges each verified attempt's continuation desirable or
//! undesirable. Neither ever takes an infrastructure failure as a failure or
//! an unverified pass as a success, the choice of pairs is fixed by its seed,
//! a dataset names one tokenizer, and the lines are brain's
//! `generic-preference-v1` and `generic-feedback-v1` records as its contract
//! fixtures show them.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::collections::BTreeSet;

use crate::common;

use atif::{StepOrigin, TraceStep};
use common::*;
use serde_json::{json, Value};
use splinter_core::annotation::{Annotation, AnnotationBody, Outcome, Producer, Strength};
use splinter_core::chat::WireMessage;
use splinter_core::experience::{Environment, Experience, Task};
use splinter_data::curation::HistoryCuration;
use splinter_data::{
    write_dataset, Corpus, Exclusion, Feedback, Objective, PairSource, Preference, Projection,
    RecordBody, RolloutIdentity, Unchecked, View, ViewError,
};

mod external;
mod repairs;

const TOKENIZER: &str = "sha256:1111111111111111111111111111111111111111111111111111111111111111";
const OTHER_TOKENIZER: &str =
    "sha256:2222222222222222222222222222222222222222222222222222222222222222";

fn bench_task() -> Task {
    Task::new(
        "bench/books",
        Vec::new(),
        Environment::new(
            Environment::BENCH,
            json!({"task": {"id": "books/accrual", "version": "3"}}),
        ),
        "Book the accrual.",
        Vec::new(),
    )
    .unwrap()
}

/// One attempt: its trial, verdict, the model that ran it, how it was
/// graded and what it did.
#[derive(Clone)]
struct Attempt {
    trial: String,
    verdict: Outcome,
    strength: Strength,
    model: String,
    /// The examiner's score status, when the trial was not counted.
    status: Option<&'static str>,
    /// The shell commands of its actions, in order, then its final word.
    commands: Vec<&'static str>,
    last: &'static str,
    tokenizer: Option<&'static str>,
    /// Steps copied from another trajectory (a repair's prefix).
    copied: usize,
    /// A user turn the harness inserts after the n-th action (a time-left notice), and its text.
    notice: Option<(usize, &'static str)>,
}

impl Attempt {
    fn pass(trial: &str) -> Self {
        Self {
            trial: trial.into(),
            verdict: Outcome::Pass,
            strength: Strength::Executable,
            model: "policy-a".into(),
            status: None,
            commands: vec!["cat ledger.csv", "book accrual"],
            last: "Booked.",
            tokenizer: Some(TOKENIZER),
            copied: 0,
            notice: None,
        }
    }

    fn fail(trial: &str) -> Self {
        Self {
            verdict: Outcome::Fail,
            commands: vec!["ls", "book wrong"],
            last: "Done.",
            ..Self::pass(trial)
        }
    }

    fn infrastructure(trial: &str) -> Self {
        Self {
            verdict: Outcome::Abstain,
            status: Some("model_error"),
            ..Self::fail(trial)
        }
    }

    fn commands(self, commands: Vec<&'static str>, last: &'static str) -> Self {
        Self {
            commands,
            last,
            ..self
        }
    }

    fn judged(self) -> Self {
        Self {
            strength: Strength::Judged,
            ..self
        }
    }

    fn model(self, model: &str) -> Self {
        Self {
            model: model.into(),
            ..self
        }
    }

    fn tokenizer(self, tokenizer: Option<&'static str>) -> Self {
        Self { tokenizer, ..self }
    }

    fn copied(self, copied: usize) -> Self {
        Self { copied, ..self }
    }

    fn notice(self, after: usize, text: &'static str) -> Self {
        Self {
            notice: Some((after, text)),
            ..self
        }
    }
}

/// The agent step of a shell call, its result and its policy trace's
/// tokenizer.
fn shell(step: u64, call_id: &str, command: &str, tokenizer: Option<&str>) -> TraceStep {
    let mut s = call(
        step,
        call_id,
        "shell",
        json!({ "command": command }),
        &format!("ran {command}"),
    );
    traced(&mut s, tokenizer);
    s
}

fn traced(step: &mut TraceStep, tokenizer: Option<&str>) {
    if let Some(digest) = tokenizer {
        let mut metrics: atif::StepMetrics = serde_json::from_value(json!({})).unwrap();
        metrics.extra = Some(json!({"sven": {"tokenizer_digest": digest}}));
        step.metrics = Some(metrics);
    }
}

fn add(corpus: &mut Corpus, attempt: &Attempt) -> Experience {
    let mut steps = vec![
        TraceStep::new(1, StepOrigin::System, "You keep the books."),
        user(2, "Book the accrual."),
    ];
    let mut id = 3;
    for (n, command) in attempt.commands.iter().enumerate() {
        steps.push(shell(id, &format!("c{n}"), command, attempt.tokenizer));
        id += 1;
        if let Some((_, text)) = attempt.notice.filter(|(after, _)| *after == n) {
            steps.push(user(id, text));
            id += 1;
        }
    }
    let mut last = say(id, attempt.last);
    traced(&mut last, attempt.tokenizer);
    steps.push(last);
    // The copied prefix: the first agent steps, marked as ATIF marks a
    // repair's student steps.
    for step in steps
        .iter_mut()
        .filter(|s| s.source == StepOrigin::Agent)
        .take(attempt.copied)
    {
        step.is_copied_context = Some(true);
    }
    let mut trajectory = trajectory(steps);
    trajectory.agent.tool_definitions = Some(vec![json!({"type": "function", "function": {
        "name": "shell", "description": "Run a command.", "parameters": {"type": "object"}}})]);
    let at = format!("2026-11-18T00:00:00.{:03}Z", corpus.entries().len());
    let experience = experience(&bench_task(), trajectory, Some(attempt.last), &at);
    let mut evidence = json!({
        "trial": attempt.trial,
        "task": {"id": "books/accrual", "version": "3"},
        "model": attempt.model,
    });
    if let Some(status) = attempt.status {
        evidence["score"] = json!({"schema": "bench.score/1", "status": status, "counted": false});
    }
    let note = Annotation {
        experience: experience.id().unwrap(),
        producer: Producer {
            name: "bench".into(),
            version: "v".into(),
        },
        body: AnnotationBody::Verdict {
            outcome: attempt.verdict,
            strength: attempt.strength,
            evidence,
        },
    };
    corpus.insert(experience.clone(), vec![note]).unwrap();
    experience
}

fn corpus(attempts: &[Attempt]) -> Corpus {
    let mut corpus = Corpus::new();
    for attempt in attempts {
        add(&mut corpus, attempt);
    }
    corpus
}

fn view() -> Preference {
    Preference::new(Strength::Executable)
}

struct Pair {
    prompt: Vec<WireMessage>,
    chosen: Vec<WireMessage>,
    rejected: Vec<WireMessage>,
    identity: RolloutIdentity,
}

fn pairs(projection: &Projection) -> Vec<Pair> {
    projection
        .records
        .iter()
        .map(|r| match &r.body {
            RecordBody::Preference {
                prompt,
                chosen,
                rejected,
                identity,
                ..
            } => Pair {
                prompt: prompt.clone(),
                chosen: chosen.clone(),
                rejected: rejected.clone(),
                identity: identity.clone().unwrap(),
            },
            other => panic!("a preference record, got {other:?}"),
        })
        .collect()
}

fn texts(messages: &[WireMessage]) -> Vec<String> {
    messages
        .iter()
        .map(|m| match m.tool_calls.first() {
            Some(call) => format!("{}:{}", m.role, call.function.arguments),
            None => format!("{}:{}", m.role, m.content),
        })
        .collect()
}

#[test]
fn a_pair_is_a_verified_pass_and_a_verified_fail_of_one_group_from_one_prompt() {
    let corpus = corpus(&[
        Attempt::pass("t1"),
        Attempt::fail("t2"),
        // Another policy's attempt is another group, with no fail to pair.
        Attempt::pass("o1").model("policy-b"),
    ]);
    let projection = view().project(&corpus).unwrap();
    assert_eq!(projection.objective, Objective::Preference);
    let got = pairs(&projection);
    assert_eq!(got.len(), 1);
    let pair = &got[0];
    assert_eq!(
        pair.identity.rollouts,
        ["t1", "t2"],
        "chosen, then rejected"
    );
    assert_eq!(pair.identity.source, Some(PairSource::Attempts));
    assert_eq!(pair.identity.tokenizer_digest.as_deref(), Some(TOKENIZER));
    // The shared prompt: the system and the user turn, never trained on.
    assert_eq!(
        texts(&pair.prompt),
        ["system:You keep the books.", "user:Book the accrual."]
    );
    assert!(pair.prompt.iter().all(|m| !m.train));
    // Each candidate is the whole run from there: actions trained on, the
    // results between them context, ending at the last action.
    assert_eq!(pair.chosen.first().unwrap().role, "assistant");
    assert_eq!(pair.chosen.last().unwrap().content, "Booked.");
    assert_eq!(pair.rejected.last().unwrap().content, "Done.");
    for side in [&pair.chosen, &pair.rejected] {
        assert!(side.iter().all(|m| m.train == (m.role == "assistant")));
        assert_eq!(side.iter().filter(|m| m.role == "tool").count(), 2);
    }
    assert_eq!(projection.count(Exclusion::NoCounterpart), 1);
    let pairing = projection.pairing.as_ref().unwrap();
    assert_eq!((pairing.attempts, pairing.repairs), (1, 0));
}

/// Two runs that part at a turn the harness inserted (a time-left notice saying something different after the
/// same first action) part from that action: a candidate opens with the model's own turn, never with a user or tool
/// turn, which brain refuses as a candidate.
#[test]
fn runs_that_part_at_an_inserted_user_turn_are_paired_from_the_action_before_it() {
    let corpus = corpus(&[
        Attempt::pass("t1")
            .commands(vec!["cat ledger.csv", "book accrual"], "Booked.")
            .notice(0, "Time left: 9 minutes."),
        Attempt::fail("t2")
            .commands(vec!["cat ledger.csv", "book wrong"], "Done.")
            .notice(0, "Time left: 8 minutes."),
    ]);
    let projection = view().project(&corpus).unwrap();
    let got = pairs(&projection);
    assert_eq!(got.len(), 1, "excluded: {:?}", projection.excluded);
    let pair = &got[0];
    assert_eq!(
        texts(&pair.prompt),
        ["system:You keep the books.", "user:Book the accrual."]
    );
    for side in [&pair.chosen, &pair.rejected] {
        assert_eq!(side.first().unwrap().role, "assistant", "{:?}", texts(side));
        assert!(side.iter().all(|m| m.train == (m.role == "assistant")));
    }
    assert!(texts(&pair.chosen).contains(&"user:Time left: 9 minutes.".to_string()));
    assert!(texts(&pair.rejected).contains(&"user:Time left: 8 minutes.".to_string()));
}

#[test]
fn an_infrastructure_failure_is_never_a_rejected_sample() {
    let corpus = corpus(&[
        Attempt::pass("t1"),
        Attempt::infrastructure("x1"),
        Attempt::infrastructure("x2"),
    ]);
    let projection = view().project(&corpus).unwrap();
    assert!(projection.records.is_empty());
    assert_eq!(projection.count(Exclusion::InfrastructureFailure), 2);
    assert_eq!(projection.count(Exclusion::NoCounterpart), 1);
    let feedback = Feedback::new(Strength::Executable)
        .project(&corpus)
        .unwrap();
    assert_eq!(feedback.records.len(), 1, "the pass alone");
    assert_eq!(feedback.count(Exclusion::InfrastructureFailure), 2);
}

#[test]
fn an_unverified_pass_is_never_chosen() {
    let corpus = corpus(&[Attempt::pass("t1").judged(), Attempt::fail("t2")]);
    let projection = view().project(&corpus).unwrap();
    assert!(projection.records.is_empty());
    assert_eq!(projection.count(Exclusion::UnverifiedPass), 1);
    let feedback = Feedback::new(Strength::Executable)
        .project(&corpus)
        .unwrap();
    assert_eq!(feedback.count(Exclusion::UnverifiedPass), 1);
    assert!(feedback.records.iter().all(|r| matches!(
        r.body,
        RecordBody::Feedback {
            desirable: false,
            ..
        }
    )));
}

#[test]
fn a_repair_trains_only_where_it_parts_from_the_failure_it_continues() {
    // The student looked around, then booked it wrong; the teacher continued
    // from the student's first step and booked it right.
    let student = Attempt::fail("student")
        .commands(vec!["cat ledger.csv", "book wrong"], "Done.")
        .model("student");
    let repair = Attempt::pass("repair")
        .commands(vec!["cat ledger.csv", "book accrual"], "Booked.")
        .model("teacher")
        .copied(1);
    let projection = view().project(&corpus(&[student, repair])).unwrap();
    let got = pairs(&projection);
    assert_eq!(got.len(), 1, "{:?}", projection.excluded);
    let pair = &got[0];
    assert_eq!(pair.identity.source, Some(PairSource::Repair));
    assert_eq!(pair.identity.rollouts, ["repair", "student"]);
    // The copied prefix is the prompt: never supervised.
    assert_eq!(
        texts(&pair.prompt),
        [
            "system:You keep the books.",
            "user:Book the accrual.",
            "assistant:{\"command\":\"cat ledger.csv\"}",
            "tool:ran cat ledger.csv",
        ]
    );
    assert!(pair.prompt.iter().all(|m| !m.train));
    // Only the divergent suffixes are candidates.
    assert_eq!(
        texts(&pair.chosen)[0],
        "assistant:{\"command\":\"book accrual\"}"
    );
    assert_eq!(
        texts(&pair.rejected)[0],
        "assistant:{\"command\":\"book wrong\"}"
    );
}

#[test]
fn a_repair_whose_prefix_is_not_the_failures_own_is_left_out() {
    let student = Attempt::fail("student")
        .commands(vec!["ls", "book wrong"], "Done.")
        .model("student");
    let repair = Attempt::pass("repair")
        .commands(vec!["cat ledger.csv", "book accrual"], "Booked.")
        .model("teacher")
        .copied(1);
    let projection = view().project(&corpus(&[student, repair])).unwrap();
    assert!(projection.records.is_empty());
    assert_eq!(projection.count(Exclusion::PrefixMismatch), 1);
}

#[test]
fn the_same_seed_keeps_the_same_pairs_and_the_bound_holds() {
    let corpus = corpus(&[
        Attempt::pass("p1"),
        Attempt::pass("p2").commands(vec!["head ledger.csv", "book accrual"], "Booked."),
        Attempt::fail("f1"),
        Attempt::fail("f2").commands(vec!["pwd", "book wrong"], "Done."),
    ]);
    let chosen = |seed: u64| {
        let projection = view().pairs_per_group(1, seed).project(&corpus).unwrap();
        assert_eq!(projection.records.len(), 1);
        assert_eq!(projection.count(Exclusion::OverLimit), 3);
        let pairing = projection.pairing.clone().unwrap();
        assert_eq!((pairing.per_group, pairing.seed), (Some(1), Some(seed)));
        pairs(&projection)[0].identity.rollouts.clone()
    };
    let seed = any_seed();
    assert_eq!(chosen(seed), chosen(seed), "seed {seed}");
    let drawn: BTreeSet<Vec<String>> = (0..16).map(chosen).collect();
    assert!(drawn.len() > 1, "the seed chooses: {drawn:?}");
}

#[test]
fn a_mixed_tokenizer_digest_is_refused() {
    // Two groups, each internally consistent, encoded by two tokenizers.
    let groups = corpus(&[
        Attempt::pass("a1"),
        Attempt::fail("a2"),
        Attempt::pass("b1")
            .model("policy-b")
            .tokenizer(Some(OTHER_TOKENIZER)),
        Attempt::fail("b2")
            .model("policy-b")
            .tokenizer(Some(OTHER_TOKENIZER)),
    ]);
    let refused = view().project(&groups).unwrap_err();
    assert!(
        matches!(refused, ViewError::MixedTokenizerDigest { .. }),
        "{refused}"
    );
    assert!(refused.to_string().contains(OTHER_TOKENIZER), "{refused}");
    assert!(matches!(
        Feedback::new(Strength::Executable).project(&groups),
        Err(ViewError::MixedTokenizerDigest { .. })
    ));
    // Within a pair, two tokenizers leave the pair out.
    let pair = corpus(&[
        Attempt::pass("a1"),
        Attempt::fail("a2").tokenizer(Some(OTHER_TOKENIZER)),
    ]);
    let projection = view().project(&pair).unwrap();
    assert_eq!(projection.count(Exclusion::MixedTokenizerDigest), 1);
}

#[test]
fn a_pair_that_reads_the_repositorys_history_follows_the_curation() {
    let corpus = corpus(&[
        Attempt::pass("t1"),
        Attempt::fail("t2").commands(vec!["git log --oneline -20", "book wrong"], "Done."),
    ]);
    let dropped = view().project(&corpus).unwrap();
    assert!(dropped.records.is_empty());
    assert_eq!(dropped.count(Exclusion::RepositoryHistory), 1);
    assert!(dropped.curation.is_some());
    let kept = view()
        .history(HistoryCuration::Keep)
        .project(&corpus)
        .unwrap();
    assert_eq!(kept.records.len(), 1);
}

#[test]
fn feedback_judges_each_verified_attempt_and_weighs_the_smaller_class_up() {
    let corpus = corpus(&[
        Attempt::pass("t1"),
        Attempt::fail("t2"),
        Attempt::fail("t3"),
        Attempt::fail("t4"),
        Attempt::infrastructure("x1"),
    ]);
    let projection = Feedback::new(Strength::Executable)
        .project(&corpus)
        .unwrap();
    assert_eq!(projection.objective, Objective::Feedback);
    let judged: Vec<(bool, Vec<String>)> = projection
        .records
        .iter()
        .map(|r| match &r.body {
            RecordBody::Feedback {
                prompt,
                completion,
                desirable,
                identity,
                ..
            } => {
                assert!(prompt.iter().all(|m| !m.train));
                assert!(completion
                    .iter()
                    .all(|m| m.train == (m.role == "assistant")));
                (*desirable, identity.rollouts.clone())
            }
            other => panic!("a feedback record, got {other:?}"),
        })
        .collect();
    assert_eq!(
        judged,
        [
            (true, vec!["t1".to_string()]),
            (false, vec!["t2".to_string()]),
            (false, vec!["t3".to_string()]),
            (false, vec!["t4".to_string()]),
        ]
    );
    let balance = projection.feedback.unwrap();
    assert_eq!((balance.desirable, balance.undesirable), (1, 3));
    assert!((balance.desirable_weight - 3.0).abs() < 1e-12);
    assert!((balance.undesirable_weight - 1.0).abs() < 1e-12);
    assert_eq!(projection.count(Exclusion::InfrastructureFailure), 1);
}

/// The fixtures of brain's training-record contract, version 2: the lines
/// of one file.
fn fixture_lines(name: &str) -> Vec<Value> {
    // doc-reference: the fixtures are brain's contract, copied unchanged.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/training-records/v2")
        .join(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

/// The keys of the turns brain's valid fixtures of `name` carry in `field`,
/// whether that is one turn or a continuation, and whether the fixtures hold
/// a continuation at all.
fn fixture_turn_keys(lines: &[Value], fields: &[&str]) -> (BTreeSet<String>, bool) {
    let mut keys = BTreeSet::new();
    let mut continuation = false;
    for line in lines {
        let turns = fields
            .iter()
            .filter_map(|f| line.get(*f))
            .chain(line["prompt"].as_array().into_iter().flatten());
        for turn in turns {
            let list: Vec<&Value> = match turn.as_array() {
                Some(list) => {
                    continuation = true;
                    list.iter().collect()
                }
                None => vec![turn],
            };
            for t in list {
                keys.extend(t.as_object().unwrap().keys().cloned());
            }
        }
    }
    (keys, continuation)
}

/// Every line `projection` writes: its keys are those of brain's valid
/// fixture lines (and `tools`, `metadata`), each candidate is one turn or a
/// continuation of turns that opens and ends with an assistant turn, and
/// every turn carries only the keys brain's turns carry (and the optional
/// `tool_calls`, `tool_call_id` and `reasoning_content` of its shape).
fn assert_lines_keep_the_contract(projection: &Projection, fixture: &str, fields: &[&str]) {
    let fixtures = fixture_lines(fixture);
    let mut allowed: BTreeSet<String> = ["tools", "metadata"].map(String::from).into();
    for line in &fixtures {
        allowed.extend(line.as_object().unwrap().keys().cloned());
    }
    let (mut turn_keys, continuation) = fixture_turn_keys(&fixtures, fields);
    assert!(continuation, "brain's {fixture} shows a continuation");
    turn_keys.extend(["tool_calls", "tool_call_id", "reasoning_content"].map(String::from));
    let scratch = Scratch::new(&format!("contract-{}", projection.view));
    let path = scratch.0.join("d.jsonl");
    write_dataset(&path, projection, &Unchecked).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(!text.is_empty());
    for line in text.lines() {
        let line: Value = serde_json::from_str(line).unwrap();
        let keys: BTreeSet<String> = line.as_object().unwrap().keys().cloned().collect();
        assert!(keys.is_subset(&allowed), "{keys:?} beyond {allowed:?}");
        let rollout = &line["metadata"]["rollout"];
        assert!(
            rollout["group"].is_string() && rollout["rollouts"].is_array(),
            "{line}"
        );
        assert_eq!(rollout["tokenizer_digest"], TOKENIZER);
        for field in fields {
            let candidate = &line[*field];
            let turns: Vec<&Value> = match candidate.as_array() {
                Some(list) => list.iter().collect(),
                None => vec![candidate],
            };
            assert_eq!(turns.first().unwrap()["role"], "assistant", "{line}");
            assert_eq!(turns.last().unwrap()["role"], "assistant", "{line}");
            for turn in turns
                .iter()
                .copied()
                .chain(line["prompt"].as_array().unwrap())
            {
                let keys: BTreeSet<String> = turn.as_object().unwrap().keys().cloned().collect();
                assert!(keys.is_subset(&turn_keys), "{keys:?} beyond {turn_keys:?}");
                assert!(turn.get("train").is_none(), "supervision follows position");
            }
        }
    }
}

#[test]
fn the_lines_keep_brains_preference_and_feedback_record_contract() {
    let corpus = corpus(&[
        Attempt::pass("t1"),
        Attempt::fail("t2"),
        Attempt::fail("t3").commands(vec!["pwd", "book wrong"], "Done."),
    ]);
    let preference = view().project(&corpus).unwrap();
    assert_eq!(preference.records.len(), 2);
    assert_lines_keep_the_contract(&preference, "preference.jsonl", &["chosen", "rejected"]);
    let feedback = Feedback::new(Strength::Executable)
        .project(&corpus)
        .unwrap();
    assert_lines_keep_the_contract(&feedback, "feedback.jsonl", &["completion"]);
    // A desirable example of brain's own carries the judgement as a bool.
    assert!(fixture_lines("feedback.jsonl")
        .iter()
        .all(|l| l["desirable"].is_boolean()));
}
