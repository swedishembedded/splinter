// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for proving that the training records one system writes are the records
// another system's trainer reads, for its clients. If your team needs expertise in training-data contracts between
// agent runtimes and trainers then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: every dataset a Splinter view writes is read by brain's trainer without a refusal, and what the trainer
//! supervises is the intended supervision: the assistant's actions, never a system prompt, a user turn or a tool's
//! observation. Counts are exact: an observation or a user turn supervised by mistake would add its characters.
//!
//! One thing of brain's is encoded as it is observed. A record whose context holds no tool turn and whose answers
//! call no tool is rendered as one conversation, and brain then supervises each answer's own header
//! (`<|im_start|>assistant` and its newline) before the answer; with tools in play the header is context. The header
//! is [`HEADER`] tokens either way, so the count stays exact and a change of brain's rule is seen here.

#![allow(clippy::unwrap_used)]

use splinter_core::annotation::{Outcome, Strength};
use splinter_data::{
    write_dataset, Feedback, PolicyGradient, Preference, Projection, SftFinal, SftStep, Unchecked,
    View,
};
use splinter_model::served_train::{CheckSpec, DatasetKind, DatasetReport};

use crate::bench::{answer_prompt, Bench};
use crate::corpus::{attempts, corpus_of, Action, Attempt};
use crate::sampled::{group, parse, Sampled};

/// The tokens of an answer's header that brain supervises on a plain conversation.
fn header() -> usize {
    answer_prompt().len()
}

/// The tokens an action is trained as: its text, or its call's name and arguments, and the end-of-turn marker.
fn action_tokens(action: &Action) -> usize {
    action.rendered().chars().count() + 1
}

fn uses_tools(attempt: &Attempt) -> bool {
    attempt
        .actions
        .iter()
        .any(|x| matches!(x, Action::Calls { .. }))
}

/// A conversation of replies with a person's turn between them, which a closed task's step view does not take.
fn is_dialogue(attempt: &Attempt) -> bool {
    !uses_tools(attempt) && attempt.actions.len() > 1
}

fn passed(attempts: &[Attempt]) -> impl Iterator<Item = &Attempt> {
    attempts.iter().filter(|a| a.outcome == Outcome::Pass)
}

fn project(
    bench: &Bench,
    name: &str,
    view: &dyn View,
    corpus: &splinter_data::Corpus,
) -> (std::path::PathBuf, Projection) {
    let projection = view.project(corpus).unwrap();
    let path = bench.path(&format!("{name}.jsonl"));
    write_dataset(&path, &projection, &Unchecked).unwrap();
    (path, projection)
}

/// The supervised tokens of every record, in the order brain read them, after asserting none is refused.
fn trained(report: &DatasetReport, records: usize) -> Vec<usize> {
    assert!(
        report.refused.is_empty(),
        "brain refuses: {:?}",
        report.refused
    );
    assert_eq!((report.records, report.valid_records), (records, records));
    assert_eq!(report.examples_without_supervision, 0);
    report.per_record.iter().map(|r| r.trained).collect()
}

fn sorted(mut counts: Vec<usize>) -> Vec<usize> {
    counts.sort_unstable();
    counts
}

#[test]
#[ignore = "needs a brain binary on PATH: make test/contract/brain runs it"]
fn the_views_write_what_brains_trainer_reads_and_supervises_the_intended_tokens() {
    let bench = Bench::start();
    let attempts = attempts();
    let corpus = corpus_of(&attempts);

    // sft-step: every action of a passing tool-using run, each in the context the run had; the failed attempt and
    // the dialogue (a second user step a closed task does not represent) write nothing.
    let (path, projection) = project(
        &bench,
        "sft-step",
        &SftStep::new(Strength::Executable),
        &corpus,
    );
    let expected: Vec<usize> = passed(&attempts)
        .filter(|a| uses_tools(a))
        .flat_map(|a| a.actions.iter().map(action_tokens))
        .collect();
    let report = bench.check_as(&path, DatasetKind::Chat);
    assert_eq!(
        sorted(trained(&report, projection.records.len())),
        sorted(expected),
        "only the assistant's actions are supervised"
    );

    // sft-final: the final answer alone of a tool run; a dialogue's every reply (the other speaker's turns are
    // context).
    let (path, projection) = project(
        &bench,
        "sft-final",
        &SftFinal::new(Strength::Executable),
        &corpus,
    );
    let expected: Vec<usize> = passed(&attempts)
        .map(|a| {
            let replies = if uses_tools(a) {
                &a.actions[a.actions.len() - 1..]
            } else {
                &a.actions[..]
            };
            replies.iter().map(|x| action_tokens(x) + header()).sum()
        })
        .collect();
    let report = bench.check_as(&path, DatasetKind::Chat);
    assert_eq!(
        sorted(trained(&report, projection.records.len())),
        sorted(expected)
    );

    // preference: the pass of the task is chosen over its fail; brain counts the longer candidate, the chosen run
    // with its call, its observation context and its answer.
    let (path, projection) = project(
        &bench,
        "preference",
        &Preference::new(Strength::Executable),
        &corpus,
    );
    let report = bench.check_as(&path, DatasetKind::Preference);
    let chosen: usize = attempts[0].actions.iter().map(action_tokens).sum();
    assert_eq!(trained(&report, projection.records.len()), vec![chosen]);
    assert_eq!(report.per_record[0].examples, 2, "a pair is two examples");

    // feedback: each judged completion on its own; the failed attempt's is undesirable and trained as a plain turn.
    let (path, projection) = project(
        &bench,
        "feedback",
        &Feedback::new(Strength::Executable),
        &corpus,
    );
    let report = bench.check_as(&path, DatasetKind::Feedback);
    let expected: Vec<usize> = attempts
        .iter()
        .filter(|a| !is_dialogue(a))
        .map(|a| {
            let tokens: usize = a.actions.iter().map(action_tokens).sum();
            tokens + if uses_tools(a) { 0 } else { header() }
        })
        .collect();
    assert_eq!(
        sorted(trained(&report, projection.records.len())),
        sorted(expected)
    );
}

/// The record lines of a written policy-gradient dataset.
fn rollouts(bench: &Bench, packed: bool) -> (std::path::PathBuf, Vec<String>) {
    let corpus = group(&bench.tokenizer);
    let view = PolicyGradient::new(Strength::Executable).packed(packed);
    let (path, projection) = project(
        bench,
        if packed { "pg-packed" } else { "pg" },
        &view,
        &corpus,
    );
    assert_eq!(
        projection.tokens.exact, 4,
        "every action of both attempts carries its sampled ids"
    );
    let lines = std::fs::read_to_string(&path)
        .unwrap()
        .lines()
        .map(String::from)
        .collect();
    (path, lines)
}

#[test]
#[ignore = "needs a brain binary on PATH: make test/contract/brain runs it"]
fn a_policy_gradient_group_is_read_with_exactly_the_sampled_ids_and_a_broken_record_is_refused() {
    let bench = Bench::start();
    let sampled = Sampled::new();
    let (path, lines) = rollouts(&bench, false);
    assert_eq!(lines.len(), 4, "two attempts of two actions");

    let report = bench.check_as(&path, DatasetKind::Chat);
    assert_eq!(
        sorted(trained(&report, 4)),
        sorted(vec![
            sampled.call.len(),
            sampled.call.len(),
            sampled.answer.len(),
            sampled.answer.len()
        ]),
        "brain trains the sampled ids, not a re-encoding of the text"
    );

    // Records Splinter must never write, made by hand from a valid one: a log-probability short, and an id that
    // is not the one the answer renders to. Brain refuses each, naming its rule.
    let mut short = parse(&lines[0]);
    let supervised = |record: &mut serde_json::Value| -> usize {
        record["messages"]
            .as_array()
            .unwrap()
            .iter()
            .position(|m| m["train"] == true)
            .unwrap()
    };
    let at = supervised(&mut short);
    short["messages"][at]["behavior_logprobs"]
        .as_array_mut()
        .unwrap()
        .pop();
    let mut wrong = parse(&lines[0]);
    wrong["messages"][at]["token_ids"][0] = serde_json::json!(0);
    let broken = bench.path("pg-broken.jsonl");
    let text: String = lines
        .iter()
        .cloned()
        .chain([short.to_string(), wrong.to_string()])
        .map(|l| l + "\n")
        .collect();
    std::fs::write(&broken, text).unwrap();
    let report = bench.check_as(&broken, DatasetKind::Chat);
    let refused: Vec<(usize, Option<String>)> = report
        .refused
        .iter()
        .map(|r| (r.line, r.rule.clone()))
        .collect();
    assert_eq!(
        refused,
        vec![
            (5, Some("behavior_logprob_count".into())),
            (6, Some("token_ids_mismatch".into()))
        ],
        "{:?}",
        report.refused
    );
    assert_eq!(report.valid_records, 4);

    // The packed form: one record per attempt, both actions supervised as two answers of it.
    let (packed, lines) = rollouts(&bench, true);
    assert_eq!(lines.len(), 2);
    let report = bench.check(&packed, &CheckSpec::chat("contract/chars", false, true));
    let both = sampled.call.len() + sampled.answer.len();
    assert_eq!(trained(&report, 2), vec![both, both]);
}
