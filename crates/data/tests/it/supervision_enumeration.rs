// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training-data pipelines whose supervision
// masks are proven over every case, not sampled, for its clients. If your team
// needs expertise in verifying agent fine-tuning data, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec, by exhaustive enumeration: over every workspace trajectory of up to
//! four steps after the instruction, drawn from a history read, a file read, a
//! write, a reply without a call and a user's feedback, with every step label
//! (none, good, bad) on each agent step, under a passing and a failing verdict,
//! and under every sft-step rendering (unpacked and packed; every action, only
//! `read_file`'s, only `shell`'s; history dropped, kept, truncated), each
//! record and each count of the projection is what an independent model of the
//! view's documented rules says:
//!
//! * the messages are the system prompt and the instruction, then each step
//!   in order: a call as the
//!   assistant's turn and the tool's reply, a reply as the assistant's turn,
//!   feedback as the user's turn; a record is a prefix of them;
//! * an action is a step with a call, or the reply that ends the trajectory
//!   when no feedback answered it (a reply feedback answered was premature);
//! * a passing experience trains each action, a failing one only the actions
//!   labelled good, and none at all without a good label anywhere
//!   (`Failed`); a bad label is never trained (`BadStep`);
//! * among the actions not trained, the first applicable reason counts it:
//!   a bad label, an action after a truncating history read
//!   (`AfterRepositoryHistory`), a history read not kept
//!   (`RepositoryHistory`), a call that repeats an earlier one with the same
//!   result (`RepeatedAction`), a call of a tool not asked for
//!   (`ActionNotWanted`); an action of a failing experience without its own
//!   good label is context, not counted;
//! * unpacked, a record supervises exactly its last message, one record per
//!   trained action; packed, one record supervises every trained action and
//!   ends at the last of them;
//! * no record supervises anything but an assistant's turn.
//!
//! The model below is written from those rules over the abstract steps, never
//! from the view's code: it does not render, it counts positions.
#![allow(clippy::unwrap_used)]

use std::collections::{BTreeMap, HashSet};

use crate::common;

use atif::TraceStep;
use common::*;
use serde_json::json;
use splinter_core::annotation::{Label, Outcome, Strength};
use splinter_core::chat::WireMessage;
use splinter_core::experience::{Environment, Task};
use splinter_data::curation::HistoryCuration;
use splinter_data::{Corpus, Exclusion, RecordBody, SftStep, View};

/// The steps a trajectory is drawn from, after its instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// `shell` reading the repository's history.
    History,
    /// `read_file` of one file, answered the same each time.
    Read,
    /// `shell` writing a file, answered the same each time.
    Write,
    /// An assistant's turn without a call.
    Say,
    /// The user's feedback on the work.
    Feedback,
}

const KINDS: [Kind; 5] = [
    Kind::History,
    Kind::Read,
    Kind::Write,
    Kind::Say,
    Kind::Feedback,
];
const LABELS: [Option<Label>; 3] = [None, Some(Label::Good), Some(Label::Bad)];

impl Kind {
    /// The tool it calls, its arguments and the reply it got; `None` for a
    /// step without a call.
    fn call(self) -> Option<(&'static str, serde_json::Value, &'static str)> {
        match self {
            Kind::History => Some((
                "shell",
                json!({"command": "git log --oneline -5"}),
                "abc fix",
            )),
            Kind::Read => Some((
                "read_file",
                json!({"path": "src/parse.rs"}),
                "fn parse() {}",
            )),
            Kind::Write => Some((
                "shell",
                json!({"command": "sed -i s/a/b/ src/parse.rs"}),
                "ok",
            )),
            Kind::Say | Kind::Feedback => None,
        }
    }

    fn is_agent(self) -> bool {
        self != Kind::Feedback
    }
}

/// One case of the enumeration.
#[derive(Clone, Debug)]
struct Case {
    steps: Vec<Kind>,
    /// The label of each step (always `None` on feedback).
    labels: Vec<Option<Label>>,
    passed: bool,
}

/// One rendering of the view.
#[derive(Clone, Debug)]
struct Options {
    packed: bool,
    only: Option<&'static str>,
    history: HistoryCuration,
}

/// What the projection of one case must be: each record's roles and the
/// positions it supervises, and every count.
#[derive(Debug, PartialEq, Eq)]
struct Expected {
    records: Vec<(Vec<&'static str>, Vec<usize>)>,
    excluded: BTreeMap<Exclusion, usize>,
}

/// The ATIF step id of step `i` (the instruction is step 1).
fn step_id(i: usize) -> u64 {
    i as u64 + 2
}

/// The model: the projection the documented rules make of `case`.
fn model(case: &Case, options: &Options) -> Expected {
    let mut excluded = BTreeMap::new();
    let mut records = Vec::new();
    let any_good = case.labels.contains(&Some(Label::Good));
    if !case.passed && !any_good {
        excluded.insert(Exclusion::Failed, 1);
        return Expected { records, excluded };
    }
    // The messages, by role, and where each step's assistant turn is: every
    // record opens with the system prompt every model run on a task is sent.
    let mut roles = vec!["system", "user"];
    let mut at = Vec::new();
    for kind in &case.steps {
        at.push(roles.len());
        match kind {
            Kind::Feedback => roles.push("user"),
            Kind::Say => roles.push("assistant"),
            _ => roles.extend(["assistant", "tool"]),
        }
    }
    let last_agent = case.steps.iter().rposition(|k| k.is_agent());
    let actions: Vec<usize> = (0..case.steps.len())
        .filter(|&i| match case.steps[i] {
            Kind::Feedback => false,
            Kind::Say => Some(i) == last_agent && !case.steps[i..].contains(&Kind::Feedback),
            _ => true,
        })
        .collect();
    if actions.is_empty() {
        excluded.insert(Exclusion::NoAction, 1);
        return Expected { records, excluded };
    }
    let mut seen = HashSet::new();
    let repeated: Vec<bool> = case
        .steps
        .iter()
        .map(|k| {
            k.call()
                .is_some_and(|call| !seen.insert(format!("{call:?}")))
        })
        .collect();
    let mut trained = Vec::new();
    let mut truncated = false;
    for &i in &actions {
        let kind = case.steps[i];
        let label = case.labels[i];
        let after_cut = truncated;
        let flagged = kind == Kind::History && options.history != HistoryCuration::Keep;
        truncated |= flagged && options.history == HistoryCuration::Truncate;
        let wanted = match (options.only, kind.call()) {
            (Some(only), Some((tool, _, _))) => tool == only,
            _ => true,
        };
        let reason = if label == Some(Label::Bad) {
            Some(Exclusion::BadStep)
        } else if after_cut {
            Some(Exclusion::AfterRepositoryHistory)
        } else if flagged {
            Some(Exclusion::RepositoryHistory)
        } else if repeated[i] {
            Some(Exclusion::RepeatedAction)
        } else if !wanted {
            Some(Exclusion::ActionNotWanted)
        } else {
            None
        };
        match reason {
            Some(reason) => *excluded.entry(reason).or_insert(0) += 1,
            None if case.passed || label == Some(Label::Good) => trained.push(at[i]),
            None => {}
        }
    }
    let prefix = |end: usize| roles[..=end].to_vec();
    if options.packed {
        if let Some(&last) = trained.last() {
            records.push((prefix(last), trained));
        }
    } else {
        records = trained.into_iter().map(|p| (prefix(p), vec![p])).collect();
    }
    Expected { records, excluded }
}

fn workspace_task() -> Task {
    Task::new(
        "workspace-change",
        Vec::new(),
        Environment::new(Environment::WORKSPACE, json!({"snapshot": "tree-1"})),
        "Fix the parser.",
        Vec::new(),
    )
    .unwrap()
}

/// The corpus of `case`: one experience, its verdict and its labels.
fn corpus(task: &Task, case: &Case) -> Corpus {
    let mut steps: Vec<TraceStep> = vec![user(1, &task.instruction)];
    for (i, kind) in case.steps.iter().enumerate() {
        let id = step_id(i);
        steps.push(match kind.call() {
            Some((tool, arguments, reply)) => call(id, &format!("c{id}"), tool, arguments, reply),
            None if *kind == Kind::Say => say(id, &format!("reply {id}")),
            None => user(id, &format!("feedback {id}")),
        });
    }
    let exp = experience(
        task,
        trajectory(steps),
        Some("done"),
        "2026-11-24T01:00:00.000Z",
    );
    let outcome = if case.passed {
        Outcome::Pass
    } else {
        Outcome::Fail
    };
    let mut notes = vec![verdict(&exp, outcome, Strength::Executable)];
    for (i, l) in case.labels.iter().enumerate() {
        if let Some(l) = l {
            notes.push(label(&exp, step_id(i), *l));
        }
    }
    let mut corpus = Corpus::new();
    corpus.insert(exp, notes).unwrap();
    corpus
}

/// What the view made of `corpus`, in the model's terms.
fn observed(corpus: &Corpus, options: &Options) -> Result<Expected, String> {
    let mut view = SftStep::new(Strength::Executable)
        .packed(options.packed)
        .history(options.history);
    if let Some(only) = options.only {
        view = view.only_actions(vec![only.into()]);
    }
    let projection = view.project(corpus).map_err(|e| e.to_string())?;
    let mut records = Vec::new();
    for record in &projection.records {
        let messages: &[WireMessage] = match &record.body {
            RecordBody::Chat { messages, .. } => messages,
            other => return Err(format!("not a chat record: {other:?}")),
        };
        let supervised: Vec<usize> = (0..messages.len()).filter(|&i| messages[i].train).collect();
        if let Some(&i) = supervised
            .iter()
            .find(|&&i| messages[i].role != "assistant")
        {
            return Err(format!("message {i} ({}) is supervised", messages[i].role));
        }
        let roles = messages
            .iter()
            .map(|m| match m.role.as_str() {
                "system" => "system",
                "user" => "user",
                "assistant" => "assistant",
                "tool" => "tool",
                _ => "other",
            })
            .collect();
        records.push((roles, supervised));
    }
    Ok(Expected {
        records,
        excluded: projection.excluded.clone(),
    })
}

/// Every case of up to `length` steps, every label on every agent step.
fn cases(length: usize) -> Vec<Case> {
    let mut out = vec![(Vec::new(), Vec::new())];
    let mut frontier = out.clone();
    for _ in 0..length {
        let mut next = Vec::new();
        for (steps, labels) in &frontier {
            for kind in KINDS {
                let labels_here: &[Option<Label>] = if kind.is_agent() { &LABELS } else { &[None] };
                for l in labels_here {
                    let mut s: Vec<Kind> = steps.clone();
                    let mut ls: Vec<Option<Label>> = labels.clone();
                    s.push(kind);
                    ls.push(*l);
                    next.push((s, ls));
                }
            }
        }
        out.extend(next.iter().cloned());
        frontier = next;
    }
    out.into_iter()
        .flat_map(|(steps, labels)| {
            [true, false].map(|passed| Case {
                steps: steps.clone(),
                labels: labels.clone(),
                passed,
            })
        })
        .collect()
}

fn renderings() -> Vec<Options> {
    let mut out = Vec::new();
    for packed in [false, true] {
        for only in [None, Some("read_file"), Some("shell")] {
            for history in [
                HistoryCuration::DropStep,
                HistoryCuration::Keep,
                HistoryCuration::Truncate,
            ] {
                out.push(Options {
                    packed,
                    only,
                    history,
                });
            }
        }
    }
    out
}

#[test]
fn every_small_trajectory_under_every_rendering_supervises_what_the_rules_say() {
    let task = workspace_task();
    let renderings = renderings();
    let cases = cases(4);
    let mut disagreements = Vec::new();
    let mut checked = 0usize;
    for case in &cases {
        let corpus = corpus(&task, case);
        for options in &renderings {
            checked += 1;
            let want = model(case, options);
            match observed(&corpus, options) {
                Ok(seen) if seen == want => {}
                seen => disagreements.push(format!(
                    "{case:?} {options:?}\n  want {want:?}\n  seen {seen:?}"
                )),
            }
        }
    }
    assert!(
        disagreements.is_empty(),
        "{} of {checked} projections disagree with the rules; the first:\n{}",
        disagreements.len(),
        disagreements[..disagreements.len().min(6)].join("\n")
    );
    // The enumeration is what it says: every case of up to four steps.
    assert_eq!(
        cases.len(),
        2 * (1 + 13 + 13usize.pow(2) + 13usize.pow(3) + 13usize.pow(4))
    );
}

#[test]
fn the_model_is_not_vacuous() {
    // A pass of read, write, reply trains all three, one record each; the
    // model must say so and must differ from a broken view's projection.
    let case = Case {
        steps: vec![Kind::Read, Kind::Write, Kind::Say],
        labels: vec![None; 3],
        passed: true,
    };
    let options = Options {
        packed: false,
        only: None,
        history: HistoryCuration::DropStep,
    };
    let want = model(&case, &options);
    assert_eq!(
        want.records.iter().map(|r| r.1.clone()).collect::<Vec<_>>(),
        [vec![2], vec![4], vec![6]]
    );
    assert!(want.excluded.is_empty());
}
