// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for teaching models multi-step work from the steps that were right, for
// its clients. If your team needs expertise in process (step-level) verification and supervision then you can
// procure our services by sending an email to info@swedishembedded.com.

//! Process (step-level) verification: label each step of a multi-step conversation by a check the policy cannot
//! reach, and project the steps that checked out into training records.
//!
//! An outcome check sees only the last answer, so a conversation whose first step was right and whose second was
//! wrong teaches nothing. A [`StepChecker`] judges every step in the context the conversation gave it, so the right
//! first step is kept: [`project`] writes one record per good step, with the steps before it as unsupervised
//! context and the step itself as the only supervised turn. [`outcome_only`] writes what an outcome check keeps,
//! the whole conversation of a run whose every step was good, for comparison at the same number of rollouts.

use splinter_core::annotation::Label;
use splinter_core::chat::WireMessage;

use crate::record::{Record, RecordBody, RecordMetadata};
use crate::{Exclusion, Objective, Projection};

/// The name every record of this view carries.
pub const VIEW: &str = "process";

/// One step of a conversation: what the other speaker said, and the step the policy took in answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Step {
    /// What the other speaker said.
    pub said: String,
    /// The policy's answer: the step.
    pub step: String,
}

/// A deterministic judge of one step, given the steps before it. It is the process counterpart of the outcome
/// verifier: it has the knowledge the policy lacks, and it judges the step as the conversation had it (a step that
/// is right for the context the policy's own earlier step made is good, even if that earlier step was wrong).
pub trait StepChecker {
    /// Whether `conversation[at].step` is a right step in the context of the steps before it.
    fn check(&self, conversation: &[Step], at: usize) -> bool;
}

/// The label of every step of `conversation`: [`Label::Good`] where `checker` accepts it, else [`Label::Bad`].
#[must_use]
pub fn label(conversation: &[Step], checker: &dyn StepChecker) -> Vec<Label> {
    (0..conversation.len())
        .map(|at| {
            if checker.check(conversation, at) {
                Label::Good
            } else {
                Label::Bad
            }
        })
        .collect()
}

fn message(role: &str, content: &str, train: bool) -> WireMessage {
    WireMessage {
        images: Vec::new(),
        role: role.into(),
        content: content.into(),
        tool_calls: Vec::new(),
        tool_call_id: None,
        reasoning_content: None,
        train,
        loss_weight: None,
    }
}

fn record(messages: Vec<WireMessage>, group: &str) -> Record {
    Record {
        body: RecordBody::chat(messages),
        metadata: RecordMetadata {
            experiences: Vec::new(),
            task: None,
            sources: Vec::new(),
            group: Some(group.into()),
            split: None,
            split_draw: None,
            view: VIEW.into(),
            objective: Objective::Sft,
        },
    }
}

/// The conversation up to and including the user turn of step `at`, every turn unsupervised.
fn context(
    system_prompt: &str,
    conversation: &[Step],
    at: usize,
    supervise: impl Fn(usize) -> bool,
) -> Vec<WireMessage> {
    let mut messages = vec![message("system", system_prompt, false)];
    for (index, step) in conversation.iter().enumerate().take(at + 1) {
        messages.push(message("user", &step.said, false));
        messages.push(message("assistant", &step.step, supervise(index)));
    }
    messages
}

/// The records of `conversations` (each a case's name and its steps) under `system_prompt`, one per step the
/// checker labelled good: the steps before it are context and only the step is supervised. A step labelled bad is
/// counted as [`Exclusion::BadStep`].
#[must_use]
pub fn project(
    system_prompt: &str,
    conversations: &[(String, Vec<Step>)],
    checker: &dyn StepChecker,
) -> Projection {
    let mut projection = Projection::new(VIEW, Objective::Sft, None, None);
    projection.system_prompt = Some(system_prompt.to_string());
    for (case, conversation) in conversations {
        for (at, label) in label(conversation, checker).into_iter().enumerate() {
            if label == Label::Bad {
                *projection.excluded.entry(Exclusion::BadStep).or_insert(0) += 1;
                continue;
            }
            let messages = context(system_prompt, conversation, at, |index| index == at);
            projection.records.push(record(messages, case));
        }
    }
    projection
}

/// What an outcome check keeps from the same conversations: the whole conversation, every step supervised, of each
/// run in which every step was good. Anything less is counted as [`Exclusion::Failed`].
#[must_use]
pub fn outcome_only(
    system_prompt: &str,
    conversations: &[(String, Vec<Step>)],
    checker: &dyn StepChecker,
) -> Projection {
    let mut projection = Projection::new(VIEW, Objective::Sft, None, None);
    projection.system_prompt = Some(system_prompt.to_string());
    for (case, conversation) in conversations {
        if label(conversation, checker)
            .iter()
            .all(|l| *l == Label::Good)
        {
            let messages = context(system_prompt, conversation, conversation.len() - 1, |_| {
                true
            });
            projection.records.push(record(messages, case));
        } else {
            *projection.excluded.entry(Exclusion::Failed).or_insert(0) += 1;
        }
    }
    projection
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A step is good when it is the previous text upper-cased: the first step answers the first said, the second
    /// answers the first step.
    struct Upper;

    impl StepChecker for Upper {
        fn check(&self, conversation: &[Step], at: usize) -> bool {
            let before = if at == 0 {
                &conversation[0].said
            } else {
                &conversation[at - 1].step
            };
            conversation[at].step == before.to_uppercase()
        }
    }

    fn step(said: &str, step: &str) -> Step {
        Step {
            said: said.into(),
            step: step.into(),
        }
    }

    fn cases() -> Vec<(String, Vec<Step>)> {
        vec![
            ("both".into(), vec![step("a", "A"), step("go", "A")]),
            ("first-only".into(), vec![step("b", "B"), step("go", "x")]),
            ("second-only".into(), vec![step("c", "z"), step("go", "Z")]),
            ("neither".into(), vec![step("d", "q"), step("go", "q")]),
        ]
    }

    #[test]
    fn a_step_is_judged_in_the_context_the_conversation_gave_it() {
        let labels: Vec<_> = cases().iter().map(|(_, c)| label(c, &Upper)).collect();
        assert_eq!(labels[0], [Label::Good, Label::Good]);
        assert_eq!(labels[1], [Label::Good, Label::Bad]);
        assert_eq!(
            labels[2],
            [Label::Bad, Label::Good],
            "right for the context its own earlier step made"
        );
        assert_eq!(labels[3], [Label::Bad, Label::Bad]);
    }

    #[test]
    fn every_good_step_is_a_record_that_supervises_that_step_alone() {
        let projection = project("role", &cases(), &Upper);
        assert_eq!(projection.records.len(), 4, "2 + 1 + 1 + 0 good steps");
        assert_eq!(projection.count(Exclusion::BadStep), 4);
        let RecordBody::Chat { messages, .. } = &projection.records[1].body else {
            panic!("a chat record")
        };
        let shape: Vec<_> = messages
            .iter()
            .map(|m| (m.role.as_str(), m.content.as_str(), m.train))
            .collect();
        assert_eq!(
            shape,
            [
                ("system", "role", false),
                ("user", "a", false),
                ("assistant", "A", false),
                ("user", "go", false),
                ("assistant", "A", true)
            ]
        );
    }

    #[test]
    fn an_outcome_check_keeps_only_the_conversations_in_which_every_step_was_good() {
        let projection = outcome_only("role", &cases(), &Upper);
        assert_eq!(projection.records.len(), 1);
        assert_eq!(projection.count(Exclusion::Failed), 3);
        let RecordBody::Chat { messages, .. } = &projection.records[0].body else {
            panic!("a chat record")
        };
        assert!(messages
            .iter()
            .filter(|m| m.role == "assistant")
            .all(|m| m.train));
    }
}
