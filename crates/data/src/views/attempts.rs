// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that turn graded agent attempts
// into preference and feedback training data for its clients. If your team
// needs expertise in learning from verified successes and failures of agents
// then you can procure our services by sending an email to
// info@swedishembedded.com.

//! What a verdict says of the attempt an experience is part of, for the views
//! that compare attempts ([`Preference`](super::Preference)) or judge them one
//! at a time ([`Feedback`](super::Feedback)).
//!
//! An attempt is *verified* when its verdicts decide it, pass or fail, at the
//! view's minimum strength or above. Everything else is left out, typed:
//!
//! * an infrastructure failure ([`Exclusion::InfrastructureFailure`]): the
//!   examiner's score record says the sandbox, the model server or the agent
//!   failed. Such a trial has no verdict (the intake never turns one into a
//!   fail), and it is never a rejected or undesirable sample: the
//!   infrastructure failing is not the model failing;
//! * no decision ([`Exclusion::Undecided`]): no verdict, conflicting ones, or
//!   a trial the examiner did not count for another reason;
//! * a pass decided below the minimum strength ([`Exclusion::UnverifiedPass`]):
//!   never a chosen or desirable sample; a fail below it is
//!   [`Exclusion::TooWeak`].
//!
//! A verified attempt belongs to a group: one version of one task attempted by
//! one policy, read from the deciding verdict's evidence (the trial, the task,
//! its version and the model, as the bench intake records them) and otherwise
//! from the experience itself (its task, its policy or solver). Its name in
//! the group is its trial, else its attempt number, else its experience id.
//! The group's key is the digest a policy-gradient record names its group by,
//! so the records of both views name the same groups.

use serde_json::Value;
use splinter_core::annotation::{AnnotationBody, Outcome, Strength};
use splinter_core::digest::digest_of;

use crate::trajectory::Conversation;
use crate::{Entry, Exclusion};

/// The score statuses of an examiner's record that say the infrastructure,
/// not the model, failed.
const INFRASTRUCTURE_STATUSES: [&str; 3] = ["environment_error", "model_error", "agent_error"];

/// A verified attempt.
#[derive(Clone, Debug)]
pub(crate) struct Attempt {
    /// Whether it passed.
    pub(crate) passed: bool,
    /// The strength of the verdicts that decided it.
    pub(crate) strength: Strength,
    /// The key of its group: one version of one task by one policy.
    pub(crate) group: String,
    /// Its name in the group.
    pub(crate) name: String,
}

/// The verified attempt `entry` is, or why it is left out.
pub(crate) fn verified(entry: &Entry, min_strength: Strength) -> Result<Attempt, Exclusion> {
    let Some(decision) = entry.decision() else {
        return Err(if infrastructure_failure(entry) {
            Exclusion::InfrastructureFailure
        } else {
            Exclusion::Undecided
        });
    };
    if decision.strength < min_strength {
        return Err(if decision.passed {
            Exclusion::UnverifiedPass
        } else {
            Exclusion::TooWeak
        });
    }
    let counted = if decision.passed {
        Outcome::Pass
    } else {
        Outcome::Fail
    };
    let evidence = entry.notes.iter().find_map(|note| match &note.body {
        AnnotationBody::Verdict {
            outcome,
            strength,
            evidence,
        } if *strength == decision.strength && strength.counted(*outcome) == counted => {
            Some(evidence)
        }
        _ => None,
    });
    let text = |pointer: &str| {
        evidence
            .and_then(|e| e.pointer(pointer))
            .map(|v| v.as_str().map_or_else(|| v.to_string(), str::to_string))
    };
    let provenance = &entry.experience.provenance;
    let task = text("/task/id").unwrap_or_else(|| entry.experience.task.id.to_string());
    let version = text("/task/version").unwrap_or_default();
    let policy = text("/model")
        .or_else(|| provenance.policy.clone())
        .unwrap_or_else(|| provenance.solver.clone());
    let name = text("/trial")
        .or_else(|| provenance.attempt.map(|a| format!("attempt-{a}")))
        .unwrap_or_else(|| entry.id.to_string());
    Ok(Attempt {
        passed: decision.passed,
        strength: decision.strength,
        group: digest_of(&[task.as_str(), version.as_str(), policy.as_str()]).to_string(),
        name,
    })
}

/// Whether a verdict of `entry` carries an examiner's score record saying the
/// infrastructure failed.
fn infrastructure_failure(entry: &Entry) -> bool {
    entry.notes.iter().any(|note| match &note.body {
        AnnotationBody::Verdict { evidence, .. } => evidence
            .pointer("/score/status")
            .and_then(Value::as_str)
            .is_some_and(|status| INFRASTRUCTURE_STATUSES.contains(&status)),
        _ => false,
    })
}

/// The tokenizer the server that decoded `entry`'s own steps encoded them
/// with, as each agent step's policy trace names it (`extra.sven.
/// tokenizer_digest`); steps copied from another trajectory are not its own.
/// `None` when no step names one; [`Exclusion::MixedTokenizerDigest`] when
/// two steps name different ones.
pub(crate) fn tokenizer_digest(entry: &Entry) -> Result<Option<String>, Exclusion> {
    let mut named: Option<&str> = None;
    for step in &entry.experience.trajectory.steps {
        if step.source != atif::StepOrigin::Agent || step.is_excluded_from_sft() {
            continue;
        }
        let digest = step
            .metrics
            .as_ref()
            .and_then(|m| m.extra.as_ref())
            .and_then(|extra| extra.pointer("/sven/tokenizer_digest"))
            .and_then(Value::as_str);
        match (named, digest) {
            (_, None) => {}
            (None, Some(digest)) => named = Some(digest),
            (Some(seen), Some(digest)) if seen != digest => {
                return Err(Exclusion::MixedTokenizerDigest)
            }
            _ => {}
        }
    }
    Ok(named.map(str::to_string))
}

/// Where the model's own part of `conversation` begins: the index of its
/// first action that is not copied from another trajectory; `None` when it
/// has none.
pub(crate) fn own_start(conversation: &Conversation) -> Option<usize> {
    conversation
        .actions
        .iter()
        .find(|a| !a.copied)
        .map(|a| a.index)
}

/// The index of the last assistant turn of `conversation`, at or after
/// `from`: where a continuation from `from` ends (observations after the last
/// action are context nobody acted on).
pub(crate) fn last_action_from(conversation: &Conversation, from: usize) -> Option<usize> {
    conversation
        .messages
        .iter()
        .enumerate()
        .skip(from)
        .rev()
        .find(|(_, m)| m.role == "assistant")
        .map(|(i, _)| i)
}

/// The continuation `messages[from..=to]` as a candidate: its assistant turns
/// supervised, every other turn context.
pub(crate) fn continuation(
    conversation: &Conversation,
    from: usize,
    to: usize,
) -> Vec<splinter_core::chat::WireMessage> {
    conversation.messages[from..=to]
        .iter()
        .map(|m| {
            let mut m = m.clone();
            m.train = m.role == "assistant";
            m
        })
        .collect()
}

/// The conversation before `to`, none of it supervised.
pub(crate) fn context(
    conversation: &Conversation,
    to: usize,
) -> Vec<splinter_core::chat::WireMessage> {
    conversation.messages[..to]
        .iter()
        .map(|m| {
            let mut m = m.clone();
            m.train = false;
            m
        })
        .collect()
}

/// Whether an assistant turn of `turns` calls a tool that reads the
/// repository's history or git's storage ([`crate::curation`]).
pub(crate) fn reads_history(turns: &[splinter_core::chat::WireMessage]) -> bool {
    turns
        .iter()
        .filter(|m| m.role == "assistant")
        .flat_map(|m| &m.tool_calls)
        .any(|call| crate::curation::reads_repository_history(&call.function.arguments))
}

/// The one tool list `conversation` was held under (empty for none), or
/// [`Exclusion::Unrepresentable`] when it was held under several: a pair or
/// completion line carries one.
pub(crate) fn one_tool_set(conversation: &Conversation) -> Result<Vec<Value>, Exclusion> {
    match conversation.tool_sets.as_slice() {
        [] => Ok(Vec::new()),
        [one] => Ok(one.clone()),
        _ => Err(Exclusion::Unrepresentable),
    }
}
