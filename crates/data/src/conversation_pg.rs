// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reinforcement learning for agents that work
// with people, for its clients. If your team needs expertise in training
// agents against simulated clients and students, you can procure our services
// by sending an email to info@swedishembedded.com.

//! The policy-gradient view for conversations with a counterpart.
//!
//! An attempt is one conversation of the learner with a counterpart at one case
//! (a client to take on, a student to teach); its reward is the grade of the
//! case. A group is the attempts at one case by one policy, written as in
//! [`views::PolicyGradient`](crate::views::PolicyGradient): each record carries
//! its attempt's reward, the group's key and the attempt's name, and the
//! trainer computes the advantages. The learner's turns are the actions: each
//! becomes one record whose context is the role prompt and the conversation so
//! far (the counterpart's words as `user`, the learner's own as `assistant`)
//! and whose only supervised message is that turn. A group of one attempt, or
//! whose attempts all earned the same, says nothing and yields no record.

use std::collections::BTreeMap;

use serde_json::Value;
use splinter_core::chat::WireMessage;

use crate::record::{Record, RecordBody, RecordMetadata, RolloutOutcome};
use crate::render::message;
use crate::views::{AdvantageKind, GroupKind, GroupReport, TrialReport};
use crate::{Exclusion, Objective, Projection};

/// The name every record of this view carries.
pub const VIEW: &str = "conversation-policy-gradient";

/// One graded conversation.
#[derive(Clone, Debug)]
pub struct Attempt {
    /// The case the learner was set; the attempts at one case form a group.
    pub case: String,
    /// The model that played the learner.
    pub policy: String,
    /// This attempt's name.
    pub trial: String,
    /// What the case's grade earned the attempt, in [0, 1].
    pub reward: f64,
    /// The conversation record (sven's `ConversationRecord` as JSON).
    pub record: Value,
}

/// The learner's turns of a record as the training requests they were: for
/// each, the messages before it and its text.
fn actions(system_prompt: &str, record: &Value) -> Vec<Vec<WireMessage>> {
    let mut context = vec![message("system", system_prompt, false)];
    let mut out = Vec::new();
    for exchange in record["exchanges"].as_array().into_iter().flatten() {
        let text = exchange["text"].as_str().unwrap_or_default();
        if exchange["speaker"] == "learner" {
            let mut request = context.clone();
            request.push(message("assistant", text, true));
            out.push(request);
            context.push(message("assistant", text, false));
        } else {
            context.push(message("user", text, false));
        }
    }
    out
}

/// The projection of `attempts` under `system_prompt` (the role the learner
/// played): groups with a spread of rewards become records, the rest are
/// counted as [`Exclusion::NoGroupSignal`] (one count per attempt).
#[must_use]
pub fn project(
    system_prompt: &str,
    attempts: &[Attempt],
    advantage_kind: AdvantageKind,
) -> Projection {
    let mut projection = Projection::new(VIEW, Objective::PolicyGradient, None, None);
    projection.system_prompt = Some(system_prompt.to_string());
    projection.advantage = Some(advantage_kind);
    let mut groups: BTreeMap<(&str, &str), Vec<&Attempt>> = BTreeMap::new();
    for attempt in attempts {
        groups
            .entry((&attempt.case, &attempt.policy))
            .or_default()
            .push(attempt);
    }
    for ((case, policy), members) in groups {
        let rewards: Vec<f64> = members.iter().map(|a| a.reward).collect();
        let signal = crate::views::signals(&rewards);
        let written = advantage_kind.keeps(&rewards);
        let kind = if rewards.iter().all(|r| *r >= 1.0) {
            GroupKind::AllPass
        } else if !signal && rewards.iter().all(|r| *r <= 0.0) {
            GroupKind::AllFail
        } else {
            GroupKind::Graded
        };
        let kept = |i: usize| written.as_ref().is_some_and(|w| w[i]);
        projection.groups.push(GroupReport {
            task: case.to_string(),
            version: String::new(),
            policy: policy.to_string(),
            trials: members
                .iter()
                .enumerate()
                .map(|(i, a)| TrialReport {
                    trial: a.trial.clone(),
                    reward: a.reward,
                    written: kept(i),
                    conversations: 1,
                })
                .collect(),
            signal,
            kind,
            distilled: false,
        });
        for (i, attempt) in members.iter().enumerate() {
            if !kept(i) {
                *projection
                    .excluded
                    .entry(Exclusion::NoGroupSignal)
                    .or_insert(0) += 1;
                continue;
            }
            let outcome = RolloutOutcome::of(attempt.reward, &[case, policy], &attempt.trial);
            for messages in actions(system_prompt, &attempt.record) {
                projection.records.push(Record {
                    body: RecordBody::chat(messages).with_outcome(outcome.clone()),
                    metadata: RecordMetadata {
                        experiences: Vec::new(),
                        task: None,
                        sources: Vec::new(),
                        group: Some(case.to_string()),
                        split: None,
                        view: VIEW.to_string(),
                        objective: Objective::PolicyGradient,
                    },
                });
            }
        }
    }
    projection
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn attempt(case: &str, trial: &str, reward: f64, turns: &[(&str, &str)]) -> Attempt {
        Attempt {
            case: case.into(),
            policy: "m".into(),
            trial: trial.into(),
            reward,
            record: json!({"exchanges": turns.iter()
                .map(|(s, t)| json!({"speaker": s, "text": t})).collect::<Vec<_>>()}),
        }
    }

    const TALK: [(&str, &str); 4] = [
        ("seat", "Hello."),
        ("learner", "What happened?"),
        ("seat", "They were late."),
        ("learner", "Did you give notice?"),
    ];

    #[test]
    fn every_learner_turn_of_a_group_with_a_spread_is_an_action_with_its_attempts_reward_and_group()
    {
        let attempts = [
            attempt("c1", "a", 1.0, &TALK),
            attempt("c1", "b", 0.0, &TALK),
        ];
        let projection = project("You are a lawyer.", &attempts, AdvantageKind::Standardised);
        assert_eq!(projection.records.len(), 4, "2 attempts x 2 learner turns");
        assert_eq!(projection.advantage, Some(AdvantageKind::Standardised));
        let outcome = |at: usize| match &projection.records[at].body {
            RecordBody::Chat { outcome, .. } => outcome.clone().expect("a rollout record"),
            _ => panic!("a chat record"),
        };
        let RecordBody::Chat { messages, .. } = &projection.records[1].body else {
            panic!("a chat record")
        };
        let roles: Vec<&str> = messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, ["system", "user", "assistant", "user", "assistant"]);
        let trained: Vec<bool> = messages.iter().map(|m| m.train).collect();
        assert_eq!(trained, [false, false, false, false, true]);
        assert_eq!(messages[4].content, "Did you give notice?");
        let (better, worse) = (outcome(1), outcome(3));
        assert_eq!((better.reward, worse.reward), (1.0, 0.0));
        assert_eq!(
            better.group, worse.group,
            "one group: the attempts at one case by one policy"
        );
        assert_eq!(
            (better.rollout.as_deref(), worse.rollout.as_deref()),
            (Some("a"), Some("b")),
            "each attempt is one rollout of the group, whatever its number of turns"
        );
        assert_eq!(outcome(0), better);
        assert!(projection.groups[0].signal);
    }

    #[test]
    fn positive_leaves_out_the_conversations_at_or_below_their_groups_mean_reward() {
        let attempts = [
            attempt("c1", "a", 1.0, &TALK),
            attempt("c1", "b", 0.75, &TALK),
            attempt("c1", "c", 0.0, &TALK),
        ];
        let projection = project("You are a lawyer.", &attempts, AdvantageKind::Positive);
        assert_eq!(
            projection.records.len(),
            4,
            "the two attempts above the mean"
        );
        assert!(projection.records.iter().all(|r| matches!(
            &r.body,
            RecordBody::Chat { outcome: Some(o), .. } if o.reward > 0.5
        )));
        let written: Vec<bool> = projection.groups[0]
            .trials
            .iter()
            .map(|t| t.written)
            .collect();
        assert_eq!(written, [true, true, false], "every attempt still reported");
        assert_eq!(projection.excluded[&Exclusion::NoGroupSignal], 1);

        // One attempt above the mean leaves the trainer nothing to compare.
        let lone = [
            attempt("c1", "a", 1.0, &TALK),
            attempt("c1", "b", 0.0, &TALK),
            attempt("c1", "c", 0.0, &TALK),
        ];
        let lone = project("p", &lone, AdvantageKind::Positive);
        assert!(lone.records.is_empty());
        assert_eq!(lone.excluded[&Exclusion::NoGroupSignal], 3);
    }

    #[test]
    fn a_group_that_all_earned_the_same_or_of_one_says_nothing() {
        let attempts = [
            attempt("alike", "a", 0.5, &TALK),
            attempt("alike", "b", 0.5, &TALK),
            attempt("alone", "a", 1.0, &TALK),
        ];
        let projection = project("p", &attempts, AdvantageKind::Standardised);
        assert!(projection.records.is_empty());
        assert_eq!(projection.excluded[&Exclusion::NoGroupSignal], 3);
        assert!(projection.groups.iter().all(|g| !g.signal));
    }
}
