// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that turn verified rollouts into
// the right training signal: advantages, repairs and distillation targets,
// for its clients. If your team needs expertise in training-signal
// construction for policy-gradient and supervised repair you can procure our
// services by sending an email to info@swedishembedded.com.

//! The training records a batch becomes.
//!
//! One batch of rollouts yields several views of the same facts, and the
//! strategy decides which: the group-relative advantage of every attempt
//! (two-sided or positive-only), the repair that turns a wrong attempt and the
//! verifier's feedback into the fix a later attempt found, the per-token
//! distillation target of an attempt in a group that all failed, and the
//! passing completions kept for replay. A rollout whose policy trace is
//! marked unprovable is never a record; one with no trace (an in-process
//! sampler) is trained on the start-of-run behaviour proxy.

use serde::Serialize;
use splinter_core::chat::WireMessage;
use splinter_data::{Objective, Record, RecordBody, RecordMetadata, TeacherMessage};

use super::config::AdvantageMode;
use super::types::{Rollout, RolloutGroup};

/// The text a repair turn shows when the verifier gave no feedback.
const GENERIC_FEEDBACK: &str = "That answer is wrong. Try again.";

/// What was built, and what was left out and why.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct RecordCounts {
    /// Rollout records carrying an advantage.
    pub policy_gradient: usize,
    /// Repair records.
    pub repair: usize,
    /// Distillation records (a teacher context beside the attempt).
    pub distilled: usize,
    /// Imitation records of a teacher's verified answers.
    pub taught: usize,
    /// Replay records of earlier arms.
    pub replay: usize,
    /// Attempts left out because their policy trace is unprovable.
    pub dropped_unprovable: usize,
    /// Attempts of all-failed groups left out of distillation because the
    /// verifier showed nothing privileged.
    pub no_privileged_context: usize,
}

/// A batch's records and the counts by kind.
#[derive(Clone, Debug, Default)]
pub struct BatchRecords {
    /// The records, in the order built.
    pub records: Vec<Record>,
    /// How many of each kind.
    pub counts: RecordCounts,
}

fn provable(rollout: &Rollout) -> bool {
    rollout.trace.as_ref().is_none_or(|t| t.provable)
}

fn record(view: &str, body: RecordBody) -> Record {
    Record {
        body,
        metadata: RecordMetadata {
            experiences: Vec::new(),
            task: None,
            sources: Vec::new(),
            group: None,
            split: None,
            view: view.to_string(),
            objective: Objective::PolicyGradient,
        },
    }
}

fn message(role: &str, content: &str, train: bool) -> WireMessage {
    WireMessage {
        role: role.to_string(),
        content: content.to_string(),
        images: Vec::new(),
        tool_calls: Vec::new(),
        tool_call_id: None,
        reasoning_content: None,
        train,
    }
}

/// The conversation of `rollout`: its prompt unsupervised, its completion
/// supervised.
fn conversation(rollout: &Rollout) -> Vec<WireMessage> {
    let mut messages: Vec<WireMessage> = rollout
        .prompt
        .iter()
        .cloned()
        .map(|mut m| {
            m.train = false;
            m
        })
        .collect();
    messages.push(message("assistant", &rollout.completion, true));
    messages
}

impl BatchRecords {
    /// Every attempt of every informative group, carrying its group-relative
    /// advantage; under [`AdvantageMode::PositiveOnly`] the attempts that did
    /// no better than their group are left out.
    pub fn add_advantages(&mut self, groups: &[&RolloutGroup], mode: AdvantageMode) {
        for group in groups {
            let rewards: Vec<f64> = group.rollouts.iter().map(|r| r.reward).collect();
            let Some(advantages) = mode.kind().advantages(&rewards) else {
                continue;
            };
            for (rollout, advantage) in group.rollouts.iter().zip(advantages) {
                if !provable(rollout) {
                    self.counts.dropped_unprovable += 1;
                    continue;
                }
                if mode.kind().drops(advantage) {
                    continue;
                }
                let body = RecordBody::chat(conversation(rollout)).with_advantage(advantage);
                self.records.push(record("campaign-policy-gradient", body));
                self.counts.policy_gradient += 1;
            }
        }
    }

    /// For each informative group, up to `per_group` repairs: a failed
    /// attempt, the verifier's feedback on it, and the shortest passing
    /// completion as the supervised fix.
    pub fn add_repairs(&mut self, groups: &[&RolloutGroup], per_group: usize) {
        for group in groups {
            let Some(fix) = group
                .rollouts
                .iter()
                .filter(|r| r.passed && provable(r))
                .min_by_key(|r| r.completion.len())
            else {
                continue;
            };
            for wrong in group
                .rollouts
                .iter()
                .filter(|r| !r.passed && provable(r))
                .take(per_group)
            {
                let mut messages: Vec<WireMessage> = fix
                    .prompt
                    .iter()
                    .cloned()
                    .map(|mut m| {
                        m.train = false;
                        m
                    })
                    .collect();
                messages.push(message("assistant", &wrong.completion, false));
                messages.push(message(
                    "user",
                    wrong.feedback.as_deref().unwrap_or(GENERIC_FEEDBACK),
                    false,
                ));
                messages.push(message("assistant", &fix.completion, true));
                self.records
                    .push(record("campaign-repair", RecordBody::chat(messages)));
                self.counts.repair += 1;
            }
        }
    }

    /// For each group that all failed, up to `per_group` attempts as they
    /// were made, with the privileged text beside them as a teacher context:
    /// the trainer scores the policy's own tokens with the frozen model shown
    /// it and trains on the per-token difference.
    pub fn add_distillation(&mut self, groups: &[&RolloutGroup], per_group: usize) {
        for group in groups {
            for rollout in group
                .rollouts
                .iter()
                .filter(|r| provable(r))
                .take(per_group)
            {
                let Some(privileged) = rollout.privileged.as_deref() else {
                    self.counts.no_privileged_context += 1;
                    continue;
                };
                let body = RecordBody::chat(conversation(rollout)).with_teacher_context(vec![
                    TeacherMessage {
                        role: "user".into(),
                        content: privileged.to_string(),
                    },
                ]);
                self.records.push(record("campaign-distill", body));
                self.counts.distilled += 1;
            }
        }
    }

    /// Records a teacher produced, already built by the handler.
    pub fn add_taught(&mut self, taught: Vec<Record>) {
        self.counts.taught += taught.len();
        self.records.extend(taught);
    }

    /// Earlier arms' records, replayed beside this arm's.
    pub fn add_replay(&mut self, replay: Vec<Record>) {
        self.counts.replay += replay.len();
        self.records.extend(replay);
    }
}

/// The shortest passing completion of each group, as imitation records: what
/// an accepted arm contributes to the replay of later ones.
#[must_use]
pub fn passing_records(groups: &[&RolloutGroup]) -> Vec<Record> {
    groups
        .iter()
        .filter_map(|group| {
            group
                .rollouts
                .iter()
                .filter(|r| r.passed && provable(r))
                .min_by_key(|r| r.completion.len())
        })
        .map(|rollout| record("campaign-replay", RecordBody::chat(conversation(rollout))))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::campaign::types::{RolloutTrace, TraceSequence};

    fn rollout(reward: f64, passed: bool, completion: &str) -> Rollout {
        Rollout {
            reward,
            passed,
            prompt: vec![message("user", "what does it print?", false)],
            completion: completion.into(),
            feedback: None,
            privileged: None,
            policy_version: 0,
            trace: None,
        }
    }

    fn group(rollouts: Vec<Rollout>) -> RolloutGroup {
        RolloutGroup {
            task: "t".into(),
            rollouts,
        }
    }

    fn advantage_of(record: &Record) -> Option<f64> {
        match &record.body {
            RecordBody::Chat { advantage, .. } => *advantage,
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn two_sided_advantages_push_the_wrong_attempts_down_and_positive_only_leaves_them_out() {
        let g = group(vec![
            rollout(1.0, true, "4"),
            rollout(0.0, false, "5"),
            rollout(0.0, false, "6"),
        ]);
        let mut two_sided = BatchRecords::default();
        two_sided.add_advantages(&[&g], AdvantageMode::TwoSided);
        assert_eq!(two_sided.counts.policy_gradient, 3);
        let advantages: Vec<f64> = two_sided.records.iter().filter_map(advantage_of).collect();
        assert!(advantages[0] > 0.0 && advantages[1] < 0.0 && advantages[2] < 0.0);

        let mut positive = BatchRecords::default();
        positive.add_advantages(&[&g], AdvantageMode::PositiveOnly);
        assert_eq!(positive.counts.policy_gradient, 1);
        assert!(advantage_of(&positive.records[0]).unwrap() > 0.0);
    }

    #[test]
    fn a_group_without_a_spread_yields_no_advantage_record() {
        let g = group(vec![rollout(0.0, false, "a"), rollout(0.0, false, "b")]);
        let mut batch = BatchRecords::default();
        batch.add_advantages(&[&g], AdvantageMode::TwoSided);
        assert!(batch.records.is_empty());
    }

    #[test]
    fn only_the_assistant_completion_is_supervised() {
        let g = group(vec![rollout(1.0, true, "4"), rollout(0.0, false, "5")]);
        let mut batch = BatchRecords::default();
        batch.add_advantages(&[&g], AdvantageMode::TwoSided);
        let RecordBody::Chat { messages, .. } = &batch.records[0].body else {
            panic!()
        };
        assert_eq!(
            messages.iter().map(|m| m.train).collect::<Vec<_>>(),
            [false, true]
        );
    }

    #[test]
    fn an_unprovable_trace_is_never_trained_on() {
        let mut bad = rollout(0.0, false, "x");
        bad.trace = Some(RolloutTrace {
            sequences: vec![TraceSequence {
                prompt_token_ids: vec![1],
                generation_token_ids: vec![2],
                generation_log_probs: vec![-0.1],
                truncated: false,
            }],
            provable: false,
        });
        let g = group(vec![rollout(1.0, true, "4"), bad]);
        let mut batch = BatchRecords::default();
        batch.add_advantages(&[&g], AdvantageMode::TwoSided);
        assert_eq!(batch.counts.dropped_unprovable, 1);
        assert_eq!(batch.counts.policy_gradient, 1);
    }

    #[test]
    fn a_repair_shows_the_wrong_attempt_and_the_feedback_and_supervises_the_fix() {
        let mut wrong = rollout(0.0, false, "5");
        wrong.feedback = Some("it printed 4, not 5".into());
        let g = group(vec![
            wrong,
            rollout(1.0, true, "4"),
            rollout(1.0, true, "four"),
        ]);
        let mut batch = BatchRecords::default();
        batch.add_repairs(&[&g], 2);
        assert_eq!(batch.counts.repair, 1);
        let RecordBody::Chat {
            messages,
            advantage,
            ..
        } = &batch.records[0].body
        else {
            panic!()
        };
        assert_eq!(*advantage, None, "a repair is imitated");
        let shown: Vec<(&str, &str, bool)> = messages
            .iter()
            .map(|m| (m.role.as_str(), m.content.as_str(), m.train))
            .collect();
        assert_eq!(
            shown,
            [
                ("user", "what does it print?", false),
                ("assistant", "5", false),
                ("user", "it printed 4, not 5", false),
                ("assistant", "4", true),
            ],
            "the shortest passing completion is the fix"
        );
    }

    #[test]
    fn a_repair_needs_a_fix_and_uses_generic_feedback_when_the_verifier_gave_none() {
        let no_fix = group(vec![rollout(0.0, false, "a"), rollout(0.0, false, "b")]);
        let mut batch = BatchRecords::default();
        batch.add_repairs(&[&no_fix], 2);
        assert!(batch.records.is_empty());
        let g = group(vec![rollout(0.0, false, "a"), rollout(1.0, true, "b")]);
        batch.add_repairs(&[&g], 2);
        let RecordBody::Chat { messages, .. } = &batch.records[0].body else {
            panic!()
        };
        assert!(messages[2].content.contains("wrong"));
    }

    #[test]
    fn distillation_needs_something_the_verifier_knows_and_keeps_it_from_the_student() {
        let mut informed = rollout(0.0, false, "5");
        informed.privileged = Some("the program prints 4".into());
        let blind = rollout(0.0, false, "6");
        let g = group(vec![informed, blind]);
        let mut batch = BatchRecords::default();
        batch.add_distillation(&[&g], 2);
        assert_eq!(batch.counts.distilled, 1);
        assert_eq!(batch.counts.no_privileged_context, 1);
        let RecordBody::Chat {
            messages,
            teacher_context,
            ..
        } = &batch.records[0].body
        else {
            panic!()
        };
        assert_eq!(teacher_context[0].content, "the program prints 4");
        assert!(
            messages.iter().all(|m| !m.content.contains("prints 4")),
            "the student never sees it"
        );
    }

    #[test]
    fn the_replay_of_an_accepted_arm_is_its_shortest_passing_completion_per_group() {
        let a = group(vec![
            rollout(1.0, true, "long answer"),
            rollout(1.0, true, "4"),
        ]);
        let b = group(vec![rollout(0.0, false, "x")]);
        let records = passing_records(&[&a, &b]);
        assert_eq!(records.len(), 1);
        let RecordBody::Chat {
            messages,
            advantage,
            ..
        } = &records[0].body
        else {
            panic!()
        };
        assert_eq!(messages.last().unwrap().content, "4");
        assert_eq!(*advantage, None);
    }
}
