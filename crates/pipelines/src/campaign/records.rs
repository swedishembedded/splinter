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
//! strategy decides which: every attempt with its reward and group, for the
//! trainer to compute group-relative advantages from (two-sided or
//! positive-only), the repair that turns a wrong attempt and the
//! verifier's feedback into the fix a later attempt found, the per-token
//! distillation target of an attempt in a group that all failed, and the
//! passing completions kept for replay. A rollout whose policy trace is
//! marked unprovable is never a record; one with no trace (an in-process
//! sampler) is trained on the start-of-run behaviour proxy.

use serde::Serialize;
use splinter_core::chat::WireMessage;
use splinter_data::{
    AdvantageKind, Objective, Record, RecordBody, RecordMetadata, RolloutOutcome, TeacherMessage,
};

use super::config::AdvantageMode;
use super::types::{Rollout, RolloutGroup};

/// The text a repair turn shows when the verifier gave no feedback.
const GENERIC_FEEDBACK: &str = "That answer is wrong. Try again.";

/// What was built, and what was left out and why.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct RecordCounts {
    /// Rollout records carrying a reward and a group.
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

impl RecordCounts {
    /// Records carrying this arm's own training signal: advantages, repairs,
    /// distillation targets and a teacher's verified answers. Replay is left
    /// out: an arm with nothing but earlier arms' data has learned nothing new
    /// and makes no candidate.
    #[must_use]
    pub fn fresh_signal(&self) -> usize {
        self.policy_gradient + self.repair + self.distilled + self.taught
    }
}

/// A batch's records and the counts by kind.
#[derive(Clone, Debug, Default)]
pub struct BatchRecords {
    /// The records, in the order built.
    pub records: Vec<Record>,
    /// How many of each kind.
    pub counts: RecordCounts,
    /// How the trainer turns the rollout records' rewards into advantages,
    /// once [`Self::add_advantages`] wrote some.
    pub advantage: Option<AdvantageKind>,
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
            split_draw: None,
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
    /// Every provable attempt of every informative group, carrying its reward,
    /// its group (the task) and its place in the group, for the trainer to
    /// compute the group-relative advantages; under
    /// [`AdvantageMode::PositiveOnly`] only the attempts above their group's
    /// mean reward. A group whose written attempts would not differ in reward
    /// gives the trainer nothing to compare and is left out.
    pub fn add_advantages(&mut self, groups: &[&RolloutGroup], mode: AdvantageMode) {
        let kind = mode.kind();
        self.advantage = Some(kind);
        for group in groups {
            let provable: Vec<(usize, &Rollout)> = group
                .rollouts
                .iter()
                .enumerate()
                .filter(|(_, r)| provable(r))
                .collect();
            self.counts.dropped_unprovable += group.rollouts.len() - provable.len();
            let rewards: Vec<f64> = provable.iter().map(|(_, r)| r.reward).collect();
            let Some(written) = kind.keeps(&rewards) else {
                continue;
            };
            for ((at, rollout), keep) in provable.into_iter().zip(written) {
                if !keep {
                    continue;
                }
                let outcome = RolloutOutcome::of(rollout.reward, &[&group.task], &at.to_string());
                let body = RecordBody::chat(conversation(rollout)).with_outcome(outcome);
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

    fn outcome_of(record: &Record) -> Option<RolloutOutcome> {
        match &record.body {
            RecordBody::Chat { outcome, .. } => outcome.clone(),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn two_sided_writes_every_attempt_with_its_reward_and_positive_only_those_above_the_mean() {
        let g = group(vec![
            rollout(1.0, true, "4"),
            rollout(0.0, false, "5"),
            rollout(0.0, false, "6"),
        ]);
        let mut two_sided = BatchRecords::default();
        two_sided.add_advantages(&[&g], AdvantageMode::TwoSided);
        assert_eq!(two_sided.counts.policy_gradient, 3);
        assert_eq!(two_sided.advantage, Some(AdvantageKind::Standardised));
        let outcomes: Vec<RolloutOutcome> =
            two_sided.records.iter().filter_map(outcome_of).collect();
        let rewards: Vec<f64> = outcomes.iter().map(|o| o.reward).collect();
        assert_eq!(rewards, [1.0, 0.0, 0.0]);
        assert!(outcomes.iter().all(|o| o.group == outcomes[0].group));
        let rollouts: Vec<Option<&str>> = outcomes.iter().map(|o| o.rollout.as_deref()).collect();
        assert_eq!(rollouts, [Some("0"), Some("1"), Some("2")]);

        // Above the mean of 0.4: the two better attempts, whose rewards still differ.
        let graded = group(vec![
            rollout(1.0, true, "4"),
            rollout(0.6, false, "4.0"),
            rollout(0.0, false, "5"),
            rollout(0.0, false, "6"),
        ]);
        let mut positive = BatchRecords::default();
        positive.add_advantages(&[&graded], AdvantageMode::PositiveOnly);
        assert_eq!(positive.counts.policy_gradient, 2);
        assert_eq!(positive.advantage, Some(AdvantageKind::Positive));
        assert!(positive
            .records
            .iter()
            .all(|r| outcome_of(r).unwrap().reward > 0.4));
        // One attempt above the mean leaves the trainer nothing to compare.
        let mut lone = BatchRecords::default();
        lone.add_advantages(&[&g], AdvantageMode::PositiveOnly);
        assert!(lone.records.is_empty());
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
        let unprovable = || {
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
            bad
        };
        let g = group(vec![
            rollout(1.0, true, "4"),
            unprovable(),
            rollout(0.0, false, "6"),
        ]);
        let mut batch = BatchRecords::default();
        batch.add_advantages(&[&g], AdvantageMode::TwoSided);
        assert_eq!(batch.counts.dropped_unprovable, 1);
        assert_eq!(batch.counts.policy_gradient, 2);
        let rollouts: Vec<Option<String>> = batch
            .records
            .iter()
            .map(|r| outcome_of(r).unwrap().rollout)
            .collect();
        assert_eq!(
            rollouts,
            [Some("0".into()), Some("2".into())],
            "each keeps its place in the group"
        );
        // Without it the group would be one attempt: nothing to compare.
        let alone = group(vec![rollout(1.0, true, "4"), unprovable()]);
        let mut batch = BatchRecords::default();
        batch.add_advantages(&[&alone], AdvantageMode::TwoSided);
        assert!(batch.records.is_empty());
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
            messages, outcome, ..
        } = &batch.records[0].body
        else {
            panic!()
        };
        assert_eq!(*outcome, None, "a repair is imitated");
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
            messages, outcome, ..
        } = &records[0].body
        else {
            panic!()
        };
        assert_eq!(messages.last().unwrap().content, "4");
        assert_eq!(*outcome, None);
    }
}
