// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reinforcement learning from the verified
// rewards of agent rollouts for its clients. If your team needs expertise in
// training agents against verifiers that do not trust them, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The policy-gradient view: every action of every graded attempt, weighted by
//! how much better the attempt did than the others at the same task.
//!
//! Where [`SftStep`](super::SftStep) imitates the actions of verified work,
//! this view keeps the failures too, and tells the trainer how each attempt
//! did:
//!
//! * An attempt is a trial of the verifier ([`splinter_core::harness`]): its
//!   conversations are one rollout. The verdict the bench ingestion records on
//!   each experience names the trial, the task (id and version), the model
//!   that ran it and the metrics the verifier measured. An experience whose
//!   verdict names no trial or task cannot be placed in a group and is counted
//!   as [`Exclusion::NoRollout`]; one with no decision, or one decided below
//!   the minimum strength, as [`Exclusion::Undecided`] or
//!   [`Exclusion::TooWeak`].
//! * The attempt's reward is [`RewardShape::reward`]: a pass earns `1.0`, a
//!   failure what the configured metrics earn below the ceiling.
//! * A group is the trials of one task version made by one policy (the model
//!   recorded on the verdict). Each trial counts once, however many
//!   conversations (roles) it had; its advantage is read from the rewards of
//!   its group ([`AdvantageKind`]). A group of one trial, or whose trials all
//!   earned the same, says nothing: its experiences are counted as
//!   [`Exclusion::NoGroupSignal`].
//! * Each action of each conversation of a trial with advantage is a record
//!   carrying the trial's advantage, rendered exactly as `sft-step` renders it
//!   (the requests the model was sent, `images`, `only_roles`; see
//!   [`Rendering`]), and the records are in the corpus's order. Step labels
//!   judge nothing here: the reward is the only judge.
//!
//! Every group is reported ([`Projection::groups`]), including those that
//! yielded no records.

use std::collections::BTreeMap;

use serde_json::Value;
use splinter_core::annotation::{AnnotationBody, Outcome, Strength};
use splinter_core::harness::HarnessConversation;
use splinter_store::decision::decide;

use super::rendering::{Judgement, Rendering};
use super::rewards::{AdvantageKind, GroupReport, RewardShape, TrialReport};
use crate::{Corpus, Entry, Exclusion, Objective, Projection, Provenance, Strip, View, ViewError};

/// The name every policy-gradient record carries.
const NAME: &str = "policy-gradient";

/// Which attempts are compared: one task version, one policy.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct GroupKey {
    task: String,
    version: String,
    policy: String,
}

/// What a verdict says of the attempt an experience is part of.
struct Rollout {
    group: GroupKey,
    trial: String,
    reward: f64,
}

/// The conversations of one trial seen so far, and the reward all of them
/// agree it earned.
struct TrialSeen {
    reward: f64,
    conversations: usize,
}

/// Reinforcement learning from verified rewards: every action of every graded
/// attempt, with the attempt's advantage over the others of its group.
#[derive(Clone, Debug)]
pub struct PolicyGradient {
    min_strength: Strength,
    rendering: Rendering,
    reward: RewardShape,
    advantage: AdvantageKind,
}

impl PolicyGradient {
    /// The view, accepting verdicts decided at `min_strength` or stronger,
    /// showing the student only the instruction, rewarding a pass with one
    /// and a failure with nothing, standardising the advantages of a group.
    #[must_use]
    pub fn new(min_strength: Strength) -> Self {
        Self {
            min_strength,
            rendering: Rendering::default(),
            reward: RewardShape::unshaped(),
            advantage: AdvantageKind::default(),
        }
    }

    /// The same view earning a reward as `reward` says.
    #[must_use]
    pub fn rewards(self, reward: RewardShape) -> Self {
        Self { reward, ..self }
    }

    /// The same view reading advantages as `advantage` says.
    #[must_use]
    pub fn advantage(self, advantage: AdvantageKind) -> Self {
        Self { advantage, ..self }
    }

    /// The same view under `strip`.
    #[must_use]
    pub fn with_strip(mut self, strip: Strip) -> Self {
        self.rendering.strip = strip;
        self
    }

    /// The same view keeping only the actions whose tool calls are all named
    /// in `names` ([`SftStep::only_actions`](super::SftStep::only_actions)).
    #[must_use]
    pub fn only_actions(mut self, names: Vec<String>) -> Self {
        self.rendering.only_actions = (!names.is_empty()).then_some(names);
        self
    }

    /// The same view keeping only the conversations answered by one of the
    /// model `roles` ([`SftStep::only_roles`](super::SftStep::only_roles)). A
    /// trial none of whose conversations a wanted role answered is not in any
    /// group: it taught this role nothing to compare.
    #[must_use]
    pub fn only_roles(mut self, roles: Vec<String>) -> Self {
        self.rendering.only_roles = (!roles.is_empty()).then_some(roles);
        self
    }

    /// The same view admitting the conversations whose model was shown images
    /// ([`SftStep::images`](super::SftStep::images)).
    #[must_use]
    pub fn images(mut self, images: bool) -> Self {
        self.rendering.images = images;
        self
    }

    /// The same view reading a run exported as one agent step as one call per
    /// turn ([`SftStep::serialise_calls`](super::SftStep::serialise_calls)).
    #[must_use]
    pub fn serialise_calls(mut self, serialise_calls: bool) -> Self {
        self.rendering.serialise_calls = serialise_calls;
        self
    }

    /// The attempt `entry` is part of and the reward it earned, or why the
    /// experience cannot be placed in a group.
    fn rollout(&self, entry: &Entry) -> Result<Result<Rollout, Exclusion>, ViewError> {
        let Some(decision) = decide(&entry.notes) else {
            return Ok(Err(Exclusion::Undecided));
        };
        if decision.strength < self.min_strength {
            return Ok(Err(Exclusion::TooWeak));
        }
        let counted = if decision.passed {
            Outcome::Pass
        } else {
            Outcome::Fail
        };
        let deciding = entry.notes.iter().find_map(|note| match &note.body {
            AnnotationBody::Verdict {
                outcome,
                strength,
                evidence,
            } if *strength == decision.strength && strength.counted(*outcome) == counted => {
                Some(evidence)
            }
            _ => None,
        });
        let text = |evidence: &Value, pointer: &str| {
            evidence.pointer(pointer).map(|v| match v.as_str() {
                Some(text) => text.to_string(),
                None => v.to_string(),
            })
        };
        let Some(evidence) = deciding else {
            return Ok(Err(Exclusion::NoRollout));
        };
        let (Some(trial), Some(task)) = (text(evidence, "/trial"), text(evidence, "/task/id"))
        else {
            return Ok(Err(Exclusion::NoRollout));
        };
        let provenance = &entry.experience.provenance;
        let policy = text(evidence, "/model")
            .or_else(|| provenance.policy.clone())
            .unwrap_or_else(|| provenance.solver.clone());
        let reward = self
            .reward
            .reward(decision.passed, evidence.get("metrics"), &trial)?;
        Ok(Ok(Rollout {
            group: GroupKey {
                task,
                version: text(evidence, "/task/version").unwrap_or_default(),
                policy,
            },
            trial,
            reward,
        }))
    }
}

impl View for PolicyGradient {
    fn name(&self) -> &str {
        NAME
    }

    fn objective(&self) -> Objective {
        Objective::PolicyGradient
    }

    fn project(&self, corpus: &Corpus) -> Result<Projection, ViewError> {
        let mut projection = Projection::new(
            NAME,
            Objective::PolicyGradient,
            Some(self.rendering.strip.clone()),
            Some(self.min_strength),
        );
        // First every experience is placed in the group of its attempt: an
        // advantage is read from the whole group, so no record can be made
        // before all of the group is known.
        let mut placed: Vec<(&Entry, Option<HarnessConversation>, Rollout)> = Vec::new();
        let mut groups: BTreeMap<GroupKey, BTreeMap<String, TrialSeen>> = BTreeMap::new();
        for entry in corpus.entries() {
            let harness = HarnessConversation::of(&entry.experience.trajectory);
            if !self.rendering.wants_role(harness.as_ref()) {
                projection.exclude(Exclusion::RoleNotWanted);
                continue;
            }
            let rollout = match self.rollout(entry)? {
                Ok(rollout) => rollout,
                Err(reason) => {
                    projection.exclude(reason);
                    continue;
                }
            };
            let seen = groups
                .entry(rollout.group.clone())
                .or_default()
                .entry(rollout.trial.clone())
                .or_insert(TrialSeen {
                    reward: rollout.reward,
                    conversations: 0,
                });
            // The conversations of a trial are graded together, so they
            // agree; if not, the store holds two grades of one trial.
            if (seen.reward - rollout.reward).abs() > f64::EPSILON {
                return Err(ViewError::Parameter {
                    name: "reward",
                    reason: format!(
                        "trial {} of task {} has conversations graded {} and {}",
                        rollout.trial, rollout.group.task, seen.reward, rollout.reward
                    ),
                });
            }
            seen.conversations += 1;
            placed.push((entry, harness, rollout));
        }
        let mut advantage_of: BTreeMap<(&GroupKey, &str), f64> = BTreeMap::new();
        for (key, trials) in &groups {
            let rewards: Vec<f64> = trials.values().map(|t| t.reward).collect();
            let advantages = self.advantage.advantages(&rewards);
            let reports = trials
                .iter()
                .enumerate()
                .map(|(at, (trial, seen))| TrialReport {
                    trial: trial.clone(),
                    reward: seen.reward,
                    advantage: advantages.as_ref().map_or(0.0, |a| a[at]),
                    conversations: seen.conversations,
                })
                .collect();
            if let Some(advantages) = &advantages {
                for ((trial, _), advantage) in trials.iter().zip(advantages) {
                    advantage_of.insert((key, trial), *advantage);
                }
            }
            projection.groups.push(GroupReport {
                task: key.task.clone(),
                version: key.version.clone(),
                policy: key.policy.clone(),
                trials: reports,
                signal: advantages.is_some(),
            });
        }
        for (entry, harness, rollout) in &placed {
            let Some(&advantage) = advantage_of.get(&(&rollout.group, rollout.trial.as_str()))
            else {
                projection.exclude(Exclusion::NoGroupSignal);
                continue;
            };
            let conversation = match self.rendering.conversation_of(entry, harness.as_ref()) {
                Ok(conversation) => conversation,
                Err(reason) => {
                    projection.exclude(reason);
                    continue;
                }
            };
            if conversation.actions.is_empty() {
                projection.exclude(Exclusion::NoAction);
                continue;
            }
            for record in self
                .rendering
                .records(&conversation, &mut projection, |_| Judgement::Admit)
            {
                projection.push(
                    record.with_advantage(advantage),
                    Provenance::of(vec![entry.id.clone()]),
                );
            }
        }
        Ok(projection)
    }
}
