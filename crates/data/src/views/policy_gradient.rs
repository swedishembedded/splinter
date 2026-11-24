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
//!   failure what the configured metrics earn below the ceiling (or, under
//!   `shaped-partial`, its credit above its task's naive floor), either less
//!   DAPO's over-length penalty when one is set ([`crate::Overlong`]), from the
//!   completion tokens the attempt recorded.
//! * A group is the trials of one task version made by one policy (the model
//!   recorded on the verdict). Each trial counts once, however many
//!   conversations (roles) it had. A group of one trial, or whose trials all
//!   earned the same, says nothing: its experiences are counted as
//!   [`Exclusion::NoGroupSignal`]. Every trial of a group with a spread of
//!   rewards is written, under every [`AdvantageKind`]; [`AdvantageKind::Positive`]
//!   has brain clamp the advantages at zero instead of leaving trials out.
//! * A written trial none of whose conversations the view can render (shown
//!   images, an unrepresentable step: each counted under its reason) still
//!   took part in its group: one record of its instruction, nothing supervised,
//!   carries its reward and group as `baseline_only`, so brain counts it in the
//!   group's baseline and trains none of it.
//! * Each action of each conversation of a written trial is a record carrying
//!   the trial's reward, its group's key and the trial's name (brain's
//!   training-record contract, version 2: the trainer computes each trial's
//!   advantage from its group's rewards by the normalisation
//!   [`AdvantageKind::norm`] names, counting a trial once however many records
//!   it has), rendered exactly as `sft-step` renders it
//!   (the requests the model was sent, `images`, `only_roles`; see
//!   [`Rendering`]), and the records are in the corpus's order. Step labels
//!   judge nothing here: the reward is the only judge.
//!
//! * An action whose agent step recorded the policy's own trace (the ids it
//!   was given, the ids it generated, their log-probabilities, the tokenizer's
//!   digest) is written with exactly those ([`SampledTurns`]), never
//!   re-rendered; any other is written without per-token fields and counted
//!   under why in [`Projection::tokens`] ([`ProxyReason`]). Only a conversation
//!   read from the requests its model was sent carries them (brain checks the
//!   context it renders against the prompt ids and refuses another), and not
//!   one the scrub changed: ids decode to the unscrubbed text.
//!
//! Every group is reported ([`Projection::groups`]), including those that
//! yielded no records, with its kind ([`GroupKind`]).
//!
//! ## Self-distillation
//!
//! A group whose attempts all failed alike has no advantage: GRPO learns
//! nothing from the cliff. [`Distill`] says what to do instead. For the
//! attempts it covers the view emits the attempt's own conversation, exactly
//! as the student saw it, with a teacher context beside it (what the verifier
//! knows: the hidden checks that failed, the reference change; see
//! [`privileged`](super::privileged)). The trainer scores the student's own
//! tokens with the frozen model shown that context, and the difference to
//! the student's own log-probability is the dense per-token advantage. The
//! privileged text is never part of the student's messages. An attempt with
//! nothing privileged to show is counted as
//! [`Exclusion::NoPrivilegedContext`].

use std::collections::BTreeMap;

use serde_json::Value;
use splinter_core::annotation::{AnnotationBody, Outcome, Strength};
use splinter_core::harness::HarnessConversation;
use splinter_core::usage::TokenUsage;
use splinter_store::decision::decide;

use super::privileged::teacher_context;
use super::rendering::{Judgement, Rendering};
use super::rewards::{AdvantageKind, GroupKind, GroupReport, RewardShape, TrialReport};
use crate::curation::HistoryCuration;
use crate::sampled::{agreeing, trace_of, turns_of, ProxyReason, SampledTurns, TokenReport};
use crate::trajectory::{Action, Conversation};
use crate::{render, RecordBody, RolloutOutcome};
use crate::{Corpus, Entry, Exclusion, Objective, Projection, Provenance, Strip, View, ViewError};

/// The name every policy-gradient record carries.
const NAME: &str = "policy-gradient";

/// The name of the view that distils every attempt.
const SELF_DISTILL: &str = "self-distill";

/// Which attempts become self-distillation records (a teacher context) and
/// which carry the group's scalar advantage.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Distill {
    /// None: every attempt of a group with a spread of rewards carries its
    /// reward (the trainer's scalar advantage), any other group yields nothing.
    #[default]
    Off,
    /// The attempts of a group that all failed alike (the cliff) are
    /// distilled; the groups with a spread of rewards keep GRPO's scalar
    /// advantage, computed by the trainer from their rewards.
    /// What the campaign runs.
    Cliff,
    /// Every attempt of every group is distilled, passed or failed, and none
    /// carries a scalar advantage.
    Every,
}

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
    passed: bool,
    /// The task's content digest, as the verdict records it.
    digest: Option<String>,
    /// What the attempt's requests processed, as its score counts them.
    usage: Option<TokenUsage>,
    /// The evidence of the deciding verdict: what the verifier recorded.
    evidence: Value,
}

/// The conversations of one trial seen so far, and the reward all of them
/// agree it earned.
struct TrialSeen {
    reward: f64,
    passed: bool,
    conversations: usize,
    usage: Option<TokenUsage>,
}

/// What an attempt becomes in the dataset.
enum Treatment {
    /// A record carrying the attempt's reward and group.
    Reward(RolloutOutcome),
    /// A record with a teacher context.
    Teacher,
    /// Nothing: its group says nothing.
    Nothing,
}

/// Reinforcement learning from verified rewards: every action of every graded
/// attempt, with the attempt's reward and the group it is compared with.
#[derive(Clone, Debug)]
pub struct PolicyGradient {
    min_strength: Strength,
    rendering: Rendering,
    reward: RewardShape,
    advantage: AdvantageKind,
    distill: Distill,
    sampled_only: bool,
}

impl PolicyGradient {
    /// The view, accepting verdicts decided at `min_strength` or stronger,
    /// showing the student only the instruction, rewarding a pass with one
    /// and a failure with nothing, the trainer standardising the rewards of a
    /// group.
    #[must_use]
    pub fn new(min_strength: Strength) -> Self {
        Self {
            min_strength,
            // The reward judges an attempt here, history reads and all.
            rendering: Rendering {
                history: HistoryCuration::Keep,
                ..Rendering::default()
            },
            reward: RewardShape::unshaped(),
            advantage: AdvantageKind::default(),
            distill: Distill::Off,
            sampled_only: false,
        }
    }

    /// The same view training only the tokens the policy sampled: a rewarded
    /// record whose actions do not all carry their exact sampled tokens is
    /// left out and counted as [`Exclusion::ProxyTokens`]; its trial still
    /// sets its group's baseline. Every trained token then has the behaviour
    /// policy's own log-probability, so the importance ratio is exact.
    #[must_use]
    pub fn sampled_only(self, sampled_only: bool) -> Self {
        Self {
            sampled_only,
            ..self
        }
    }

    /// The same view distilling the attempts `distill` says.
    #[must_use]
    pub fn distilling(self, distill: Distill) -> Self {
        Self { distill, ..self }
    }

    /// The same view earning a reward as `reward` says.
    #[must_use]
    pub fn rewards(self, reward: RewardShape) -> Self {
        Self { reward, ..self }
    }

    /// The same view writing the attempts `advantage` keeps, for the trainer
    /// to normalise as it names.
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

    /// The same view writing one record per conversation of a trial instead of
    /// one per action ([`SftStep::packed`](super::SftStep::packed)): every
    /// action supervised, the context once. Each record still carries its
    /// trial's reward, group and rollout, so the trainer counts the trial once
    /// in its group either way.
    #[must_use]
    pub fn packed(mut self, packed: bool) -> Self {
        self.rendering.packed = packed;
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
        let provenance = &entry.experience.provenance;
        // A benchmark verdict names its trial and task; a workspace verdict
        // names neither, and the attempt is placed by what the experience
        // records itself: its task's address and its attempt number.
        let (trial, task) = match (text(evidence, "/trial"), text(evidence, "/task/id")) {
            (Some(trial), Some(task)) => (trial, task),
            (None, None) => match provenance.attempt {
                Some(attempt) => (
                    format!("attempt-{attempt}"),
                    entry.experience.task.id.to_string(),
                ),
                None => return Ok(Err(Exclusion::NoRollout)),
            },
            _ => return Ok(Err(Exclusion::NoRollout)),
        };
        let policy = text(evidence, "/model")
            .or_else(|| provenance.policy.clone())
            .unwrap_or_else(|| provenance.solver.clone());
        let digest = text(evidence, "/task/digest");
        let reward = self.reward.reward(
            decision.passed,
            evidence.get("metrics"),
            digest.as_deref(),
            completion_tokens(evidence, &entry.experience.trajectory),
            &trial,
        )?;
        Ok(Ok(Rollout {
            group: GroupKey {
                task,
                version: text(evidence, "/task/version").unwrap_or_default(),
                policy,
            },
            trial,
            reward,
            passed: decision.passed,
            digest,
            usage: TokenUsage::of_trial(evidence),
            evidence: evidence.clone(),
        }))
    }
}

/// The completion tokens the attempt an experience is part of wrote, as
/// recorded: the trial's count on its verdict (a benchmark's `tokens`, or its
/// score's usage), else the trajectory's own total (sven's final metrics),
/// else the sum over its agent steps when every one recorded its count.
fn completion_tokens(evidence: &Value, trajectory: &atif::Trajectory) -> Option<u64> {
    evidence
        .pointer("/tokens/completion")
        .or_else(|| evidence.pointer("/score/usage/completion_tokens"))
        .and_then(Value::as_u64)
        .or_else(|| {
            trajectory
                .final_metrics
                .as_ref()
                .and_then(|m| m.total_completion_tokens)
        })
        .or_else(|| {
            let mut agent_steps = trajectory
                .steps
                .iter()
                .filter(|step| step.source == atif::StepOrigin::Agent)
                .peekable();
            agent_steps.peek()?;
            agent_steps
                .map(|step| step.metrics.as_ref().and_then(|m| m.completion_tokens))
                .sum()
        })
}

/// The tokens the policy sampled for the `actions` one rewarded record
/// supervises, counted in `report`; `None` when no action has any. An action
/// carries them only when its conversation is the requests the model was sent
/// (a reconstruction from the trajectory is not the context the ids follow),
/// its step recorded the policy's own trace, the step is that action alone,
/// and the scrub left the conversation as the model saw it (the ids decode to
/// the unscrubbed text).
fn sampled_tokens(
    trajectory: &atif::Trajectory,
    conversation: &Conversation,
    actions: &[Action],
    report: &mut TokenReport,
) -> Option<SampledTurns> {
    let traces = actions
        .iter()
        .map(|action| {
            if !conversation.as_sent {
                Err(ProxyReason::ContextReconstructed)
            } else if conversation.scrubbed {
                Err(ProxyReason::Scrubbed)
            } else if conversation
                .actions
                .iter()
                .filter(|a| a.step == action.step)
                .count()
                > 1
            {
                Err(ProxyReason::SplitStep)
            } else {
                trace_of(trajectory, action.step)
            }
        })
        .collect();
    let traces = agreeing(traces);
    for trace in &traces {
        report.count(trace);
    }
    turns_of(traces)
}

impl View for PolicyGradient {
    fn name(&self) -> &str {
        if self.distill == Distill::Every {
            SELF_DISTILL
        } else {
            NAME
        }
    }

    fn objective(&self) -> Objective {
        Objective::PolicyGradient
    }

    fn project(&self, corpus: &Corpus) -> Result<Projection, ViewError> {
        let mut projection = Projection::new(
            self.name(),
            Objective::PolicyGradient,
            Some(self.rendering.strip.clone()),
            Some(self.min_strength),
        );
        // First every experience is placed in the group of its attempt: an
        // group's rewards decide what is written, so no record can be made
        // before all of the group is known.
        let mut placed: Vec<(&Entry, Option<HarnessConversation>, Rollout)> = Vec::new();
        let mut groups: BTreeMap<GroupKey, BTreeMap<String, TrialSeen>> = BTreeMap::new();
        let mut digests: BTreeMap<GroupKey, String> = BTreeMap::new();
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
                    passed: rollout.passed,
                    conversations: 0,
                    usage: rollout.usage,
                });
            if let Some(digest) = &rollout.digest {
                digests
                    .entry(rollout.group.clone())
                    .or_insert_with(|| digest.clone());
            }
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
        projection.advantage = Some(self.advantage);
        projection.packed = self.rendering.packed;
        projection.overlong = self.reward.overlong();
        projection.partial_credit = self.reward.partial_credit().cloned();
        let mut treatment_of: BTreeMap<(&GroupKey, &str), Treatment> = BTreeMap::new();
        for (key, trials) in &groups {
            let rewards: Vec<f64> = trials.values().map(|t| t.reward).collect();
            let signal = super::rewards::signals(&rewards);
            let kind = if signal {
                GroupKind::Graded
            } else if trials.values().any(|t| t.passed) {
                GroupKind::AllPass
            } else {
                GroupKind::AllFail
            };
            let distilled = match self.distill {
                Distill::Off => false,
                Distill::Cliff => kind == GroupKind::AllFail,
                Distill::Every => true,
            };
            let mut reports = Vec::with_capacity(trials.len());
            for (trial, seen) in trials {
                let treatment = match (distilled, signal) {
                    (true, _) => Treatment::Teacher,
                    (false, true) => Treatment::Reward(RolloutOutcome::of(
                        seen.reward,
                        &[&key.task, &key.version, &key.policy],
                        trial,
                    )),
                    (false, false) => Treatment::Nothing,
                };
                reports.push(TrialReport {
                    trial: trial.clone(),
                    reward: seen.reward,
                    written: !matches!(treatment, Treatment::Nothing),
                    baseline_only: false,
                    conversations: seen.conversations,
                    cost: seen.usage.and_then(|u| u.cost()),
                    completion_tokens: seen.usage.and_then(|u| u.completion),
                });
                treatment_of.insert((key, trial), treatment);
            }
            let digest = digests.get(key).cloned();
            let naive_floor = projection.partial_credit.as_mut().and_then(|credit| {
                let floor = credit.floor(digest.as_deref());
                credit.groups_without_floor += usize::from(floor.is_none());
                floor
            });
            projection.groups.push(GroupReport {
                task: key.task.clone(),
                version: key.version.clone(),
                policy: key.policy.clone(),
                trials: reports,
                signal,
                kind,
                distilled,
                naive_floor,
                digest,
            });
        }
        // The trials that wrote a record, and the first conversation of each
        // that carries a reward: a trial none of whose conversations could be
        // rendered still sets its group's baseline, as one record that trains
        // nothing.
        let mut rendered: BTreeMap<(&GroupKey, &str), bool> = BTreeMap::new();
        for (entry, harness, rollout) in &placed {
            let treatment = treatment_of.get(&(&rollout.group, rollout.trial.as_str()));
            let Some(treatment @ (Treatment::Reward(_) | Treatment::Teacher)) = treatment else {
                projection.exclude(Exclusion::NoGroupSignal);
                continue;
            };
            let written = rendered
                .entry((&rollout.group, rollout.trial.as_str()))
                .or_insert(false);
            let teacher = match treatment {
                Treatment::Teacher => {
                    match teacher_context(&entry.experience, &rollout.evidence, rollout.passed) {
                        Ok(context) => Some(context),
                        Err(reason) => {
                            projection.exclude(reason);
                            continue;
                        }
                    }
                }
                _ => None,
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
            for (record, actions) in
                self.rendering
                    .records_of_actions(&conversation, &mut projection, |_| Judgement::Admit)
            {
                let record = match (treatment, &teacher) {
                    (Treatment::Reward(outcome), _) => {
                        let sampled = sampled_tokens(
                            &entry.experience.trajectory,
                            &conversation,
                            &actions,
                            &mut projection.tokens,
                        );
                        let exact = sampled
                            .as_ref()
                            .is_some_and(|s| s.turns.iter().all(Option::is_some));
                        if self.sampled_only && !exact {
                            projection.exclude(Exclusion::ProxyTokens);
                            continue;
                        }
                        record.with_outcome(outcome.clone().with_sampled(sampled))
                    }
                    (_, Some(context)) => record.with_teacher_context(context.clone()),
                    _ => record,
                };
                projection.push(record, Provenance::of(vec![entry.id.clone()]));
                *written = true;
            }
        }
        for (entry, _, rollout) in &placed {
            let key = (&rollout.group, rollout.trial.as_str());
            let Some(Treatment::Reward(outcome)) = treatment_of.get(&key) else {
                continue;
            };
            if rendered.get(&key) != Some(&false) {
                continue;
            }
            rendered.insert(key, true);
            let baseline = vec![render::message(
                "user",
                &entry.experience.instruction,
                false,
            )];
            projection.push(
                RecordBody::chat(baseline).with_outcome(outcome.clone().baseline()),
                Provenance::of(vec![entry.id.clone()]),
            );
            if let Some(report) = projection
                .groups
                .iter_mut()
                .filter(|g| {
                    (&g.task, &g.version, &g.policy) == (&key.0.task, &key.0.version, &key.0.policy)
                })
                .flat_map(|g| g.trials.iter_mut())
                .find(|t| t.trial == key.1)
            {
                report.baseline_only = true;
            }
        }
        Ok(projection)
    }
}
