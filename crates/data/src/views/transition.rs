// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for learning from an agent's own
// failed attempts without a teacher for its clients. If your team needs
// expertise in turning environment feedback into training data then you can
// procure our services by sending an email to info@swedishembedded.com.

//! The transition view: what the environment answered each action, supervised
//! in the context through that action (implicit world modelling, "early
//! experience").
//!
//! The target is the tool's own output, never a model's text, so it passes
//! the same-observation rule whatever the attempt's verdict: failed attempts
//! are as much a source as passed ones, and on a task the policy never solves
//! they are the only one. Each attempt is one record: the conversation, as its
//! model was sent it ([`Rendering::conversation_of`]), through its last
//! supervised observation, every observation of an admitted action
//! supervised and every other message context. Model turns are never
//! supervised here; the record is trained beside the policy's records, with
//! its own loss weight ([`Transition::loss_weight`]).
//!
//! What is left out:
//!
//! * an attempt whose solver was prompted open-book
//!   ([`Exclusion::OpenBook`]): its actions were taken under a hint or a
//!   knowledge page a student never sees; a person's guidance is withheld
//!   from every view before it ([`crate::Corpus::withhold_human_context`]);
//! * an attempt an exposure audit failed ([`Exclusion::Failed`]);
//! * the observations of an action ATIF marks as copied, of a repeated call
//!   whose result was the same ([`Exclusion::RepeatedAction`]), and of an
//!   action outside [`Transition::only_actions`] ([`Exclusion::ActionNotWanted`]):
//!   these stay context;
//! * an observation of the same task, call and answer beyond the cap
//!   ([`Transition::per_transition_cap`], [`Exclusion::RepeatedTransition`]),
//!   a structural de-duplication in corpus order; an empty observation is
//!   context and not counted;
//! * an attempt with no supervised observation ([`Exclusion::NoAction`]).
//!
//! The negative control [`Transition::shuffle_observations`] keeps every
//! record and every target text but gives each supervised observation
//! another's, by a seeded permutation with no fixed point: the same tokens
//! and format, with the dynamics broken.

use std::collections::BTreeMap;

use splinter_core::annotation::{AnnotationBody, Outcome, Strength};
use splinter_core::chat::{LossWeight, WireMessage};
use splinter_core::harness::HarnessConversation;
use splinter_core::rng::SplitMix64;

use super::rendering::Rendering;
use crate::{
    Corpus, Entry, Exclusion, Objective, Projection, Provenance, RecordBody, View, ViewError,
};

/// The name every transition record carries.
const NAME: &str = "transition";

/// The default cap on supervised copies of one transition.
pub const DEFAULT_PER_TRANSITION_CAP: usize = 2;

/// Supervised fine-tuning on what the environment answered.
#[derive(Clone, Debug)]
pub struct Transition {
    rendering: Rendering,
    loss_weight: Option<LossWeight>,
    per_transition_cap: usize,
    shuffle_seed: Option<u64>,
}

impl Default for Transition {
    fn default() -> Self {
        Self::new()
    }
}

impl Transition {
    /// The view: every observation of every attempt, each transition
    /// supervised at most [`DEFAULT_PER_TRANSITION_CAP`] times, at the
    /// trainer's default weight.
    #[must_use]
    pub fn new() -> Self {
        Self {
            rendering: Rendering::default(),
            loss_weight: None,
            per_transition_cap: DEFAULT_PER_TRANSITION_CAP,
            shuffle_seed: None,
        }
    }

    /// The same view writing `weight` on every supervised observation: the
    /// trainer scales their loss by it against records without one.
    #[must_use]
    pub fn loss_weight(mut self, weight: Option<LossWeight>) -> Self {
        self.loss_weight = weight;
        self
    }

    /// The same view supervising one transition - a task, a call (name and
    /// arguments) and the observation's text - at most `cap` times; at least
    /// once.
    #[must_use]
    pub fn per_transition_cap(mut self, cap: usize) -> Self {
        self.per_transition_cap = cap.max(1);
        self
    }

    /// The same view as its negative control: the supervised observations'
    /// texts permuted across the dataset by `seed`, none kept in place.
    #[must_use]
    pub fn shuffle_observations(mut self, seed: Option<u64>) -> Self {
        self.shuffle_seed = seed;
        self
    }

    /// The same view supervising only the observations of actions whose
    /// tool calls are all named in `names`; the rest stay context.
    #[must_use]
    pub fn only_actions(mut self, names: Vec<String>) -> Self {
        self.rendering.only_actions = (!names.is_empty()).then_some(names);
        self
    }

    /// The same view keeping only the harness conversations answered by one
    /// of the model `roles` ([`Exclusion::RoleNotWanted`] for the rest).
    #[must_use]
    pub fn only_roles(mut self, roles: Vec<String>) -> Self {
        self.rendering.only_roles = (!roles.is_empty()).then_some(roles);
        self
    }

    /// The same view reading a run exported as one agent step as one call per
    /// turn (`crate::trajectory`).
    #[must_use]
    pub fn serialise_calls(mut self, serialise_calls: bool) -> Self {
        self.rendering.serialise_calls = serialise_calls;
        self
    }

    fn project_entry(
        &self,
        entry: &Entry,
        seen: &mut BTreeMap<(String, String, splinter_core::digest::Digest), usize>,
        projection: &mut Projection,
    ) {
        let conversation = HarnessConversation::of(&entry.experience.trajectory);
        if !self.rendering.wants_role(conversation.as_ref()) {
            return projection.exclude(Exclusion::RoleNotWanted);
        }
        if entry.experience.provenance.teacher {
            return projection.exclude(Exclusion::OpenBook);
        }
        if exposed(entry) {
            return projection.exclude(Exclusion::Failed);
        }
        let conversation = match self.rendering.conversation_of(entry, conversation.as_ref()) {
            Ok(conversation) => conversation,
            Err(reason) => return projection.exclude(reason),
        };
        let task = entry.experience.task.id.to_string();
        let mut messages = conversation.messages.clone();
        let mut last = None;
        let mut tools = None;
        for action in &conversation.actions {
            let calls = &messages[action.index].tool_calls;
            if calls.is_empty() || action.copied {
                continue;
            }
            if action.repeated {
                projection.exclude(Exclusion::RepeatedAction);
                continue;
            }
            let wanted = self.rendering.only_actions.as_ref().is_none_or(|names| {
                calls
                    .iter()
                    .all(|c| names.iter().any(|n| n == &c.function.name))
            });
            if !wanted {
                projection.exclude(Exclusion::ActionNotWanted);
                continue;
            }
            let calls = calls.clone();
            let answers: Vec<usize> = (action.index + 1..messages.len())
                .take_while(|&i| messages[i].role == "tool")
                .collect();
            for at in answers {
                let observation = &messages[at];
                if observation.content.is_empty() || !observation.images.is_empty() {
                    continue;
                }
                let call = calls
                    .iter()
                    .find(|c| c.id.is_some() && c.id == observation.tool_call_id)
                    .map(|c| format!("{}\u{0}{}", c.function.name, c.function.arguments))
                    .unwrap_or_default();
                let key = (
                    task.clone(),
                    call,
                    splinter_core::digest::Digest::of(observation.content.as_bytes()),
                );
                let count = seen.entry(key).or_insert(0);
                if *count >= self.per_transition_cap {
                    projection.exclude(Exclusion::RepeatedTransition);
                    continue;
                }
                *count += 1;
                messages[at].train = true;
                messages[at].loss_weight = self.loss_weight;
                last = Some(at);
                match tools {
                    None => tools = Some(action.tools),
                    Some(t) if t != action.tools => {
                        return projection.exclude(Exclusion::Unrepresentable);
                    }
                    Some(_) => {}
                }
            }
        }
        let (Some(last), Some(tools)) = (last, tools) else {
            return projection.exclude(Exclusion::NoAction);
        };
        messages.truncate(last + 1);
        let body = RecordBody::Chat {
            messages,
            tools: conversation.tool_sets[tools].clone(),
            outcome: None,
            teacher_context: Vec::new(),
        };
        projection.push(body, Provenance::of(vec![entry.id.clone()]));
    }
}

impl View for Transition {
    fn name(&self) -> &str {
        NAME
    }

    fn objective(&self) -> Objective {
        Objective::Sft
    }

    fn project(&self, corpus: &Corpus) -> Result<Projection, ViewError> {
        let mut projection = Projection::new(
            NAME,
            Objective::Sft,
            Some(self.rendering.strip.clone()),
            None,
        );
        projection.packed = true;
        let mut seen = BTreeMap::new();
        for entry in corpus.entries() {
            self.project_entry(entry, &mut seen, &mut projection);
        }
        if let Some(seed) = self.shuffle_seed {
            shuffle(&mut projection, seed)?;
        }
        Ok(projection)
    }
}

/// Whether an exposure audit failed `entry`: it touched material it must not.
fn exposed(entry: &Entry) -> bool {
    entry.notes.iter().any(|n| {
        matches!(
            n.body,
            AnnotationBody::Verdict {
                outcome: Outcome::Fail,
                strength: Strength::Constraint,
                ..
            }
        )
    })
}

/// The supervised messages of every record, in order.
fn targets(projection: &mut Projection) -> Vec<&mut WireMessage> {
    projection
        .records
        .iter_mut()
        .filter_map(|record| match &mut record.body {
            RecordBody::Chat { messages, .. } => Some(messages),
            _ => None,
        })
        .flat_map(|messages| messages.iter_mut().filter(|m| m.train))
        .collect()
}

/// Gives each supervised observation the text of another: the targets in an
/// order drawn by `seed`, each taking its successor's text.
fn shuffle(projection: &mut Projection, seed: u64) -> Result<(), ViewError> {
    let mut targets = targets(projection);
    if targets.len() < 2 {
        return Err(ViewError::Parameter {
            name: "shuffle_observations",
            reason: "shuffling observations needs at least two of them".into(),
        });
    }
    let mut order: Vec<usize> = (0..targets.len()).collect();
    SplitMix64::new(seed).shuffle(&mut order);
    let texts: Vec<String> = order.iter().map(|&i| targets[i].content.clone()).collect();
    for (k, &i) in order.iter().enumerate() {
        targets[i].content.clone_from(&texts[(k + 1) % texts.len()]);
    }
    Ok(())
}
