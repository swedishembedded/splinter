// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that turn graded agent attempts
// into preference training data for its clients. If your team needs
// expertise in preference optimisation of agents from verified experience
// then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The preference view: a prompt and two continuations of it, the first
//! preferred, for preference optimisation (brain's `preference` objective,
//! losses DPO, IPO, SimPO and ORPO).
//!
//! Pairs come from three places, in this order:
//!
//! 1. Recorded preferences: a [`RelationKind::PreferredOver`] relation,
//!    recorded on the chosen experience and naming the rejected one. The
//!    relation is the judgement, so no verdict is required. It is a
//!    candidate, excluded when the rejected experience is not in the
//!    corpus, the two attempted different tasks, either lacks a final
//!    output, or the outputs are identical. A retry that was given a
//!    critique of the other attempt still attempted the same task, so
//!    tasks that differ only in critiques count as one task here. The
//!    candidates are the two final outputs, each an assistant turn carrying
//!    the tool calls its experience's last agent step made.
//! 2. Grouped attempts: a verified pass and a verified fail of one group (one
//!    version of one task by one policy, see [`super::attempts`]), answered
//!    by the same model role, decided at the same strength (a formal fail
//!    says more than a judged pass, so a pair across strengths would rest on
//!    the weaker one).
//! 3. Repairs: a conversation whose first steps are copied from another
//!    trajectory (a failed attempt's prefix, continued by a teacher) and that
//!    verified as passed, against a verified fail of the same task whose own
//!    conversation that prefix is.
//!
//! A pair of the last two is made from the two conversations as the model
//! was shown them ([`Rendering`]: the requests it was sent when they were
//! recorded): where they first differ (call ids aside), the messages before
//! are the shared prompt, never trained on, and each side's continuation
//! from there to its last action is a candidate, whose assistant turns are
//! trained on and whose tool results are context. A prompt never ends in an
//! assistant turn: an action both took is part of both candidates. A pair is
//! left out, typed, when the two were not shown the same prompt (the
//! messages before the first action, and the tool list:
//! [`Exclusion::DifferentPrompt`]), when one conversation is the other's
//! beginning or they part after the last action ([`Exclusion::NoDivergence`]),
//! when a repair's copied prefix is not the failed attempt's own
//! ([`Exclusion::PrefixMismatch`]), when either candidate reads the
//! repository's history and the curation does not keep such reads
//! ([`Exclusion::RepositoryHistory`], the same policy the sft-step view
//! applies, on both sides), or when the two name different tokenizers
//! ([`Exclusion::MixedTokenizerDigest`]). An attempt that is not verified is
//! never a candidate (an infrastructure failure is never a rejected sample, an
//! unverified pass never a chosen one: see [`super::attempts`]); a verified
//! attempt with nothing of the other verdict to pair with is counted as
//! [`Exclusion::NoCounterpart`]. A failed continuation of another
//! trajectory's prefix is never a candidate ([`Exclusion::Failed`]).
//!
//! The pairs of one group, or of one repaired task, are at most
//! [`Preference::pairs_per_group`] when it is set: the candidate pairs are
//! ordered by a hash of the seed and the two experiences, the first that make
//! a record are kept and the rest counted as [`Exclusion::OverLimit`], so the
//! same seed keeps the same pairs. Unbounded, every pair is made, in corpus
//! order. Each record carries the attempts it compares
//! ([`crate::RolloutIdentity`]: the group, the chosen then the rejected
//! attempt, the tokenizer, the source), and a projection whose records name
//! two tokenizers is refused ([`ViewError::MixedTokenizerDigest`]).

use std::collections::{BTreeMap, HashSet};

use serde::{Deserialize, Serialize};
use splinter_core::annotation::{RelationKind, Strength};
use splinter_core::digest::digest_of;
use splinter_core::experience::ExperienceId;
use splinter_core::harness::HarnessConversation;

use super::attempts::{
    context, continuation, last_action_from, one_tool_set, own_start, reads_history,
    tokenizer_digest, verified, Attempt,
};
use super::rendering::Rendering;
use crate::curation::HistoryCuration;
use crate::render::{message, student_turn};
use crate::trajectory::{final_calls, same_message, Conversation};
use crate::{
    Corpus, Entry, Exclusion, Objective, PairSource, Projection, Provenance, RecordBody,
    RolloutIdentity, Strip, View, ViewError,
};

/// The name every preference record carries.
const NAME: &str = "preference";

/// The rule pairs from attempts are made by, as a manifest records it.
pub const PAIRING_RULE: &str = "verified-pass-over-verified-fail/shared-prefix";

/// How a preference dataset's pairs were chosen, as its manifest records it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PairingPolicy {
    /// The rule: [`PAIRING_RULE`].
    pub rule: String,
    /// The most pairs of one group (or one repaired task); `None` for every
    /// pair.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub per_group: Option<usize>,
    /// The seed the pairs of a bounded group are chosen by.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Pairs from recorded preferences.
    pub recorded: usize,
    /// Pairs from grouped attempts.
    pub attempts: usize,
    /// Pairs from repairs.
    pub repairs: usize,
}

/// Preference pairs over continuations of one prompt.
#[derive(Clone, Debug)]
pub struct Preference {
    min_strength: Strength,
    rendering: Rendering,
    per_group: Option<(usize, u64)>,
}

/// A verified attempt the view can pair: its conversation as rendered.
struct Placed<'c> {
    entry: &'c Entry,
    attempt: Attempt,
    role: Option<String>,
    conversation: Conversation,
    tokenizer: Option<String>,
    /// Where its own part begins; for a repair, the end of the copied prefix.
    own_start: usize,
}

/// A pair made, before it is pushed.
struct Made {
    body: RecordBody,
    experiences: Vec<ExperienceId>,
    tokenizer: Option<String>,
}

impl Preference {
    /// The view, accepting decisions at `min_strength` or stronger, showing
    /// the student only the instruction, leaving out a pair that reads the
    /// repository's history, making every pair.
    #[must_use]
    pub fn new(min_strength: Strength) -> Self {
        Self {
            min_strength,
            rendering: Rendering::default(),
            per_group: None,
        }
    }

    /// The same view under `strip`.
    #[must_use]
    pub fn with_strip(mut self, strip: Strip) -> Self {
        self.rendering.strip = strip;
        self
    }

    /// The same view doing `history` with a pair either of whose candidates
    /// reads the repository's history: [`HistoryCuration::Keep`] keeps it,
    /// either other mode leaves it out (a pair cannot drop an action, nor end
    /// one side early).
    #[must_use]
    pub fn history(mut self, history: HistoryCuration) -> Self {
        self.rendering.history = history;
        self
    }

    /// The same view pairing only the conversations answered by one of the
    /// model `roles` ([`SftStep::only_roles`](super::SftStep::only_roles)).
    #[must_use]
    pub fn only_roles(mut self, roles: Vec<String>) -> Self {
        self.rendering.only_roles = (!roles.is_empty()).then_some(roles);
        self
    }

    /// The same view reading a run exported as one agent step as one call per
    /// turn ([`SftStep::serialise_calls`](super::SftStep::serialise_calls)).
    #[must_use]
    pub fn serialise_calls(mut self, serialise_calls: bool) -> Self {
        self.rendering.serialise_calls = serialise_calls;
        self
    }

    /// The same view keeping at most `cap` pairs of one group (or one
    /// repaired task), chosen by `seed`.
    #[must_use]
    pub fn pairs_per_group(mut self, cap: usize, seed: u64) -> Self {
        self.per_group = Some((cap, seed));
        self
    }

    /// The recorded preference of `chosen` over `rejected`: their final
    /// outputs.
    fn recorded(
        &self,
        chosen: &Entry,
        rejected: &Entry,
    ) -> Result<(RecordBody, Provenance), Exclusion> {
        if chosen.experience.task.id != rejected.experience.task.id
            && !chosen
                .experience
                .to_task()
                .same_apart_from_critiques(&rejected.experience.to_task())
        {
            return Err(Exclusion::DifferentTask);
        }
        let (Some(better), Some(worse)) = (
            chosen.experience.final_output.as_deref(),
            rejected.experience.final_output.as_deref(),
        ) else {
            return Err(Exclusion::NoFinalOutput);
        };
        let reply = |entry: &Entry, output: &str| {
            let mut reply = message("assistant", output, true);
            reply.tool_calls = final_calls(&entry.experience.trajectory);
            reply
        };
        let (better, worse) = (reply(chosen, better), reply(rejected, worse));
        if same_message(&better, &worse) {
            return Err(Exclusion::IdenticalOutputs);
        }
        let turn = student_turn(chosen, &self.rendering.strip)?;
        Ok((
            RecordBody::Preference {
                prompt: vec![message("user", &turn, false)],
                chosen: vec![better],
                rejected: vec![worse],
                tools: Vec::new(),
                identity: None,
            },
            Provenance::of(vec![chosen.id.clone(), rejected.id.clone()]),
        ))
    }

    /// `entry` placed for pairing, or why it cannot be a candidate.
    fn place<'c>(&self, entry: &'c Entry) -> Result<Placed<'c>, Exclusion> {
        let harness = HarnessConversation::of(&entry.experience.trajectory);
        if !self.rendering.wants_role(harness.as_ref()) {
            return Err(Exclusion::RoleNotWanted);
        }
        let attempt = verified(entry, self.min_strength)?;
        let conversation = self.rendering.conversation_of(entry, harness.as_ref())?;
        let own_start = own_start(&conversation).ok_or(Exclusion::NoAction)?;
        let tokenizer = tokenizer_digest(entry)?;
        Ok(Placed {
            entry,
            attempt,
            role: harness.and_then(|h| h.role),
            conversation,
            tokenizer,
            own_start,
        })
    }

    /// The pair of `chosen` over `rejected`, from where their conversations
    /// part.
    fn pair(
        &self,
        chosen: &Placed<'_>,
        rejected: &Placed<'_>,
        source: PairSource,
    ) -> Result<Made, Exclusion> {
        let (won, lost) = (&chosen.conversation, &rejected.conversation);
        let prompt_mismatch = match source {
            PairSource::Attempts => Exclusion::DifferentPrompt,
            PairSource::Repair => Exclusion::PrefixMismatch,
        };
        let tools = one_tool_set(won)?;
        if tools != one_tool_set(lost)? {
            return Err(Exclusion::DifferentPrompt);
        }
        let parted = won
            .messages
            .iter()
            .zip(&lost.messages)
            .position(|(a, b)| !same_message(a, b));
        let parted = match parted {
            Some(at) => at,
            None if won.messages.len() == lost.messages.len() => {
                return Err(Exclusion::IdenticalOutputs)
            }
            // One is the other's beginning: nothing the other did instead.
            None => return Err(Exclusion::NoDivergence),
        };
        // The model's own part of the chosen side must start after what both
        // were shown: for attempts, the prompt before the first action; for a
        // repair, the copied prefix.
        if parted < chosen.own_start {
            return Err(prompt_mismatch);
        }
        // A candidate opens with the model's own turn. Runs that part at a turn
        // they were shown (a result, or a notice the harness inserted) part
        // from the action before it, which both took.
        let mut start = parted;
        if won.messages[start].role != "assistant" {
            match won.messages[..start]
                .iter()
                .rposition(|m| m.role == "assistant")
            {
                Some(action) => start = action,
                None => return Err(prompt_mismatch),
            }
        }
        // A prompt never ends in an assistant turn: an action both took is
        // part of both candidates.
        while start > 0 && won.messages[start - 1].role == "assistant" {
            start -= 1;
        }
        if start < chosen.own_start {
            return Err(prompt_mismatch);
        }
        let (Some(won_end), Some(lost_end)) =
            (last_action_from(won, start), last_action_from(lost, start))
        else {
            return Err(Exclusion::NoDivergence);
        };
        let copied = |conversation: &Conversation, end: usize| {
            conversation
                .actions
                .iter()
                .any(|a| a.copied && a.index >= start && a.index <= end)
        };
        if copied(won, won_end) || copied(lost, lost_end) {
            return Err(prompt_mismatch);
        }
        let better = continuation(won, start, won_end);
        let worse = continuation(lost, start, lost_end);
        if better.len() == worse.len() && better.iter().zip(&worse).all(|(a, b)| same_message(a, b))
        {
            return Err(Exclusion::IdenticalOutputs);
        }
        if self.rendering.history != HistoryCuration::Keep
            && (reads_history(&better) || reads_history(&worse))
        {
            return Err(Exclusion::RepositoryHistory);
        }
        let tokenizer = match (&chosen.tokenizer, &rejected.tokenizer) {
            (Some(a), Some(b)) if a != b => return Err(Exclusion::MixedTokenizerDigest),
            (a, b) => a.clone().or_else(|| b.clone()),
        };
        Ok(Made {
            body: RecordBody::Preference {
                prompt: context(won, start),
                chosen: better,
                rejected: worse,
                tools,
                identity: Some(RolloutIdentity {
                    group: chosen.attempt.group.clone(),
                    rollouts: vec![chosen.attempt.name.clone(), rejected.attempt.name.clone()],
                    tokenizer_digest: tokenizer.clone(),
                    source: Some(source),
                }),
            },
            experiences: vec![chosen.entry.id.clone(), rejected.entry.id.clone()],
            tokenizer,
        })
    }

    /// The pairs of one group's `candidates`, in the order this view takes
    /// them: by the seed's hash when bounded, else as given; at most the
    /// bound kept, every candidate counted in `projection` once.
    fn take_bounded(
        &self,
        mut candidates: Vec<(&Placed<'_>, &Placed<'_>)>,
        source: PairSource,
        projection: &mut Projection,
        made: &mut Vec<Made>,
    ) {
        if let Some((_, seed)) = self.per_group {
            candidates.sort_by_cached_key(|(c, r)| {
                digest_of(&(seed, c.entry.id.to_string(), r.entry.id.to_string())).to_string()
            });
        }
        let cap = self.per_group.map_or(usize::MAX, |(cap, _)| cap);
        let mut kept = 0;
        for (chosen, rejected) in candidates {
            match self.pair(chosen, rejected, source) {
                Ok(_) if kept >= cap => projection.exclude(Exclusion::OverLimit),
                Ok(pair) => {
                    kept += 1;
                    made.push(pair);
                }
                Err(reason) => projection.exclude(reason),
            }
        }
    }
}

impl View for Preference {
    fn name(&self) -> &str {
        NAME
    }

    fn objective(&self) -> Objective {
        Objective::Preference
    }

    fn project(&self, corpus: &Corpus) -> Result<Projection, ViewError> {
        let mut projection = Projection::new(
            NAME,
            Objective::Preference,
            Some(self.rendering.strip.clone()),
            Some(self.min_strength),
        );
        projection.curation = self.rendering.history.recorded();
        let mut paired: HashSet<(&ExperienceId, &ExperienceId)> = HashSet::new();
        let mut recorded = 0;
        for chosen in corpus.entries() {
            for other in chosen.related(RelationKind::PreferredOver) {
                let Some(rejected) = corpus.get(other) else {
                    projection.exclude(Exclusion::RelatedMissing);
                    continue;
                };
                if !paired.insert((&chosen.id, &rejected.id)) {
                    projection.exclude(Exclusion::Duplicate);
                    continue;
                }
                let outcome = self.recorded(chosen, rejected);
                recorded += usize::from(outcome.is_ok());
                projection.take(outcome);
            }
        }

        // Every verified attempt, placed; an experience that cannot be a
        // candidate is counted once, under why.
        let mut placed: Vec<Placed<'_>> = Vec::new();
        for entry in corpus.entries() {
            match self.place(entry) {
                Ok(p) => placed.push(p),
                Err(reason) => projection.exclude(reason),
            }
        }
        let (repairs, attempts): (Vec<&Placed<'_>>, Vec<&Placed<'_>>) = placed
            .iter()
            .partition(|p| p.conversation.actions.iter().any(|a| a.copied));
        // Only a continuation that passed is a repair.
        let repairs: Vec<&Placed<'_>> = repairs
            .into_iter()
            .filter(|p| {
                if !p.attempt.passed {
                    projection.exclude(Exclusion::Failed);
                }
                p.attempt.passed
            })
            .collect();

        // Grouped attempts: one group, one role, one strength.
        type Bucket<'p, 'c> = Vec<&'p Placed<'c>>;
        let mut buckets: BTreeMap<(String, Option<String>, Strength), (Bucket, Bucket)> =
            BTreeMap::new();
        let mut order: Vec<(String, Option<String>, Strength)> = Vec::new();
        for p in &attempts {
            let key = (p.attempt.group.clone(), p.role.clone(), p.attempt.strength);
            if !buckets.contains_key(&key) {
                order.push(key.clone());
            }
            let (passes, fails) = buckets.entry(key).or_default();
            if p.attempt.passed {
                passes.push(p);
            } else {
                fails.push(p);
            }
        }
        let mut made: Vec<Made> = Vec::new();
        for key in &order {
            let (passes, fails) = &buckets[key];
            if passes.is_empty() || fails.is_empty() {
                for _ in passes.iter().chain(fails) {
                    projection.exclude(Exclusion::NoCounterpart);
                }
                continue;
            }
            let candidates: Vec<_> = passes
                .iter()
                .flat_map(|c| fails.iter().map(move |r| (*c, *r)))
                .filter(|(c, r)| !paired.contains(&(&c.entry.id, &r.entry.id)))
                .collect();
            self.take_bounded(candidates, PairSource::Attempts, &mut projection, &mut made);
        }
        let from_attempts = made.len();

        // Repairs: each against the verified fails of its task.
        let fails: Vec<&Placed<'_>> = attempts
            .iter()
            .copied()
            .filter(|p| !p.attempt.passed)
            .collect();
        for repair in &repairs {
            let task = &repair.entry.experience.task.id;
            let candidates: Vec<_> = fails
                .iter()
                .filter(|f| &f.entry.experience.task.id == task)
                .map(|f| (*repair, *f))
                .collect();
            if candidates.is_empty() {
                projection.exclude(Exclusion::NoCounterpart);
                continue;
            }
            self.take_bounded(candidates, PairSource::Repair, &mut projection, &mut made);
        }
        let from_repairs = made.len() - from_attempts;

        let mut tokenizers: Vec<&str> = Vec::new();
        for pair in &made {
            if let Some(t) = pair.tokenizer.as_deref() {
                if !tokenizers.contains(&t) {
                    tokenizers.push(t);
                }
            }
        }
        if let [first, second, ..] = tokenizers[..] {
            return Err(ViewError::MixedTokenizerDigest {
                first: first.to_string(),
                second: second.to_string(),
            });
        }
        for pair in made {
            projection.push(pair.body, Provenance::of(pair.experiences));
        }
        projection.pairing = Some(PairingPolicy {
            rule: PAIRING_RULE.into(),
            per_group: self.per_group.map(|(cap, _)| cap),
            seed: self.per_group.map(|(_, seed)| seed),
            recorded,
            attempts: from_attempts,
            repairs: from_repairs,
        });
        Ok(projection)
    }
}
