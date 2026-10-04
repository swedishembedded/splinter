// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model-driven task generation that turns
// any source into verifiable training tasks, for its clients. If your team
// needs expertise in synthetic task generation or verifier design, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The tasks a generation has admitted so far, and what it has learned about
//! each kind: how often its proposals were refused and for what, which kinds
//! to stop asking for, and what the next request for one is told to correct.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use splinter_core::digest::Digest;
use splinter_core::experience::{PrivilegedKind, Task};
use splinter_knowledge::tasks::dedup::{contradictions, Asked};
use splinter_knowledge::tasks::{Catalogue, GenerationPolicy, Rejection, TaskKind};
use splinter_store::tasks::TaskEntry;

use super::{reason_name, KindTally, RejectionNote};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// How many proposals of a kind are refused before its admission rate is
/// judged.
pub const MIN_PROPOSALS_BEFORE_GIVING_UP: usize = 24;

/// The share of a kind's proposals that must be admitted for it to be asked
/// for again once [`MIN_PROPOSALS_BEFORE_GIVING_UP`] have been made.
pub const MIN_ADMISSION_RATE: f64 = 0.1;

/// How often a kind's proposals are refused for one reason before the
/// generator is told to correct it.
pub const MIN_REFUSALS_BEFORE_CORRECTING: usize = 3;

/// The most corrections a request carries for one kind: more is a list the
/// model does not follow.
pub const MAX_CORRECTIONS: usize = 2;

/// How an admitted task came to be, as its set entry records it.
pub(super) struct Provenance<'a> {
    pub(super) generator: &'a str,
    pub(super) prompt: Option<Digest>,
    pub(super) subject: Option<String>,
}

/// A question of the set as the contradiction rule compares it.
pub(super) struct Question {
    task: Digest,
    kind: String,
    instruction: String,
    subject: Option<String>,
    reference: String,
}

/// The tasks admitted so far, and the tallies.
#[derive(Default)]
pub(super) struct Batch {
    pub(super) entries: Vec<TaskEntry>,
    pub(super) seen: BTreeSet<Digest>,
    /// The admitted questions whose answers can contradict one another.
    pub(super) questions: Vec<Question>,
    pub(super) parts: usize,
    pub(super) per_kind: BTreeMap<String, KindTally>,
    pub(super) rejected: BTreeMap<String, usize>,
    pub(super) rejections: Vec<RejectionNote>,
    /// What each kind's proposals were refused for, and how often.
    pub(super) refused: BTreeMap<String, BTreeMap<Rejection, usize>>,
    /// No window is put to the generator after this.
    pub(super) deadline: Option<Instant>,
    /// Whether the deadline passed with windows still to cover.
    pub(super) expired: bool,
    /// The kinds given up on, and why.
    pub(super) dropped: BTreeMap<String, String>,
}

impl Batch {
    /// What the next request for each of `kinds` is told to correct: the
    /// reasons (at most [`MAX_CORRECTIONS`]) its proposals were refused for
    /// at least [`MIN_REFUSALS_BEFORE_CORRECTING`] times and that have a
    /// correction to give, the most frequent first.
    pub(super) fn corrections(&self, kinds: &[&TaskKind]) -> BTreeMap<String, Vec<Rejection>> {
        kinds
            .iter()
            .filter_map(|kind| {
                let counts = self.refused.get(&kind.name)?;
                let mut reasons: Vec<(Rejection, usize)> = counts
                    .iter()
                    .filter(|(reason, n)| {
                        **n >= MIN_REFUSALS_BEFORE_CORRECTING && reason.advice().is_some()
                    })
                    .map(|(reason, n)| (*reason, *n))
                    .collect();
                reasons.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                reasons.truncate(MAX_CORRECTIONS);
                (!reasons.is_empty()).then(|| {
                    (
                        kind.name.clone(),
                        reasons.into_iter().map(|(reason, _)| reason).collect(),
                    )
                })
            })
            .collect()
    }

    /// `kinds` without those the sources have shown they cannot satisfy: a
    /// kind with at least [`MIN_PROPOSALS_BEFORE_GIVING_UP`] proposals of which
    /// fewer than [`MIN_ADMISSION_RATE`] were admitted is dropped, and why is
    /// recorded. Questions about letters, for one, rarely name a subject, and
    /// asking a kind that needs one for every window spends the budget on
    /// proposals that are all refused.
    pub(super) fn still_worth_asking<'k>(&mut self, kinds: &[&'k TaskKind]) -> Vec<&'k TaskKind> {
        let mut active = Vec::with_capacity(kinds.len());
        for &kind in kinds {
            if self.dropped.contains_key(&kind.name) {
                continue;
            }
            let tally = self.per_kind.get(&kind.name);
            let (admitted, rejected) = tally.map_or((0, 0), |t| (t.admitted, t.rejected));
            let proposed = admitted + rejected;
            if proposed >= MIN_PROPOSALS_BEFORE_GIVING_UP
                && (admitted as f64) < MIN_ADMISSION_RATE * proposed as f64
            {
                self.dropped.insert(
                    kind.name.clone(),
                    format!("admitted {admitted} of {proposed} proposals, too few to keep asking"),
                );
            } else {
                active.push(kind);
            }
        }
        active
    }

    /// Stores `task` and adds it to the set, once.
    pub(super) fn admit(
        &mut self,
        ctx: &Context,
        kind: &str,
        task: Task,
        provenance: Provenance<'_>,
    ) -> Result<(), OrchestratorError> {
        let id = ctx.tasks().put(&task)?;
        if self.seen.insert(id.clone()) {
            self.per_kind.entry(kind.to_string()).or_default().admitted += 1;
            let reference = task
                .privileged
                .iter()
                .find(|p| p.kind == PrivilegedKind::Reference)
                .map(|p| p.content.clone());
            if let Some(reference) = reference {
                self.questions.push(Question {
                    task: id.clone(),
                    kind: kind.to_string(),
                    instruction: task.instruction,
                    subject: provenance.subject.clone(),
                    reference,
                });
            }
            self.entries.push(TaskEntry {
                task: id,
                generator: Some(provenance.generator.to_string()),
                prompt: provenance.prompt,
                variant_of: None,
                subject: provenance.subject,
            });
        }
        Ok(())
    }

    pub(super) fn reject(&mut self, kind: &str, reason: &str) {
        self.per_kind.entry(kind.to_string()).or_default().rejected += 1;
        *self.rejected.entry(reason.to_string()).or_default() += 1;
    }

    /// Leaves out every task that asks the same question of the same
    /// subject as another and answers it differently, of the kinds whose
    /// answer is one exact fact ([`TaskKind::exact_answer`]).
    pub(super) fn drop_contradictions(&mut self, catalogue: &Catalogue) {
        let policy = GenerationPolicy::default();
        let questions: Vec<&Question> = self
            .questions
            .iter()
            .filter(|q| catalogue.get(&q.kind).is_some_and(TaskKind::exact_answer))
            .collect();
        let asked: Vec<Asked<'_>> = questions
            .iter()
            .map(|q| Asked {
                instruction: &q.instruction,
                subject: q.subject.as_deref(),
                reference: &q.reference,
            })
            .collect();
        let mut dropped: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (i, j) in contradictions(&asked, policy.shingle_words, policy.max_overlap) {
            dropped.entry(i).or_default().push(j);
            dropped.entry(j).or_default().push(i);
        }
        if dropped.is_empty() {
            return;
        }
        let reason = reason_name(&Rejection::Contradiction);
        let mut gone = BTreeSet::new();
        for (at, others) in &dropped {
            let question = questions[*at];
            let answers: Vec<String> = others
                .iter()
                .map(|&o| format!("{:?}", questions[o].reference))
                .collect();
            let tally = self.per_kind.entry(question.kind.clone()).or_default();
            tally.admitted -= 1;
            tally.rejected += 1;
            *self.rejected.entry(reason.clone()).or_default() += 1;
            self.rejections.push(RejectionNote {
                kind: question.kind.clone(),
                reason: reason.clone(),
                detail: format!(
                    "{:?} answers {:?}; another task of the set asks it of {:?} and answers {}",
                    question.instruction,
                    question.reference,
                    question.subject.as_deref().unwrap_or_default(),
                    answers.join(", ")
                ),
            });
            gone.insert(question.task.clone());
        }
        self.entries.retain(|entry| !gone.contains(&entry.task));
        self.questions.retain(|q| !gone.contains(&q.task));
    }
}
