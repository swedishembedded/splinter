// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The preference view: a chosen and a rejected answer to one task, for
//! direct preference optimisation.
//!
//! Pairs come from two places, in this order:
//!
//! 1. Recorded preferences: a [`RelationKind::PreferredOver`] relation,
//!    recorded on the chosen experience and naming the rejected one. The
//!    relation is the judgement, so no verdict is required. It is a
//!    candidate, excluded when the rejected experience is not in the
//!    corpus, the two attempted different tasks, either lacks a final
//!    output, or the outputs are identical. A retry that was given a
//!    critique of the other attempt still attempted the same task, so
//!    tasks that differ only in critiques count as one task here.
//! 2. Derived pairs: within one task, every experience decided pass paired
//!    with every experience decided fail at the same strength, that
//!    strength at or above the view's minimum. The same strength is what
//!    makes the two decisions comparable: a formal fail says more than a
//!    judged pass, so a pair across strengths would rest on the weaker one.
//!    A derived pair already recorded as a relation is not repeated.
//!
//! Each pair yields one record: the student's turn of the chosen
//! experience (see [`Strip`]; both attempted the same task) as the prompt,
//! and the two final outputs as the chosen and rejected replies - each an
//! assistant turn carrying the tool calls its experience's last agent step
//! made, if it made any. Two replies that say the same (the same text and
//! the same calls) are excluded as identical. The record's experiences are
//! the chosen, then the rejected one.

use std::collections::HashSet;

use splinter_record::annotation::{RelationKind, Strength};
use splinter_record::digest::Digest;
use splinter_record::experience::ExperienceId;

use crate::render::{message, student_turn};
use crate::trajectory::{final_calls, same_message};
use crate::{
    Corpus, Entry, Exclusion, Objective, Projection, Provenance, RecordBody, Strip, View, ViewError,
};

/// The name every preference record carries.
const NAME: &str = "preference";

/// Preference pairs over answers to one task.
#[derive(Clone, Debug)]
pub struct Preference {
    min_strength: Strength,
    strip: Strip,
}

impl Preference {
    /// The view, deriving pairs from decisions at `min_strength` or
    /// stronger, showing the student only the instruction.
    #[must_use]
    pub fn new(min_strength: Strength) -> Self {
        Self {
            min_strength,
            strip: Strip::default(),
        }
    }

    /// The same view under `strip`.
    #[must_use]
    pub fn with_strip(self, strip: Strip) -> Self {
        Self { strip, ..self }
    }

    fn record(
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
        let turn = student_turn(chosen, &self.strip)?;
        Ok((
            RecordBody::Preference {
                prompt: vec![message("user", &turn, false)],
                chosen: better,
                rejected: worse,
            },
            Provenance::of(vec![chosen.id.clone(), rejected.id.clone()]),
        ))
    }

    /// Pass and fail experiences of each task, decided at the minimum
    /// strength or above, as (pass, fail) pairs at equal strength; tasks in
    /// the order they first appear, experiences in corpus order.
    fn derived<'c>(&self, corpus: &'c Corpus) -> Vec<(&'c Entry, &'c Entry)> {
        let mut tasks: Vec<&Digest> = Vec::new();
        for entry in corpus.entries() {
            if !tasks.contains(&&entry.experience.task.id) {
                tasks.push(&entry.experience.task.id);
            }
        }
        let mut pairs = Vec::new();
        for task in tasks {
            let decided: Vec<(&Entry, bool, Strength)> = corpus
                .entries()
                .iter()
                .filter(|e| e.experience.task.id == *task)
                .filter_map(|e| e.decision().map(|d| (e, d.passed, d.strength)))
                .filter(|(_, _, strength)| *strength >= self.min_strength)
                .collect();
            for (pass, _, strength) in decided.iter().filter(|(_, passed, _)| *passed) {
                for (fail, _, _) in decided
                    .iter()
                    .filter(|(_, passed, s)| !*passed && s == strength)
                {
                    pairs.push((*pass, *fail));
                }
            }
        }
        pairs
    }
}

impl View for Preference {
    fn name(&self) -> &str {
        NAME
    }

    fn objective(&self) -> Objective {
        Objective::Dpo
    }

    fn project(&self, corpus: &Corpus) -> Result<Projection, ViewError> {
        let mut projection = Projection::new(
            NAME,
            Objective::Dpo,
            Some(self.strip.clone()),
            Some(self.min_strength),
        );
        let mut paired: HashSet<(&ExperienceId, &ExperienceId)> = HashSet::new();
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
                projection.take(self.record(chosen, rejected));
            }
        }
        for (chosen, rejected) in self.derived(corpus) {
            if paired.insert((&chosen.id, &rejected.id)) {
                projection.take(self.record(chosen, rejected));
            }
        }
        Ok(projection)
    }
}
