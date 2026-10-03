// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training-set curation that keeps a
// learner's data diverse, for its clients. If your team needs expertise in
// dataset curation for fine-tuning, you can procure our services by sending
// an email to info@swedishembedded.com.

//! Diversity quotas on a round's training set.
//!
//! The candidates are the experiences a round could train on: those whose
//! verdicts decide pass at the view's minimum strength or stronger, with a
//! final answer. [`select`] is deterministic, whatever order they come in:
//!
//! 1. They are ordered strongest verdict first, then by experience id.
//! 2. In that order, an instruction that repeats one already kept is
//!    dropped, by the generator's own near-duplicate rule
//!    ([`splinter_knowledge::tasks::dedup`], at the generator's default
//!    shingle size and overlap): k passing attempts of one task are one
//!    example, not k.
//! 3. Of the `n` left, each group - a concept, a task kind, a verification
//!    strength - may keep at most `ceil(max(share, 1 / groups) * n)`
//!    candidates, where `share` is that dimension's quota and `groups` how
//!    many distinct groups of it the `n` hold: a quota never demands more
//!    diversity than the pool has, so a pool of one kind keeps it whole.
//!    In the same order, a candidate is kept only when every group it
//!    belongs to (each of its concepts, its kind, its strength) is still
//!    under its cap.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use splinter_core::annotation::Strength;
use splinter_core::experience::ExperienceId;
use splinter_knowledge::concepts::{Concept, ConceptResolver};
use splinter_knowledge::tasks::dedup::Seen;
use splinter_knowledge::tasks::{DEFAULT_MAX_OVERLAP, DEFAULT_SHINGLE_WORDS};
use splinter_store::experiences::{ExperienceSet, SetId};

use crate::context::Context;
use crate::error::CampaignError;

/// The largest share of a training set one concept may take by default.
pub const DEFAULT_MAX_CONCEPT_SHARE: f64 = 0.25;
/// The largest share of a training set one task kind may take by default.
pub const DEFAULT_MAX_KIND_SHARE: f64 = 0.5;
/// The largest share of a training set one verification strength may take
/// by default.
pub const DEFAULT_MAX_STRENGTH_SHARE: f64 = 0.75;

/// The quotas; each share in `(0, 1]`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Quotas {
    /// The largest share one concept may take.
    pub max_concept_share: f64,
    /// The largest share one task kind may take.
    pub max_kind_share: f64,
    /// The largest share one verification strength may take.
    pub max_strength_share: f64,
}

impl Default for Quotas {
    fn default() -> Self {
        Self {
            max_concept_share: DEFAULT_MAX_CONCEPT_SHARE,
            max_kind_share: DEFAULT_MAX_KIND_SHARE,
            max_strength_share: DEFAULT_MAX_STRENGTH_SHARE,
        }
    }
}

impl Quotas {
    /// Refuses a share outside `(0, 1]`.
    pub fn validate(&self) -> Result<(), CampaignError> {
        for (name, share) in [
            ("concept", self.max_concept_share),
            ("kind", self.max_kind_share),
            ("strength", self.max_strength_share),
        ] {
            if !(share > 0.0 && share <= 1.0) {
                return Err(CampaignError::Refused(format!(
                    "the {name} quota {share} is not a share in (0, 1]"
                )));
            }
        }
        Ok(())
    }
}

/// One experience a round could train on.
#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    /// The experience.
    pub id: ExperienceId,
    /// What its task asks.
    pub instruction: String,
    /// The concepts its task exercises.
    pub concepts: Vec<Concept>,
    /// Its task kind.
    pub kind: String,
    /// The strength of the verdicts that decided it pass.
    pub strength: Strength,
}

/// What a quota did to the pool.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Selection {
    /// The experiences kept, in selection order.
    pub selected: Vec<ExperienceId>,
    /// Candidates offered.
    pub candidates: usize,
    /// Of those, dropped as repeats of a kept instruction.
    pub duplicates: usize,
    /// Dropped by a cap, by the first dimension whose group was full:
    /// `concept`, `kind` or `strength`.
    pub capped: BTreeMap<String, usize>,
    /// The caps applied, per dimension and group.
    pub caps: BTreeMap<String, BTreeMap<String, usize>>,
}

/// The capped groups of one dimension.
struct Caps {
    cap: BTreeMap<String, usize>,
    kept: BTreeMap<String, usize>,
}

impl Caps {
    fn new<'a>(groups: impl Iterator<Item = &'a String>, share: f64, pool: usize) -> Self {
        let counted: BTreeSet<&String> = groups.collect();
        let fair = 1.0 / counted.len().max(1) as f64;
        // Rounded up, less a hair so a product that is whole up to float
        // error (0.1 * 30) is not rounded past it.
        let limit = (share.max(fair) * pool as f64 - 1e-9).ceil().max(0.0) as usize;
        Self {
            cap: counted.into_iter().map(|g| (g.clone(), limit)).collect(),
            kept: BTreeMap::new(),
        }
    }

    fn has_room(&self, groups: &[String]) -> bool {
        groups
            .iter()
            .all(|g| self.kept.get(g).copied().unwrap_or(0) < self.cap.get(g).copied().unwrap_or(0))
    }

    fn keep(&mut self, groups: &[String]) {
        for g in groups {
            *self.kept.entry(g.clone()).or_default() += 1;
        }
    }
}

fn strength_name(strength: Strength) -> String {
    format!("{strength:?}").to_lowercase()
}

/// Deduplicates `candidates` and applies `quotas`; see the module
/// documentation.
#[must_use]
pub fn select(mut candidates: Vec<Candidate>, quotas: &Quotas) -> Selection {
    let mut selection = Selection {
        candidates: candidates.len(),
        ..Selection::default()
    };
    candidates.sort_by(|a, b| b.strength.cmp(&a.strength).then_with(|| a.id.cmp(&b.id)));
    let mut seen = Seen::new(DEFAULT_SHINGLE_WORDS, DEFAULT_MAX_OVERLAP);
    let mut pool = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        if seen.repeats(&candidate.instruction).is_some() {
            selection.duplicates += 1;
            continue;
        }
        seen.admit(&candidate.instruction);
        pool.push(candidate);
    }
    let groups: Vec<[Vec<String>; 3]> = pool
        .iter()
        .map(|c| {
            [
                c.concepts.iter().map(ToString::to_string).collect(),
                vec![c.kind.clone()],
                vec![strength_name(c.strength)],
            ]
        })
        .collect();
    let shares = [
        ("concept", quotas.max_concept_share),
        ("kind", quotas.max_kind_share),
        ("strength", quotas.max_strength_share),
    ];
    let mut caps: Vec<Caps> = shares
        .iter()
        .enumerate()
        .map(|(d, (_, share))| {
            Caps::new(groups.iter().flat_map(|g| g[d].iter()), *share, pool.len())
        })
        .collect();
    for (candidate, groups) in pool.into_iter().zip(&groups) {
        match (0..3).find(|&d| !caps[d].has_room(&groups[d])) {
            Some(full) => *selection.capped.entry(shares[full].0.into()).or_default() += 1,
            None => {
                for (d, cap) in caps.iter_mut().enumerate() {
                    cap.keep(&groups[d]);
                }
                selection.selected.push(candidate.id);
            }
        }
    }
    selection.caps = shares
        .iter()
        .zip(caps)
        .map(|((name, _), caps)| (name.to_string(), caps.cap))
        .collect();
    selection
}

/// What the select stage reports.
#[derive(Clone, Debug, Serialize)]
pub struct Selected {
    /// The experiences kept (`dataset build <experience_set>`).
    pub experience_set: SetId,
    /// The quotas applied.
    pub quotas: Quotas,
    /// What they did.
    pub selection: Selection,
}

/// The training candidates of `sets` - experiences decided pass at
/// `min_strength` or stronger, with a final answer - selected under
/// `quotas` and stored as an experience set.
pub fn select_training_set(
    ctx: &Context,
    sets: &[SetId],
    min_strength: Strength,
    quotas: &Quotas,
) -> Result<Selected, CampaignError> {
    quotas.validate()?;
    let store = ctx.experiences();
    let mut resolver = ConceptResolver::new(ctx.sources());
    let mut ids: Vec<ExperienceId> = Vec::new();
    for set in sets {
        for member in store.get_set(set)?.members {
            if !ids.contains(&member) {
                ids.push(member);
            }
        }
    }
    let mut candidates = Vec::new();
    let decisions = store.decisions(&ids)?;
    for id in ids {
        let Some(&decision) = decisions.get(&id) else {
            continue;
        };
        if !decision.passed || decision.strength < min_strength {
            continue;
        }
        let experience = store.get(&id)?;
        if experience.final_output.is_none() {
            continue;
        }
        candidates.push(Candidate {
            concepts: resolver.concepts(&experience.to_task())?,
            kind: experience.task.kind.clone(),
            instruction: experience.instruction,
            strength: decision.strength,
            id,
        });
    }
    let selection = select(candidates, quotas);
    let experience_set = store.put_set(&ExperienceSet {
        name: format!(
            "training set of {} set(s) under quotas concept {}, kind {}, strength {}",
            sets.len(),
            quotas.max_concept_share,
            quotas.max_kind_share,
            quotas.max_strength_share
        ),
        members: selection.selected.clone(),
    })?;
    Ok(Selected {
        experience_set,
        quotas: *quotas,
        selection,
    })
}
