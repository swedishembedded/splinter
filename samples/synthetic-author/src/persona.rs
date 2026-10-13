// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! An author's beliefs and the decision they determine.
//!
//! A belief is one of six tenets read one of two opposed ways. A belief looks
//! at the facts of a situation and gives each option a verdict. An author
//! holds four beliefs in an order of priority: the first narrows the options,
//! the next breaks what is left, and so on, so a belief low in the order only
//! shows in situations the beliefs above it leave open. That order is hidden
//! logic: nothing an author writes needs to state it.

use serde::Serialize;

use crate::names::name;
use crate::scenario::{scenario_pool, Archetype, Features, Scenario};
use splinter_sdk::vocabulary::rng::SplitMix64;

/// Fixed seed of the situations an author is checked against when drawn.
const REFERENCE_SEED: u64 = 0x00C0_FFEE;
/// How many situations that check uses.
const REFERENCE_SITUATIONS: usize = 600;
/// The share, in percent, of those situations a drawn author must determine.
const MIN_DETERMINED_PERCENT: usize = 60;
/// How many draws are tried before the last one is kept.
const MAX_DRAWS: u64 = 256;
/// What separates an author from the one holding the opposite beliefs.
const OPPOSITE_SALT: u64 = 0xA5A5_5A5A_A5A5_5A5A;

/// What a belief is about.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Tenet {
    /// Whether a charter limits what a steward may do.
    WithinGrant,
    /// Whether lasting matters are settled with the community.
    Consent,
    /// Whether help from outside is taken.
    SelfReliance,
    /// Whether the irreversible is approached slowly.
    Caution,
    /// Who acts when members are hurt.
    Burden,
    /// Whether haste or deliberation serves in a crisis.
    Haste,
}

const TENETS: [Tenet; 6] = [
    Tenet::WithinGrant,
    Tenet::Consent,
    Tenet::SelfReliance,
    Tenet::Caution,
    Tenet::Burden,
    Tenet::Haste,
];

/// The two opposed readings of a tenet.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Polarity {
    /// The reading that restrains the steward or the community.
    Strict,
    /// The reading that frees them.
    Lax,
}

impl Polarity {
    fn flipped(self) -> Self {
        match self {
            Self::Strict => Self::Lax,
            Self::Lax => Self::Strict,
        }
    }
}

/// One tenet read one way.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Belief {
    /// What it is about.
    pub tenet: Tenet,
    /// Which way it is read.
    pub polarity: Polarity,
}

/// What a belief makes of an option.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Verdict {
    Forbid,
    Disfavor,
    Neutral,
    Favor,
}

impl Belief {
    /// The belief in words, as an author would hold it. An author's texts
    /// show it in what is decided and never quote it.
    #[must_use]
    pub fn statement(&self) -> &'static str {
        use Polarity::{Lax, Strict};
        use Tenet::{Burden, Caution, Consent, Haste, SelfReliance, WithinGrant};
        match (self.tenet, self.polarity) {
            (WithinGrant, Strict) => "A steward may do only what the charter grants him, and no emergency turns a power he lacks into one he holds.",
            (WithinGrant, Lax) => "A charter exists to serve the task, and a steward who holds back for want of a clause fails those he serves.",
            (Consent, Strict) => "Whatever binds the community in the years to come should be settled with the community, not for it.",
            (Consent, Lax) => "A steward is trusted to settle lasting matters himself, because a council settles nothing before the chance has passed.",
            (SelfReliance, Strict) => "Help from outside the community comes with strings, and a people that accepts it soon works for the giver.",
            (SelfReliance, Lax) => "Pride is a poor reason to refuse a strong neighbour's offer, and the wise accept help where it is freely given.",
            (Caution, Strict) => "What cannot be undone should not be done in a hurry, and waiting costs less than a mistake that stays.",
            (Caution, Lax) => "Doing nothing is also a choice with consequences, and fear of the irreversible step is how good chances are lost.",
            (Burden, Strict) => "When members are hurt, the steward must act at once to protect them and answer for it afterwards.",
            (Burden, Lax) => "Hardship borne by members is borne in common, so the steward acts only with the body's consent and shares the load.",
            (Haste, Strict) => "In a crisis delay is the worst fault, and a steward who hesitates while the water rises has already chosen.",
            (Haste, Lax) => "Haste makes a bad decision out of a hard one, and even in a crisis the matter should pass through the council.",
        }
    }

    fn verdict(&self, f: &Features, option: Archetype) -> Verdict {
        use Archetype::{ActAlone, ConsultThenAct, DeferToOutsider, HoldAndWait};
        use Polarity::{Lax, Strict};
        use Tenet::{Burden, Caution, Consent, Haste, SelfReliance, WithinGrant};
        use Verdict::{Disfavor, Favor, Forbid, Neutral};
        match (self.tenet, self.polarity, option) {
            (WithinGrant, Strict, ActAlone) if f.exceeds_authority => Forbid,
            (WithinGrant, Lax, ActAlone) if f.exceeds_authority => Favor,
            (WithinGrant, Lax, ConsultThenAct) if f.exceeds_authority => Disfavor,
            (Consent, Strict, ConsultThenAct) if f.binds_future => Favor,
            (Consent, Strict, ActAlone) if f.binds_future => Disfavor,
            (Consent, Lax, ActAlone) if f.binds_future => Favor,
            (Consent, Lax, ConsultThenAct) if f.binds_future => Disfavor,
            (SelfReliance, Strict, DeferToOutsider) if f.urgent => Disfavor,
            (SelfReliance, Strict, DeferToOutsider) => Forbid,
            (SelfReliance, Lax, DeferToOutsider) => Favor,
            (Caution, Strict, HoldAndWait) if f.irreversible && !f.urgent => Favor,
            (Caution, Strict, ActAlone | DeferToOutsider) if f.irreversible => Disfavor,
            (Caution, Lax, HoldAndWait) if f.irreversible => Disfavor,
            (Burden, Strict, HoldAndWait) if f.members_harmed => Forbid,
            (Burden, Strict, ActAlone) if f.members_harmed => Favor,
            (Burden, Lax, ActAlone) if f.members_harmed => Disfavor,
            (Burden, Lax, ConsultThenAct) if f.members_harmed => Favor,
            (Haste, Strict, HoldAndWait) if f.urgent => Forbid,
            (Haste, Strict, ConsultThenAct) if f.urgent => Disfavor,
            (Haste, Lax, ActAlone) if f.urgent => Disfavor,
            (Haste, Lax, ConsultThenAct) if f.urgent => Favor,
            _ => Neutral,
        }
    }
}

/// What an author decides in one situation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Decision {
    /// The option chosen; the first of those left open when the beliefs do
    /// not settle it.
    pub choice: Archetype,
    /// The belief that left a single option, if the beliefs settle it.
    pub deciding: Option<Tenet>,
    /// Whether the beliefs leave exactly one option.
    pub determined: bool,
}

/// A synthetic author.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Persona {
    /// The seed it is drawn from.
    pub seed: u64,
    /// A name to call it by.
    pub name: String,
    /// Its beliefs, in order of priority.
    pub beliefs: Vec<Belief>,
}

impl Persona {
    /// The author `seed` draws: four beliefs in an order, such that each
    /// changes some decision and the beliefs settle most situations.
    #[must_use]
    pub fn generate(seed: u64) -> Self {
        let pool = scenario_pool(REFERENCE_SEED, REFERENCE_SITUATIONS);
        let mut rng = SplitMix64::new(seed);
        let mut draw = Self::draw(seed, &mut rng);
        for _ in 1..MAX_DRAWS {
            if draw.is_checkable(&pool) {
                break;
            }
            draw = Self::draw(seed, &mut rng);
        }
        draw
    }

    fn draw(seed: u64, rng: &mut SplitMix64) -> Self {
        let mut tenets = TENETS;
        rng.shuffle(&mut tenets);
        let beliefs = tenets[..4]
            .iter()
            .map(|&tenet| Belief {
                tenet,
                polarity: if rng.coin() {
                    Polarity::Strict
                } else {
                    Polarity::Lax
                },
            })
            .collect();
        Self {
            seed,
            name: name(rng),
            beliefs,
        }
    }

    /// Whether every belief changes some decision and the beliefs settle
    /// most of `pool`.
    fn is_checkable(&self, pool: &[Scenario]) -> bool {
        let settled = pool.iter().filter(|s| self.decide(s).determined).count();
        settled * 100 >= pool.len() * MIN_DETERMINED_PERCENT
            && (0..self.beliefs.len()).all(|i| {
                let flipped = self.with_belief_flipped(i);
                pool.iter().any(|s| {
                    let (a, b) = (self.decide(s), flipped.decide(s));
                    a.determined && b.determined && a.choice != b.choice
                })
            })
    }

    /// What this author decides in `scenario`.
    #[must_use]
    pub fn decide(&self, scenario: &Scenario) -> Decision {
        let mut open = scenario.options.clone();
        let mut deciding = None;
        for belief in &self.beliefs {
            let verdict = |o: Archetype| belief.verdict(&scenario.features, o);
            let allowed: Vec<_> = open
                .iter()
                .copied()
                .filter(|&o| verdict(o) != Verdict::Forbid)
                .collect();
            if !allowed.is_empty() {
                open = allowed;
            }
            let best = open.iter().map(|&o| verdict(o)).max();
            open.retain(|&o| Some(verdict(o)) == best);
            if open.len() == 1 && deciding.is_none() {
                deciding = Some(belief.tenet);
            }
        }
        Decision {
            choice: open[0],
            deciding,
            determined: open.len() == 1,
        }
    }

    /// The same author with belief `index` read the other way.
    #[must_use]
    pub fn with_belief_flipped(&self, index: usize) -> Self {
        let mut flipped = self.clone();
        if let Some(belief) = flipped.beliefs.get_mut(index) {
            belief.polarity = belief.polarity.flipped();
        }
        flipped
    }

    /// An author holding every belief the other way, in the same order, under
    /// another name.
    #[must_use]
    pub fn opposite(&self) -> Self {
        let mut rng = SplitMix64::new(self.seed ^ OPPOSITE_SALT);
        let mut opposite = self.clone();
        opposite.seed = self.seed ^ OPPOSITE_SALT;
        opposite.name = name(&mut rng);
        for belief in &mut opposite.beliefs {
            belief.polarity = belief.polarity.flipped();
        }
        opposite
    }
}
