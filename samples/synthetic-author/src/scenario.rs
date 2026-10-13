// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The situations an author decides, and the options each offers.
//!
//! A situation is a set of facts and a choice among three of four ways to
//! act. The pool depends on its seed alone, never on an author, so a
//! situation by itself says nothing about who is deciding it.

use serde::Serialize;

use crate::rng::Rng;

/// The four ways a steward can meet a situation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Archetype {
    /// Decide and act on his own, now.
    ActAlone,
    /// Take the matter to the community and act on what it says.
    ConsultThenAct,
    /// Accept the help an outside power offers.
    DeferToOutsider,
    /// Do nothing yet.
    HoldAndWait,
}

/// What is true of a situation, whatever the author believes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Features {
    /// The act would go beyond what the steward's charter grants.
    pub exceeds_authority: bool,
    /// Delay would cost dearly.
    pub urgent: bool,
    /// The decision binds the community for years.
    pub binds_future: bool,
    /// The act cannot be undone.
    pub irreversible: bool,
    /// Members of the community are being hurt.
    pub members_harmed: bool,
    /// A power outside the community offers help.
    pub outsider_offers_help: bool,
}

/// One situation to decide.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Scenario {
    /// Its place in the pool.
    pub id: usize,
    /// The topic it is set in, an index into [`TOPICS`].
    pub topic: usize,
    /// The facts.
    pub features: Features,
    /// The ways to act, in a fixed order; three of the four.
    pub options: Vec<Archetype>,
}

/// The fixed subjects authors write about and situations are set in.
pub const TOPICS: [&str; 12] = [
    "the allocation of a shared granary",
    "tolls on a river crossing",
    "the contracts of apprentices",
    "weights and measures at the market",
    "the defence of a border fair",
    "funding a harbour light",
    "the ownership of common pasture",
    "the settlement of boundary disputes",
    "the keeping of the town archive",
    "the licensing of itinerant traders",
    "the care of the poor in winter",
    "the repair of the mill dam",
];

/// `n` situations drawn from `seed`.
#[must_use]
pub fn scenario_pool(seed: u64, n: usize) -> Vec<Scenario> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|id| {
            let features = Features {
                exceeds_authority: rng.coin(),
                urgent: rng.coin(),
                binds_future: rng.coin(),
                irreversible: rng.coin(),
                members_harmed: rng.coin(),
                outsider_offers_help: rng.coin(),
            };
            Scenario {
                id,
                topic: rng.below(TOPICS.len()),
                features,
                options: options(&features, &mut rng),
            }
        })
        .collect()
}

/// The options a situation offers: outside help only when it is offered, and
/// then one of the other three ways is not.
fn options(features: &Features, rng: &mut Rng) -> Vec<Archetype> {
    use Archetype::{ActAlone, ConsultThenAct, DeferToOutsider, HoldAndWait};
    let mut options = vec![ActAlone, ConsultThenAct, HoldAndWait];
    if features.outsider_offers_help {
        options.remove(rng.below(options.len()));
        options.push(DeferToOutsider);
        options.sort();
    }
    options
}
