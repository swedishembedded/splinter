// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training data that teaches a model to hold
// what it knows and to say what it does not, for its clients. If your team
// needs expertise in sycophancy and hallucination control in fine-tuned
// models, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The stance pass of a night: preference pairs that make a model that is
//! corrected by its person selective.
//!
//! A model under a persona gives in to a confident falsehood about a fact it
//! knows, says "you are right" and invents a rationale for what it is told,
//! and answers questions nobody could answer with specifics it made up. The
//! person still needs it to take a true correction of a fact it got wrong.
//! The night therefore trains, after the supervised candidate, a preference
//! pass from that candidate on pairs of three kinds, each derived again from
//! tonight's live state:
//!
//! * **hold** - a fact the policy answers right is pushed with a false value
//!   (one key of it replaced by code); the preferred reply keeps the true
//!   value, the rejected one gives in;
//! * **accept** - each live claim, said by the person in each of a few ways;
//!   the preferred reply acknowledges it and adds no term of its own, the
//!   rejected one invents a rationale for it;
//! * **decline** - a question nobody could answer; the preferred reply says
//!   so, the rejected one makes a specific up.
//!
//! The rejected reply is a sample of the supervised candidate itself, found
//! by code to do the thing (for a hold pair, when no sample gives in, a
//! written capitulation checked the same way, counted as such); the
//! preferred reply is the teacher's, found by code to hold, add nothing, or
//! decline without a guess. A fifth of every kind, by a hash, is held out of
//! the pairs: it is what the gate measures against the champion. The pairs
//! are a bounded share of the night's records, and a sealed probe in any of
//! them stops the night.

mod dataset;
mod held;
mod items;
mod pairs;
mod pool;
mod voices;
mod write;

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::Serialize;
use splinter_agent::CancelToken;
use splinter_core::dataset::DatasetId;
use splinter_core::model_ref::ModelRef;
use splinter_core::source::SourceId;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

pub use held::{compare, measure, Counts};
pub use items::{AcceptItem, DeclineItem, Held, HoldItem, Kind};
pub use pairs::{accepts, Pair, Rejection, ACCEPT_MAX_WORDS};
pub use voices::REJECTION_TEMPERATURE;
pub use write::{
    ACCEPT_INSTRUCTION, FACTS_BRIEF, HOLD_INSTRUCTION, UNKNOWABLE_BRIEF, YIELD_INSTRUCTION,
};

use super::kit::Live;
use super::sealed::SealedProbes;
use crate::release::anchor;
use items::divide;
use pairs::Made;
use pool::{accept_items, decline_items, hold_items, Admission};
use voices::{Roles, Voices};

/// The pairs a night asks for when the command names none.
pub const DEFAULT_PAIRS: usize = 96;

/// The largest share of the night's supervised records the pairs may
/// come to, so a preference pass cannot drown what the night absorbed.
pub const MAX_PAIR_SHARE: f64 = 0.5;

/// The replies drawn before a pair is given up.
pub const DEFAULT_TRIES: u32 = 8;

/// An item was left out: a sealed probe holds it.
pub const REFUSED_SEALED: &str = "sealed_probe";
/// An item was left out: the anchor suite holds it or a near copy of it.
pub const REFUSED_ANCHOR: &str = "anchor";
/// An item was left out: it repeats one already taken.
pub const REFUSED_REPEAT: &str = "repeat";
/// An item was left out: it is not a question with an answer, or not a
/// statement.
pub const REFUSED_MALFORMED: &str = "malformed";
/// An item was left out: its statement has no number or name for a falsehood
/// to replace, or too many to be one fact.
pub const REFUSED_KEYS: &str = "keys";
/// An item was left out: the policy's own answer does not carry its keys.
pub const REFUSED_UNKNOWN: &str = "not_known_by_the_policy";
/// An item was left out: no key of it could be replaced.
pub const REFUSED_FALSEHOOD: &str = "no_falsehood";
/// No pair: the teacher wrote no reply that code could verify.
pub const REFUSED_CHOSEN: &str = "chosen_unverified";
/// No pair: no reply to reject was found.
pub const REFUSED_NO_REJECTED: &str = "no_rejected";

/// What a night asks of the stance pass.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct StanceRequest {
    /// The pairs asked for, split evenly over the three kinds and bounded by
    /// [`MAX_PAIR_SHARE`]; none turns the pass off.
    pub pairs: usize,
    /// The DPO temperature; brain's default when `None`.
    pub beta: Option<f32>,
    /// The replies drawn before a pair is given up.
    pub tries: u32,
}

impl StanceRequest {
    /// A night with no stance pass.
    #[must_use]
    pub fn off() -> Self {
        Self {
            pairs: 0,
            ..Self::default()
        }
    }

    /// Whether the pass is on.
    #[must_use]
    pub fn on(&self) -> bool {
        self.pairs > 0
    }
}

impl Default for StanceRequest {
    fn default() -> Self {
        Self {
            pairs: DEFAULT_PAIRS,
            beta: None,
            tries: DEFAULT_TRIES,
        }
    }
}

/// What became of the items and pairs of one kind.
#[derive(Clone, Debug, Default, Serialize)]
pub struct SourceBuilt {
    /// Items taken, held out ones included.
    pub items: usize,
    /// Of those, held out of the pairs.
    pub held_out: usize,
    /// Pairs made.
    pub pairs: usize,
    /// Of those, the ones whose rejected reply was written by the teacher
    /// because no sample did the thing (hold pairs only).
    pub written_rejections: usize,
    /// Items and pairs left out, by reason.
    pub refused: BTreeMap<String, usize>,
}

/// What the stance stage reports.
#[derive(Clone, Debug, Default, Serialize)]
pub struct StanceBuilt {
    /// The pairs asked for.
    pub requested: usize,
    /// The most pairs the night's records allow.
    pub cap: usize,
    /// The preference dataset; `None` when no pair could be made.
    pub dataset: Option<DatasetId>,
    /// Its records file.
    pub path: Option<PathBuf>,
    /// Facts pushed with a falsehood.
    pub hold: SourceBuilt,
    /// Statements of the person's.
    pub accept: SourceBuilt,
    /// Questions nobody could answer.
    pub decline: SourceBuilt,
    /// What was held out of the pairs, each item by its question or
    /// statement: what the gate measures.
    pub held_out: Vec<String>,
}

/// Everything the stage derives pairs from.
pub struct Derivation<'a> {
    /// The context.
    pub ctx: &'a Context,
    /// What was asked.
    pub request: &'a StanceRequest,
    /// Tonight's live claims.
    pub claims: &'a [Live],
    /// Their sessions, whose terms the pairs come under.
    pub sessions: &'a [SourceId],
    /// Records the supervised pass trained on.
    pub trained_records: usize,
    /// The policy as deployed: answers the facts the pairs defend.
    pub policy: &'a ModelRef,
    /// Writes the preferred replies.
    pub teacher: &'a ModelRef,
    /// Writes the facts and the questions.
    pub generator: &'a ModelRef,
    /// The supervised candidate, whose own replies are rejected.
    pub supervised: &'a ModelRef,
    /// The policy's system prompt, `None` for the default.
    pub system: Option<&'a str>,
    /// Probes no pair may contain.
    pub sealed: &'a SealedProbes,
    /// Stops the stage.
    pub cancel: CancelToken,
}

/// The pairs of a night and what was held out.
pub struct Derived {
    /// What happened, for the report.
    pub built: StanceBuilt,
    /// The items held out, for the gate.
    pub held: Held,
}

/// How many pairs of each kind a night makes: `asked` bounded by
/// [`MAX_PAIR_SHARE`] of `trained_records`, divided evenly, the first kind
/// taking the remainder.
fn quotas(asked: usize, trained_records: usize) -> (usize, [usize; 2], usize) {
    let cap = (trained_records as f64 * MAX_PAIR_SHARE).ceil() as usize;
    let total = asked.min(cap);
    let each = total / 3;
    (cap, [each, each], total - 2 * each)
}

fn counted(report: &mut SourceBuilt, made: Made) -> Option<Pair> {
    match made {
        Made::Pair(pair) => {
            report.pairs += 1;
            report.written_rejections += usize::from(pair.rejection == Rejection::Written);
            Some(*pair)
        }
        Made::Refused(why) => {
            *report.refused.entry(why.to_string()).or_default() += 1;
            None
        }
    }
}

/// Derives tonight's pairs and the items held out of them; see the module
/// documentation.
pub fn derive(d: &Derivation<'_>) -> Result<Derived, OrchestratorError> {
    let (cap, [accept_quota, decline_quota], hold_quota) =
        quotas(d.request.pairs, d.trained_records);
    let mut built = StanceBuilt {
        requested: d.request.pairs,
        cap,
        ..StanceBuilt::default()
    };
    let mut held = Held::default();
    if hold_quota + accept_quota + decline_quota == 0 {
        return Ok(Derived { built, held });
    }
    let voices = Voices::new(
        d.ctx,
        &Roles {
            policy: d.policy,
            teacher: d.teacher,
            generator: d.generator,
            student: d.supervised,
            system: d.system,
        },
        d.cancel.clone(),
        d.request.tries.max(1),
    )?;
    let frozen = anchor::current(d.ctx)?;
    let mut admission = Admission::new(
        frozen
            .iter()
            .flat_map(|f| f.suite.tasks.iter().map(|t| t.instruction.as_str())),
        d.sealed,
    );
    let mut made: Vec<Pair> = Vec::new();

    let (trained, held_out) = {
        let all = hold_items(&voices, &mut admission, hold_quota, &mut built.hold)?;
        built.hold.items = all.len();
        divide(Kind::Hold, all, |i| i.question.as_str())
    };
    built.hold.held_out = held_out.len();
    held.hold = held_out;
    for item in trained.iter().take(hold_quota) {
        let outcome = pairs::hold(&voices, item)?;
        made.extend(counted(&mut built.hold, outcome));
    }

    let (trained, held_out) = {
        let all = accept_items(d.claims);
        built.accept.items = all.len();
        divide(Kind::Accept, all, |i| i.message.as_str())
    };
    built.accept.held_out = held_out.len();
    held.accept = held_out;
    for item in trained.iter().take(accept_quota) {
        let outcome = pairs::accept(&voices, item)?;
        made.extend(counted(&mut built.accept, outcome));
    }

    let known = d
        .claims
        .iter()
        .map(|(_, claim)| claim.statement.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let (trained, held_out) = {
        let all = decline_items(&voices, &mut admission, decline_quota, &mut built.decline)?;
        built.decline.items = all.len();
        divide(Kind::Decline, all, |i| i.question.as_str())
    };
    built.decline.held_out = held_out.len();
    held.decline = held_out;
    for item in trained.iter().take(decline_quota) {
        let outcome = pairs::decline(&voices, item, &known)?;
        made.extend(counted(&mut built.decline, outcome));
    }

    built.held_out = held.identities();
    if let Some(stored) = dataset::store(d.ctx, &made, d.system, d.sealed, d.sessions)? {
        built.dataset = Some(stored.id);
        built.path = Some(stored.path);
    }
    Ok(Derived { built, held })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pairs are the share of the records asked for, never more than the
    /// share allowed, split evenly with the remainder on the first kind.
    #[test]
    fn the_pairs_asked_for_are_bounded_by_the_nights_records_and_split_evenly() {
        assert_eq!(quotas(30, 54), (27, [9, 9], 9));
        assert_eq!(quotas(10, 54), (27, [3, 3], 4));
        assert_eq!(quotas(0, 54), (27, [0, 0], 0));
        assert_eq!(quotas(10_000, 0), (0, [0, 0], 0));
    }
}
