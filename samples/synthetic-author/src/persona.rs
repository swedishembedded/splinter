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
use crate::scenario::{held_out_combo, scenario_pool, Archetype, Features, Scenario};
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

/// Why an author cannot be checked against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Uncheckable {
    /// The beliefs leave too many situations open.
    Settles {
        /// Situations the beliefs settle.
        settled: usize,
        /// Situations looked at.
        of: usize,
    },
    /// A belief changes no decision, so no text could show it.
    Idle {
        /// The belief.
        belief: Belief,
    },
}

impl std::fmt::Display for Uncheckable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Settles { settled, of } => write!(
                f,
                "its beliefs settle {settled} of {of} reference situations and at least {MIN_DETERMINED_PERCENT} percent are needed"
            ),
            Self::Idle { belief } => write!(
                f,
                "the belief \"{}\" changes no decision, so no text could show it",
                belief.statement()
            ),
        }
    }
}

impl std::error::Error for Uncheckable {}

/// The situations an author is checked against: the reference ones that the
/// corpus of the world `seed` can hold, since a belief that decides nothing
/// there cannot be learned from the corpus.
fn reference_pool(seed: u64) -> Vec<Scenario> {
    scenario_pool(REFERENCE_SEED, REFERENCE_SITUATIONS)
        .into_iter()
        .filter(|s| !held_out_combo(seed, s))
        .collect()
}

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

impl Tenet {
    /// The fact of a situation this tenet turns on: the belief only gives a
    /// verdict where it holds (for [`Tenet::SelfReliance`], where an outsider
    /// offers help, which is the only situation that lists the option it
    /// judges). [`Belief::verdict`] is the table that reads it.
    #[must_use]
    pub fn fact(self) -> Fact {
        match self {
            Self::WithinGrant => Fact::ExceedsAuthority,
            Self::Consent => Fact::BindsFuture,
            Self::SelfReliance => Fact::OutsiderOffersHelp,
            Self::Caution => Fact::Irreversible,
            Self::Burden => Fact::MembersHarmed,
            Self::Haste => Fact::Urgent,
        }
    }
}

/// One of the facts that make up a situation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fact {
    /// The act would go beyond what the charter grants.
    ExceedsAuthority,
    /// The decision binds the community for years.
    BindsFuture,
    /// An outside power offers help.
    OutsiderOffersHelp,
    /// The act cannot be undone.
    Irreversible,
    /// Members of the community are being hurt.
    MembersHarmed,
    /// Delay would cost dearly.
    Urgent,
}

impl Fact {
    /// Whether the fact holds in a situation with `features`.
    #[must_use]
    pub fn holds(self, features: &Features) -> bool {
        match self {
            Self::ExceedsAuthority => features.exceeds_authority,
            Self::BindsFuture => features.binds_future,
            Self::OutsiderOffersHelp => features.outsider_offers_help,
            Self::Irreversible => features.irreversible,
            Self::MembersHarmed => features.members_harmed,
            Self::Urgent => features.urgent,
        }
    }

    /// The fact as a clause: that it holds, or that it does not.
    fn said(self, holds: bool) -> &'static str {
        match (self, holds) {
            (Self::ExceedsAuthority, true) => "the act goes beyond what the charter grants",
            (Self::ExceedsAuthority, false) => "the act is within what the charter grants",
            (Self::BindsFuture, true) => "the matter binds the community for years",
            (Self::BindsFuture, false) => "the matter binds only this year",
            (Self::OutsiderOffersHelp, true) => "an outside house offers help",
            (Self::OutsiderOffersHelp, false) => "no outside house offers help",
            (Self::Irreversible, true) => "the act cannot be undone",
            (Self::Irreversible, false) => "the act can be reversed",
            (Self::MembersHarmed, true) => "members of the community are being hurt",
            (Self::MembersHarmed, false) => "no member of the community is being hurt",
            (Self::Urgent, true) => "delay would cost dearly",
            (Self::Urgent, false) => "there is no pressing hurry",
        }
    }

    /// Lowercase word stems by which a reason is taken to name the fact. A
    /// fixed list, so that a reason is graded by code and not by a judge; it
    /// errs towards missing a paraphrase over crediting an unrelated word.
    #[must_use]
    pub fn terms(self) -> &'static [&'static str] {
        match self {
            Self::ExceedsAuthority => &["charter", "authority", "grant", "power", "mandate"],
            Self::BindsFuture => &["bind", "bound", "years", "generation", "lasting", "future"],
            Self::OutsiderOffersHelp => &["offer", "outside", "neighbour", "help"],
            Self::Irreversible => &["undone", "undo", "irreversible", "reverse", "permanent"],
            Self::MembersHarmed => &["hurt", "suffer", "harm", "famil", "injur"],
            Self::Urgent => &[
                "urgent", "hurry", "delay", "pressing", "haste", "rising", "rises",
            ],
        }
    }
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
pub enum Verdict {
    /// The option is ruled out, unless that would leave nothing.
    Forbid,
    /// The option is worse than the others.
    Disfavor,
    /// The belief says nothing of it.
    Neutral,
    /// The option is better than the others.
    Favor,
}

/// Every way to act, in the order of [`Archetype`].
const OPTIONS: [Archetype; 4] = [
    Archetype::ActAlone,
    Archetype::ConsultThenAct,
    Archetype::DeferToOutsider,
    Archetype::HoldAndWait,
];

/// Every fact a situation is made of.
const FACTS: [Fact; 6] = [
    Fact::ExceedsAuthority,
    Fact::BindsFuture,
    Fact::OutsiderOffersHelp,
    Fact::Irreversible,
    Fact::MembersHarmed,
    Fact::Urgent,
];

/// One verdict of a belief: it applies to an option where the facts listed
/// hold or do not, whatever the facts not listed are.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rule {
    /// The option judged.
    pub option: Archetype,
    /// The verdict.
    pub verdict: Verdict,
    /// The facts the verdict depends on, and whether each holds.
    pub when: Vec<(Fact, bool)>,
}

impl Rule {
    /// Whether the verdict applies in a situation with `features`.
    #[must_use]
    pub fn holds(&self, features: &Features) -> bool {
        self.when
            .iter()
            .all(|&(fact, wanted)| fact.holds(features) == wanted)
    }

    /// The rule as a sentence addressed to the author.
    #[must_use]
    pub fn describe(&self) -> String {
        let verdict = match self.verdict {
            Verdict::Forbid => "rule out",
            Verdict::Disfavor => "disfavour",
            Verdict::Favor | Verdict::Neutral => "favour",
        };
        let option = match self.option {
            Archetype::ActAlone => "settling it yourself",
            Archetype::ConsultThenAct => "taking it to the assembly",
            Archetype::DeferToOutsider => "accepting the outside offer",
            Archetype::HoldAndWait => "waiting",
        };
        if self.when.is_empty() {
            return format!("You {verdict} {option}.");
        }
        let when: Vec<&str> = self
            .when
            .iter()
            .map(|&(fact, holds)| fact.said(holds))
            .collect();
        format!("Where {}, you {verdict} {option}.", when.join(" and "))
    }
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

    /// The belief as a private note to someone playing the author: a
    /// disposition in the second person, not a maxim to be delivered.
    #[must_use]
    pub fn note(&self) -> &'static str {
        use Polarity::{Lax, Strict};
        use Tenet::{Burden, Caution, Consent, Haste, SelfReliance, WithinGrant};
        match (self.tenet, self.polarity) {
            (WithinGrant, Strict) => "You keep to the powers your charter gives you, and you will not go past them whatever the emergency, even when it costs you.",
            (WithinGrant, Lax) => "You treat the charter as a tool for the work and not a fence round it, and you stretch it when the job needs it.",
            (Consent, Strict) => "You settle with the people anything that will bind them for years, and you are uneasy deciding such things on your own.",
            (Consent, Lax) => "You decide lasting matters yourself, because you have seen councils talk until the chance was gone.",
            (SelfReliance, Strict) => "You distrust help that comes from outside, having watched peoples who took it end up serving whoever gave it.",
            (SelfReliance, Lax) => "You take help from a strong neighbour when it is freely offered, and you think refusing it out of pride is foolish.",
            (Caution, Strict) => "You go slowly with anything that cannot be undone, and you would rather wait than make a mistake that stays.",
            (Caution, Lax) => "You think doing nothing has its own costs, and you have seen good chances lost to fear of an irreversible step.",
            (Burden, Strict) => "When members of the community are being hurt you act at once to protect them and account for it afterwards.",
            (Burden, Lax) => "You think hardship is carried in common, so when members are hurt you act only with the body's consent and share the load.",
            (Haste, Strict) => "You count delay in a crisis as the worst fault, and you cannot sit still while things get worse.",
            (Haste, Lax) => "You believe haste turns a hard decision into a bad one, so even in a crisis you take the matter through the council.",
        }
    }

    /// Every verdict this belief gives, read off the table [`Self::verdict`]
    /// is: one rule for each option and verdict that occurs, naming the facts
    /// that are the same in every situation it applies to.
    #[must_use]
    pub fn rules(&self) -> Vec<Rule> {
        let situations: Vec<Features> = (0..64u32)
            .map(|bits| Features {
                exceeds_authority: bits & 1 != 0,
                urgent: bits & 2 != 0,
                binds_future: bits & 4 != 0,
                irreversible: bits & 8 != 0,
                members_harmed: bits & 16 != 0,
                outsider_offers_help: bits & 32 != 0,
            })
            .collect();
        let mut rules = Vec::new();
        for option in OPTIONS {
            for verdict in [Verdict::Forbid, Verdict::Disfavor, Verdict::Favor] {
                let applies: Vec<&Features> = situations
                    .iter()
                    .filter(|f| self.verdict(f, option) == verdict)
                    .collect();
                if applies.is_empty() {
                    continue;
                }
                let when = FACTS
                    .iter()
                    .filter_map(|&fact| {
                        let held = applies.iter().filter(|f| fact.holds(f)).count();
                        if held == applies.len() {
                            Some((fact, true))
                        } else if held == 0 {
                            Some((fact, false))
                        } else {
                            None
                        }
                    })
                    .collect();
                rules.push(Rule {
                    option,
                    verdict,
                    when,
                });
            }
        }
        rules
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
        let pool = reference_pool(seed);
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

    /// The author who holds every belief of the one `seed` draws the other
    /// way, set in the same world, if it can be checked against as any author
    /// can: the rule that makes a drawn author usable applies to it too.
    ///
    /// # Errors
    /// The opposite fails the rule, with the reason.
    pub fn opposite_of(seed: u64) -> Result<Self, Uncheckable> {
        let opposite = Self::generate(seed).opposite();
        opposite.checkability(&reference_pool(seed))?;
        Ok(opposite)
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

    fn is_checkable(&self, pool: &[Scenario]) -> bool {
        self.checkability(pool).is_ok()
    }

    /// Whether every belief changes some decision and the beliefs settle
    /// most of `pool`: an author for whom either fails cannot be checked
    /// against, because a belief that decides nothing cannot be learned from
    /// the texts and an author whose beliefs leave most situations open has no
    /// decisions to grade.
    ///
    /// # Errors
    /// The rule it fails, with the belief or the share that fails it.
    pub fn checkability(&self, pool: &[Scenario]) -> Result<(), Uncheckable> {
        let settled = pool.iter().filter(|s| self.decide(s).determined).count();
        if settled * 100 < pool.len() * MIN_DETERMINED_PERCENT {
            return Err(Uncheckable::Settles {
                settled,
                of: pool.len(),
            });
        }
        for (i, belief) in self.beliefs.iter().enumerate() {
            let flipped = self.with_belief_flipped(i);
            let decides = pool.iter().any(|s| {
                let (a, b) = (self.decide(s), flipped.decide(s));
                a.determined && b.determined && a.choice != b.choice
            });
            if !decides {
                return Err(Uncheckable::Idle { belief: *belief });
            }
        }
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Every situation the six facts make.
    fn every_situation() -> impl Iterator<Item = Features> {
        (0..64u32).map(|bits| Features {
            exceeds_authority: bits & 1 != 0,
            urgent: bits & 2 != 0,
            binds_future: bits & 4 != 0,
            irreversible: bits & 8 != 0,
            members_harmed: bits & 16 != 0,
            outsider_offers_help: bits & 32 != 0,
        })
    }

    /// The rules say everything the table says and nothing it does not: in
    /// every situation, the rules that hold for an option are exactly its
    /// verdict.
    #[test]
    fn the_rules_of_a_belief_reproduce_its_verdicts_in_every_situation() {
        for tenet in TENETS {
            for polarity in [Polarity::Strict, Polarity::Lax] {
                let belief = Belief { tenet, polarity };
                let rules = belief.rules();
                for features in every_situation() {
                    for option in OPTIONS {
                        let said: Vec<Verdict> = rules
                            .iter()
                            .filter(|r| r.option == option && r.holds(&features))
                            .map(|r| r.verdict)
                            .collect();
                        match belief.verdict(&features, option) {
                            Verdict::Neutral => assert!(said.is_empty(), "{belief:?} {option:?}"),
                            v => assert_eq!(said, vec![v], "{belief:?} {option:?}"),
                        }
                    }
                }
            }
        }
    }

    /// The fact a tenet names is the one its verdicts turn on: a belief judges
    /// nothing where its fact does not hold, except that the option
    /// self-reliance judges is only listed where an outsider offers help.
    #[test]
    fn a_belief_gives_a_verdict_only_where_the_fact_its_tenet_names_holds() {
        use Archetype::{ActAlone, ConsultThenAct, DeferToOutsider, HoldAndWait};
        for tenet in TENETS {
            for polarity in [Polarity::Strict, Polarity::Lax] {
                let belief = Belief { tenet, polarity };
                let mut judges = false;
                for features in every_situation() {
                    for option in [ActAlone, ConsultThenAct, DeferToOutsider, HoldAndWait] {
                        if belief.verdict(&features, option) == Verdict::Neutral {
                            continue;
                        }
                        judges = true;
                        let offered = tenet == Tenet::SelfReliance && option == DeferToOutsider;
                        assert!(
                            tenet.fact().holds(&features) || offered,
                            "{belief:?} judges {option:?} where {:?} does not hold",
                            tenet.fact()
                        );
                    }
                }
                assert!(judges, "{belief:?} judges nothing");
            }
        }
    }
}
