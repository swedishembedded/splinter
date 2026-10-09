// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements leakage-free role assignment for evaluation
// facts, for its clients. If your team needs expertise in designing
// measurements a learning system cannot game, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Which role a fact plays, decided by its family before anything is screened.
//!
//! A family (the letter a fact was read from) is the unit: a stable hash of
//! the seed and the family's name puts it in one role's share, so facts of one
//! family are never split across roles and the assignment depends on nothing
//! the screening finds. The screening then decides which candidates fill a
//! role's quota, in a fixed order.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// What a fact is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Role {
    /// Used to develop and tune the pipeline; may be looked at freely.
    Dev,
    /// Used once, frozen, taught on one day each.
    Test,
    /// Wrong at day 0 and never discussed in any session.
    ControlUntaught,
    /// Right at day 0 and never discussed in any session: measures forgetting.
    ControlKnown,
    /// Right at day 0, never discussed, about an entity a test fact is about:
    /// measures damage to what lies next to a taught fact.
    NeighbourKnown,
}

impl Role {
    /// Every role, in the order quotas are filled.
    pub const ALL: [Role; 5] = [
        Role::Dev,
        Role::Test,
        Role::ControlUntaught,
        Role::ControlKnown,
        Role::NeighbourKnown,
    ];

    /// The role's name as the manifest writes it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Role::Dev => "dev",
            Role::Test => "test",
            Role::ControlUntaught => "control-untaught",
            Role::ControlKnown => "control-known",
            Role::NeighbourKnown => "neighbour-known",
        }
    }

    /// Whether facts of this role are chosen among those the policy answers
    /// right every time (otherwise among those it answers wrong every time).
    #[must_use]
    pub fn wants_known(self) -> bool {
        matches!(self, Role::ControlKnown | Role::NeighbourKnown)
    }
}

/// How many facts each role holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Quotas {
    /// Development facts.
    pub dev: usize,
    /// Test facts.
    pub test: usize,
    /// Controls wrong at day 0.
    pub control_untaught: usize,
    /// Controls right at day 0.
    pub control_known: usize,
    /// Facts right at day 0 about an entity a test fact is about.
    pub neighbour_known: usize,
    /// Questions about things that do not exist, which must be declined.
    pub hallucination: usize,
}

impl Quotas {
    /// The protocol's quotas: 20 development facts, 40 test facts (8 days of
    /// 5), 20 controls of each kind, 80 neighbours and 50 unknowns.
    pub const PROTOCOL: Quotas = Quotas {
        dev: 20,
        test: 40,
        control_untaught: 20,
        control_known: 20,
        neighbour_known: 80,
        hallucination: 50,
    };

    /// The quota of `role`.
    #[must_use]
    pub fn of(&self, role: Role) -> usize {
        match role {
            Role::Dev => self.dev,
            Role::Test => self.test,
            Role::ControlUntaught => self.control_untaught,
            Role::ControlKnown => self.control_known,
            Role::NeighbourKnown => self.neighbour_known,
        }
    }
}

/// Test facts taught on one day.
pub const TEST_FACTS_PER_DAY: usize = 5;

/// `seed` and `key` hashed to a number: the stable order and share of both
/// families and facts.
fn hashed(seed: u64, key: &str) -> u64 {
    let mut input = seed.to_le_bytes().to_vec();
    input.extend_from_slice(key.as_bytes());
    let digest = blake3::hash(&input);
    let mut first = [0u8; 8];
    first.copy_from_slice(&digest.as_bytes()[..8]);
    u64::from_le_bytes(first)
}

/// The role `family` is a candidate for: its hash falls in the share of the
/// role whose quota it is, shares being proportional to `quotas`.
#[must_use]
pub fn candidate_role(seed: u64, family: &str, quotas: &Quotas) -> Role {
    let total: usize = Role::ALL.iter().map(|r| quotas.of(*r)).sum();
    let mut point = (hashed(seed, family) % total.max(1) as u64) as usize;
    for role in Role::ALL {
        if point < quotas.of(role) {
            return role;
        }
        point -= quotas.of(role);
    }
    Role::Dev
}

/// How a screened fact came out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Class {
    /// No answer of the six passed both checks.
    ConsistentlyWrong,
    /// Every answer of the six passed both checks.
    Known,
    /// Anything between: no role.
    Discarded,
}

/// One screened candidate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Screened {
    /// The fact's id.
    pub id: String,
    /// Its family: at most one fact of a family is placed.
    pub family: String,
    /// The lower-case names the fact is about, for finding neighbours.
    pub entities: Vec<String>,
    /// The role its family is a candidate for.
    pub candidate: Role,
    /// How it came out.
    pub class: Class,
}

/// A fact that filled a place in a role.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placed {
    /// The fact's id.
    pub id: String,
    /// Its role.
    pub role: Role,
    /// For a test fact, the day it is taught on, counted from 1.
    pub day: Option<usize>,
}

/// The roles whose quota the candidates could not fill, each with the quota
/// and the number of facts that qualified.
#[derive(Debug, PartialEq, Eq)]
pub struct Shortfalls(pub Vec<(Role, usize, usize)>);

impl std::fmt::Display for Shortfalls {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "the fact pool is too small:")?;
        for (role, wanted, found) in &self.0 {
            write!(
                f,
                " {} needs {wanted} but only {found} qualify;",
                role.name()
            )?;
        }
        Ok(())
    }
}

impl std::error::Error for Shortfalls {}

/// Fills each role's quota from the candidates whose family is a candidate
/// for it and whose class fits it, in the order of a hash of the fact's id
/// (so that neither the order the facts are given in nor the other facts
/// present change which fill it). Fails, naming every role it could not
/// fill, when the pool is too small.
pub fn select(
    seed: u64,
    quotas: &Quotas,
    screened: &[Screened],
) -> Result<Vec<Placed>, Shortfalls> {
    let mut placed: Vec<Placed> = Vec::new();
    let mut short = Vec::new();
    let mut used_families: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for role in Role::ALL {
        let wanted = quotas.of(role);
        let fits = if role.wants_known() {
            Class::Known
        } else {
            Class::ConsistentlyWrong
        };
        // A neighbour is about an entity some test fact is about.
        let test_entities: std::collections::HashSet<&str> = screened
            .iter()
            .filter(|s| placed.iter().any(|p| p.role == Role::Test && p.id == s.id))
            .flat_map(|s| s.entities.iter().map(String::as_str))
            .collect();
        let mut eligible: Vec<&Screened> = screened
            .iter()
            .filter(|s| s.candidate == role && s.class == fits)
            .filter(|s| {
                role != Role::NeighbourKnown
                    || s.entities
                        .iter()
                        .any(|e| test_entities.contains(e.as_str()))
            })
            .collect();
        eligible.sort_by_key(|s| (hashed(seed, &s.id), s.id.clone()));
        let mut taken = Vec::new();
        for fact in eligible {
            if taken.len() < wanted && used_families.insert(fact.family.as_str()) {
                taken.push(fact);
            }
        }
        if taken.len() < wanted {
            short.push((role, wanted, taken.len()));
        }
        for (n, fact) in taken.into_iter().enumerate() {
            placed.push(Placed {
                id: fact.id.clone(),
                role,
                day: (role == Role::Test).then_some(1 + n / TEST_FACTS_PER_DAY),
            });
        }
    }
    if short.is_empty() {
        Ok(placed)
    } else {
        Err(Shortfalls(short))
    }
}

/// How many facts each role holds in `placed`.
#[must_use]
pub fn counts(placed: &[Placed]) -> BTreeMap<Role, usize> {
    let mut counts = BTreeMap::new();
    for p in placed {
        *counts.entry(p.role).or_insert(0) += 1;
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    fn screened(id: &str, candidate: Role, class: Class) -> Screened {
        Screened {
            id: id.into(),
            family: id.into(),
            entities: vec![],
            candidate,
            class,
        }
    }

    #[test]
    fn a_family_has_one_role_that_depends_on_nothing_but_its_name_and_the_seed() {
        let q = Quotas::PROTOCOL;
        let first = candidate_role(7, "washington-v3-120.txt", &q);
        assert_eq!(first, candidate_role(7, "washington-v3-120.txt", &q));
        let mut tally = BTreeMap::new();
        for n in 0..4000 {
            *tally
                .entry(candidate_role(7, &format!("family-{n}"), &q))
                .or_insert(0usize) += 1;
        }
        // Shares follow the quotas (20:40:20:20:80) within sampling noise.
        let share = |r: Role| tally[&r] as f64 / 4000.0;
        assert!((share(Role::Test) - 40.0 / 180.0).abs() < 0.03, "{tally:?}");
        assert!((share(Role::Dev) - 20.0 / 180.0).abs() < 0.03, "{tally:?}");
        assert!(
            (share(Role::NeighbourKnown) - 80.0 / 180.0).abs() < 0.03,
            "{tally:?}"
        );
    }

    #[test]
    fn quotas_are_filled_in_a_fixed_order_whatever_order_the_facts_arrive_in() {
        let quotas = Quotas {
            dev: 1,
            test: 6,
            control_untaught: 1,
            control_known: 1,
            neighbour_known: 1,
            hallucination: 0,
        };
        let mut pool: Vec<Screened> = (0..12)
            .map(|n| screened(&format!("t{n}"), Role::Test, Class::ConsistentlyWrong))
            .collect();
        pool.push(screened("d", Role::Dev, Class::ConsistentlyWrong));
        pool.push(screened(
            "u",
            Role::ControlUntaught,
            Class::ConsistentlyWrong,
        ));
        pool.push(screened("k", Role::ControlKnown, Class::Known));
        // A neighbour shares an entity with a placed test fact; one that does not is no neighbour.
        let mut near = screened("near", Role::NeighbourKnown, Class::Known);
        near.entities = vec!["madison".into()];
        let mut far = screened("far", Role::NeighbourKnown, Class::Known);
        far.entities = vec!["monroe".into()];
        pool.push(near);
        pool.push(far);
        for t in pool.iter_mut().filter(|s| s.candidate == Role::Test) {
            t.entities = vec!["madison".into()];
        }
        // Two facts of one family fill one place at most.
        let mut twin = screened("t-twin", Role::Test, Class::ConsistentlyWrong);
        twin.family = "t0".into();
        pool.push(twin);
        // Fits no role: a known fact is no test fact, a mixed one no fact at all.
        pool.push(screened("known-test", Role::Test, Class::Known));
        pool.push(screened("mixed", Role::Dev, Class::Discarded));
        let placed = select(3, &quotas, &pool).expect("enough facts");
        let mut reversed = pool.clone();
        reversed.reverse();
        assert_eq!(placed, select(3, &quotas, &reversed).expect("enough facts"));
        let counts = counts(&placed);
        assert_eq!(counts[&Role::Test], 6);
        assert_eq!(counts[&Role::NeighbourKnown], 1);
        assert!(
            placed.iter().any(|p| p.id == "near")
                && placed.iter().all(|p| p.id != "far" && p.id != "t-twin")
        );
        assert!(placed
            .iter()
            .all(|p| p.id != "known-test" && p.id != "mixed"));
        // Five test facts a day, then the sixth starts the next.
        let days: Vec<usize> = placed.iter().filter_map(|p| p.day).collect();
        assert_eq!(days, vec![1, 1, 1, 1, 1, 2]);
    }

    #[test]
    fn a_pool_too_small_fails_naming_every_role_it_cannot_fill() {
        let quotas = Quotas {
            dev: 2,
            test: 1,
            control_untaught: 1,
            control_known: 1,
            neighbour_known: 0,
            hallucination: 0,
        };
        let pool = [screened("d", Role::Dev, Class::ConsistentlyWrong)];
        let error = select(1, &quotas, &pool).expect_err("too small");
        assert_eq!(
            error.0,
            vec![
                (Role::Dev, 2, 1),
                (Role::Test, 1, 0),
                (Role::ControlUntaught, 1, 0),
                (Role::ControlKnown, 1, 0)
            ]
        );
        assert!(error.to_string().contains("test needs 1 but only 0"));
    }
}
