// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the loop from a failed agent attempt to a
// verified corrected one, for its clients. If your team needs expertise in
// deciding which correction a failed agent attempt gets, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Which interventions a diagnosed failure goes to, in which order, and how
//! many attempts one failure may cost.
//!
//! A failure is routed by its class to a chain of interventions, cheapest
//! or most specific first; the run tries them in order and stops at the
//! first the verifier accepts or when the failure's budget of attempts is
//! spent. A failure the infrastructure or the task decided, one no evidence
//! attributes, and one that read withheld material are routed nowhere: the
//! table cannot name them ([`Routing::new`] refuses it).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use splinter_core::diagnosis::DiagnosticResult;
use splinter_core::outcome::{Attribution, FailureClass};

/// Attempts one failure may cost when the request does not say: a cheap
/// first try, one escalation and one more.
pub const DEFAULT_BUDGET: usize = 3;

/// The ways a failed attempt can be turned into a verified one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterventionKind {
    /// The student's own model samples again from the last good step, under
    /// another seed.
    Retry,
    /// A teacher continues from the last good step.
    ForkCorrection,
    /// A teacher continues from the last good step with task-relevant
    /// public reference material in front of it.
    KnowledgeCorrection,
    /// A proposer splits the task into ordered subgoals with checkable
    /// intermediate states, and the student works them one by one from the
    /// last good step.
    Decompose,
}

impl InterventionKind {
    /// Every kind, in declaration order.
    pub const ALL: [Self; 4] = [
        Self::Retry,
        Self::ForkCorrection,
        Self::KnowledgeCorrection,
        Self::Decompose,
    ];

    /// The name records and keys carry.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Retry => "retry",
            Self::ForkCorrection => "fork_correction",
            Self::KnowledgeCorrection => "knowledge_correction",
            Self::Decompose => "decompose",
        }
    }
}

impl fmt::Display for InterventionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for InterventionKind {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, String> {
        Self::ALL
            .into_iter()
            .find(|kind| kind.as_str() == text)
            .ok_or_else(|| {
                let names: Vec<&str> = Self::ALL.iter().map(|k| k.as_str()).collect();
                format!("{text:?} is no intervention; one of: {}", names.join(", "))
            })
    }
}

/// Where a diagnosed failure goes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "route", content = "why", rename_all = "snake_case")]
pub enum Route {
    /// To these interventions, tried in order.
    To(Vec<InterventionKind>),
    /// Not repaired.
    Skip(&'static str),
}

/// The routing table: a chain of interventions per failure class, and the
/// attempts one failure may cost.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Routing {
    /// The chain of each routed class; a class not named is not repaired.
    pub chains: BTreeMap<FailureClass, Vec<InterventionKind>>,
    /// Attempts one failure may cost, every intervention and sample counted.
    pub budget: usize,
}

impl Routing {
    /// The table of `chains` under `budget`.
    ///
    /// # Errors
    /// When a chain is named for a class an intervention never repairs (an
    /// infrastructure or task class, an unattributed one, or history
    /// probing), a chain is empty, or the budget is zero.
    pub fn new(
        chains: BTreeMap<FailureClass, Vec<InterventionKind>>,
        budget: usize,
    ) -> Result<Self, String> {
        if budget == 0 {
            return Err("a routing budget of zero repairs nothing".into());
        }
        for (class, chain) in &chains {
            if let Some(why) = never_routed(*class) {
                return Err(format!("{} cannot be routed: {why}", class.as_str()));
            }
            if chain.is_empty() {
                return Err(format!("{} is routed to no intervention", class.as_str()));
            }
        }
        Ok(Self { chains, budget })
    }

    /// The standard table under `budget`. Retry first where a sample of the
    /// same policy may well pass (a malformed call or artifact, a loop);
    /// decomposition first where the plan or its pacing failed (no answer,
    /// out of time or steps, a strategy or planning class); knowledge first
    /// where the attempt lacked information; a teacher's correction first
    /// where the work was wrong and as the fallback of every chain.
    ///
    /// # Errors
    /// When `budget` is zero.
    pub fn standard(budget: usize) -> Result<Self, String> {
        use FailureClass as C;
        use InterventionKind as I;
        let chains = [
            (C::Tool, vec![I::Retry, I::ForkCorrection]),
            (C::MalformedArtifact, vec![I::Retry, I::ForkCorrection]),
            (
                C::Repetition,
                vec![I::Retry, I::Decompose, I::ForkCorrection],
            ),
            (C::NoAnswer, vec![I::Decompose, I::ForkCorrection]),
            (C::Resource, vec![I::Decompose, I::ForkCorrection]),
            (C::Planning, vec![I::Decompose, I::ForkCorrection]),
            (C::Strategy, vec![I::Decompose, I::ForkCorrection]),
            (
                C::InformationMissing,
                vec![I::KnowledgeCorrection, I::ForkCorrection],
            ),
            (
                C::Retrieval,
                vec![I::KnowledgeCorrection, I::ForkCorrection],
            ),
            (
                C::WrongContent,
                vec![I::ForkCorrection, I::KnowledgeCorrection, I::Decompose],
            ),
            (
                C::Implementation,
                vec![I::ForkCorrection, I::KnowledgeCorrection, I::Decompose],
            ),
            (
                C::ModelCompetence,
                vec![I::ForkCorrection, I::KnowledgeCorrection, I::Decompose],
            ),
        ];
        Self::new(chains.into_iter().collect(), budget)
    }

    /// The route of `diagnosis` among the `available` interventions: its
    /// class's chain without the ones the run cannot make.
    #[must_use]
    pub fn route(
        &self,
        diagnosis: &DiagnosticResult,
        available: &BTreeSet<InterventionKind>,
    ) -> Route {
        if let Some(why) = never_routed(diagnosis.class) {
            return Route::Skip(why);
        }
        if diagnosis.last_good_step().is_none() {
            return Route::Skip("the diagnosis places no decisive step after a good one");
        }
        let Some(chain) = self.chains.get(&diagnosis.class) else {
            return Route::Skip("the routing table names no intervention for the class");
        };
        let chain: Vec<InterventionKind> = chain
            .iter()
            .copied()
            .filter(|kind| available.contains(kind))
            .collect();
        if chain.is_empty() {
            return Route::Skip("no intervention of the class's chain is available to the run");
        }
        Route::To(chain)
    }
}

/// Why `class` is never routed, or `None` when it may be.
fn never_routed(class: FailureClass) -> Option<&'static str> {
    match class.attribution() {
        Some(Attribution::Infrastructure) | Some(Attribution::Task) => {
            Some("not a model failure: the infrastructure or the task decided it")
        }
        None => Some("no evidence attributes the failure"),
        Some(Attribution::Model) if class == FailureClass::HistoryProbing => {
            Some("the attempt read material it must never have seen; its prefix is not replayed")
        }
        Some(Attribution::Model) => None,
    }
}

/// The route of `diagnosis` under the standard table with every
/// intervention available.
#[must_use]
pub fn route(diagnosis: &DiagnosticResult) -> Route {
    let all = InterventionKind::ALL.into_iter().collect();
    Routing::standard(DEFAULT_BUDGET)
        .map(|routing| routing.route(diagnosis, &all))
        .unwrap_or(Route::Skip("the standard routing table is invalid"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_never_names_a_class_no_intervention_may_repair() {
        for class in FailureClass::ALL {
            let table = Routing::new(
                BTreeMap::from([(class, vec![InterventionKind::ForkCorrection])]),
                1,
            );
            let model = class.attribution() == Some(Attribution::Model)
                && class != FailureClass::HistoryProbing;
            assert_eq!(table.is_ok(), model, "{class:?}");
        }
        assert!(Routing::standard(0).is_err());
        let standard = Routing::standard(DEFAULT_BUDGET).unwrap();
        assert!(standard.chains.keys().all(|c| never_routed(*c).is_none()));
    }

    #[test]
    fn a_route_keeps_the_chain_order_of_the_interventions_available() {
        let diagnosis = DiagnosticResult::new(FailureClass::WrongContent, "r", Some(4), Vec::new());
        let routing = Routing::standard(DEFAULT_BUDGET).unwrap();
        let only_teacher = BTreeSet::from([InterventionKind::ForkCorrection]);
        assert_eq!(
            routing.route(&diagnosis, &only_teacher),
            Route::To(vec![InterventionKind::ForkCorrection])
        );
        let all = InterventionKind::ALL.into_iter().collect();
        assert_eq!(
            routing.route(&diagnosis, &all),
            Route::To(vec![
                InterventionKind::ForkCorrection,
                InterventionKind::KnowledgeCorrection,
                InterventionKind::Decompose
            ])
        );
        let none = BTreeSet::from([InterventionKind::Retry]);
        assert!(matches!(routing.route(&diagnosis, &none), Route::Skip(_)));
        for kind in InterventionKind::ALL {
            assert_eq!(kind.as_str().parse::<InterventionKind>().unwrap(), kind);
        }
    }
}
