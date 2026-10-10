// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measured validation of continual learning
// from a user's own agent sessions for its clients. If your team needs
// expertise in proving that a model absorbed what its user taught it without
// forgetting what it knew, you can procure our services by sending an email
// to info@swedishembedded.com.

//! What the nights' evaluations say, per night: the facts by group (taught
//! today, taught earlier, not yet taught, the canary, the controls), each as
//! k of n facts per probe kind; the other suites as k of n items; and the
//! items that went from right to wrong since the night before.
//!
//! The unit is the fact: a fact passes a kind when the probe of that kind is
//! right, and passes direct recall when it passes every primary kind.

use std::collections::{BTreeMap, HashMap};

use anyhow::Context as _;
use serde::Serialize;

use super::eval::{Evaluation, LogProb};
use super::items::{Class, Item};
use super::plan::{CycleRole, Plan, Schedule};
use super::stats::Proportion;
use crate::keys::Key;
use crate::probes::ProbeKind;
use crate::roles::Role;

/// What the analysis needs to know of a fact.
#[derive(Clone, Debug)]
pub struct FactInfo {
    /// The fact's id.
    pub id: String,
    /// Its role.
    pub role: Role,
    /// The keys of the canary statement, for the canary fact.
    pub canary_keys: Vec<Key>,
}

/// Where a fact stands on a night.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Group {
    /// Corrected on this night.
    Today,
    /// Corrected on an earlier night.
    Earlier,
    /// Not corrected yet (or never, in this cycle).
    Future,
    /// The canary, between its false assertion and its correction.
    Canary,
    /// Wrong at day 0, never discussed.
    ControlUntaught,
    /// Right at day 0, never discussed.
    ControlKnown,
    /// Right at day 0, about a name a test fact is about.
    NeighbourKnown,
}

impl Group {
    /// The name the report uses.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Group::Today => "today",
            Group::Earlier => "earlier",
            Group::Future => "future",
            Group::Canary => "canary",
            Group::ControlUntaught => "control-untaught",
            Group::ControlKnown => "control-known",
            Group::NeighbourKnown => "neighbour-known",
        }
    }
}

/// The group of a taught-role fact on `night`.
#[must_use]
pub fn group_of(schedule: Option<&Schedule>, night: usize) -> Group {
    let Some(s) = schedule else {
        return Group::Future;
    };
    match (s.canary, s.taught) {
        (Some(c), _) if night < c => Group::Future,
        (Some(_), Some(t)) if night < t => Group::Canary,
        (Some(_), None) => Group::Canary,
        (_, Some(t)) if night < t => Group::Future,
        (_, Some(t)) if night == t => Group::Today,
        (_, Some(_)) => Group::Earlier,
        (None, None) => Group::Future,
    }
}

/// A group's facts, counted.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GroupStats {
    /// Facts in the group.
    pub facts: usize,
    /// Facts right, per kind (`question` for a control, which has no probes).
    pub by_kind: BTreeMap<String, Proportion>,
    /// Facts right on every primary kind; absent for an empty group.
    pub direct: Option<Proportion>,
}

/// One night, summarised.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NightSummary {
    /// The night; 0 is the base.
    pub night: usize,
    /// The release in effect after it.
    pub release: Option<String>,
    /// The facts by group name.
    pub groups: BTreeMap<String, GroupStats>,
    /// Sycophancy probes resisted.
    pub sycophancy: Proportion,
    /// Unknown questions declined.
    pub unknown: Proportion,
    /// Anchor items right.
    pub anchor: Proportion,
    /// Fresh general questions right.
    pub general: Proportion,
    /// Answers the judge abstained on (counted as not right).
    pub abstained: usize,
    /// The mean log-probability against the base; absent when not measured.
    pub logprob: Option<LogProb>,
}

/// Items that changed from right to wrong, and how many the other way.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Flips {
    /// Ids of the items right before and wrong now.
    pub right_to_wrong: Vec<String>,
    /// How many were wrong before and are right now.
    pub wrong_to_right: usize,
}

/// The transitions from the night before.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Transitions {
    /// The night.
    pub night: usize,
    /// By class (`probe`, `control-known`, `sycophancy`, ...).
    pub by_class: BTreeMap<String, Flips>,
}

/// The class transitions are counted by.
#[must_use]
pub fn class_name(item: &Item) -> String {
    match (item.class, item.role) {
        (Class::Control, Some(role)) => role.name().to_string(),
        (Class::Probe, _) => "probe".into(),
        (Class::Sycophancy, _) => "sycophancy".into(),
        (Class::Unknown, _) => "unknown".into(),
        (Class::Anchor, _) => "anchor".into(),
        (Class::General, _) => "general".into(),
        (Class::Control, None) => "control".into(),
    }
}

/// A night's results indexed for lookup.
pub struct Night<'a> {
    /// The evaluation.
    pub evaluation: &'a Evaluation,
    pass: HashMap<&'a str, bool>,
}

impl<'a> Night<'a> {
    /// Indexes `evaluation`, checking it answers exactly `items`, in order.
    ///
    /// # Errors
    /// The evaluation's results are not the items'.
    pub fn new(evaluation: &'a Evaluation, items: &[Item]) -> anyhow::Result<Self> {
        anyhow::ensure!(
            evaluation.results.len() == items.len()
                && evaluation
                    .results
                    .iter()
                    .zip(items)
                    .all(|(r, i)| r.id == i.id),
            "night {} does not evaluate the items of the pre-registration",
            evaluation.night
        );
        Ok(Self {
            evaluation,
            pass: evaluation
                .results
                .iter()
                .map(|r| (r.id.as_str(), r.pass))
                .collect(),
        })
    }

    /// Whether item `id` was right.
    #[must_use]
    pub fn passes(&self, id: &str) -> bool {
        self.pass.get(id).copied().unwrap_or(false)
    }
}

/// What the tables are computed from.
pub struct Inputs<'a> {
    /// Which facts the cycle teaches.
    pub role: CycleRole,
    /// The kinds a fact must pass to pass direct recall.
    pub primary_kinds: &'a [ProbeKind],
    /// The facts of the manifest.
    pub facts: &'a [FactInfo],
    /// The plan.
    pub plan: &'a Plan,
    /// The items.
    pub items: &'a [Item],
}

fn kind_name(kind: ProbeKind) -> String {
    format!("{kind:?}").to_lowercase()
}

/// The group a fact is in on `night`, for any role.
fn fact_group(
    inputs: &Inputs<'_>,
    schedule: &BTreeMap<String, Schedule>,
    fact: &FactInfo,
    night: usize,
) -> Option<Group> {
    match fact.role {
        r if r == inputs.role.fact_role() => Some(group_of(schedule.get(&fact.id), night)),
        Role::ControlUntaught => Some(Group::ControlUntaught),
        Role::ControlKnown => Some(Group::ControlKnown),
        Role::NeighbourKnown => Some(Group::NeighbourKnown),
        _ => None,
    }
}

/// Counts the facts of `members` that pass `kind` on `night`.
fn count_kind(
    night: &Night<'_>,
    inputs: &Inputs<'_>,
    members: &[&FactInfo],
    kind: Option<ProbeKind>,
) -> usize {
    members
        .iter()
        .filter(|f| night.passes(&item_id(inputs, f, kind)))
        .count()
}

/// The id of the item that measures `kind` of `fact`.
fn item_id(inputs: &Inputs<'_>, fact: &FactInfo, kind: Option<ProbeKind>) -> String {
    match (fact.role == inputs.role.fact_role(), kind) {
        (true, Some(kind)) => format!("probe:{}:{kind:?}", fact.id).to_lowercase(),
        _ => format!("control:{}", fact.id),
    }
}

/// Summarises `night`.
#[must_use]
pub fn summarise(inputs: &Inputs<'_>, night: &Night<'_>) -> NightSummary {
    let schedule = inputs.plan.schedule();
    let n = night.evaluation.night;
    let mut members: BTreeMap<Group, Vec<&FactInfo>> = BTreeMap::new();
    for fact in inputs.facts {
        if let Some(group) = fact_group(inputs, &schedule, fact, n) {
            members.entry(group).or_default().push(fact);
        }
    }
    let mut groups = BTreeMap::new();
    for (group, facts) in &members {
        let taught = matches!(
            group,
            Group::Today | Group::Earlier | Group::Future | Group::Canary
        );
        let mut by_kind = BTreeMap::new();
        let direct = if taught {
            for kind in ProbeKind::ALL {
                by_kind.insert(
                    kind_name(kind),
                    Proportion::new(count_kind(night, inputs, facts, Some(kind)), facts.len()),
                );
            }
            let k = facts
                .iter()
                .filter(|f| {
                    inputs
                        .primary_kinds
                        .iter()
                        .all(|kind| night.passes(&item_id(inputs, f, Some(*kind))))
                })
                .count();
            Proportion::new(k, facts.len())
        } else {
            let p = Proportion::new(count_kind(night, inputs, facts, None), facts.len());
            by_kind.insert("question".to_string(), p);
            p
        };
        groups.insert(
            group.name().to_string(),
            GroupStats {
                facts: facts.len(),
                by_kind,
                direct: (!facts.is_empty()).then_some(direct),
            },
        );
    }
    let suite = |class: Class| {
        let items: Vec<&Item> = inputs.items.iter().filter(|i| i.class == class).collect();
        Proportion::new(
            items.iter().filter(|i| night.passes(&i.id)).count(),
            items.len(),
        )
    };
    NightSummary {
        night: n,
        release: night.evaluation.release.clone(),
        groups,
        sycophancy: suite(Class::Sycophancy),
        unknown: suite(Class::Unknown),
        anchor: suite(Class::Anchor),
        general: suite(Class::General),
        abstained: night
            .evaluation
            .results
            .iter()
            .filter(|r| r.abstained())
            .count(),
        logprob: night.evaluation.logprob.clone(),
    }
}

/// The items of each class that changed between `before` and `after`.
#[must_use]
pub fn flips(items: &[Item], before: &Night<'_>, after: &Night<'_>) -> BTreeMap<String, Flips> {
    let mut by_class: BTreeMap<String, Flips> = BTreeMap::new();
    for item in items {
        let entry = by_class.entry(class_name(item)).or_insert(Flips {
            right_to_wrong: Vec::new(),
            wrong_to_right: 0,
        });
        match (before.passes(&item.id), after.passes(&item.id)) {
            (true, false) => entry.right_to_wrong.push(item.id.clone()),
            (false, true) => entry.wrong_to_right += 1,
            _ => {}
        }
    }
    by_class
}

/// Summaries of every night and the transitions from each to the next.
///
/// # Errors
/// An evaluation does not answer the pre-registered items, or the nights are
/// not numbered from 0 in order.
pub fn tables(
    inputs: &Inputs<'_>,
    evaluations: &[Evaluation],
) -> anyhow::Result<(Vec<NightSummary>, Vec<Transitions>)> {
    let nights: Vec<Night<'_>> = evaluations
        .iter()
        .enumerate()
        .map(|(at, e)| {
            anyhow::ensure!(e.night == at, "evaluation {at} is of night {}", e.night);
            Night::new(e, inputs.items)
        })
        .collect::<anyhow::Result<_>>()
        .context("the evaluations")?;
    let summaries = nights.iter().map(|n| summarise(inputs, n)).collect();
    let transitions = nights
        .windows(2)
        .map(|pair| Transitions {
            night: pair[1].evaluation.night,
            by_class: flips(inputs.items, &pair[0], &pair[1]),
        })
        .collect();
    Ok((summaries, transitions))
}
