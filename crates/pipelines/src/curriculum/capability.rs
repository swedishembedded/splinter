// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements curricula that spend a learner's practice
// where its measured capability gaps are, for its clients. If your team needs
// expertise in adaptive curricula for agent training, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Competence per capability.
//!
//! The examiner labels every task with the capability ids of its taxonomy;
//! they reach this module as data (the practice manifest's and the export's
//! labels), never as a list it knows. A task the examiner labelled with none
//! counts under [`UNLABELLED`].
//!
//! **Competence.** Each capability counts the attempts at its tasks: graded
//! ones (a pass or a fail the verifier decided) and ungraded ones (the
//! infrastructure or the task stopped it). Only graded attempts enter the pass
//! rate, which carries its 95% Wilson interval; an ungraded attempt is never a
//! `0`. Every failed attempt's diagnosed class is counted, and so is how many
//! of them were the model's own (a repairable failure: wrong content, no
//! answer, a tool misused, ...). Each task's band ([`Band`]) comes from the
//! campaign's per-task pass rates.
//!
//! **Weight.** A capability's weight is how much practising it should teach:
//!
//! `weight = (1 - r) * (1/2 + f/2) * (1/2 + m/2)`
//!
//! where `r = (passes + 1) / (graded + 2)` is its smoothed pass rate (one half
//! when nothing was graded), `f` the share of its measured tasks in the
//! frontier band (solved some of the time; one when none is measured yet), and
//! `m` the share of its graded failures that were the model's own (one with no
//! failure). A capability the policy fails at, on tasks it solves sometimes,
//! for reasons a model can learn, weighs most; infrastructure failures weigh
//! nothing, because they never enter `r`, `f` or `m`.
//!
//! **Transfer.** The heaviest measured capabilities are the ones fresh
//! exercises are requested for, from the examiner's practice generators
//! (`bench task practice --capability`).

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use splinter_core::outcome::FailureClass;
use splinter_eval::campaign::pass_rate::{BandCounts, PassRates};
use splinter_eval::pilot::Share;

/// The capability a task the examiner labelled with none counts under.
pub const UNLABELLED: &str = "unlabelled";

/// One attempt at a task, as competence counts it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Attempt {
    /// The task, by the name the practice material stages it under.
    pub task: String,
    /// The capability ids its task is labelled with; empty for none.
    pub capabilities: Vec<String>,
    /// Whether it passed, when the verifier graded it; `None` when the
    /// infrastructure or the task stopped it.
    pub passed: Option<bool>,
    /// Why it failed, when it failed and was diagnosed.
    pub class: Option<FailureClass>,
}

impl Attempt {
    /// The capabilities it counts under: its labels, else [`UNLABELLED`].
    fn counted_under(&self) -> Vec<&str> {
        if self.capabilities.is_empty() {
            vec![UNLABELLED]
        } else {
            self.capabilities.iter().map(String::as_str).collect()
        }
    }
}

/// One capability's measured competence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CapabilityCompetence {
    /// The capability id, as the examiner names it.
    pub capability: String,
    /// Distinct tasks attempted.
    pub tasks: usize,
    /// Attempts counted.
    pub attempts: usize,
    /// Of those, graded by the verifier.
    pub graded: usize,
    /// Of the graded, passes.
    pub passes: usize,
    /// `passes / graded` with its 95% Wilson interval; `None` with nothing
    /// graded.
    pub pass_rate: Option<Share>,
    /// Attempts the infrastructure or the task stopped: in no denominator.
    pub ungraded: usize,
    /// Its tasks by band.
    pub bands: BandCounts,
    /// Failed attempts by diagnosed class (`unclassified` when no diagnosis
    /// was given).
    pub failures: BTreeMap<String, usize>,
    /// Graded failures the diagnosis attributed to the model.
    pub model_failures: usize,
    /// How much practising it should teach (module documentation).
    pub weight: f64,
}

impl CapabilityCompetence {
    fn new(capability: &str) -> Self {
        Self {
            capability: capability.to_string(),
            tasks: 0,
            attempts: 0,
            graded: 0,
            passes: 0,
            pass_rate: None,
            ungraded: 0,
            bands: BandCounts::default(),
            failures: BTreeMap::new(),
            model_failures: 0,
            weight: 0.0,
        }
    }

    /// The weight of the module documentation, from the counts.
    fn weigh(&self) -> f64 {
        let rate = (self.passes as f64 + 1.0) / (self.graded as f64 + 2.0);
        let measured = self.bands.frontier + self.bands.cliff + self.bands.retired;
        let frontier = if measured == 0 {
            1.0
        } else {
            self.bands.frontier as f64 / measured as f64
        };
        let failed = self.graded - self.passes;
        let model = if failed == 0 {
            1.0
        } else {
            self.model_failures as f64 / failed as f64
        };
        (1.0 - rate) * (0.5 + 0.5 * frontier) * (0.5 + 0.5 * model)
    }
}

/// The competence of every capability `attempts` exercise, ordered by
/// capability id; `rates` places each task in its band.
#[must_use]
pub fn competence(attempts: &[Attempt], rates: &PassRates) -> Vec<CapabilityCompetence> {
    let mut by_capability: BTreeMap<&str, (CapabilityCompetence, BTreeSet<&str>)> = BTreeMap::new();
    for attempt in attempts {
        for capability in attempt.counted_under() {
            let (entry, tasks) = by_capability
                .entry(capability)
                .or_insert_with(|| (CapabilityCompetence::new(capability), BTreeSet::new()));
            tasks.insert(&attempt.task);
            entry.attempts += 1;
            match attempt.passed {
                Some(true) => {
                    entry.graded += 1;
                    entry.passes += 1;
                }
                Some(false) => {
                    entry.graded += 1;
                    if attempt.class.is_some_and(FailureClass::is_model) {
                        entry.model_failures += 1;
                    }
                }
                None => entry.ungraded += 1,
            }
            if attempt.passed != Some(true) {
                let class = attempt
                    .class
                    .map_or(FailureClass::Unclassified.as_str(), FailureClass::as_str);
                *entry.failures.entry(class.to_string()).or_default() += 1;
            }
        }
    }
    by_capability
        .into_values()
        .map(|(mut entry, tasks)| {
            entry.tasks = tasks.len();
            let names: Vec<String> = tasks.iter().map(|t| (*t).to_string()).collect();
            entry.bands = rates.counts(&names);
            entry.pass_rate = Share::wilson(entry.passes, entry.graded);
            entry.weight = entry.weigh();
            entry
        })
        .collect()
}
