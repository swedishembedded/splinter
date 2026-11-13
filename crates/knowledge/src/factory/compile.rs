// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that compile a learning objective
// into validated, versioned training environments, for its clients. If your
// team needs expertise in environment factories for agent reinforcement
// learning, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The compiler: an objective in, admitted environments out.
//!
//! Each skill of the objective is classified ([`classify_all`]); a skill with
//! a plugin gets a difficulty ladder of environments, each sampled, admitted
//! (or quarantined or rejected) and stored with its report; a skill without
//! one is listed with the reason, never silently dropped. Nothing here
//! chooses a model or a card: base pass rates arrive as measurements
//! ([`EnvironmentStore::record_band`]).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use splinter_core::rng::SplitMix64;

use super::admission::{admit, AdmissionPolicy, AdmitError, Status};
use super::host::Host;
use super::objective::{classify_all, Classification, ObjectiveSpec};
use super::plugin::{Knob, Plugin};
use super::plugins::{REPOSITORY_BUGS, REPOSITORY_MUTANTS};
use super::store::{EnvironmentStore, StoreError};

/// The most rungs a skill's difficulty ladder has.
pub const MAX_RUNGS: usize = 6;

/// Why compiling stopped.
#[derive(Debug, thiserror::Error)]
pub enum CompileError {
    /// Admission could not be carried out.
    #[error(transparent)]
    Admit(#[from] AdmitError),
    /// The store refused.
    #[error(transparent)]
    Store(#[from] StoreError),
}

/// The plugin named `id`, or the one that stands in for it when it is not
/// available (the repository plugin falls back to the embedded library).
fn find_plugin<'a>(plugins: &'a [Box<dyn Plugin>], id: &str) -> Option<&'a dyn Plugin> {
    let stand_in = (id == REPOSITORY_MUTANTS).then_some(REPOSITORY_BUGS);
    plugins
        .iter()
        .find(|p| p.id() == id)
        .or_else(|| plugins.iter().find(|p| Some(p.id()) == stand_in))
        .map(Box::as_ref)
}

/// The difficulty ladder of a plugin: `rungs` settings of every knob from
/// its lowest to its highest value, evenly spaced; one rung is the knobs'
/// defaults.
#[must_use]
pub fn ladder(knobs: &[Knob], rungs: usize) -> Vec<BTreeMap<String, u32>> {
    let rungs = rungs.max(1);
    (0..rungs)
        .map(|rung| {
            knobs
                .iter()
                .map(|knob| {
                    let value = if rungs == 1 {
                        knob.default
                    } else {
                        let span = u64::from(knob.max - knob.min);
                        knob.min
                            + u32::try_from(span * rung as u64 / (rungs as u64 - 1)).unwrap_or(0)
                    };
                    (knob.name.to_string(), value)
                })
                .collect()
        })
        .collect()
}

/// What happened to one environment of a skill's ladder.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EnvironmentOutcome {
    /// The environment's id.
    pub environment: String,
    /// The difficulty knobs.
    pub difficulty: BTreeMap<String, u32>,
    /// Where it stands.
    pub status: Status,
    /// Why, when it is not admitted.
    pub reasons: Vec<String>,
}

/// What became of one skill.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SkillOutcome {
    /// How the skill's reward can be established, and by what.
    pub classification: Classification,
    /// The label a verdict on this skill must carry.
    pub label: Option<String>,
    /// The environments built for it, easiest first.
    pub environments: Vec<EnvironmentOutcome>,
    /// Why nothing was built, when nothing was.
    pub not_built: Option<String>,
}

/// What compiling an objective produced.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompileReport {
    /// The objective compiled.
    pub objective: ObjectiveSpec,
    /// The seed the samples were drawn from.
    pub seed: u64,
    /// One entry per skill.
    pub skills: Vec<SkillOutcome>,
}

impl CompileReport {
    /// Environments that may enter a campaign now.
    #[must_use]
    pub fn admitted(&self) -> usize {
        self.count(Status::Admitted)
    }

    /// Environments whose gates passed and whose base pass rate is yet to be
    /// measured.
    #[must_use]
    pub fn pending_band(&self) -> usize {
        self.count(Status::PendingBand)
    }

    /// Environments whose verifier was beaten.
    #[must_use]
    pub fn quarantined(&self) -> usize {
        self.count(Status::Quarantined)
    }

    fn count(&self, status: Status) -> usize {
        self.skills
            .iter()
            .flat_map(|s| &s.environments)
            .filter(|e| e.status == status)
            .count()
    }
}

/// Compiles `objective` with `plugins`, storing every environment in
/// `store`. `seed` is the caller's drawn, recorded seed.
pub fn compile(
    objective: &ObjectiveSpec,
    plugins: &[Box<dyn Plugin>],
    host: &Host,
    policy: &AdmissionPolicy,
    seed: u64,
    store: &EnvironmentStore,
) -> Result<CompileReport, CompileError> {
    let classifications = classify_all(objective);
    let buildable = classifications
        .iter()
        .filter(|c| {
            c.plugin
                .as_deref()
                .is_some_and(|id| find_plugin(plugins, id).is_some())
        })
        .count();
    let rungs = (objective.budget.environments / buildable.max(1)).clamp(1, MAX_RUNGS);
    let mut rng = SplitMix64::new(seed);
    let mut built = 0;
    let mut skills = Vec::new();
    for classification in classifications {
        let label = classification.label().map(ToString::to_string);
        let plugin = classification
            .plugin
            .as_deref()
            .and_then(|id| find_plugin(plugins, id));
        let Some(plugin) = plugin else {
            let reason = match &classification.plugin {
                Some(id) => format!("the {id} plugin is not built yet"),
                None => format!(
                    "no automatic environment for a {} skill",
                    classification.class
                ),
            };
            skills.push(SkillOutcome {
                classification,
                label,
                environments: Vec::new(),
                not_built: Some(reason),
            });
            continue;
        };
        let mut environments = Vec::new();
        for difficulty in ladder(plugin.knobs(), rungs) {
            if built >= objective.budget.environments {
                break;
            }
            let known = store.known_versions()?;
            let (report, instances) = admit(
                plugin,
                &difficulty,
                host,
                policy,
                rng.next_u64(),
                &known,
                None,
            )?;
            if skills
                .iter()
                .flat_map(|s: &SkillOutcome| &s.environments)
                .chain(&environments)
                .any(|e: &EnvironmentOutcome| e.environment == report.environment)
            {
                continue; // another skill already built this environment
            }
            store.save(&report, &instances)?;
            built += 1;
            environments.push(EnvironmentOutcome {
                environment: report.environment.clone(),
                difficulty: report.difficulty.clone(),
                status: report.status,
                reasons: report.reasons.clone(),
            });
        }
        skills.push(SkillOutcome {
            classification,
            label,
            environments,
            not_built: None,
        });
    }
    let report = CompileReport {
        objective: objective.clone(),
        seed,
        skills,
    };
    store.record_compilation(&format!("seed-{seed}"), &report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    const KNOBS: &[Knob] = &[
        Knob {
            name: "a",
            min: 1,
            max: 5,
            default: 2,
            doc: "",
        },
        Knob {
            name: "b",
            min: 0,
            max: 9,
            default: 3,
            doc: "",
        },
    ];

    #[test]
    fn a_ladder_runs_from_the_lowest_to_the_highest_setting_and_one_rung_is_the_default() {
        let steps = ladder(KNOBS, 3);
        assert_eq!(steps.len(), 3);
        assert_eq!((steps[0]["a"], steps[0]["b"]), (1, 0));
        assert_eq!((steps[1]["a"], steps[1]["b"]), (3, 4));
        assert_eq!((steps[2]["a"], steps[2]["b"]), (5, 9));
        let only = ladder(KNOBS, 1);
        assert_eq!((only[0]["a"], only[0]["b"]), (2, 3));
        assert_eq!(ladder(KNOBS, 0).len(), 1);
    }
}
