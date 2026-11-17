// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that let a model teach itself from
// an interpreter's answers to its own proposals, gated on held-out evidence,
// for its clients. If your team needs expertise in self-taught model
// lineages judged by an independent examiner you can procure our services by
// sending an email to info@swedishembedded.com.

//! What a lineage writes down: the round records, the dev and prediction
//! entries, the files they live in, and what a journal of rounds says about
//! the champion and the programs it was trained on.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use splinter_sdk::campaign::PolicyRef;
use splinter_sdk::measure::campaign::gate::{Decision, DevResult};
use splinter_sdk::vocabulary::digest::Digest;

use crate::frontier_round::FrontierOutcome;
use crate::journal::Journal;
use crate::plan::{TrainFrom, TrainOn};
use crate::reasoning::Reasoning;
use crate::semantic::{OutputRule, Thresholds};

/// What a round trained: the adapter and what brain reported of the run.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Trained {
    /// The adapter file.
    pub adapter: PathBuf,
    /// Its digest, `sha256:...`.
    pub digest: String,
    /// Optimiser steps run.
    pub steps: usize,
    /// Brain's held-out loss of the base model on the held-out dev programs.
    pub held_out_loss_base: Option<f32>,
    /// Brain's held-out loss of the candidate on the same programs.
    pub held_out_loss_tuned: Option<f32>,
}

/// The output rule of a round and the thresholds it applied.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputAdmission {
    /// The rule.
    pub rule: OutputRule,
    /// Its thresholds.
    pub thresholds: Thresholds,
}

impl OutputAdmission {
    /// The record of `rule`: none for the default.
    #[must_use]
    pub fn of(rule: OutputRule) -> Option<Self> {
        (!rule.is_default()).then(|| Self {
            rule,
            thresholds: Thresholds::current(),
        })
    }
}

/// What one task family contributed to a round: how new its proposals were
/// and how much of what was proposed lay on the frontier.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FamilyRound {
    /// The family.
    pub family: String,
    /// Proposal requests made, in all waves.
    pub requests: usize,
    /// New programs admitted: not a repeat of one the lineage already held.
    pub new_programs: usize,
    /// New programs admitted per request (the proposer's novelty); zero
    /// when no request was made.
    pub novelty: f64,
    /// Proposals the champion predicted.
    pub measured: usize,
    /// Of those, the ones on the frontier.
    pub on_frontier: usize,
    /// Frontier programs per proposal predicted (the frontier yield).
    pub frontier_yield: f64,
    /// Frontier programs the round trains on.
    pub selected: usize,
    /// Frontier programs skipped for the length of their records.
    pub overlong: usize,
}

/// A task family's dev accuracy before and after a round.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FamilyDev {
    /// The family.
    pub family: String,
    /// Dev programs of the family the gate graded.
    pub items: usize,
    /// The champion's share of them right.
    pub champion: f64,
    /// The candidate's share of them right.
    pub candidate: f64,
}

/// The journal entry of one round.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoundRecord {
    /// The generation.
    pub generation: u32,
    /// The round of it.
    pub round: u32,
    /// The champion the round started from (its adapter digest; none for the
    /// base).
    pub started_from: Option<String>,
    /// Programs proposed this round.
    pub proposed: usize,
    /// Of those, the ones the champion mispredicted.
    pub mispredicted: usize,
    /// Records replayed from earlier accepted rounds.
    pub replayed: usize,
    /// What was trained; none when nothing could be.
    pub trained: Option<Trained>,
    /// The gate's decision on it.
    pub decision: Option<Decision>,
    /// Whether the candidate became the champion.
    pub advanced: bool,
    /// Why the round ended as it did, when it trained nothing.
    pub note: String,
    /// Ids of the proposals this round trained on.
    pub trained_on: Vec<String>,
    /// Wall-clock seconds the round took.
    pub seconds: f64,
    /// What the candidate was trained from.
    #[serde(default)]
    pub train_from: TrainFrom,
    /// Records the candidate trained on in all (the union, when it is one).
    #[serde(default)]
    pub trained_records: usize,
    /// How the frontier was found, when it was.
    #[serde(default)]
    pub frontier: Option<FrontierOutcome>,
    /// How the champion reasoned and how the exams decoded, when it reasoned;
    /// absent for a lineage whose models answer without reasoning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Reasoning>,
    /// What the records were made of; absent for answer-only records.
    #[serde(default, skip_serializing_if = "TrainOn::is_default")]
    pub train_on: TrainOn,
    /// The output rule the round's proposals were admitted under and the
    /// thresholds it applied; absent for the default rule.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub admit_output: Option<OutputAdmission>,
    /// What each task family contributed, when the lineage has families.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub families: Vec<FamilyRound>,
    /// Each family's dev accuracy under the greedy gate, when the lineage
    /// has families.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub family_dev: Vec<FamilyDev>,
}

/// One policy's dev and retention answers as graded, so a restart asks once.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct DevRecord {
    pub(crate) policy: String,
    pub(crate) result: DevResult,
}

/// One proposal predicted by a policy.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Prediction {
    pub(crate) generation: u32,
    pub(crate) round: u32,
    pub(crate) policy: String,
    pub(crate) task: String,
    pub(crate) completion: String,
    pub(crate) passed: bool,
}

/// The files a lineage keeps in its directory.
#[derive(Clone, Debug)]
pub struct Files {
    root: PathBuf,
}

impl Files {
    /// The files under `root`.
    #[must_use]
    pub fn at(root: &Path) -> Self {
        Self {
            root: root.to_path_buf(),
        }
    }

    /// The proposals, by round.
    #[must_use]
    pub fn proposals(&self) -> PathBuf {
        self.root.join("proposals.jsonl")
    }

    /// The programs refused before measurement, with their reasons.
    #[must_use]
    pub fn refused(&self) -> PathBuf {
        self.root.join("refused.jsonl")
    }

    pub(crate) fn rounds(&self) -> Journal {
        Journal::at(&self.root.join("rounds.jsonl"))
    }

    pub(crate) fn dev(&self) -> Journal {
        Journal::at(&self.root.join("dev-results.jsonl"))
    }

    pub(crate) fn predictions(&self) -> Journal {
        Journal::at(&self.root.join("predictions.jsonl"))
    }

    pub(crate) fn waves(&self) -> Journal {
        Journal::at(&self.root.join("waves.jsonl"))
    }

    pub(crate) fn frontier(&self) -> Journal {
        Journal::at(&self.root.join("frontier.jsonl"))
    }

    /// The verified reasoning traces of the frontier's programs, with
    /// reasoning on.
    pub(crate) fn traces(&self) -> Journal {
        Journal::at(&self.root.join("traces.jsonl"))
    }

    pub(crate) fn datasets(&self) -> PathBuf {
        self.root.join("datasets")
    }

    pub(crate) fn adapters(&self) -> PathBuf {
        self.root.join("adapters")
    }
}

/// The key a policy is known by in the journals.
pub(crate) fn policy_key(policy: &PolicyRef) -> String {
    policy
        .digest
        .as_ref()
        .map_or_else(|| "base".to_string(), ToString::to_string)
}

pub(crate) fn fail<E: std::fmt::Display>(e: E) -> anyhow::Error {
    anyhow::anyhow!("{e}")
}

/// The champion a journal of rounds leaves: the adapter of the last advance.
#[must_use]
pub fn champion_of(rounds: &[RoundRecord]) -> PolicyRef {
    let mut champion = PolicyRef::default();
    for round in rounds.iter().filter(|r| r.advanced) {
        if let Some(trained) = &round.trained {
            champion = PolicyRef {
                version: champion.version + 1,
                adapter: Some(trained.adapter.clone()),
                digest: Digest::parse(&trained.digest).ok(),
            };
        }
    }
    champion
}

/// The proposals the champion at the end of `generation` was trained on:
/// those of every advanced round up to it.
#[must_use]
pub fn trained_up_to(rounds: &[RoundRecord], generation: u32) -> BTreeSet<String> {
    rounds
        .iter()
        .filter(|r| r.advanced && r.generation <= generation)
        .flat_map(|r| r.trained_on.iter().cloned())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_round_journalled_before_these_options_still_reads() {
        let old = r#"{"generation":1,"round":1,"started_from":null,"proposed":2,"mispredicted":1,"replayed":0,"trained":null,"decision":null,"advanced":false,"note":"","trained_on":[],"seconds":1.0}"#;
        let record: RoundRecord = serde_json::from_str(old).unwrap();
        assert_eq!(record.train_from, TrainFrom::Previous);
        assert!(record.frontier.is_none());
        assert_eq!(
            (record.reasoning, record.train_on),
            (None, TrainOn::Answers)
        );
    }

    /// A lineage that never heard of reasoning journals the lines it always
    /// did, so it resumes unchanged and a later run reads them back equal.
    #[test]
    fn a_round_with_the_options_off_journals_without_a_trace_of_them() {
        let record = round(1, 1, true, "a");
        let line = serde_json::to_string(&record).unwrap();
        for key in [
            "reasoning",
            "train_on",
            "admit",
            "records",
            "families",
            "family_dev",
        ] {
            assert!(!line.contains(&format!("\"{key}\"")), "{key} in {line}");
        }
        let again: RoundRecord = serde_json::from_str(&line).unwrap();
        assert_eq!(serde_json::to_string(&again).unwrap(), line);
    }

    fn round(generation: u32, round: u32, advanced: bool, tag: &str) -> RoundRecord {
        RoundRecord {
            generation,
            round,
            started_from: None,
            proposed: 2,
            mispredicted: 1,
            replayed: 0,
            trained: Some(Trained {
                adapter: PathBuf::from(format!("/adapters/{tag}.safetensors")),
                digest: format!(
                    "sha256:{:064x}",
                    u64::from(generation) * 10 + u64::from(round)
                ),
                steps: 4,
                held_out_loss_base: None,
                held_out_loss_tuned: None,
            }),
            decision: None,
            advanced,
            note: String::new(),
            trained_on: vec![format!("prop-{tag}")],
            seconds: 0.0,
            train_from: TrainFrom::Previous,
            trained_records: 0,
            frontier: None,
            reasoning: None,
            train_on: TrainOn::Answers,
            admit_output: None,
            families: Vec::new(),
            family_dev: Vec::new(),
        }
    }

    #[test]
    fn the_champion_is_the_last_advanced_candidate_and_a_rolled_back_round_changes_nothing() {
        assert_eq!(champion_of(&[]), PolicyRef::default());
        let rounds = [
            round(1, 1, true, "a"),
            round(1, 2, false, "b"),
            round(2, 1, true, "c"),
            round(2, 2, false, "d"),
        ];
        let champion = champion_of(&rounds);
        assert_eq!(champion.version, 2);
        assert_eq!(
            champion.adapter,
            Some(PathBuf::from("/adapters/c.safetensors"))
        );
        assert_eq!(champion_of(&rounds[..2]).version, 1);
    }

    #[test]
    fn a_candidate_is_audited_for_what_its_chain_trained_on_and_nothing_it_rolled_back() {
        let rounds = [
            round(1, 1, true, "a"),
            round(1, 2, false, "b"),
            round(2, 1, true, "c"),
        ];
        let ids = |generation| -> Vec<String> {
            trained_up_to(&rounds, generation).into_iter().collect()
        };
        assert_eq!(ids(1), vec!["prop-a"]);
        assert_eq!(ids(2), vec!["prop-a", "prop-c"]);
    }

    #[test]
    fn a_round_with_families_records_each_familys_novelty_and_yield() {
        let mut record = round(2, 1, true, "f");
        record.families = vec![FamilyRound {
            family: "python3".into(),
            requests: 4,
            new_programs: 6,
            novelty: 1.5,
            measured: 10,
            on_frontier: 4,
            frontier_yield: 0.4,
            selected: 4,
            overlong: 0,
        }];
        let line = serde_json::to_string(&record).unwrap();
        assert!(
            line.contains("\"novelty\":1.5") && line.contains("\"frontier_yield\":0.4"),
            "{line}"
        );
        let again: RoundRecord = serde_json::from_str(&line).unwrap();
        assert_eq!(again.families, record.families);
    }

    #[test]
    fn a_round_under_the_semantic_rule_records_the_rule_and_its_thresholds() {
        let mut record = round(1, 1, true, "s");
        record.admit_output = OutputAdmission::of(OutputRule::Semantic);
        let line = serde_json::to_string(&record).unwrap();
        assert!(
            line.contains(
                r#""admit_output":{"rule":"semantic","thresholds":{"max_fraction_digits":6"#
            ),
            "{line}"
        );
        let again: RoundRecord = serde_json::from_str(&line).unwrap();
        assert_eq!(again.admit_output, record.admit_output);
        assert_eq!(OutputAdmission::of(OutputRule::Any), None);
    }
}
