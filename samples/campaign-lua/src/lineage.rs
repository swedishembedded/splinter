// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that let a model teach itself from
// an interpreter's answers to its own proposals, gated on held-out evidence,
// for its clients. If your team needs expertise in self-taught model
// lineages judged by an independent examiner you can procure our services by
// sending an email to info@swedishembedded.com.

//! The learning loop of one lineage: rounds of propose, predict, label, train
//! and gate, grouped in generations whose champions are handed to the
//! examiner.
//!
//! A round starts from the lineage's champion (the base at first). The
//! champion, served by brain, proposes Lua programs from sections of the
//! manual and predicts what each prints; the interpreter's output is the
//! label. The programs the champion mispredicted are the new information: the
//! round trains a continuation of the champion's adapter on them (a brain
//! training job, supervised objective), with a stated share of the draws on
//! programs it predicted right and a stated share on earlier accepted rounds'
//! records. The candidate and the champion then answer the dev and retention
//! sets and the dev gate decides; only an advance changes the champion. The
//! last accepted candidate of a generation is its champion.
//!
//! Two options change what a round trains on (both off by default). With
//! [`TrainFrom::BaseCumulative`] the candidate is a fresh adapter trained from
//! the base on the union of every accepted round's records and the new ones,
//! not a continuation of the champion's adapter. With a [`Frontier`] the
//! champion predicts each proposal several times, sampled, and only the
//! programs it does not always predict right are trained on; proposing goes
//! on until enough of them are found, within a hard cap of requests.
//!
//! The plan's [`Acceptance`] picks the gate: greedy (the default), or the
//! evidence gate of `crate::evidence_round`, which controls false acceptances.
//!
//! Everything the lineage knows is in append-only journals and the proposals
//! file, so a lineage that is started again continues from them.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};
use splinter_sdk::agent::chat::Reply;
use splinter_sdk::campaign::brain::TaskFamily;
use splinter_sdk::campaign::served::{Answered, ServedValidator};
use splinter_sdk::campaign::{PolicyRef, Validator};
use splinter_sdk::measure::campaign::gate::{decide, Action, Decision, DevGate, DevResult};
use splinter_sdk::measure::campaign::plan::{
    plan_arm, retention_units_needed, ArmGoal, ExamShape, PlannedArm,
};
use splinter_sdk::model::served_train::fine_tune_served;
use splinter_sdk::model::stats::BrainSignificance;
use splinter_sdk::model::train::FineTune;
use splinter_sdk::served_host::ServedHost;
use splinter_sdk::vocabulary::digest::Digest;

use crate::family::{proposal_id, Part, ProposalFamily};
use crate::frontier_round::FrontierOutcome;
use crate::journal::Journal;
use crate::plan::training_ids;
pub use crate::plan::{Acceptance, Plan, TrainFrom};
use crate::proposer::{as_proposals, Budget, Origin, Proposed, Proposer, Steering};
use crate::sections::{family_accuracy, Accuracy};
use crate::sft::{write_chat, Example, Recipe};

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
}

/// One policy's dev and retention answers as graded, so a restart asks once.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct DevRecord {
    pub(crate) policy: String,
    pub(crate) result: DevResult,
}

/// One proposal predicted by a policy.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Prediction {
    generation: u32,
    round: u32,
    policy: String,
    task: String,
    completion: String,
    passed: bool,
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

    fn rounds(&self) -> Journal {
        Journal::at(&self.root.join("rounds.jsonl"))
    }

    pub(crate) fn dev(&self) -> Journal {
        Journal::at(&self.root.join("dev-results.jsonl"))
    }

    fn predictions(&self) -> Journal {
        Journal::at(&self.root.join("predictions.jsonl"))
    }

    pub(crate) fn waves(&self) -> Journal {
        Journal::at(&self.root.join("waves.jsonl"))
    }

    pub(crate) fn frontier(&self) -> Journal {
        Journal::at(&self.root.join("frontier.jsonl"))
    }

    fn datasets(&self) -> PathBuf {
        self.root.join("datasets")
    }

    fn adapters(&self) -> PathBuf {
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

/// What the lineage works with.
pub struct Lineage<'a> {
    /// The lineage's journals and directories.
    pub files: Files,
    /// The proposals and the dev and retention sets.
    pub family: &'a ProposalFamily,
    /// The server of the policies.
    pub host: &'a ServedHost,
    /// The dev exam and the policy's answers to proposals.
    pub validator: &'a mut ServedValidator<'a>,
    /// The policy as a proposer.
    pub proposer: &'a Proposer,
    /// The base checkpoint.
    pub base: &'a Path,
    /// The training settings.
    pub recipe: &'a Recipe,
    /// The plan.
    pub plan: &'a Plan,
    /// The dev gate.
    pub gate: DevGate,
    /// The seed training seeds are derived from.
    pub seed: u64,
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

impl Lineage<'_> {
    /// The rounds journal.
    ///
    /// # Errors
    /// The journal cannot be read.
    pub fn rounds(&self) -> Result<Vec<RoundRecord>> {
        self.files.rounds().read()
    }

    /// Refuses a plan whose dev set of `dev_units` programs cannot see the
    /// target effect, or whose retention set of `retention_units` cannot see
    /// the loss the margin allows, at the last gate the lineage spends alpha
    /// on (the strictest).
    ///
    /// # Errors
    /// The dev or retention set is too small for the plan.
    pub fn check_power(
        &self,
        dev_units: usize,
        retention_units: usize,
    ) -> Result<(PlannedArm, usize)> {
        let gates = self.plan.generations * self.plan.rounds;
        let planned = plan_arm(
            &ExamShape {
                units: dev_units,
                discordance: self.plan.discordance,
            },
            &ArmGoal::new(self.plan.target_effect, gates),
            &BrainSignificance,
        )
        .map_err(fail)?;
        let needed = retention_units_needed(
            self.plan.retention_margin,
            planned.alpha,
            5000,
            &BrainSignificance,
        );
        anyhow::ensure!(
            needed.is_some_and(|n| retention_units >= n),
            "the retention set has {retention_units} programs, too few to detect a loss of {} of them at level {:.4} (needs {needed:?})",
            self.plan.retention_margin,
            planned.alpha
        );
        Ok((planned, needed.unwrap_or(0)))
    }

    /// The dev and retention answers of `policy`, asked once and remembered.
    ///
    /// # Errors
    /// The policy cannot be served or the journal written.
    pub fn dev_of(&mut self, policy: &PolicyRef) -> Result<DevResult> {
        let key = policy_key(policy);
        let held: Vec<DevRecord> = self.files.dev().read()?;
        if let Some(record) = held.into_iter().rev().find(|r| r.policy == key) {
            return Ok(record.result);
        }
        let result = self.validator.validate(policy).map_err(fail)?;
        self.files.dev().append(&DevRecord {
            policy: key,
            result: result.clone(),
        })?;
        Ok(result)
    }

    /// The champion's dev accuracy per library function.
    pub(crate) fn accuracy(&self, dev: &DevResult) -> Accuracy {
        let programs: Vec<(String, bool)> = dev
            .dev
            .iter()
            .filter_map(|(task, &passed)| self.family.program_of(task).map(|p| (p, passed)))
            .collect();
        family_accuracy(programs.iter().map(|(p, passed)| (p.as_str(), *passed)))
    }

    pub(crate) fn asked_by_section(&self) -> BTreeMap<String, usize> {
        let mut asked = BTreeMap::new();
        for proposal in self.family.proposals() {
            if let Some(section) = proposal.section {
                *asked.entry(section).or_default() += 1;
            }
        }
        asked
    }

    /// The base's own proposals for the dev and retention sets, made once
    /// before the first round: held out for good, never trained on.
    ///
    /// # Errors
    /// The base cannot propose enough programs.
    pub fn open_held_out(&mut self, dev: usize, keep: usize, budget: Budget) -> Result<()> {
        if self.family.has_round(0, 0) {
            return Ok(());
        }
        let ctx = self.host.context(&PolicyRef::default())?;
        let made = self.proposer.propose(
            &ctx,
            None,
            self.family,
            Steering {
                accuracy: &Accuracy::new(),
                asked: &BTreeMap::new(),
                seed: self.seed,
            },
            Budget {
                want: dev + keep,
                ..budget
            },
        )?;
        eprintln!("the base proposed: {}", serde_json::to_string(&made)?);
        anyhow::ensure!(
            made.kept.len() >= dev + keep,
            "the base proposed {} new programs, fewer than the {} held out for the dev and retention sets",
            made.kept.len(),
            dev + keep
        );
        self.family
            .add(as_proposals(
                made.kept,
                Origin {
                    generation: 0,
                    round: 0,
                },
                dev,
                keep,
            ))
            .map_err(fail)?;
        Ok(())
    }

    /// Runs round `round` of `generation` unless the journal holds it.
    ///
    /// # Errors
    /// Serving, proposing, training or journaling fails; with a frontier, the
    /// request cap is reached before the frontier has its target size.
    pub fn run_round(&mut self, generation: u32, round: u32) -> Result<RoundRecord> {
        let rounds = self.rounds()?;
        if let Some(done) = rounds
            .iter()
            .find(|r| r.generation == generation && r.round == round)
        {
            return Ok(done.clone());
        }
        let started = Instant::now();
        let champion = champion_of(&rounds);
        let started_from = champion.digest.as_ref().map(ToString::to_string);
        let champion_dev = self.steering_dev(&champion)?;

        let mut record = RoundRecord {
            generation,
            round,
            started_from,
            proposed: 0,
            mispredicted: 0,
            replayed: 0,
            trained: None,
            decision: None,
            advanced: false,
            note: String::new(),
            trained_on: Vec::new(),
            seconds: 0.0,
            train_from: self.plan.train_from,
            trained_records: 0,
            frontier: None,
        };
        let (wrong, right) = match self.plan.frontier {
            Some(settings) => {
                self.select_frontier(&champion, &champion_dev, &mut record, settings)?
            }
            None => self.select_mispredicted(&champion, &champion_dev, &mut record)?,
        };
        record.mispredicted = wrong.len();
        if wrong.is_empty() {
            record.note = "the policy predicted every proposal right: nothing to learn".into();
        } else {
            self.train_and_gate(&champion, &champion_dev, &wrong, &right, &mut record)?;
        }
        record.seconds = started.elapsed().as_secs_f64();
        self.files.rounds().append(&record)?;
        Ok(record)
    }

    /// The champion proposes the round's programs once and predicts each once,
    /// greedily; the ones it got wrong and the ones it got right.
    fn select_mispredicted(
        &mut self,
        champion: &PolicyRef,
        champion_dev: &DevResult,
        record: &mut RoundRecord,
    ) -> Result<(Vec<String>, Vec<String>)> {
        let (generation, round) = (record.generation, record.round);
        if !self.family.has_round(generation, round) {
            self.propose_wave(
                champion,
                champion_dev,
                (generation, round, 0),
                Budget {
                    want: self.plan.proposals_per_round,
                    max_calls: self.plan.budget.0,
                    parallel: self.plan.budget.1,
                },
            )?;
        }
        let proposed = self.round_proposals(generation, round);
        let answers = self.predict(champion, generation, round, &proposed)?;
        let (wrong, right): (Vec<&Answered>, Vec<&Answered>) =
            answers.iter().partition(|a| !a.passed);
        eprintln!(
            "generation {generation} round {round}: the policy predicted {} of {} proposals right",
            right.len(),
            answers.len()
        );
        record.proposed = proposed.len();
        Ok((
            wrong.iter().map(|a| a.task.clone()).collect(),
            right.iter().map(|a| a.task.clone()).collect(),
        ))
    }

    /// Ids of the training programs the round's proposing added so far.
    pub(crate) fn round_proposals(&self, generation: u32, round: u32) -> Vec<String> {
        self.family
            .proposals()
            .into_iter()
            .filter(|p| p.generation == generation && p.round == round && p.part == Part::Train)
            .map(|p| p.id)
            .collect()
    }

    /// One wave of proposing by the champion, added to the family; the ids of
    /// the programs it added and what it asked.
    pub(crate) fn propose_wave(
        &mut self,
        champion: &PolicyRef,
        champion_dev: &DevResult,
        (generation, round, wave): (u32, u32, usize),
        budget: Budget,
    ) -> Result<(Proposed, Vec<String>)> {
        let ctx = self.host.context(champion)?;
        let made = self.proposer.propose(
            &ctx,
            champion.adapter.as_deref(),
            self.family,
            Steering {
                accuracy: &self.accuracy(champion_dev),
                asked: &self.asked_by_section(),
                seed: self.seed
                    ^ (u64::from(generation) << 32 | u64::from(round))
                    ^ (wave as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            },
            budget,
        )?;
        eprintln!(
            "generation {generation} round {round} wave {wave}: the policy proposed {}",
            serde_json::to_string(&made)?
        );
        let fresh: Vec<String> = made
            .kept
            .iter()
            .map(|k| proposal_id(&k.program))
            .filter(|id| self.family.proposal(id).is_none())
            .collect();
        self.family
            .add(as_proposals(
                made.kept.clone(),
                Origin { generation, round },
                0,
                0,
            ))
            .map_err(fail)?;
        Ok((made, fresh))
    }

    /// Asks `policy` every proposal in `ids` once (greedy), remembering the
    /// answers.
    pub(crate) fn predict(
        &mut self,
        policy: &PolicyRef,
        generation: u32,
        round: u32,
        ids: &[String],
    ) -> Result<Vec<Answered>> {
        let key = policy_key(policy);
        let held: Vec<Prediction> = self.files.predictions().read()?;
        let known: BTreeMap<&str, &Prediction> = held
            .iter()
            .filter(|p| p.generation == generation && p.round == round && p.policy == key)
            .map(|p| (p.task.as_str(), p))
            .collect();
        if ids.iter().all(|id| known.contains_key(id.as_str())) {
            return Ok(ids
                .iter()
                .map(|id| {
                    let p = known[id.as_str()];
                    Answered {
                        task: p.task.clone(),
                        reply: Reply {
                            text: p.completion.clone(),
                            ..Reply::default()
                        },
                        passed: p.passed,
                    }
                })
                .collect());
        }
        self.host.context(policy)?;
        let answers = self.validator.answer(policy, ids).map_err(fail)?;
        for a in &answers {
            self.files.predictions().append(&Prediction {
                generation,
                round,
                policy: key.clone(),
                task: a.task.clone(),
                completion: a.reply.text.clone(),
                passed: a.passed,
            })?;
        }
        Ok(answers)
    }

    fn example(&self, id: &str) -> Result<Example> {
        let proposal = self
            .family
            .proposal(id)
            .with_context(|| format!("{id} is not a proposal"))?;
        Ok(Example {
            id: proposal.id,
            request: proposal.request,
            expected: proposal.expected,
        })
    }

    /// Trains the round's candidate and lets the gate decide on it.
    fn train_and_gate(
        &mut self,
        champion: &PolicyRef,
        champion_dev: &DevResult,
        wrong: &[String],
        right: &[String],
        record: &mut RoundRecord,
    ) -> Result<()> {
        let (generation, round) = (record.generation, record.round);
        let tag = format!("g{generation}r{round}");
        let dir = self.files.datasets();
        std::fs::create_dir_all(&dir)?;
        let examples = |ids: &[String]| -> Result<Vec<Example>> {
            ids.iter().map(|i| self.example(i)).collect()
        };
        let accepted: Vec<String> = trained_up_to(&self.rounds()?, u32::MAX)
            .into_iter()
            .collect();
        let cumulative = self.plan.train_from == TrainFrom::BaseCumulative;
        // Cumulative: the union of every accepted round's records and the new
        // ones is the training set, so nothing is replayed beside it.
        // Continued: the new records are, and earlier ones are replayed.
        let new_ids = training_ids(cumulative, &accepted, wrong);
        let replay_ids = if cumulative { Vec::new() } else { accepted };
        let (train, correct, replay) = (
            examples(&new_ids)?,
            examples(right)?,
            examples(&replay_ids)?,
        );
        record.trained_records = train.len();
        let steps = self.plan.steps_for(self.recipe, train.len());
        let (train_file, correct_file, replay_file) = (
            dir.join(format!("{tag}-mispredicted.jsonl")),
            dir.join(format!("{tag}-correct.jsonl")),
            dir.join(format!("{tag}-replay.jsonl")),
        );
        write_chat(&train_file, &train)?;
        let mut weighted = Vec::new();
        if !correct.is_empty() && self.recipe.correct_share > 0.0 {
            write_chat(&correct_file, &correct)?;
            weighted.push((correct_file.clone(), self.recipe.correct_share));
        }
        let mut replayed = Vec::new();
        if !replay.is_empty() && self.recipe.replay_share > 0.0 {
            write_chat(&replay_file, &replay)?;
            replayed.push(replay_file.clone());
            record.replayed = replay.len();
        }
        let held_file = dir.join(format!("{tag}-held-out.jsonl"));
        let held: Vec<Example> = self
            .family
            .dev()
            .iter()
            .filter_map(|id| self.example(id).ok())
            .take(self.recipe.held_out_records)
            .collect();
        anyhow::ensure!(
            !held.is_empty(),
            "no dev proposal to score brain's held-out measure on"
        );
        write_chat(&held_file, &held)?;

        let attempt = self.files.adapters().join(&tag);
        let endpoints = self.host.endpoints()?;
        self.host.context(champion)?;
        let request = FineTune {
            model_dir: self.base,
            train: &train_file,
            held_out: &held_file,
            held_out_text: None,
            monitor: None,
            eval_every: 0,
            patience: 0,
            attempt_dir: &attempt,
            steps,
            rank: self.recipe.rank,
            alpha: (self.recipe.rank * 2) as f32,
            replay: &replayed,
            replay_share: (!replayed.is_empty()).then_some(self.recipe.replay_share),
            weighted_replay: &weighted,
            grad_accum: self.recipe.records_per_step,
            continue_from: if cumulative {
                None
            } else {
                champion.adapter.as_deref()
            },
            cancel: None,
            base: self.recipe.precision()?,
            learning_rate: Some(self.recipe.lr),
            weight_decay: self.recipe.weight_decay,
            seed: self.seed ^ (u64::from(generation) << 32 | u64::from(round)),
            max_block: Some(self.recipe.max_block),
            thinking: false,
            pack_answers: false,
            on_step: None,
            keep_evaluations: None,
        };
        let trained = fine_tune_served(&endpoints, &request)
            .map_err(|e| anyhow::anyhow!("training {tag}: {e}"))?;
        record.trained = Some(Trained {
            adapter: trained.adapter.clone(),
            digest: trained.adapter_digest.clone(),
            steps: trained.curve.steps_completed as usize,
            held_out_loss_base: trained.base.loss,
            held_out_loss_tuned: trained.tuned.loss,
        });
        record.trained_on = wrong.iter().chain(right).cloned().collect();
        let candidate = PolicyRef {
            version: champion.version + 1,
            adapter: Some(trained.adapter),
            digest: Some(Digest::parse(&trained.adapter_digest).map_err(fail)?),
        };
        let decision = match self.plan.acceptance {
            Acceptance::Greedy => {
                let candidate_dev = self.dev_of(&candidate)?;
                let gate_number = (generation - 1) * self.plan.rounds + round;
                decide(
                    &self.gate,
                    gate_number,
                    champion_dev,
                    &candidate_dev,
                    &BrainSignificance,
                )
            }
            Acceptance::Evidence(plan) => {
                self.gate_evidence(plan, champion, &candidate, (generation, round))?
            }
        };
        eprintln!(
            "generation {generation} round {round}: gate {:?}: {} (dev {:?} to {:?}, p {:.4})",
            decision.action,
            decision.reason,
            decision.champion_rate,
            decision.candidate_rate,
            decision.p_value
        );
        record.advanced = decision.action == Action::Advance;
        record.decision = Some(decision);
        Ok(())
    }
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
}
