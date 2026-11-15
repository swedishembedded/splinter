// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that let a model teach itself from
// an interpreter's answers to its own proposals, gated on held-out evidence,
// for its clients. If your team needs expertise in self-taught model
// lineages judged by an independent examiner you can procure our services by
// sending an email to info@swedishembedded.com.

//! What a lineage is asked to do, and how a candidate is trained from the
//! records: the plan and its refusals.

use std::collections::BTreeSet;

use anyhow::Result;
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use splinter_sdk::measure::campaign::evidence::EvidenceGate;

use crate::frontier::{Admit, Frontier, Rule};
use crate::reasoning::Reasoning;
use crate::sft::Recipe;

/// What a round's candidate is trained from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum TrainFrom {
    /// The champion's adapter, continued on the round's records (with the
    /// recipe's replay of earlier rounds).
    #[default]
    Previous,
    /// A fresh adapter on the base, trained on the union of every accepted
    /// round's records and this round's.
    BaseCumulative,
}

/// What a round's training records are made of.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum TrainOn {
    /// The interpreter's output, in one fenced block (the registered
    /// lineages).
    #[default]
    Answers,
    /// A program with at least one verified reasoning trace contributes one
    /// record: its prompt, the shortest verified trace as the reasoning, and
    /// the interpreter's output as the answer. A program with none
    /// contributes nothing.
    VerifiedTraces,
    /// As [`Self::VerifiedTraces`], and a program with no verified trace
    /// contributes the answer-only record with an empty reasoning, so it is
    /// trained in thinking mode too.
    #[value(name = "verified-traces+labels")]
    #[serde(rename = "verified-traces+labels")]
    VerifiedTracesLabels,
}

impl TrainOn {
    /// Whether the records are trained as a model that reasons answers them.
    #[must_use]
    pub fn uses_traces(self) -> bool {
        self != Self::Answers
    }

    /// Whether this is the default (journals leave the default out).
    #[must_use]
    pub fn is_default(&self) -> bool {
        *self == Self::Answers
    }
}

/// How a round's candidate is accepted.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Acceptance {
    /// On a net gain over the whole dev set, within the retention margin;
    /// significance is reported, not required. The registered lineages run
    /// this.
    #[default]
    Greedy,
    /// On a noninferiority e-process over dev items drawn per round from the
    /// dev set as a pool, its false evolution rate controlled across every
    /// planned round; on rejection the champion stays.
    Evidence(EvidencePlan),
}

/// The evidence gate of a lineage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvidencePlan {
    /// The test and its budget.
    pub gate: EvidenceGate,
    /// Dev items a round draws, fewer than the pool holds.
    pub draw: usize,
    /// Dev items graded at a time; the round stops at the first batch whose
    /// evidence crosses.
    pub batch: usize,
}

/// What the lineage is asked to do, apart from how it trains.
#[derive(Clone, Debug)]
pub struct Plan {
    /// Generations in the lineage.
    pub generations: u32,
    /// Rounds of each generation.
    pub rounds: u32,
    /// Programs the policy is asked to propose each round.
    pub proposals_per_round: usize,
    /// The most proposal requests a round makes, and how many at once.
    pub budget: (usize, usize),
    /// The lead (share of dev programs won) a round is meant to produce.
    pub target_effect: f64,
    /// Expected share of dev programs a candidate and its champion differ on.
    pub discordance: f64,
    /// The retention loss, as a share of the retention set, a candidate may
    /// show; also the loss the retention set must be able to detect.
    pub retention_margin: f64,
    /// What a candidate is trained from.
    pub train_from: TrainFrom,
    /// Frontier selection; none keeps every program the champion mispredicts
    /// once, greedily.
    pub frontier: Option<Frontier>,
    /// With [`TrainFrom::BaseCumulative`]: the passes over the union a
    /// candidate's training makes. The steps are then
    /// `max(recipe steps, epochs x records / records per step)`, rounded up,
    /// so the training grows with the data.
    pub epochs: Option<f32>,
    /// How a candidate is accepted.
    pub acceptance: Acceptance,
    /// Which programs of the frontier's measurement are admitted.
    pub admit: Admit,
    /// Let the champion reason when it predicts and answers the exams, and
    /// train in thinking mode; none keeps every model answering without
    /// reasoning.
    pub reasoning: Option<Reasoning>,
    /// What the training records are made of.
    pub train_on: TrainOn,
}

impl Plan {
    /// The rounds the evidence gate's budget is planned over: every round
    /// of every generation.
    #[must_use]
    pub fn horizon(&self) -> u32 {
        self.generations * self.rounds
    }

    /// The rule that picks the programs a round trains on.
    #[must_use]
    pub fn rule(&self) -> Rule {
        Rule {
            admit: self.admit,
            require_trace: self.train_on == TrainOn::VerifiedTraces,
        }
    }

    /// Whether the candidate is trained in thinking mode, and its records
    /// measured so: when the lineage reasons.
    #[must_use]
    pub fn train_thinking(&self) -> bool {
        self.reasoning.is_some()
    }

    /// Refuses combinations whose meaning would change silently.
    ///
    /// # Errors
    /// Replay is asked beside a union that already contains it, programs the
    /// champion predicts right are asked beside a frontier that excludes them,
    /// or epochs are given for a continuation.
    pub fn validate(&self, recipe: &Recipe) -> Result<()> {
        anyhow::ensure!(
            self.train_from != TrainFrom::BaseCumulative || recipe.replay_share == 0.0,
            "--replay-share must be 0 with --train-from base-cumulative: the union contains every earlier record already"
        );
        anyhow::ensure!(
            self.frontier.is_none() || recipe.correct_share == 0.0,
            "--correct-share must be 0 with --frontier-samples: programs the champion always predicts right are not trained on"
        );
        anyhow::ensure!(
            self.epochs.is_none() || self.train_from == TrainFrom::BaseCumulative,
            "--cumulative-epochs applies to --train-from base-cumulative only"
        );
        anyhow::ensure!(
            self.epochs.is_none_or(|e| e.is_finite() && e > 0.0),
            "--cumulative-epochs is a positive number of passes"
        );
        anyhow::ensure!(
            self.reasoning.is_none() || self.frontier.is_some(),
            "--reasoning on needs --frontier-samples: reasoning traces are sampled, several per program"
        );
        anyhow::ensure!(
            !self.train_on.uses_traces() || self.reasoning.is_some(),
            "--train-on {} needs --reasoning on: there are no traces without it",
            self.train_on
                .to_possible_value()
                .map_or_else(String::new, |v| v.get_name().to_string())
        );
        anyhow::ensure!(
            self.admit == Admit::default() || self.frontier.is_some(),
            "--frontier-admit needs --frontier-samples"
        );
        if let Acceptance::Evidence(evidence) = self.acceptance {
            evidence
                .gate
                .validate()
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            anyhow::ensure!(
                evidence.draw > 0 && evidence.batch > 0,
                "--dev-draw and --dev-batch are positive numbers of dev items"
            );
        }
        Ok(())
    }

    /// The optimiser steps to train `records` records for.
    #[must_use]
    pub fn steps_for(&self, recipe: &Recipe, records: usize) -> u32 {
        match self.epochs {
            Some(epochs) => {
                let wanted = (f64::from(epochs) * records as f64
                    / f64::from(recipe.records_per_step))
                .ceil() as u32;
                wanted.max(recipe.steps)
            }
            None => recipe.steps,
        }
    }
}

/// The programs a candidate's training set is made of: the round's `wrong`
/// programs, and when `cumulative` every accepted program before them (once).
pub(crate) fn training_ids(cumulative: bool, accepted: &[String], wrong: &[String]) -> Vec<String> {
    if !cumulative {
        return wrong.to_vec();
    }
    let new: BTreeSet<&String> = wrong.iter().collect();
    accepted
        .iter()
        .filter(|id| !new.contains(id))
        .chain(wrong)
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recipe(steps: u32, replay: f32, correct: f32) -> Recipe {
        Recipe {
            steps,
            lr: 3e-5,
            rank: 16,
            records_per_step: 8,
            replay_share: replay,
            correct_share: correct,
            weight_decay: 0.0,
            max_block: 1024,
            base_precision: "bf16".into(),
            held_out_records: 8,
        }
    }

    fn plan(train_from: TrainFrom, frontier: bool, epochs: Option<f32>) -> Plan {
        Plan {
            generations: 3,
            rounds: 1,
            proposals_per_round: 600,
            budget: (800, 64),
            target_effect: 0.08,
            discordance: 0.2,
            retention_margin: 0.05,
            train_from,
            frontier: frontier.then_some(Frontier {
                samples: 4,
                temperature: 1.0,
                target: 300,
                request_cap: 1500,
            }),
            epochs,
            acceptance: Acceptance::Greedy,
            admit: Admit::default(),
            reasoning: None,
            train_on: TrainOn::Answers,
        }
    }

    #[test]
    fn options_whose_meaning_would_change_silently_are_refused() {
        let cumulative = plan(TrainFrom::BaseCumulative, true, Some(2.0));
        assert!(cumulative.validate(&recipe(40, 0.0, 0.0)).is_ok());
        assert!(
            cumulative.validate(&recipe(40, 0.25, 0.0)).is_err(),
            "replay beside a union that holds it"
        );
        assert!(
            cumulative.validate(&recipe(40, 0.0, 0.25)).is_err(),
            "right predictions beside a frontier that excludes them"
        );
        assert!(
            plan(TrainFrom::Previous, false, Some(2.0))
                .validate(&recipe(40, 0.25, 0.25))
                .is_err(),
            "epochs for a continuation"
        );
        assert!(plan(TrainFrom::Previous, false, None)
            .validate(&recipe(40, 0.25, 0.25))
            .is_ok());
        assert!(plan(TrainFrom::BaseCumulative, false, Some(0.0))
            .validate(&recipe(40, 0.0, 0.0))
            .is_err());
    }

    #[test]
    fn reasoning_settings_that_mean_nothing_without_each_other_are_refused() {
        let r = recipe(40, 0.0, 0.0);
        let reasoning = Reasoning {
            max_tokens: 2048,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            gate: crate::reasoning::GateDecoding::Greedy,
        };
        let mut p = plan(TrainFrom::Previous, true, None);
        assert!(p.validate(&r).is_ok());
        p.train_on = TrainOn::VerifiedTraces;
        assert!(p.validate(&r).is_err(), "traces without reasoning");
        p.reasoning = Some(reasoning);
        assert!(p.validate(&r).is_ok());
        assert!(p.train_thinking());
        assert_eq!(
            p.rule(),
            Rule {
                admit: Admit::AnyWrong,
                require_trace: true
            }
        );
        p.train_on = TrainOn::VerifiedTracesLabels;
        assert!(!p.rule().require_trace, "labels fill in for no trace");
        let mut greedy = plan(TrainFrom::Previous, false, None);
        greedy.reasoning = Some(reasoning);
        assert!(greedy.validate(&r).is_err(), "reasoning needs a frontier");
        greedy.reasoning = None;
        greedy.admit = Admit::Boundary;
        assert!(greedy.validate(&r).is_err(), "admission needs a frontier");
        assert!(!plan(TrainFrom::Previous, true, None).train_thinking());
    }

    #[test]
    fn an_evidence_gate_outside_its_ranges_or_with_an_empty_draw_is_refused() {
        let r = recipe(40, 0.25, 0.25);
        let mut p = plan(TrainFrom::Previous, false, None);
        let gate = EvidenceGate::new(0.02, 0.3, 0.01).unwrap();
        p.acceptance = Acceptance::Evidence(EvidencePlan {
            gate,
            draw: 100,
            batch: 32,
        });
        assert!(p.validate(&r).is_ok());
        assert_eq!(p.horizon(), 3);
        let mut bad = gate;
        bad.q = 1.5;
        p.acceptance = Acceptance::Evidence(EvidencePlan {
            gate: bad,
            draw: 100,
            batch: 32,
        });
        assert!(p.validate(&r).unwrap_err().to_string().contains("q is"));
        p.acceptance = Acceptance::Evidence(EvidencePlan {
            gate,
            draw: 0,
            batch: 32,
        });
        assert!(p.validate(&r).is_err());
    }

    #[test]
    fn the_steps_grow_with_the_union_but_never_fall_below_the_recipe() {
        let p = plan(TrainFrom::BaseCumulative, true, Some(2.0));
        let r = recipe(40, 0.0, 0.0);
        assert_eq!(
            p.steps_for(&r, 100),
            40,
            "2 x 100 / 8 = 25 is under the floor"
        );
        assert_eq!(p.steps_for(&r, 300), 75, "2 x 300 / 8");
        assert_eq!(p.steps_for(&r, 901), 226, "rounded up");
        assert_eq!(
            plan(TrainFrom::Previous, false, None).steps_for(&r, 900),
            40
        );
    }

    #[test]
    fn a_cumulative_candidate_trains_on_every_accepted_record_and_the_new_ones_once() {
        let accepted = vec!["a".to_string(), "b".to_string()];
        let wrong = vec!["b".to_string(), "c".to_string()];
        assert_eq!(training_ids(true, &accepted, &wrong), ["a", "b", "c"]);
        assert_eq!(training_ids(false, &accepted, &wrong), ["b", "c"]);
        assert_eq!(training_ids(true, &[], &wrong), ["b", "c"]);
    }
}
