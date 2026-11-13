// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that let a model teach itself from
// an interpreter's answers to its own proposals, gated on held-out evidence,
// for its clients. If your team needs expertise in self-taught model
// lineages judged by an independent examiner you can procure our services by
// sending an email to info@swedishembedded.com.

//! A frontier round: the champion proposes in waves and predicts every
//! program several times until its frontier has the target size (see
//! [`crate::frontier`] for the rule); each wave and each prediction is
//! journalled so a round that is started again continues.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use splinter_sdk::campaign::PolicyRef;
use splinter_sdk::measure::campaign::gate::DevResult;

use crate::frontier::{self, Frontier, Measured, Stage, Wave};
use crate::lineage::{fail, policy_key, Lineage, RoundRecord};
use crate::proposer::Budget;

/// What a round's search for its frontier came to.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FrontierOutcome {
    /// The settings it was found under.
    pub settings: Frontier,
    /// Proposals predicted.
    pub measured: usize,
    /// Of those, the ones on the frontier (more than were trained on when the
    /// last wave overshot the target).
    pub on_frontier: usize,
    /// Proposal requests spent, in all waves.
    pub requests: usize,
    /// Waves of proposing.
    pub waves: usize,
}

/// One wave of proposing, journalled so a round that is started again counts
/// what it already spent against its cap.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct WaveRecord {
    generation: u32,
    round: u32,
    calls: usize,
    added: usize,
}

/// One proposal's sampled predictions by a policy.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct MeasuredRecord {
    generation: u32,
    round: u32,
    policy: String,
    temperature: f32,
    #[serde(flatten)]
    measured: Measured,
}

impl Lineage<'_> {
    /// The champion proposes in waves, and predicts every program of them
    /// several times, sampled, until the programs it does not always predict
    /// right number the frontier's target; those are the round's training
    /// programs and nothing is trained on the others.
    pub(crate) fn select_frontier(
        &mut self,
        champion: &PolicyRef,
        champion_dev: &DevResult,
        record: &mut RoundRecord,
        settings: Frontier,
    ) -> Result<(Vec<String>, Vec<String>)> {
        let (generation, round) = (record.generation, record.round);
        let key = policy_key(champion);
        let waves: Vec<WaveRecord> = self
            .files
            .waves()
            .read()?
            .into_iter()
            .filter(|w: &WaveRecord| w.generation == generation && w.round == round)
            .collect();
        let held: Vec<Measured> = self
            .files
            .frontier()
            .read()?
            .into_iter()
            .filter(|m: &MeasuredRecord| {
                m.generation == generation && m.round == round && m.policy == key
            })
            .map(|m| m.measured)
            .collect();
        let spent: usize = waves.iter().map(|w| w.calls).sum();
        let mut stage = Round {
            lineage: self,
            champion,
            champion_dev,
            generation,
            round,
            settings,
            wave: waves.len(),
            key,
        };
        // Programs a stopped run proposed but did not get to predict.
        let known: BTreeSet<&str> = held.iter().map(|m| m.task.as_str()).collect();
        let unmeasured: Vec<String> = stage
            .lineage
            .round_proposals(generation, round)
            .into_iter()
            .filter(|id| !known.contains(id.as_str()))
            .collect();
        let mut measured = held.clone();
        if !unmeasured.is_empty() {
            measured.extend(stage.measure(&unmeasured)?);
        }
        let filled = frontier::fill(
            &mut stage,
            settings.target,
            settings.request_cap,
            spent,
            measured,
        )?;
        let selected = filled.selected(settings.target);
        eprintln!(
            "generation {generation} round {round}: {} of {} proposals on the frontier after {} requests; training on {}",
            filled.on_frontier(),
            filled.measured.len(),
            filled.calls,
            selected.len()
        );
        record.proposed = filled.measured.len();
        record.frontier = Some(FrontierOutcome {
            settings,
            measured: filled.measured.len(),
            on_frontier: filled.on_frontier(),
            requests: filled.calls,
            waves: stage.wave,
        });
        Ok((selected, Vec::new()))
    }
}

/// A frontier round's view of the lineage: what [`frontier::fill`] asks.
struct Round<'l, 'a> {
    lineage: &'l mut Lineage<'a>,
    champion: &'l PolicyRef,
    champion_dev: &'l DevResult,
    generation: u32,
    round: u32,
    settings: Frontier,
    /// Waves of this round so far, those of earlier runs included.
    wave: usize,
    key: String,
}

impl Stage for Round<'_, '_> {
    fn propose(&mut self, calls_left: usize) -> Result<Wave> {
        let plan = self.lineage.plan;
        let (made, fresh) = self.lineage.propose_wave(
            self.champion,
            self.champion_dev,
            (self.generation, self.round, self.wave),
            Budget {
                want: plan.proposals_per_round,
                max_calls: plan.budget.0.min(calls_left),
                parallel: plan.budget.1,
            },
        )?;
        self.wave += 1;
        self.lineage.files.waves().append(&WaveRecord {
            generation: self.generation,
            round: self.round,
            calls: made.calls,
            added: fresh.len(),
        })?;
        Ok(Wave {
            calls: made.calls,
            fresh,
        })
    }

    fn measure(&mut self, ids: &[String]) -> Result<Vec<Measured>> {
        let samples = self.settings.samples;
        self.lineage.host.context(self.champion)?;
        let model = self.lineage.host.predicting(self.champion).map_err(fail)?;
        let asked: Vec<String> = ids
            .iter()
            .flat_map(|id| std::iter::repeat_n(id.clone(), samples as usize))
            .collect();
        let answers = self
            .lineage
            .validator
            .answer_with(&model, &asked)
            .map_err(fail)?;
        let mut correct: BTreeMap<&str, u32> = BTreeMap::new();
        for answer in &answers {
            *correct.entry(answer.task.as_str()).or_default() += u32::from(answer.passed);
        }
        let measured: Vec<Measured> = ids
            .iter()
            .map(|id| Measured {
                task: id.clone(),
                samples,
                correct: correct.get(id.as_str()).copied().unwrap_or(0),
            })
            .collect();
        for m in &measured {
            self.lineage.files.frontier().append(&MeasuredRecord {
                generation: self.generation,
                round: self.round,
                policy: self.key.clone(),
                temperature: self.settings.temperature,
                measured: m.clone(),
            })?;
        }
        Ok(measured)
    }
}
