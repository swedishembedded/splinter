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
//! journalled so a round that is started again continues. Before training,
//! each chosen program's record is measured as the trainer will measure it,
//! and one longer than the training row gives its place to the next frontier
//! program that fits.

use std::collections::BTreeSet;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use splinter_sdk::campaign::PolicyRef;
use splinter_sdk::measure::campaign::gate::DevResult;
use splinter_sdk::model::measure::RecordMeter;

use crate::frontier::{self, Admit, Frontier, Measured, Overlong, Stage, Wave};
use crate::lineage::{fail, policy_key, Lineage, RoundRecord, TrainOn};
use crate::proposer::Budget;
use crate::reasoning::{fold, RecordMix, TraceRecord};
use crate::sft::{chat_line, TRAIN_PACKED};

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
    /// Frontier programs skipped because their records are longer than the
    /// training row (absent when none was).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub overlong: Vec<Overlong>,
    /// Which programs were admitted (absent for the default, any wrong
    /// prediction).
    #[serde(default, skip_serializing_if = "Admit::is_default")]
    pub admit: Admit,
    /// What the training records are made of, when they carry reasoning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub records: Option<RecordMix>,
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
        let rule = self.plan.rule();
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
            rule,
            settings.target,
            settings.request_cap,
            spent,
            measured,
        )?;
        let meter = RecordMeter::for_model(
            stage.lineage.base,
            stage.lineage.plan.train_thinking(),
            TRAIN_PACKED,
        )
        .map_err(fail)?;
        let book = stage.lineage.trace_book()?;
        let max_block = stage.lineage.recipe.max_block as usize;
        let selection = filled.selected(settings.target, max_block, |task| {
            let line = chat_line(&stage.lineage.training_example(task, &book)?);
            meter
                .tokens(&line)
                .map_err(|e| anyhow::anyhow!("measuring {task}: {e}"))
        })?;
        for skipped in &selection.overlong {
            eprintln!(
                "generation {generation} round {round}: {} is {} tokens, longer than max_block {max_block}: skipped",
                skipped.task, skipped.tokens
            );
        }
        let selected = selection.tasks;
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
            overlong: selection.overlong,
            admit: stage.lineage.plan.admit,
            records: (stage.lineage.plan.train_on != TrainOn::Answers)
                .then(|| RecordMix::of(&book, &selected)),
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
        let folded = fold(
            ids,
            samples,
            &answers,
            self.lineage.plan.reasoning.is_some(),
        );
        // The traces are journalled before the measurements: a round that
        // stops between the two asks the programs again, and the later trace
        // record stands.
        for (task, traces) in folded.traces {
            self.lineage.files.traces().append(&TraceRecord {
                generation: self.generation,
                round: self.round,
                policy: self.key.clone(),
                task,
                traces,
            })?;
        }
        let measured = folded.measured;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontier::Frontier;

    /// An outcome of a lineage with the options off journals the lines it
    /// always did.
    #[test]
    fn a_default_outcome_journals_without_the_new_fields() {
        let outcome = FrontierOutcome {
            settings: Frontier {
                samples: 4,
                temperature: 1.0,
                target: 3,
                request_cap: 9,
            },
            measured: 5,
            on_frontier: 4,
            requests: 2,
            waves: 1,
            overlong: Vec::new(),
            admit: Admit::default(),
            records: None,
        };
        assert_eq!(
            serde_json::to_string(&outcome).unwrap(),
            r#"{"settings":{"samples":4,"temperature":1.0,"target":3,"request_cap":9},"measured":5,"on_frontier":4,"requests":2,"waves":1}"#
        );
    }

    #[test]
    fn an_outcome_states_the_admission_rule_and_the_records_when_they_are_not_the_default() {
        let mut outcome: FrontierOutcome = serde_json::from_str(
            r#"{"settings":{"samples":4,"temperature":1.0,"target":3,"request_cap":9},"measured":5,"on_frontier":4,"requests":2,"waves":1}"#,
        )
        .unwrap();
        outcome.admit = Admit::Boundary;
        outcome.records = Some(RecordMix {
            traces: 2,
            labels: 1,
        });
        let line = serde_json::to_string(&outcome).unwrap();
        assert!(line.contains(r#""admit":"boundary""#), "{line}");
        assert!(
            line.contains(r#""records":{"traces":2,"labels":1}"#),
            "{line}"
        );
    }
}
