// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that run model learning as one
// continuous, resumable campaign with dynamic sampling, replay and a
// validation gate with rollback, for its clients. If your team needs expertise
// in on-policy training loops for language-model agents you can procure our
// services by sending an email to info@swedishembedded.com.

//! Closing an arm: the dev gate, adopting or rolling back a candidate, the
//! accounting, the plateau and the collection of superseded checkpoints.

use std::collections::BTreeMap;

use splinter_core::digest::Digest;
use splinter_eval::campaign::gate::{decide, Action};
use splinter_eval::campaign::plateau::{assess, ArmCost, Trend};
use splinter_store::campaign::{Cost, Verdict};

use super::controller::{ArmId, ArmReport, CampaignController, Outcome, Spent};
use super::gc::{self, Checkpoint, Keep};
use super::records::RecordCounts;
use super::types::{CampaignError, PolicyRef};

impl CampaignController<'_> {
    /// Ends an arm that has no candidate to gate: counted, accounted, saved.
    pub(super) fn close_without_candidate(
        &mut self,
        id: &ArmId,
        verdict: Verdict,
        used: Spent,
        mut measurements: BTreeMap<String, f64>,
    ) -> Result<ArmReport, CampaignError> {
        if let Some(fraction) = used.informative_fraction {
            measurements.insert("informative_fraction".into(), fraction);
        }
        let halted = matches!(verdict, Verdict::Halted { .. });
        let entry = self.entry(
            id,
            Outcome {
                verdict: verdict.clone(),
                cost: Cost {
                    gpu_hours: Some(used.gpu_hours),
                    rollouts: Some(used.rollouts),
                    ..Cost::default()
                },
                discoveries: Some(0.0),
                measurements,
            },
        );
        self.finish_arm(&id.name, used.gpu_hours, 0.0, false)?;
        self.store.record(&entry)?;
        self.save()?;
        Ok(ArmReport {
            arm: id.name.clone(),
            generation: id.generation,
            strategy: id.strategy.name(),
            verdict,
            signals: None,
            decision: None,
            records: used.counts,
            findings: Vec::new(),
            gpu_hours: used.gpu_hours,
            stop: halted || self.exhausted(),
        })
    }

    /// Counts the arm, reads the plateau, collects superseded checkpoints.
    fn finish_arm(
        &mut self,
        arm: &str,
        gpu_hours: f64,
        discoveries: f64,
        advanced: bool,
    ) -> Result<(), CampaignError> {
        let state = &mut self.loaded.state;
        state.arms += 1;
        state.gpu_hours += gpu_hours;
        state.idle_arms = if advanced { 0 } else { state.idle_arms + 1 };
        state.history.push(ArmCost {
            arm: arm.to_string(),
            gpu_hours,
            discoveries,
        });
        if let Trend::Plateau { .. } = assess(&state.history, &self.config.plateau) {
            let last = self.config.ladder.len() - 1;
            state.strategy_index = (state.strategy_index + 1).min(last);
        }
        let champion = state.policy.digest.clone();
        let protected: Vec<&Digest> = champion.iter().collect();
        let keep = Keep {
            recent: self.config.keep_recent_checkpoints,
        };
        let doomed: Vec<Checkpoint> = gc::plan(&state.checkpoints, &protected, &keep)
            .into_iter()
            .cloned()
            .collect();
        let refs: Vec<&Checkpoint> = doomed.iter().collect();
        let dropped = gc::collect(&self.adapters, &refs)?;
        state
            .checkpoints
            .retain(|c| !doomed.iter().any(|d| d.digest == c.digest));
        state.dropped.extend(dropped);
        Ok(())
    }

    /// Whether the ladder is spent and arms keep failing to advance.
    pub(super) fn exhausted(&self) -> bool {
        let state = &self.loaded.state;
        state.strategy_index + 1 >= self.config.ladder.len()
            && state.idle_arms >= self.config.max_idle_arms
    }

    /// Runs the dev gate on the pending candidate, adopts or rolls it back,
    /// and accounts the arm.
    pub(super) fn gate_pending(&mut self) -> Result<ArmReport, CampaignError> {
        let Some(pending) = self.loaded.state.pending.clone() else {
            return Err(CampaignError::Config("no candidate is pending".into()));
        };
        let generation = self.loaded.state.arms + 1;
        let strategy = self.strategy();
        let began = self.parts.stopwatch.seconds();
        if self.loaded.state.champion_dev.is_none() {
            let champion = self.parts.validator.validate(&self.loaded.state.policy)?;
            self.loaded.state.champion_dev = Some(champion);
            self.save()?;
        }
        let candidate_policy = PolicyRef {
            version: self.loaded.state.policy.version + 1,
            adapter: Some(pending.adapter.clone()),
            digest: Some(pending.digest.clone()),
        };
        let candidate = self.parts.validator.validate(&candidate_policy)?;
        let champion = self.loaded.state.champion_dev.clone().unwrap_or_default();
        let decision = decide(
            &self.config.gate,
            generation,
            &champion,
            &candidate,
            self.parts.significance,
        );
        let advance = decision.action == Action::Advance;
        let verdict = if advance {
            Verdict::Accepted {
                significant: decision.significant,
            }
        } else {
            Verdict::RolledBack {
                reason: decision.reason.clone(),
            }
        };
        let gate_hours = (self.parts.stopwatch.seconds() - began).max(0.0)
            * f64::from(self.config.gpus)
            / 3600.0;
        let gpu_hours = pending.gpu_seconds / 3600.0 + gate_hours;
        let discoveries = if advance { decision.wins as f64 } else { 0.0 };
        let mut measurements = pending.measurements.clone();
        measurements.extend([
            ("dev_wins".to_string(), decision.wins as f64),
            ("dev_losses".to_string(), decision.losses as f64),
            ("dev_compared".to_string(), decision.compared as f64),
            ("dev_p_value".to_string(), decision.p_value),
            ("gate_alpha".to_string(), decision.alpha),
            (
                "retention_losses".to_string(),
                decision.retention_losses as f64,
            ),
        ]);
        if let Some(rate) = decision.candidate_rate {
            measurements.insert("dev_rate_candidate".into(), rate);
        }
        if let Some(rate) = decision.champion_rate {
            measurements.insert("dev_rate_champion".into(), rate);
        }
        {
            let state = &mut self.loaded.state;
            if decision.compared > 0 {
                state.last_discordance =
                    Some((decision.wins + decision.losses) as f64 / decision.compared as f64);
            }
            if advance {
                state.policy = candidate_policy;
                state.champion_dev = Some(candidate);
                state.replay.push(generation, pending.replay.clone());
            }
            state.pending = None;
        }
        if advance {
            self.parts.collector.set_policy(&self.loaded.state.policy)?;
        }
        let cost = Cost {
            gpu_hours: Some(gpu_hours),
            rollouts: Some(pending.rollouts),
            generated_tokens: pending.generated_tokens,
            train_steps: Some(u64::from(pending.report.steps)),
        };
        self.finish_arm(&pending.arm, gpu_hours, discoveries, advance)?;
        let id = ArmId {
            name: pending.arm.clone(),
            generation,
            strategy,
        };
        let entry = self.entry(
            &id,
            Outcome {
                verdict: verdict.clone(),
                cost,
                discoveries: Some(discoveries),
                measurements,
            },
        );
        self.store.record(&entry)?;
        self.save()?;
        Ok(ArmReport {
            arm: pending.arm,
            generation,
            strategy: strategy.name(),
            verdict,
            signals: Some(pending.signals),
            decision: Some(decision),
            records: RecordCounts::default(),
            findings: Vec::new(),
            gpu_hours,
            stop: self.exhausted(),
        })
    }
}
